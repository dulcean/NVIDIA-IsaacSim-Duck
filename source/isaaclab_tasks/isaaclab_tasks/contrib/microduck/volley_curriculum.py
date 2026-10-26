# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-out evaluation gates for the MicroDuck receive, recover, and rally curriculum.

This module has no simulator dependency. Training episodes never advance the
curriculum: the training driver supplies a separately measured deterministic
evaluation report, then persists this state alongside the selected checkpoint.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    """A receiving distribution and its required unassisted rally length.

    Lateral fraction zero is a vertical serve above the receiving forehead;
    fraction one starts the serve in the partner's direction. Flight time is
    the reset-to-receive interval, in seconds. Spread is a target XY radius in
    meters. Coach stages require an actual target crossing and recovery, not a
    predicted ballistic score. Pair stages require actual alternating returns.
    """

    name: str
    lateral_range: tuple[float, float]
    flight_time_range: tuple[float, float]
    serve_spread: float
    required_returns: int
    paired: bool


STAGES = (
    StageSpec("receive_vertical", (0.0, 0.0), (0.25, 0.32), 0.005, 0, False),
    StageSpec("receive_lateral", (1.0, 1.0), (0.25, 0.35), 0.01, 0, False),
    StageSpec("receive_varied", (0.7, 1.0), (0.25, 0.40), 0.02, 0, False),
    StageSpec("pair_2_returns", (0.7, 1.0), (0.25, 0.40), 0.02, 2, True),
    StageSpec("pair_4_returns", (0.7, 1.0), (0.25, 0.40), 0.02, 4, True),
    StageSpec("pair_8_returns", (0.7, 1.0), (0.25, 0.40), 0.02, 8, True),
)


@dataclass(frozen=True)
class EvaluationReport:
    """Counts from one fresh held-out rollout of a single curriculum stage.

    A side denotes the initial receiving robot. Success includes the complete
    stage objective and recovery of both robots, and excludes safety failures.
    Unsafe counts are episode counts of the union of physical safety failures,
    rather than a sum of overlapping reason flags. The evaluator must use the
    stage's required return count, including unassisted contacts in pair stages.
    """

    evaluation_id: int
    stage: int
    required_returns: int
    attempts_by_side: tuple[int, int]
    successes_by_side: tuple[int, int]
    unsafe_episodes: int
    deterministic: bool = True
    split: str = "heldout"
    complete: bool = True


@dataclass(frozen=True)
class CurriculumDecision:
    """Result of an evaluation; no decision silently lowers the difficulty."""

    status: str
    reason: str
    stage: int
    complete: bool


class CurriculumState:
    """Advance after repeated balanced, safe held-out evaluations.

    A strong result on only one receiver, too few episodes, training rewards,
    repeated delivery of the same report, or easier return requirements cannot
    advance a stage. Failed evaluations hold the current stage and reset its
    pass streak. Repeated failures produce ``stalled`` for the driver to report
    or recover a saved best checkpoint; they never promote or demote a stage.
    """

    def __init__(
        self,
        stage: int = 0,
        consecutive_passes: int = 0,
        min_attempts_per_side: int = 64,
        success_threshold: float = 0.8,
        max_unsafe_rate: float = 0.05,
        passes_required: int = 2,
        stall_evaluations: int = 4,
    ):
        if type(stage) is not int or not 0 <= stage < len(STAGES):
            raise ValueError("Curriculum stage is outside the supported range.")
        if any(
            type(value) is not int or value < 1 for value in (min_attempts_per_side, passes_required, stall_evaluations)
        ):
            raise ValueError("Curriculum sample, pass, and stall requirements must be positive integers.")
        if not 0 < success_threshold <= 1 or not 0 <= max_unsafe_rate < 1:
            raise ValueError("Curriculum success and unsafe thresholds must be valid rates.")
        if type(consecutive_passes) is not int or not 0 <= consecutive_passes < passes_required:
            raise ValueError("Initial pass streak must be smaller than the advancement requirement.")
        self.stage = stage
        self.consecutive_passes = consecutive_passes
        self.min_attempts_per_side = min_attempts_per_side
        self.success_threshold = success_threshold
        self.max_unsafe_rate = max_unsafe_rate
        self.passes_required = passes_required
        self.stall_evaluations = stall_evaluations
        self.failed_evaluations = 0
        self.last_evaluation_id = -1
        self.complete = False

    @property
    def spec(self) -> StageSpec:
        """The scene/reset contract for the current stage."""
        return STAGES[self.stage]

    def update_evaluation(self, report: EvaluationReport) -> CurriculumDecision:
        """Validate fresh counts and update the held-out pass streak once."""
        self._validate_report(report)
        self.last_evaluation_id = report.evaluation_id
        if self.complete:
            return CurriculumDecision("complete", "The final stage already passed its held-out gate.", self.stage, True)
        if not report.complete or min(report.attempts_by_side) < self.min_attempts_per_side:
            self.consecutive_passes = 0
            return CurriculumDecision(
                "insufficient", "Both receiving sides need the minimum held-out episode count.", self.stage, False
            )
        side_rates = [
            success / attempts for success, attempts in zip(report.successes_by_side, report.attempts_by_side)
        ]
        unsafe_rate = report.unsafe_episodes / sum(report.attempts_by_side)
        passed = min(side_rates) >= self.success_threshold and unsafe_rate <= self.max_unsafe_rate
        if not passed:
            self.consecutive_passes = 0
            self.failed_evaluations += 1
            status = "stalled" if self.failed_evaluations >= self.stall_evaluations else "hold"
            reason = (
                f"Side success rates {side_rates[0]:.3f}/{side_rates[1]:.3f}; unsafe episode rate {unsafe_rate:.3f}."
            )
            return CurriculumDecision(status, reason, self.stage, False)
        self.failed_evaluations = 0
        self.consecutive_passes += 1
        if self.consecutive_passes < self.passes_required:
            return CurriculumDecision("passed", "Another independent held-out pass is required.", self.stage, False)
        self.consecutive_passes = 0
        if self.stage == len(STAGES) - 1:
            self.complete = True
            return CurriculumDecision(
                "complete", "The final unassisted rally and recovery gate passed.", self.stage, True
            )
        self.stage += 1
        return CurriculumDecision("advanced", "Repeated held-out passes unlocked the next stage.", self.stage, False)

    def _validate_report(self, report: EvaluationReport):
        """Reject malformed or easier evaluations before modifying persisted progress."""
        if not isinstance(report, EvaluationReport):
            raise TypeError("Expected an EvaluationReport from the deterministic evaluator.")
        if not report.deterministic or report.split != "heldout":
            raise ValueError("Only deterministic held-out evaluations can advance the curriculum.")
        if type(report.complete) is not bool:
            raise ValueError("Evaluation completion must be an explicit boolean.")
        if type(report.evaluation_id) is not int or report.evaluation_id <= self.last_evaluation_id:
            raise ValueError("Evaluation IDs must be fresh and strictly increasing.")
        if report.stage != self.stage or report.required_returns != self.spec.required_returns:
            raise ValueError("Evaluation stage and return requirement must match the active curriculum stage.")
        if len(report.attempts_by_side) != 2 or len(report.successes_by_side) != 2:
            raise ValueError("Evaluation counts must include exactly two receiving sides.")
        counts = (*report.attempts_by_side, *report.successes_by_side, report.unsafe_episodes)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Evaluation counts must be non-negative integers.")
        if any(success > attempts for success, attempts in zip(report.successes_by_side, report.attempts_by_side)):
            raise ValueError("Successful episodes cannot exceed attempts on either side.")
        if report.unsafe_episodes + sum(report.successes_by_side) > sum(report.attempts_by_side):
            raise ValueError("A successful episode cannot also count as unsafe.")

    def _gate_settings(self) -> dict[str, Any]:
        return {
            "min_attempts_per_side": self.min_attempts_per_side,
            "success_threshold": self.success_threshold,
            "max_unsafe_rate": self.max_unsafe_rate,
            "passes_required": self.passes_required,
            "stall_evaluations": self.stall_evaluations,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize progress with its gate and stage contracts for safe resumption."""
        return {
            "version": 1,
            "stages": [asdict(spec) for spec in STAGES],
            "gate": self._gate_settings(),
            "stage": self.stage,
            "consecutive_passes": self.consecutive_passes,
            "failed_evaluations": self.failed_evaluations,
            "last_evaluation_id": self.last_evaluation_id,
            "complete": self.complete,
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Restore progress, refusing silently changed task definitions or gates."""
        if state.get("version") != 1 or state.get("gate") != self._gate_settings():
            raise ValueError("Saved curriculum version or evaluation gates do not match.")
        # JSON converts tuple-valued ranges to lists, so compare a normalized form.
        saved_stages = state.get("stages", [])
        if len(saved_stages) != len(STAGES):
            raise ValueError("Saved curriculum stage definitions do not match.")
        for saved, expected in zip(saved_stages, STAGES):
            candidate = dict(saved)
            for key in ("lateral_range", "flight_time_range"):
                candidate[key] = tuple(candidate.get(key, ()))
            if candidate != asdict(expected):
                raise ValueError("Saved curriculum stage definitions do not match.")
        candidate = CurriculumState(
            stage=state["stage"], consecutive_passes=state["consecutive_passes"], **self._gate_settings()
        )
        for key, minimum in (("failed_evaluations", 0), ("last_evaluation_id", -1)):
            value = state[key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid saved curriculum field: {key}.")
            setattr(candidate, key, value)
        if type(state["complete"]) is not bool or (state["complete"] and candidate.stage != len(STAGES) - 1):
            raise ValueError("Only the final curriculum stage may be marked complete.")
        candidate.complete = state["complete"]
        self.__dict__.update(candidate.__dict__)
