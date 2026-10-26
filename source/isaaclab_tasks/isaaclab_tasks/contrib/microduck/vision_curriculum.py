# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent held-out contracts for learning gaze before RGB-only volleyball.

The first lessons require measured visual acquisition, a legal forehead hit,
then a physical target crossing. They do not reuse teacher-policy graduation
evidence. Every lesson also requires both robots to finish grounded recovery;
the final lessons retain the actual alternating 2/4/8-return requirements.
"""

from dataclasses import dataclass

from .volley_curriculum import CurriculumState, StageSpec
from .volley_curriculum import EvaluationReport as TeacherEvaluationReport


@dataclass(frozen=True)
class VisionStageSpec(StageSpec):
    """A reset distribution and the measured objective required before recovery.

    ``watch_visible_ball`` requires sustained, fresh true-ball pixel evidence;
    prediction validity or a high predicted ballistic score cannot satisfy it.
    ``receive_forehead`` requires a legal contact on the front head patch.
    ``aim_return`` requires an observed crossing of the coach's target disk.
    ``rally`` requires the specified count of actual alternating returns.
    Source height is measured above the incoming ball's arrival point; source
    distance adds a frontal offset along the receiver-to-partner direction.
    """

    objective: str
    source_height_min: float
    source_height_max: float
    source_distance: float


VISION_STAGES = (
    VisionStageSpec(
        "watch_visible_ball", (1.0, 1.0), (0.45, 0.50), 0.005, 0, False, "watch_visible_ball", 0.005, 0.015, 0.25
    ),
    VisionStageSpec(
        "receive_forehead", (1.0, 1.0), (0.35, 0.50), 0.005, 0, False, "receive_forehead", 0.06, 0.12, 0.25
    ),
    VisionStageSpec("aim_return", (0.7, 1.0), (0.35, 0.45), 0.01, 0, False, "aim_return", 0.12, 0.18, 0.45),
    VisionStageSpec("pair_2_returns", (0.7, 1.0), (0.35, 0.45), 0.02, 2, True, "rally", 0.18, 0.25, 0.50),
    VisionStageSpec("pair_4_returns", (0.7, 1.0), (0.35, 0.45), 0.02, 4, True, "rally", 0.18, 0.25, 0.50),
    VisionStageSpec("pair_8_returns", (0.7, 1.0), (0.35, 0.45), 0.02, 8, True, "rally", 0.18, 0.25, 0.50),
)


@dataclass(frozen=True)
class VisionEvaluationReport(TeacherEvaluationReport):
    """Held-out episode counts that explicitly identify the measured visual objective."""

    objective: str = ""


class VisionCurriculumState(CurriculumState):
    """Use the same balanced safety gate with an independent visual lesson contract.

    Stage names, reset ranges and objectives are serialized with each saved
    checkpoint. Resuming with teacher stages or changed visual objectives fails
    before any progress is restored. Training iterations never advance a stage.
    """

    stages = VISION_STAGES

    def _validate_report(self, report: TeacherEvaluationReport):
        super()._validate_report(report)
        if not isinstance(report, VisionEvaluationReport) or report.objective != self.spec.objective:
            raise ValueError("Visual evaluation objective must match the active visual curriculum stage.")
