# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts preventing false advancement of the volleyball curriculum."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/volley_curriculum.py"
)
_SPEC = importlib.util.spec_from_file_location("microduck_volley_curriculum", _PATH)
curriculum = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = curriculum
_SPEC.loader.exec_module(curriculum)


def report(evaluation_id, stage=0, successes=(60, 60), unsafe=0, attempts=(64, 64)):
    """Build independent held-out episode counts, never task rewards."""
    return curriculum.EvaluationReport(
        evaluation_id=evaluation_id,
        stage=stage,
        required_returns=curriculum.STAGES[stage].required_returns,
        attempts_by_side=attempts,
        successes_by_side=successes,
        unsafe_episodes=unsafe,
    )


def test_gate_requires_both_sides_safety_and_consecutive_passes():
    """An excellent server, unsafe episodes, or an interrupted pass streak cannot unlock harder serves."""
    state = curriculum.CurriculumState()
    assert state.update_evaluation(report(0)).status == "passed"
    # Aggregate success is above 80%, but the less capable receiver remains below its own threshold.
    assert state.update_evaluation(report(1, successes=(64, 40))).status == "hold"
    assert state.stage == 0 and state.consecutive_passes == 0
    assert state.update_evaluation(report(2, successes=(52, 52), unsafe=7)).status == "hold"
    assert state.update_evaluation(report(3)).status == "passed"
    assert state.update_evaluation(report(4, attempts=(63, 64), successes=(60, 60))).status == "insufficient"
    assert state.consecutive_passes == 0
    assert state.update_evaluation(report(5)).status == "passed"
    decision = state.update_evaluation(report(6))
    assert decision.status == "advanced" and decision.stage == 1 and not decision.complete


def test_partial_large_cohort_cannot_advance_even_after_minimum_counts_finish():
    """The early-finishing subset of a larger requested batch cannot count as a completed evaluation."""
    state = curriculum.CurriculumState()
    partial = replace(report(0), complete=False)
    assert state.update_evaluation(partial).status == "insufficient"
    assert state.consecutive_passes == 0


def test_repeated_failure_reports_stall_without_automatic_promotion_or_regression():
    """A limited training budget can detect lack of progress instead of graduating by iteration count."""
    state = curriculum.CurriculumState(stage=2)
    for index in range(4):
        decision = state.update_evaluation(report(index, stage=2, successes=(0, 0)))
    assert decision.status == "stalled" and state.stage == 2 and not state.complete
    decision = state.update_evaluation(report(4, stage=2))
    assert decision.status == "passed" and state.failed_evaluations == 0


def test_final_completion_requires_repeated_eight_return_evaluations():
    """Two-return or coach measurements cannot certify the final physical rally stage."""
    state = curriculum.CurriculumState(stage=5)
    with pytest.raises(ValueError, match="return requirement"):
        state.update_evaluation(replace(report(0, stage=5), required_returns=2))
    assert state.update_evaluation(report(0, stage=5)).status == "passed"
    decision = state.update_evaluation(report(1, stage=5))
    assert decision.status == "complete" and decision.complete and state.stage == 5
    assert state.update_evaluation(report(2, stage=5)).complete


@pytest.mark.parametrize(
    "changes",
    [
        {"deterministic": False},
        {"split": "training"},
        {"stage": 1},
        {"successes_by_side": (65, 0)},
        {"unsafe_episodes": 9},  # 120 successes + 9 unsafe cannot fit 128 distinct episodes.
        {"attempts_by_side": (True, 64)},
    ],
)
def test_invalid_evaluation_cannot_modify_persisted_progress(changes):
    """Wrong distributions, overlapping success/safety counts, and malformed reports fail closed."""
    state = curriculum.CurriculumState()
    before = state.state_dict()
    with pytest.raises(ValueError):
        state.update_evaluation(replace(report(0), **changes))
    assert state.state_dict() == before


def test_json_resume_preserves_pass_streak_and_rejects_duplicate_report():
    """Restarting the driver preserves evidence and cannot count the same evaluation twice."""
    state = curriculum.CurriculumState()
    state.update_evaluation(report(21))
    restored = curriculum.CurriculumState()
    restored.load_state_dict(json.loads(json.dumps(state.state_dict())))
    with pytest.raises(ValueError, match="fresh"):
        restored.update_evaluation(report(21))
    assert restored.stage == 0
    assert restored.update_evaluation(report(22)).stage == 1


def test_resume_refuses_changed_gate_and_stage_contract_atomically():
    """A changed success threshold or serve definition cannot silently reuse prior graduation evidence."""
    state = curriculum.CurriculumState()
    state.update_evaluation(report(0))
    saved = state.state_dict()
    changed_gate = curriculum.CurriculumState(success_threshold=0.7)
    with pytest.raises(ValueError, match="gates"):
        changed_gate.load_state_dict(saved)
    malformed = json.loads(json.dumps(saved))
    malformed["stages"][0]["serve_spread"] = 0.0
    before = state.state_dict()
    with pytest.raises(ValueError, match="stage definitions"):
        state.load_state_dict(malformed)
    assert state.state_dict() == before
