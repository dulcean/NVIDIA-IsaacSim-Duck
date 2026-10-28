# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU evaluation-collection contracts independent of simulation physics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "evaluate_microduck_volley.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_microduck_volley", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def episode(world, *, success=True, unsafe=False, episode_id=1, stage=0):
    return {
        "env_id": world,
        "episode_id": episode_id,
        "server": world % 2,
        "success": success,
        "unsafe": unsafe,
        "returns": 0,
        "stage": stage,
        "reason": "success" if success else "unsafe" if unsafe else "miss",
    }


class EpisodeSource:
    """Emit predefined asynchronous episode completions at the collector boundary."""

    def __init__(self, num_envs, schedule, profile=None):
        self.num_envs = num_envs
        self.schedule = schedule
        self.unwrapped = self
        self.cfg = SimpleNamespace(evaluation=True, curriculum_stage=0)
        if profile is not None:
            self.cfg.curriculum_profile = profile
        self.evaluation = True
        self.recording = False
        self.extras = {}
        self.resets = 0

    def reset(self):
        self.tick = 0
        self.resets += 1
        self.extras.clear()
        return None, self.extras

    def step(self, _actions):
        self.tick += 1
        if self.tick in self.schedule:
            self.extras["episode_records"] = self.schedule[self.tick]
        return None, None, [False] * self.num_envs, self.extras


class DeterministicPolicy:
    def __init__(self):
        self.training = True

    def __call__(self, _observations):
        assert not self.training

    def eval(self):
        self.training = False

    def train(self, mode):
        self.training = mode

    def reset(self, _dones=None):
        pass


def test_fixed_cohort_waits_for_slow_failures_and_ignores_auto_reset_followups():
    """Fast successful worlds cannot replace slow unsafe worlds with extra short episodes."""
    good = {
        "goal_reached": True,
        "head_hits": 1,
        "head_contacts": 1,
        "recovered_miss": False,
        "recovery_stable_by_player": [True, True],
        "recovery_hold_s": 0.26,
        "unsafe_reasons": [],
        "miss_reasons": [],
    }
    bad = {
        "goal_reached": False,
        "head_hits": 0,
        "head_contacts": 1,
        "recovered_miss": False,
        "recovery_stable_by_player": [True, False],
        "recovery_hold_s": 0.0,
        "unsafe_reasons": ["tilt"],
        "miss_reasons": ["invalid_head_surface"],
    }
    source = EpisodeSource(
        4,
        {
            1: [{**episode(0), **good}, {**episode(2), **good}],
            2: [episode(0, episode_id=2), episode(2, episode_id=2)],
            4: [
                {**episode(1, success=False, unsafe=True), **bad},
                {**episode(3, success=False, unsafe=True), **bad},
            ],
        },
    )
    policy = DeterministicPolicy()
    report, metrics = evaluation.evaluate_policy(
        source, policy, stage=0, evaluation_id=0, episodes_per_side=3, max_steps=10
    )
    assert report.attempts_by_side == (3, 3)
    assert report.successes_by_side == (3, 0)
    assert report.unsafe_episodes == 3
    assert metrics["success_rate_by_side"] == [1.0, 0.0]
    assert metrics["complete"] and not metrics["gate_passed"]
    assert metrics["steps"] == 8 and source.resets == 2
    assert metrics["diagnostic_episodes"] == 6
    assert metrics["head_hit_rate"] == metrics["mean_head_hits"] == 0.5
    assert metrics["mean_head_contacts"] == 1.0
    assert metrics["goal_reached_rate"] == 0.5
    assert metrics["recovery_stable_rate_by_player"] == [1.0, 0.5]
    assert metrics["recovery_stable_rate_by_role"] == {"receiver": 0.5, "partner": 1.0}
    assert metrics["mean_recovery_hold_s"] == pytest.approx(0.13)
    assert metrics["recovered_miss_rate"] == 0.0
    assert metrics["unsafe_reasons"] == {"tilt": 3}
    assert metrics["miss_reasons"] == {"invalid_head_surface": 3}
    assert policy.training and not source.recording


def test_step_limit_does_not_count_stale_records_or_certify_a_partial_cohort():
    """One completed side cannot graduate, even when its last extras dictionary is returned repeatedly."""
    source = EpisodeSource(2, {1: [episode(0)]})
    report, metrics = evaluation.evaluate_policy(
        source, DeterministicPolicy(), stage=0, evaluation_id=0, episodes_per_side=1, max_steps=3
    )
    assert report.attempts_by_side == (1, 0)
    assert not report.complete and metrics["max_steps_reached"]
    assert not metrics["gate_passed"] and metrics["completed_episodes"] == 1
    state = evaluation._curriculum_module().CurriculumState(min_attempts_per_side=1)
    assert state.update_evaluation(report).status == "insufficient"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage": 1}, "different stage"),
        ({"head_hits": True}, "integer counts"),
        ({"recovery_hold_s": float("nan")}, "finite and non-negative"),
        ({"recovery_stable_by_player": [True, 1]}, "boolean for each player"),
    ],
)
def test_malformed_outcome_fails_closed_and_restores_policy_mode(overrides, message):
    """Bad physical metadata cannot certify a lesson or publish misleading diagnostics."""
    source = EpisodeSource(2, {1: [{**episode(0), **overrides}]})
    policy = DeterministicPolicy()
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate_policy(source, policy, stage=0, evaluation_id=0, episodes_per_side=1)
    assert policy.training and not source.recording


@pytest.mark.parametrize(
    ("stage", "objective"),
    [(0, "watch_visible_ball"), (1, "receive_forehead"), (2, "aim_return")],
)
def test_visual_evaluation_identifies_its_actual_objective_despite_equal_zero_return_counts(stage, objective):
    """Visual acquisition, legal contact, and a measured return cannot exchange graduation evidence."""
    source = EpisodeSource(
        2,
        {1: [{**episode(side, stage=stage), "objective": objective} for side in (0, 1)]},
        profile=evaluation.VISION_PROFILE,
    )
    source.cfg.curriculum_stage = stage
    report, metrics = evaluation.evaluate_policy(
        source, DeterministicPolicy(), stage=stage, evaluation_id=0, episodes_per_side=1
    )
    contract = evaluation._curriculum_module(evaluation.VISION_PROFILE)
    assert isinstance(report, contract.VisionEvaluationReport)
    assert report.objective == metrics["objective"] == objective
    assert metrics["profile"] == evaluation.VISION_PROFILE and report.complete
    state = contract.VisionCurriculumState(stage=stage, min_attempts_per_side=1)
    assert state.update_evaluation(report).status == "passed"


@pytest.mark.parametrize("objective", [None, "aim_return"])
def test_watch_evaluation_rejects_teacher_or_other_visual_objective_records(objective):
    """A teacher success or a different zero-return visual lesson cannot masquerade as RGB acquisition."""
    record = episode(0)
    if objective is not None:
        record["objective"] = objective
    source = EpisodeSource(2, {1: [record]}, profile=evaluation.VISION_PROFILE)
    with pytest.raises(ValueError, match="objective"):
        evaluation.evaluate_policy(source, DeterministicPolicy(), stage=0, evaluation_id=0, episodes_per_side=1)


def test_partial_visual_acquisition_improves_checkpoint_selection_without_graduating():
    """Growing real exposure windows stay observable before the complete watch-and-recover objective succeeds."""
    from scripts.tools import train_microduck_volley as driver

    records = [
        {
            **episode(side, success=False),
            "objective": "watch_visible_ball",
            "goal_reached": False,
            "gaze_visible_a": 0.6 if side == 0 else 0.8,
            "gaze_visible_b": 0.9 if side == 0 else 0.4,
            "gaze_alignment": 0.7,
            "gaze_reward": 0.02,
            "watch_visible_s": 0.12 if side == 0 else 0.09,
        }
        for side in (0, 1)
    ]
    source = EpisodeSource(2, {1: records}, profile=evaluation.VISION_PROFILE)
    source.cfg.watch_duration = 0.30
    contract = evaluation._curriculum_module(evaluation.VISION_PROFILE)
    state = contract.VisionCurriculumState()
    report, metrics = evaluation.evaluate_policy(
        source, DeterministicPolicy(), stage=0, evaluation_id=0, episodes_per_side=64
    )
    assert metrics["watch_fraction_by_server"] == pytest.approx([0.4, 0.3])
    assert metrics["gaze_visible_by_role"] == pytest.approx({"receiver": 0.5, "partner": 0.85})
    partial_score = driver.progress_score(metrics)
    assert 0 < partial_score <= 0.01
    assert not metrics["gate_passed"] and state.update_evaluation(report).status == "hold"
    records[0]["watch_visible_s"], records[1]["watch_visible_s"] = 0.18, 0.165
    report, improved = evaluation.evaluate_policy(
        source, DeterministicPolicy(), stage=0, evaluation_id=1, episodes_per_side=64
    )
    assert driver.progress_score(improved) > partial_score
    assert not improved["gate_passed"] and state.update_evaluation(report).status == "hold"
    assert state.stage == 0 and state.consecutive_passes == 0


@pytest.mark.parametrize("visibility", [None, 0.59, float("nan")])
def test_final_visual_success_cannot_certify_a_blind_or_unmeasured_rally(visibility):
    """Eight physical returns alone cannot certify a final episode that lacks both players' RGB evidence."""
    record = {**episode(0, stage=5), "returns": 8, "objective": "rally", "gaze_visible_b": 0.9}
    if visibility is not None:
        record["gaze_visible_a"] = visibility
    source = EpisodeSource(2, {1: [record]}, profile=evaluation.VISION_PROFILE)
    source.cfg.curriculum_stage = 5
    source.cfg.final_min_gaze_visibility = 0.60
    with pytest.raises(ValueError, match="visibility|finite"):
        evaluation.evaluate_policy(source, DeterministicPolicy(), stage=5, evaluation_id=0, episodes_per_side=1)


def test_final_visual_episode_uses_the_declared_visibility_threshold_for_each_player():
    records = [
        {**episode(side, stage=5), "returns": 8, "objective": "rally", "gaze_visible_a": 0.6, "gaze_visible_b": 0.75}
        for side in (0, 1)
    ]
    source = EpisodeSource(2, {1: records}, profile=evaluation.VISION_PROFILE)
    source.cfg.curriculum_stage = 5
    source.cfg.final_min_gaze_visibility = 0.60
    report, metrics = evaluation.evaluate_policy(
        source, DeterministicPolicy(), stage=5, evaluation_id=0, episodes_per_side=1
    )
    assert report.successes_by_side == (1, 1) and report.objective == "rally"
    assert metrics["final_min_gaze_visibility"] == 0.60


def test_live_pass_graduation_requires_real_transfers_and_both_ball_and_partner_rgb_evidence():
    """The live profile cannot reuse virtual coach successes or certify blind eight-return rallies."""
    contract = evaluation._curriculum_module(evaluation.LIVE_PASS_PROFILE)
    records = [
        {
            **episode(side, stage=2),
            "objective": "rally",
            "partner_visible_a": 0.5,
            "partner_visible_b": 0.5,
            "partner_seen_a": 1,
            "partner_seen_b": 1,
        }
        for side in (0, 1)
    ]
    source = EpisodeSource(2, {1: records}, profile=evaluation.LIVE_PASS_PROFILE)
    source.cfg.curriculum_stage = 2
    policy = DeterministicPolicy()
    with pytest.raises(ValueError, match="too-short rally"):
        evaluation.evaluate_policy(source, policy, stage=2, evaluation_id=0, episodes_per_side=1)
    for record in records:
        record["returns"] = 1
    report, metrics = evaluation.evaluate_policy(source, policy, stage=2, evaluation_id=1, episodes_per_side=1)
    state = evaluation._state_type(contract)(stage=2, min_attempts_per_side=1)
    assert isinstance(state, contract.LivePassCurriculumState)
    assert state.stages[2].paired and report.required_returns == 1
    assert report.successes_by_side == (1, 1) and state.update_evaluation(report).status == "passed"
    assert metrics["profile"] == evaluation.LIVE_PASS_PROFILE

    final = {
        **episode(0, stage=5),
        "returns": 8,
        "objective": "rally",
        "gaze_visible_a": 0.60,
        "gaze_visible_b": 0.75,
        "partner_visible_a": 0.10,
        "partner_visible_b": 0.20,
        "partner_seen_a": 1,
        "partner_seen_b": 1,
    }
    source = EpisodeSource(2, {1: [final, {**final, "env_id": 1, "server": 1}]}, evaluation.LIVE_PASS_PROFILE)
    source.cfg.curriculum_stage = 5
    source.cfg.final_min_gaze_visibility = 0.60
    source.cfg.final_min_partner_visibility = 0.10
    for override in (
        {"returns": 7},
        {"gaze_visible_a": 0.59},
        {"partner_visible_a": 0.09},
        {"partner_seen_b": 0},
        {"partner_seen_a": None},
    ):
        source.schedule[1][0] = {**final, **override}
        with pytest.raises(ValueError, match="rally|visibility|partner"):
            evaluation.evaluate_policy(source, policy, stage=5, evaluation_id=2, episodes_per_side=1)
    source.schedule[1][0] = final
    report, _ = evaluation.evaluate_policy(source, policy, stage=5, evaluation_id=3, episodes_per_side=1)
    state = evaluation._state_type(contract)(stage=5, min_attempts_per_side=1)
    assert report.successes_by_side == (1, 1) and report.required_returns == 8
    assert state.update_evaluation(report).status == "passed"
