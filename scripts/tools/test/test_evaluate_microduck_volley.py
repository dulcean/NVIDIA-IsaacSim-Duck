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

    def __init__(self, num_envs, schedule):
        self.num_envs = num_envs
        self.schedule = schedule
        self.unwrapped = self
        self.cfg = SimpleNamespace(evaluation=True, curriculum_stage=0)
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
