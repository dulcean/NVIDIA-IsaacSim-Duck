# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent objective, safety, deadline and reset contracts for volley episodes."""

import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/volley_episode.py"
_SPEC = importlib.util.spec_from_file_location("microduck_volley_episode", _PATH)
episode = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(episode)


def test_goal_requires_complete_recovery_window_and_only_pays_once():
    """A stable goal cannot immediately pay or finish before the fixed recovery deadline."""
    state = episode.VolleyEpisode(1, "cpu")
    yes = torch.tensor([True])
    no = ~yes
    done, success, started_goal, started_miss = state.step(yes, no, no, no, yes, 0.1)
    assert started_goal.item() and not started_miss.item()
    assert state.phase.item() == 1 and state.phase_time.item() == 0
    assert not done.item() and not success.item()
    for _ in range(5):
        done, success, started_goal, started_miss = state.step(yes, no, no, no, yes, 0.1)
        assert not done.item() and not success.item()
        assert not started_goal.item() and not started_miss.item()
    done, success, _, _ = state.step(no, no, no, no, yes, 0.1)
    assert done.item() and success.item()
    done, success, _, _ = state.step(yes, no, no, no, yes, 0.1)
    assert done.item() and not success.item()


def test_miss_cannot_be_promoted_to_goal_or_restart_its_deadline():
    """A later apparent target crossing cannot turn a failed objective into success."""
    state = episode.VolleyEpisode(1, "cpu")
    yes = torch.tensor([True])
    no = ~yes
    _, _, started_goal, started_miss = state.step(yes, yes, no, no, yes, 0.2)
    assert started_miss.item() and not started_goal.item()
    for _ in range(2):
        done, success, started_goal, started_miss = state.step(yes, yes, no, no, yes, 0.2)
        assert not done.item() and not success.item()
        assert not started_goal.item() and not started_miss.item()
    done, success, _, _ = state.step(yes, no, no, no, yes, 0.2)
    assert done.item() and not success.item()
    assert state.phase.item() == 2 and state.missed.item()


def test_goal_recovery_requires_final_continuous_hold_not_early_stability():
    """A robot that briefly balances then sags cannot pass the final recovery gate."""
    state = episode.VolleyEpisode(2, "cpu")
    yes = torch.ones(2, dtype=torch.bool)
    no = ~yes
    state.step(yes, no, no, no, yes, 0.1)
    for _ in range(3):
        state.step(no, no, no, no, yes, 0.1)
    state.step(no, no, no, no, torch.tensor([True, False]), 0.1)
    state.step(no, no, no, no, yes, 0.1)
    done, success, _, _ = state.step(no, no, no, no, yes, 0.1)
    assert done.all() and success.tolist() == [True, False]


def test_safety_and_timeout_override_objective_and_ready_recovery():
    """Safety terminates while timeout truncates; neither can release a goal bonus."""
    state = episode.VolleyEpisode(4, "cpu")
    yes = torch.ones(4, dtype=torch.bool)
    no = ~yes
    unsafe = torch.tensor([True, False, False, False])
    timeout = torch.tensor([False, True, False, False])
    done, success, started_goal, _ = state.step(yes, no, unsafe, timeout, yes, 0.2)
    assert done.tolist() == [True, False, False, False]
    assert started_goal.tolist() == [False, False, True, True]
    assert not success.any()
    for _ in range(2):
        state.step(no, no, no, no, yes, 0.2)
    unsafe = torch.tensor([False, False, True, False])
    timeout = torch.tensor([False, False, False, True])
    done, success, _, _ = state.step(no, no, unsafe, timeout, yes, 0.2)
    assert done.tolist() == [True, False, True, False] and not success.any()
    done, success, started_goal, _ = state.step(yes, no, no, no, yes, 0.2)
    assert not success.any() and not started_goal.any()


def test_subset_reset_clears_terminal_world_without_restarting_its_partner():
    """Asynchronous resets must preserve other environments' unfinished recovery."""
    state = episode.VolleyEpisode(2, "cpu")
    yes = torch.ones(2, dtype=torch.bool)
    no = ~yes
    state.step(yes, no, no, no, yes, 0.2)
    state.step(no, no, torch.tensor([True, False]), no, yes, 0.2)
    state.reset(torch.tensor([0]))
    assert state.phase.tolist() == [0, 1]
    assert not state.unsafe.any() and not state.success.any()
    for _ in range(2):
        done, success, _, _ = state.step(no, no, no, no, yes, 0.2)
    assert done.tolist() == [False, True]
    assert success.tolist() == [False, True]
    _, _, started_goal, _ = state.step(torch.tensor([True, False]), no, no, no, yes, 0.2)
    assert started_goal.tolist() == [True, False]
