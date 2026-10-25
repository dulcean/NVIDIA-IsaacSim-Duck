# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for volley turn order and robot collisions."""

import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/ball_rules.py"
_SPEC = importlib.util.spec_from_file_location("microduck_ball_rules", _PATH)
rules = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rules)


def test_alternating_onset_accepts_reply_without_flight_time_gate():
    """A reply from the other robot is legal; wrong turns, wrong surfaces, and persistent touches are not."""
    valid, surface, turn = rules.classify_head_contact(
        torch.tensor([True, True, True, False, True]),
        torch.tensor([True, True, False, True, True]),
        torch.tensor([1, 0, 1, 1, 1]),
        torch.tensor([0, 1, 0, 0, 0]),
        torch.tensor([False, False, False, False, True]),
        player=1,
        num_players=2,
    )
    assert valid.tolist() == [True, False, False, False, False]
    assert surface.tolist() == [False, False, True, False, False]
    assert turn.tolist() == [False, True, False, False, False]


def test_heading_still_allows_only_one_strike():
    """Removing the pair's flight-time gate must not turn single-robot heading into juggling."""
    valid, surface, turn = rules.classify_head_contact(
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([-1, 0]),
        torch.zeros(2, dtype=torch.bool),
        player=0,
        num_players=1,
    )
    assert valid.tolist() == [True, False]
    assert surface.tolist() == [False, False]
    assert turn.tolist() == [False, True]


def test_robot_contacts_do_not_cancel_or_count_subthreshold_noise():
    """Opposing body contacts still end a rally, while tiny numerical forces do not."""
    forces = torch.zeros(4, 2, 2, 3)
    forces[0, 0, 0, 0] = 1.0
    forces[0, 1, 1, 0] = -1.0
    forces[2, 0, 0, 2] = 0.005
    forces[3, 0, 0, 2] = float("nan")
    assert rules.robot_collision_mask(forces, 0.01).tolist() == [True, False, False, True]
