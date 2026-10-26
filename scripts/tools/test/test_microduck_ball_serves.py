# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU physical contracts for incoming MicroDuck receiver serves."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/ball_serves.py"
_SPEC = importlib.util.spec_from_file_location("microduck_ball_serves", _PATH)
serves = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serves)


def test_target_success_uses_the_observed_swept_descent_instead_of_endpoint_distance_or_prediction():
    """A fast segment may cross the disk between samples; misses and upward paths cannot count."""
    previous = torch.tensor(
        [
            [-0.2, 0.0, 0.4],  # Neither endpoint is near the disk, but the segment crosses its center.
            [-0.2, 0.06, 0.4],  # Descending on the correct side, but outside the disk.
            [-0.2, 0.0, 0.2],  # Ascending through the disk.
            [-0.2, 0.0, 0.2],  # Target plane is above the entire descending segment.
            [-0.2, 0.0, 0.5],  # Target plane is below the entire descending segment.
            [0.0, 0.0, 0.3],  # Level movement on the plane cannot count as descent.
            [-0.2, 0.0, 0.4],  # Exact endpoint crossing.
            [0.0, 0.0, 0.3],  # Exact startpoint crossing.
            [float("nan"), 0.0, 0.4],
        ],
        dtype=torch.float64,
    )
    current = torch.tensor(
        [
            [0.2, 0.0, 0.2],
            [0.2, 0.06, 0.2],
            [0.2, 0.0, 0.4],
            [0.2, 0.0, 0.1],
            [0.2, 0.0, 0.4],
            [0.01, 0.0, 0.3],
            [0.0, 0.0, 0.3],
            [0.2, 0.0, 0.2],
            [0.2, 0.0, 0.2],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64).repeat(len(previous), 1)
    assert serves.descending_target_crossing(previous, current, target, 0.05).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_curriculum_arcs_reach_the_sampled_receiver_on_a_descending_collision_free_flight():
    """Vertical through lateral incoming serves reach known ball-center targets under gravity alone."""
    count = 120
    receiver = torch.tensor([[-0.114, 0.0, 0.273]], dtype=torch.float64).repeat(count, 1)
    partner = torch.tensor([[0.114, 0.0, 0.273]], dtype=torch.float64).repeat(count, 1)
    time = torch.linspace(0.2, 0.4, count, dtype=torch.float64)
    fraction = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    height = torch.linspace(0.18, 0.25, count, dtype=torch.float64)
    offset = torch.zeros_like(receiver)
    offset[:, 1] = torch.linspace(-0.01, 0.01, count, dtype=torch.float64)
    offset[:, 2] = 0.025  # Ball radius above the selected forehead surface point.
    position, velocity, target, valid = serves.incoming_ballistic_serve(
        receiver, partner, time, height, fraction, offset
    )
    assert valid.all()
    torch.testing.assert_close(target, receiver + offset)
    # Independent point-mass flight: no interpolation or position reset after launch.
    elapsed = time[:, None] * torch.linspace(0, 1, 21, dtype=torch.float64)[None, :]
    trajectory = position[:, None, :] + velocity[:, None, :] * elapsed[:, :, None]
    trajectory[:, :, 2] -= 4.905 * elapsed.square()
    torch.testing.assert_close(trajectory[:, -1], receiver + offset, atol=1e-12, rtol=0)
    assert (trajectory[:, :, 2] >= 0.05).all()
    assert (velocity[:, 2] - 9.81 * time < -1.5).all()
    assert position[0, 0] == pytest.approx(-0.114)
    assert position[-1, 0] == pytest.approx(0.114)
    assert velocity[-1, 0] < 0


def test_mirrored_receivers_produce_mirrored_serves_in_translated_worlds():
    """Player identity and cloned-world origins must not bias arrival timing or lateral direction."""
    receiver = torch.tensor([[-0.114, 0.02, 0.273], [0.114, -0.02, 0.273]])
    partner = torch.tensor([[0.114, -0.01, 0.273], [-0.114, 0.01, 0.273]])
    origin = torch.tensor([[3.0, -2.0, 0.0], [3.0, -2.0, 0.0]])
    offset = torch.tensor([[0.005, -0.003, 0.025], [-0.005, 0.003, 0.025]])
    position, velocity, target, valid = serves.incoming_ballistic_serve(
        receiver + origin,
        partner + origin,
        torch.full((2,), 0.3),
        torch.full((2,), 0.23),
        torch.full((2,), 0.8),
        offset,
    )
    assert valid.all()
    torch.testing.assert_close((position - origin)[0, :2], -(position - origin)[1, :2])
    torch.testing.assert_close(velocity[0, :2], -velocity[1, :2])
    torch.testing.assert_close(velocity[0, 2], velocity[1, 2])
    torch.testing.assert_close(target - origin, receiver + offset)


def test_invalid_serves_cannot_silently_enter_the_physical_reset_batch():
    """Reject floor penetration, wrong-way lateral serves, excessive speed, and undefined flight time."""
    count = 7
    receiver = torch.tensor([[-0.114, 0.0, 0.273]] * count)
    partner = torch.tensor([[0.114, 0.0, 0.273]] * count)
    time = torch.full((count,), 0.3)
    time[3] = 0.0
    offset = torch.zeros_like(receiver)
    offset[0, 2] = -0.26  # Target below the ball's safe center height.
    source_offset = torch.zeros_like(receiver)
    source_offset[1, 0] = -0.4  # Ball would move from behind the receiver toward its partner.
    source_offset[2, 1] = 2.0  # Unreasonably large arrival speed.
    source_offset[4, 2] = -1.0  # Source below the floor.
    offset[5, 0] = float("nan")
    _, _, _, valid = serves.incoming_ballistic_serve(
        receiver,
        partner,
        time,
        torch.full((count,), 0.23),
        torch.ones(count),
        offset,
        source_offset=source_offset,
    )
    assert valid.tolist() == [False, False, False, False, False, False, True]
