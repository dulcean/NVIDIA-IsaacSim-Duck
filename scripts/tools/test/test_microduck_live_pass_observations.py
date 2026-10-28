# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB information-isolation and root-frame contracts at the live actor boundary."""

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import torch

_ROOT = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck"
_PACKAGE = "microduck_live_actor_tests"
_namespace = ModuleType(_PACKAGE)
_namespace.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = _namespace
vision = importlib.import_module(f"{_PACKAGE}.ball_vision")
live = importlib.import_module(f"{_PACKAGE}.live_pass_observations")


def inputs():
    def track(point):
        return vision.VisualBallTracks(
            position_w=torch.tensor([point]),
            velocity_w=torch.zeros(1, 3),
            visible=torch.ones(1, dtype=torch.bool),
            age=torch.full((1,), 0.03),
            valid=torch.ones(1, dtype=torch.bool),
            confidence=torch.ones(1),
            last_seen_position_w=torch.tensor([point]),
            seen=torch.ones(1, dtype=torch.bool),
        )

    privileged = torch.arange(142, dtype=torch.float32)[None] / 100
    privileged[:, 70] = privileged[:, 141] = 1
    balls = [track((-0.114, 0.0, 0.55)), track((0.114, 0.0, 0.55))]
    balls = [replace(ball, velocity_w=torch.tensor([[0.0, 0.0, -1.0]])) for ball in balls]
    partners = [track((0.114, 0.0, 0.2726)), track((-0.114, 0.0, 0.2726))]
    root = torch.tensor([[[-0.175, 0.0, 0.125], [0.175, 0.0, 0.125]]])
    quats = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]])
    velocities = torch.tensor([[[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]]])
    heads = torch.tensor([[[-0.114, 0.0, 0.2726], [0.114, 0.0, 0.2726]]])
    return privileged, balls, partners, root, quats, velocities, heads, torch.zeros(1, 2, 3)


def test_live_actor_has_no_privileged_ball_target_or_other_player_kinematic_fallback():
    """Identical camera evidence yields identical actor features despite arbitrary hidden states."""
    values = inputs()
    actor = live.live_pass_actor_observations(*values, env_origins=torch.zeros(1, 3))
    assert actor.shape == (1, 178)
    blocks = actor.view(1, 2, 89)
    assert blocks[0, :, 79].tolist() == [1, 1] and blocks[0, :, 88].tolist() == [1, 1]
    # The RGB forehead is .289m in front of either root; the ball centre is
    # .2726+.025-.125=.1726m above it, with no simulated receiver target.
    torch.testing.assert_close(blocks[0, :, 67:70], torch.tensor([[0.289, 0.0, 0.1726]] * 2))
    torch.testing.assert_close(blocks[0, :, 76:79], torch.tensor([[-0.1, 0.0, 0.0]] * 2))
    # Desired velocity subtracts moving-root odometry; delta-v is a pure
    # difference and must only rotate. Both roots move +.1m/s in their own frame.
    torch.testing.assert_close(blocks[0, :, 84] - blocks[0, :, 81], torch.full((2,), 0.1))
    assert (blocks[0, :, 87] > 0).all() and (blocks[0, :, 87] < 1).all()
    changed = values[0].clone()
    changed[:, 61:70] = float("nan")
    changed[:, 132:141] = -1000
    isolated = live.live_pass_actor_observations(changed, *values[1:], env_origins=torch.zeros(1, 3))
    torch.testing.assert_close(isolated, actor)
    assert torch.isfinite(isolated).all()

    # The observed partner can move physically anywhere without becoming a
    # second privileged pose source for player A. Its camera track owns A's aim.
    kinematics = [value.clone() if isinstance(value, torch.Tensor) else value for value in values]
    for index in (3, 5, 6, 7):
        kinematics[index][:, 1] += 7
    changed_other_player = live.live_pass_actor_observations(*kinematics, env_origins=torch.zeros(1, 3))
    torch.testing.assert_close(changed_other_player[:, :89], actor[:, :89])

    partners = list(values[2])
    partners[0] = replace(partners[0], position_w=torch.tensor([[0.15, 0.03, 0.29]]))
    changed_camera = live.live_pass_actor_observations(
        values[0], values[1], partners, *values[3:], env_origins=torch.zeros(1, 3)
    )
    assert not torch.allclose(changed_camera[:, 67:70], actor[:, 67:70])
    torch.testing.assert_close(changed_camera[:, 89:], actor[:, 89:])


def test_missing_partner_or_ball_cannot_produce_a_plan_and_ball_search_history_survives():
    """Expired visual estimates have explicit invalid sentinels, never receiver/ball truth."""
    values = inputs()
    original = live.live_pass_actor_observations(*values, env_origins=torch.zeros(1, 3)).view(1, 2, 89)
    partners = list(values[2])
    partners[0] = replace(partners[0], age=torch.tensor([0.751]))
    partners[1] = replace(partners[1], valid=torch.zeros(1, dtype=torch.bool))
    missing_partner = live.live_pass_actor_observations(
        values[0], values[1], partners, *values[3:], env_origins=torch.zeros(1, 3)
    ).view(1, 2, 89)
    assert not missing_partner[..., 67:70].any()
    assert not missing_partner[..., 76:80].any()
    assert (missing_partner[..., 80] == 1).all()
    assert not missing_partner[..., 81:89].any()
    torch.testing.assert_close(missing_partner[..., 71:76], original[..., 71:76])

    balls = list(values[1])
    balls[0] = replace(balls[0], age=torch.tensor([0.151]))
    missing_ball = live.live_pass_actor_observations(values[0], balls, *values[2:], env_origins=torch.zeros(1, 3)).view(
        1, 2, 89
    )
    assert not missing_ball[0, 0, 61:67].any() and not missing_ball[0, 0, 81:89].any()
    assert missing_ball[0, 0, 72] == 1
    torch.testing.assert_close(missing_ball[0, 0, 73:76], original[0, 0, 73:76])
    torch.testing.assert_close(missing_ball[0, 1], original[0, 1])
