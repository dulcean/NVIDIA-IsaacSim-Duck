# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU geometry contracts for a raised, camera-facing MicroDuck strike."""

import importlib.util
from pathlib import Path

import torch

# Load the pure geometry owner without importing the simulator/task registry on a CPU-only host.
_PATH = Path(__file__).resolve().parents[3] / (
    "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/heading_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("microduck_heading_geometry", _PATH)
geometry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geometry)


def test_front_strike_excludes_roof_lens_and_unseen_ball():
    """Only the forehead patch with a raised camera facing the ball counts as a strike."""
    contacts = torch.tensor(
        [
            [0.035, 0.0, -0.075],  # Front forehead, above the lens.
            [0.042, 0.0, 0.0],  # Upper lid, behind the front surface.
            [0.0155, 0.0, -0.075],  # Lens area.
            [0.035, 0.0, -0.075],  # Front contact, camera still looking horizontally.
            [0.035, 0.0, -0.075],  # Ball outside the camera's forward cone.
            [float("nan"), 0.0, -0.075],  # No physical contact point.
        ]
    )
    ball_directions = torch.tensor([[0.0, 0.0, -1.0]] * 6)
    ball_directions[4] = torch.tensor([1.0, 0.0, 0.0])
    upward = torch.tensor([0.8, 0.8, 0.8, 0.0, 0.8, 0.8])
    actual = geometry.front_strike_mask(
        contacts,
        ball_directions,
        upward,
        (0.0, 0.0, -1.0),
        (0.027, -0.04, -0.10),
        (0.045, 0.04, -0.05),
        0.5,
        0.25,
    )
    assert actual.tolist() == [True, False, False, False, False, False]


def test_gaze_reward_points_forward():
    """A ball directly in front scores above one at the side or behind the camera."""
    directions = torch.tensor([[0.0, 0.0, -2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    torch.testing.assert_close(geometry.camera_alignment(directions, (0.0, 0.0, -1.0)), torch.tensor([1.0, 0.0, -1.0]))
