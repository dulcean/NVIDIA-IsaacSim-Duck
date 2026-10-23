# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-facing strike geometry in the MicroDuck head's local frame."""

import torch
import torch.nn.functional as functional


def camera_alignment(ball_direction_b: torch.Tensor, forward_axis: tuple[float, float, float]) -> torch.Tensor:
    """Return the cosine between the camera axis and the camera-to-ball vector."""
    forward = ball_direction_b.new_tensor(forward_axis)
    return (functional.normalize(ball_direction_b, dim=-1) * forward).sum(dim=-1)


def front_strike_mask(
    contact_b: torch.Tensor,
    ball_direction_b: torch.Tensor,
    forward_up: torch.Tensor,
    forward_axis: tuple[float, float, float],
    patch_min: tuple[float, float, float],
    patch_max: tuple[float, float, float],
    min_camera_cosine: float,
    min_forward_up: float,
) -> torch.Tensor:
    """Accept the front forehead patch only while the raised camera faces the ball.

    All points/directions are in the rigid head frame; ``forward_up`` is the camera
    axis dotted with world up. The forehead patch excludes the lens and roof.
    """
    inside = ((contact_b >= contact_b.new_tensor(patch_min)) & (contact_b <= contact_b.new_tensor(patch_max))).all(-1)
    visible = camera_alignment(ball_direction_b, forward_axis) >= min_camera_cosine
    finite = torch.isfinite(contact_b).all(-1) & torch.isfinite(ball_direction_b).all(-1)
    return inside & visible & finite & (forward_up >= min_forward_up)
