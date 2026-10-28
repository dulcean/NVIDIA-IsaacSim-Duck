# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small active-gaze training terms, separate from RGB-only actor observations.

Physical ball position may label a training reward, like task success. It must
not replace missing visual actor features. Camera poses here are the current
kinematic poses rather than the pose of the last captured image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CameraGazeGeometry:
    """Bounded acquisition alignment and calibrated raw-image projection.

    Alignment is ``(1 + cos(optical bearing))/2``: a ball directly behind the
    camera scores zero, a sideways ball scores one half, and a ball straight
    ahead scores one. ``in_frame`` additionally requires a finite, front-facing
    calibrated projection inside the image. Invalid projections use finite zero
    pixel coordinates; callers must always use the accompanying boolean mask.
    """

    alignment: torch.Tensor
    in_frame: torch.Tensor
    pixel_xy: torch.Tensor


@dataclass(frozen=True)
class ActiveGazeReward:
    """Time-integrated reward and masked mean diagnostic terms per environment."""

    reward: torch.Tensor
    eligible: torch.Tensor
    alignment_mean: torch.Tensor
    visible_mean: torch.Tensor


class MeasuredVisibilityWindow:
    """Count uninterrupted real exposure intervals, never tracker predictions.

    A first accepted image starts a window at zero. Only another accepted
    exposure within ``max_gap`` extends it. Missing images break the window;
    repeated cached frames cannot extend it. Independent world resets clear
    only their own evidence.
    """

    def __init__(self, count: int, device, max_gap: float):
        if count < 1 or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("Visibility windows require a positive batch size and capture gap.")
        self.max_gap = max_gap
        self.hold = torch.zeros(count, 2, device=device)
        self.peak = torch.zeros_like(self.hold)
        self.elapsed = torch.zeros_like(self.hold)
        self.age = torch.full_like(self.hold, float("inf"))
        self.matched = torch.zeros(count, 2, dtype=torch.bool, device=device)

    def update(self, matched: torch.Tensor, fresh: torch.Tensor, dt: float):
        """Advance physics time and consume only fresh, physically matched RGB detections."""
        if matched.shape != self.hold.shape or fresh.shape != self.hold.shape:
            raise ValueError("Visibility masks must match the two-camera batch.")
        if matched.dtype != torch.bool or fresh.dtype != torch.bool or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Visibility updates require boolean masks and positive finite time.")
        self.elapsed += dt
        self.age += dt
        continuous = matched & self.matched & (self.elapsed <= self.max_gap + 1e-6)
        self.hold.copy_(torch.where(fresh, torch.where(continuous, self.hold + self.elapsed, 0.0), self.hold))
        self.peak.copy_(torch.maximum(self.peak, self.hold))
        self.matched.copy_(torch.where(fresh, matched, self.matched))
        self.age.masked_fill_(fresh & matched, 0.0)
        self.elapsed.masked_fill_(fresh, 0.0)

    @property
    def recent(self):
        """Accepted pixels remain usable only until the next expected exposure window."""
        return self.matched & (self.age <= self.max_gap)

    def reset(self, env_ids):
        self.hold[env_ids] = 0
        self.peak[env_ids] = 0
        self.elapsed[env_ids] = 0
        self.age[env_ids] = float("inf")
        self.matched[env_ids] = False


def camera_gaze_geometry(
    ball_position_w: torch.Tensor,
    camera_positions_w: torch.Tensor,
    camera_quats_xyzw: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
) -> CameraGazeGeometry:
    """Project a training label using optical +X-right/+Y-down/+Z-forward poses.

    Shapes are ``(N, 3)`` for ball position and ``(N, 2, 3/4)`` for camera
    positions/XYZW orientations. Intrinsics have shape ``(N, 2, 3, 3)``.
    ``image_size`` is raw ``(width, height)`` before any UI rotation.
    Distortion is OpenCV ``k1, k2, p1, p2, k3`` and may broadcast over cameras.
    This is geometry for reward labels, not a visual measurement or actor input.
    """
    if ball_position_w.ndim != 2 or ball_position_w.shape[-1] != 3:
        raise ValueError("Ball training labels must have shape (batch, 3).")
    count = ball_position_w.shape[0]
    if camera_positions_w.shape != (count, 2, 3) or camera_quats_xyzw.shape != (count, 2, 4):
        raise ValueError("Gaze geometry requires two optical camera poses per environment.")
    if intrinsics.shape != (count, 2, 3, 3):
        raise ValueError("Camera intrinsics must have shape (batch, 2, 3, 3).")
    if len(image_size) != 2 or any(type(value) is not int or value < 2 for value in image_size):
        raise ValueError("Raw image width and height must be integers greater than one.")
    width, height = image_size
    dtype, device = ball_position_w.dtype, ball_position_w.device
    if any(value.device != device for value in (camera_positions_w, camera_quats_xyzw, intrinsics)):
        raise ValueError("Gaze positions, orientations, and calibration must use one device.")
    coefficient = torch.as_tensor(distortion, dtype=dtype, device=device)
    try:
        coefficient = torch.broadcast_to(coefficient, (count, 2, 5))
    except RuntimeError as error:
        raise ValueError("Distortion coefficients must broadcast to (batch, 2, 5).") from error
    finite = torch.isfinite(ball_position_w).all(-1)[:, None]
    finite = finite & torch.isfinite(camera_positions_w).all(-1) & torch.isfinite(camera_quats_xyzw).all(-1)
    finite &= torch.isfinite(intrinsics).all(dim=(-1, -2)) & torch.isfinite(coefficient).all(-1)
    finite &= camera_quats_xyzw.norm(dim=-1) > 1e-6
    finite &= (intrinsics[..., 0, 0] > 0) & (intrinsics[..., 1, 1] > 0)
    quat = F.normalize(torch.nan_to_num(camera_quats_xyzw), dim=-1)
    delta = ball_position_w[:, None] - camera_positions_w
    inverse_vector = -quat[..., :3]
    first = 2 * torch.cross(inverse_vector, delta, dim=-1)
    local = delta + quat[..., 3:] * first + torch.cross(inverse_vector, first, dim=-1)
    distance = local.norm(dim=-1)
    finite &= torch.isfinite(distance) & (distance > 1e-6)
    cosine = (local[..., 2] / distance.clamp_min(1e-6)).clamp(-1, 1)
    alignment = torch.where(finite, (cosine + 1) / 2, 0.0)
    front = local[..., 2] > 1e-6
    safe_z = torch.where(front, local[..., 2], 1.0)
    x, y = local[..., 0] / safe_z, local[..., 1] / safe_z
    k1, k2, p1, p2, k3 = coefficient.unbind(-1)
    radius_squared = x.square() + y.square()
    radial = 1 + radius_squared * (k1 + radius_squared * (k2 + radius_squared * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (radius_squared + 2 * x.square())
    yd = y * radial + p1 * (radius_squared + 2 * y.square()) + 2 * p2 * x * y
    u = intrinsics[..., 0, 0] * xd + intrinsics[..., 0, 1] * yd + intrinsics[..., 0, 2]
    v = intrinsics[..., 1, 0] * xd + intrinsics[..., 1, 1] * yd + intrinsics[..., 1, 2]
    pixels = torch.stack((u, v), dim=-1)
    projection_valid = finite & front & torch.isfinite(pixels).all(-1)
    in_frame = projection_valid & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    pixels = torch.where(projection_valid[..., None], pixels, 0.0)
    return CameraGazeGeometry(torch.nan_to_num(alignment), in_frame, pixels)


def active_gaze_reward(
    alignment: torch.Tensor,
    pixel_accepted: torch.Tensor,
    playing: torch.Tensor,
    contact: torch.Tensor,
    hit_age: torch.Tensor,
    dt: float,
    grace: float = 0.08,
    alignment_rate: float = 0.15,
    visibility_rate: float = 0.15,
) -> ActiveGazeReward:
    """Pay small per-second acquisition/visible-pixel terms outside strike recovery.

    ``pixel_accepted`` must already require a fresh real RGB detection, the true
    ball inside this camera's image, and a plausible measurement error. This
    prevents an orange robot part from earning visibility credit. An ordinary
    predicted tracker position during camera dropout is not accepted pixels.

    Only actual contact or less than ``grace`` seconds after the robot's own
    strike masks gaze. Ball distance never disables it. Both cameras receive
    equal weight: exclusions contribute zero rather than increasing the other
    camera's rate. Thus reward is bounded by ``dt * (alignment_rate +
    visibility_rate)`` regardless of how many robots are currently striking.
    """
    if alignment.ndim != 2 or alignment.shape[1] != 2:
        raise ValueError("Active gaze needs two camera terms per environment.")
    if pixel_accepted.shape != alignment.shape or contact.shape != alignment.shape or hit_age.shape != alignment.shape:
        raise ValueError("Visibility, contact, and own-hit ages must match gaze alignment shape.")
    if playing.shape != alignment.shape[:1]:
        raise ValueError("The active-rally mask must have one value per environment.")
    if any(value.dtype != torch.bool for value in (pixel_accepted, playing, contact)):
        raise ValueError("Gaze visibility, rally, and contact masks must be boolean.")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Active-gaze reward requires a finite positive time interval.")
    if not all(math.isfinite(value) and value >= 0 for value in (grace, alignment_rate, visibility_rate)):
        raise ValueError("Gaze grace and per-second reward rates must be finite and non-negative.")
    eligible = playing[:, None] & ~contact & (hit_age >= grace) & ~torch.isnan(hit_age)
    safe_alignment = torch.nan_to_num(alignment, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    alignment_mean = (safe_alignment * eligible.float()).mean(-1)
    visible_mean = (pixel_accepted & eligible).float().mean(-1)
    reward = dt * (alignment_rate * alignment_mean + visibility_rate * visible_mean)
    return ActiveGazeReward(reward, eligible, alignment_mean, visible_mean)
