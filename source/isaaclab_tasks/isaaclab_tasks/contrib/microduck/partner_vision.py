# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-only receiving-point estimates from two known-size badges on a real head.

Two independently colored spheres determine a rigid line on the head. Its
receiving point is ``2 * cyan - magenta``: this recovers the configured forehead
point even while the head tilts, without reading the other robot's pose. The
camera pose is only the observing robot's calibrated kinematic reference.
"""

from __future__ import annotations

import math

import torch

from .ball_vision import BallVisualTracker, VisualBallTracks

PARTNER_MARKER_RADIUS = 0.008
PARTNER_CYAN_POSITION = (0.055, 0.0, -0.070)
PARTNER_MAGENTA_POSITION = (0.075, 0.0, -0.070)
PARTNER_STRIKE_POSITION = (0.035, 0.0, -0.070)
PARTNER_MARKER_BASELINE = 0.020


class PartnerHeadVisualTracker:
    """Estimate a partner's forehead from paired RGB silhouettes and finite memory.

    Color masks, connected components, lens undistortion, and known-size range
    estimation use the production sphere detector. Both fresh spheres must
    agree with their physical baseline to correct a head estimate. A missing
    badge uses only bounded constant-velocity memory of the last complete RGB
    pair; an unseen or expired partner has finite zero state and ``valid=False``.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        marker_radius: float = PARTNER_MARKER_RADIUS,
        max_age: float = 0.75,
        max_speed: float = 1.5,
        velocity_smoothing: float = 0.25,
        baseline_tolerance: float = 0.025,
        distortion: tuple[float, float, float, float, float] = (0.0,) * 5,
    ):
        if not math.isfinite(baseline_tolerance) or not 0 < baseline_tolerance < 0.05:
            raise ValueError("Head badge baseline tolerance must be finite and between zero and 5 cm.")
        self.spheres = [
            BallVisualTracker(
                num_envs,
                device,
                ball_radius=marker_radius,
                max_age=max_age,
                max_speed=max_speed,
                min_radius_px=1.5,
                velocity_smoothing=velocity_smoothing,
                distortion=distortion,
                color=color,
                acceleration=(0.0, 0.0, 0.0),
            )
            for color in ("cyan", "magenta")
        ]
        self.max_age = max_age
        self.max_speed = max_speed
        self.velocity_smoothing = velocity_smoothing
        self.baseline_tolerance = baseline_tolerance
        self.position_w = torch.zeros(num_envs, 3, device=self.spheres[0].device)
        self.velocity_w = torch.zeros_like(self.position_w)
        self.visible = torch.zeros(num_envs, dtype=torch.bool, device=self.position_w.device)
        self.valid = torch.zeros_like(self.visible)
        self.age = torch.zeros(num_envs, device=self.position_w.device)
        self.confidence = torch.zeros_like(self.age)
        self.last_seen_position_w = torch.zeros_like(self.position_w)
        self.seen = torch.zeros_like(self.visible)
        self._last_measurement = torch.zeros_like(self.position_w)
        self._measurement_confidence = torch.zeros_like(self.age)

    @property
    def tracks(self) -> VisualBallTracks:
        """Return a snapshot without advancing time or using simulated partner state."""
        return VisualBallTracks(
            *(
                value.clone()
                for value in (
                    self.position_w,
                    self.velocity_w,
                    self.visible,
                    self.age,
                    self.valid,
                    self.confidence,
                    self.last_seen_position_w,
                    self.seen,
                )
            )
        )

    def reset(self, env_ids=None):
        """Erase selected worlds, including search bearings and pair history."""
        for sphere in self.spheres:
            sphere.reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        for value in (
            self.position_w,
            self.velocity_w,
            self.visible,
            self.valid,
            self.age,
            self.confidence,
            self.last_seen_position_w,
            self.seen,
            self._last_measurement,
            self._measurement_confidence,
        ):
            value[env_ids] = 0

    def _advance_pair(self, dt: float, measured: bool):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Partner tracking requires finite positive elapsed time.")
        elapsed = self.age + dt
        prior = self.valid & (elapsed <= self.max_age + 1e-6)
        prediction = self.position_w + self.velocity_w * dt
        cyan, magenta = self.spheres
        baseline = (magenta.position_w - cyan.position_w).norm(dim=-1)
        baseline_valid = (baseline >= max(0.005, PARTNER_MARKER_BASELINE - self.baseline_tolerance)) & (
            baseline <= PARTNER_MARKER_BASELINE + self.baseline_tolerance
        )
        accepted = cyan.visible & magenta.visible & baseline_valid & measured
        measurement = 2 * cyan.position_w - magenta.position_w
        accepted &= torch.isfinite(measurement).all(-1)
        speed_gate = self.max_speed * elapsed + 0.08
        accepted &= ~prior | ((measurement - prediction).norm(dim=-1) <= speed_gate)
        velocity = (measurement - self._last_measurement) / elapsed.clamp_min(1e-6)[:, None]
        velocity = self.velocity_smoothing * velocity + (1 - self.velocity_smoothing) * self.velocity_w
        velocity *= (self.max_speed / velocity.norm(dim=-1).clamp_min(self.max_speed))[:, None]
        velocity = torch.where(prior[:, None], velocity, 0.0)
        self.valid = accepted | prior
        self.visible = accepted
        self.position_w = torch.where(accepted[:, None], measurement, prediction)
        self.velocity_w = torch.where(accepted[:, None], velocity, self.velocity_w)
        self.age = torch.where(accepted, 0.0, elapsed)
        confidence = torch.minimum(cyan.confidence, magenta.confidence)
        self._measurement_confidence = torch.where(accepted, confidence, self._measurement_confidence)
        self.confidence = torch.where(
            self.valid, self._measurement_confidence * (1 - self.age / self.max_age).clamp_min(0), 0.0
        )
        self._last_measurement = torch.where(accepted[:, None], measurement, self._last_measurement)
        self.last_seen_position_w = torch.where(accepted[:, None], measurement, self.last_seen_position_w)
        self.seen |= accepted
        for value in (self.position_w, self.velocity_w, self._last_measurement):
            value.masked_fill_(~self.valid[:, None], 0.0)
        self.age.masked_fill_(~self.valid, 0.0)
        self._measurement_confidence.masked_fill_(~self.valid, 0.0)

    def update(
        self,
        rgb,
        intrinsics,
        camera_position_w,
        camera_quat_xyzw,
        dt: float = 0.02,
        measurement_mask=None,
        snapshot: bool = True,
    ) -> VisualBallTracks | None:
        """Correct only from a synchronized fresh RGB pair, or retain short memory."""
        for sphere in self.spheres:
            sphere.update(
                rgb,
                intrinsics,
                camera_position_w,
                camera_quat_xyzw,
                dt=dt,
                measurement_mask=measurement_mask,
                snapshot=False,
            )
        self._advance_pair(dt, measured=True)
        return self.tracks if snapshot else None

    def advance(self, dt: float):
        """Advance constant-velocity memory when no new camera exposure exists."""
        for sphere in self.spheres:
            sphere.advance(dt)
        self._advance_pair(dt, measured=False)

    def predict(self, dt: float) -> VisualBallTracks:
        """Advance without image evidence and return a snapshot."""
        self.advance(dt)
        return self.tracks
