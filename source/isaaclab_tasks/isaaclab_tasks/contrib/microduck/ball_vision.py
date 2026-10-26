# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-only orange-ball estimates with calibrated monocular geometry and short tracking.

The pose rotates optical camera coordinates (+X right, +Y down, +Z forward)
into world coordinates, using XYZW quaternions. No simulator state, depth,
segmentation labels, or hidden ball initialization is accepted. Color and
sphere-size assumptions need calibration for other lighting/materials. A round
orange distractor is intrinsically ambiguous in a single RGB image; ambiguous
new tracks are rejected rather than claiming object identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualBallTracks:
    """A snapshot: accepted pixels, recent prediction, or finite zero sentinels.

    ``visible`` means this update accepted an RGB measurement. ``valid`` also
    allows a short world-frame ballistic prediction after a missing image.
    ``age`` is time since the last accepted measurement, in seconds. Expired or
    uninitialized tracks have zero position/velocity/confidence and valid=False.
    """

    position_w: torch.Tensor
    velocity_w: torch.Tensor
    visible: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    last_seen_position_w: torch.Tensor | None = None
    seen: torch.Tensor | None = None


def visual_actor_observations(
    privileged: torch.Tensor,
    tracks: list[VisualBallTracks] | tuple[VisualBallTracks, VisualBallTracks],
    root_positions_w: torch.Tensor,
    root_quats_xyzw: torch.Tensor,
    root_velocities_w: torch.Tensor,
    max_age: float = 0.15,
) -> torch.Tensor:
    """Replace both privileged ball blocks with RGB estimates and expose measurement age.

    ``privileged`` contains two legacy 71-value player blocks. Simulator ball
    channels 61:67 are always overwritten, including when a track is absent.
    Other robot/task channels are preserved. Each output block adds accepted
    measurement visibility, normalized age and the last observed bearing,
    producing 152 actor features. A stale bearing remains a search hint after
    the position estimate expires; it never becomes a valid ball position.
    World camera/root kinematics are calibrated odometry inputs; no ball truth
    or privileged ball-derived fallback is used for missing measurements.
    """
    if privileged.ndim != 2 or privileged.shape[1] != 142:
        raise ValueError("Privileged observations must have shape (batch, 142).")
    count = privileged.shape[0]
    if len(tracks) != 2 or root_positions_w.shape != (count, 2, 3):
        raise ValueError("Visual observations need two tracks and two root positions per world.")
    if root_quats_xyzw.shape != (count, 2, 4) or root_velocities_w.shape != (count, 2, 3):
        raise ValueError("Root orientations/velocities must match the two-player batch.")
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("Visual observation lifetime must be finite and positive.")
    blocks = privileged.view(count, 2, 71).clone()
    actor = []
    for player, track in enumerate(tracks):
        quat = root_quats_xyzw[:, player]
        position, velocity = root_positions_w[:, player], root_velocities_w[:, player]
        valid = track.valid & torch.isfinite(track.position_w).all(-1) & torch.isfinite(track.velocity_w).all(-1)
        valid &= torch.isfinite(track.age) & (track.age >= 0) & (track.age <= max_age + 1e-6)
        valid &= torch.isfinite(position).all(-1) & torch.isfinite(velocity).all(-1)
        valid &= torch.isfinite(quat).all(-1) & (quat.norm(dim=-1) > 1e-6)
        quat = F.normalize(torch.nan_to_num(quat), dim=-1)
        vectors = torch.stack((track.position_w - position, track.velocity_w - velocity), dim=1)
        inverse_vector = -quat[:, None, :3].expand(-1, 2, -1)
        first = 2 * torch.cross(inverse_vector, vectors, dim=-1)
        local = vectors + quat[:, None, 3:] * first + torch.cross(inverse_vector, first, dim=-1)
        blocks[:, player, 61:67] = torch.where(valid[:, None], local.flatten(1), 0.0)
        age = torch.where(valid, track.age / max_age, 1.0).clamp(0, 1)
        last_position = track.last_seen_position_w if track.last_seen_position_w is not None else track.position_w
        seen = track.seen if track.seen is not None else track.valid
        delta = torch.nan_to_num(last_position - position)
        inverse = -quat[:, :3]
        cross = 2 * torch.cross(inverse, delta, dim=-1)
        bearing = delta + quat[:, 3:] * cross + torch.cross(inverse, cross, dim=-1)
        bearing = F.normalize(bearing, dim=-1)
        bearing = torch.where((seen & torch.isfinite(last_position).all(-1))[:, None], bearing, 0.0)
        actor.append(
            torch.cat((blocks[:, player], (track.visible & valid).float()[:, None], age[:, None], bearing), dim=-1)
        )
    return torch.cat(actor, dim=-1)


class BallVisualTracker:
    """Find compact orange blobs and track a known-radius sphere in world coordinates."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        ball_radius: float = 0.025,
        max_age: float = 0.15,
        min_radius_px: float = 2.0,
        max_radius_fraction: float = 0.22,
        min_range: float = 0.06,
        max_range: float = 3.0,
        max_speed: float = 6.0,
        velocity_smoothing: float = 0.5,
        distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
        max_candidates: int = 4,
    ):
        if type(num_envs) is not int or num_envs < 1 or type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("Tracking requires a positive batch size and candidate count.")
        positive = (ball_radius, max_age, min_radius_px, min_range, max_range, max_speed)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Ball dimensions, tracking intervals, and speed/range bounds must be finite and positive.")
        if min_range >= max_range or not 0 < max_radius_fraction < 0.5 or not 0 < velocity_smoothing <= 1:
            raise ValueError("Tracking range, pixel radius, or velocity smoothing bounds are invalid.")
        if len(distortion) != 5 or not all(math.isfinite(value) for value in distortion):
            raise ValueError("OpenCV distortion must contain finite k1, k2, p1, p2, k3 coefficients.")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.ball_radius = ball_radius
        self.max_age = max_age
        self.min_radius_px = min_radius_px
        self.max_radius_fraction = max_radius_fraction
        self.min_range = min_range
        self.max_range = max_range
        self.max_speed = max_speed
        self.velocity_smoothing = velocity_smoothing
        self.max_candidates = max_candidates
        self.distortion = torch.tensor(distortion, device=self.device, dtype=torch.float32)
        self.device = self.distortion.device
        self.position_w = torch.zeros(num_envs, 3, device=self.device)
        self.velocity_w = torch.zeros_like(self.position_w)
        self.visible = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros_like(self.visible)
        self.age = torch.zeros(num_envs, device=self.device)
        self.confidence = torch.zeros_like(self.age)
        self._last_measurement = torch.zeros_like(self.position_w)
        self.last_seen_position_w = torch.zeros_like(self.position_w)
        self.seen = torch.zeros_like(self.visible)
        self._measurement_confidence = torch.zeros_like(self.age)
        self._acceleration = self.position_w.new_tensor((0.0, 0.0, -9.81))
        self._pixel_cache = {}

    def reset(self, env_ids=None):
        """Clear selected tracks without carrying old pixels or velocity across an episode reset."""
        if env_ids is None:
            env_ids = slice(None)
        for value in (
            self.position_w,
            self.velocity_w,
            self.visible,
            self.valid,
            self.age,
            self.confidence,
            self._last_measurement,
            self._measurement_confidence,
            self.last_seen_position_w,
            self.seen,
        ):
            value[env_ids] = 0

    def _undistort(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert Brown-Conrady distortion and reject nonconvergent/noninvertible estimates."""
        k1, k2, p1, p2, k3 = self.distortion.unbind()
        x, y = points[..., 0], points[..., 1]
        xd, yd = x.clone(), y.clone()
        for _ in range(12):
            r2 = x.square() + y.square()
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
            dy = p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
            safe = torch.where(radial.abs() > 1e-6, radial, torch.ones_like(radial))
            x, y = ((xd - dx) / safe).clamp(-4, 4), ((yd - dy) / safe).clamp(-4, 4)
        r2 = x.square() + y.square()
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
        dy = p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
        error = (x * radial + dx - xd).square() + (y * radial + dy - yd).square()
        derivative = 1 + 3 * k1 * r2 + 5 * k2 * r2.square() + 7 * k3 * r2.pow(3)
        good = torch.isfinite(error) & (error < 1e-8) & (radial > 0.1) & (derivative > 0.1)
        return torch.stack((x, y), dim=-1), good

    def _measure(self, rgb, intrinsics, camera_position_w, camera_quat_xyzw, prediction, prior):
        height, width = rgb.shape[1:3]
        key = (height, width)
        if key not in self._pixel_cache:
            yy, xx = torch.meshgrid(
                torch.arange(height, device=self.device, dtype=torch.float32),
                torch.arange(width, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            template = torch.arange(1, height * width + 1, device=self.device, dtype=torch.float32)
            counts = torch.zeros(self.num_envs, height * width + 1, device=self.device)
            self._pixel_cache[key] = (template.view(1, height, width), yy, xx, counts)
        template, yy, xx, counts = self._pixel_cache[key]
        pixels = rgb.to(torch.float32)
        red, green, blue = pixels.unbind(-1)
        # Ratios retain the shaded hemisphere seen in actual Newton camera frames.
        # Absolute brightness alone removed half the ball and corrupted monocular size/range.
        mask = (red >= 45) & (green >= 15) & (red - green >= 15)
        mask &= (red >= 1.30 * green) & (green >= 1.5 * blue + 4)
        labels = template.expand(self.num_envs, -1, -1) * mask
        for _ in range(math.ceil(math.log2(height * width)) + 2):
            labels = F.max_pool2d(labels.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) * mask
            parents = labels.flatten(1)
            labels = parents.gather(1, (parents.long() - 1).clamp_min(0)).view_as(labels) * mask
        counts.zero_()
        counts.scatter_add_(1, labels.flatten(1).long(), mask.flatten(1).float())
        candidates = counts.topk(min(self.max_candidates, height * width), dim=1).indices
        # All candidates share one geometry/undistortion pass. This preserves the
        # component rules while avoiding four copies of small GPU kernel launches.
        selected = (labels[:, None] == candidates[:, :, None, None]) & mask[:, None]
        area = selected.sum(dim=(-1, -2)).float()
        safe_area = area.clamp_min(1)
        u = (selected * xx).sum(dim=(-1, -2)) / safe_area
        v = (selected * yy).sum(dim=(-1, -2)) / safe_area
        var_x = (selected * (xx - u[:, :, None, None]).square()).sum(dim=(-1, -2)) / safe_area
        var_y = (selected * (yy - v[:, :, None, None]).square()).sum(dim=(-1, -2)) / safe_area
        low_x = torch.where(selected, xx, float(width)).amin(dim=(-1, -2))
        high_x = torch.where(selected, xx, -1.0).amax(dim=(-1, -2))
        low_y = torch.where(selected, yy, float(height)).amin(dim=(-1, -2))
        high_y = torch.where(selected, yy, -1.0).amax(dim=(-1, -2))
        box_width, box_height = high_x - low_x + 1, high_y - low_y + 1
        fill = area / (box_width * box_height).clamp_min(1)
        aspect = box_width / box_height.clamp_min(1)
        radius = torch.sqrt(area / math.pi)
        moment_error = ((var_x + var_y) * (2 * math.pi) / safe_area - 1).abs()
        good = (radius >= self.min_radius_px) & (radius <= min(height, width) * self.max_radius_fraction)
        good &= (low_x >= 1) & (low_y >= 1) & (high_x < width - 1) & (high_y < height - 1)
        good &= (aspect >= 0.65) & (aspect <= 1.55) & (fill >= 0.60) & (fill <= 0.94)
        intrinsics = intrinsics.float()
        pose_finite = torch.isfinite(camera_position_w).all(-1) & torch.isfinite(camera_quat_xyzw).all(-1)
        pose_finite &= camera_quat_xyzw.norm(dim=-1) > 1e-6
        calibration = torch.isfinite(intrinsics).all(dim=(-1, -2))
        calibration &= (intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0)
        calibration &= (intrinsics[:, 0, 1].abs() < 1e-6) & (intrinsics[:, 1, 0].abs() < 1e-6)
        good &= (moment_error < 0.30) & pose_finite[:, None] & calibration[:, None]
        radius_x, radius_y = 2 * var_x.sqrt(), 2 * var_y.sqrt()
        endpoint_u = torch.stack((u, u - radius_x, u + radius_x, u, u), dim=-1)
        endpoint_v = torch.stack((v, v, v, v - radius_y, v + radius_y), dim=-1)
        points = torch.stack(
            (
                (endpoint_u - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0].clamp_min(1e-6),
                (endpoint_v - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1].clamp_min(1e-6),
            ),
            dim=-1,
        )
        undistorted, converged = self._undistort(points)
        rays = F.normalize(torch.cat((undistorted, torch.ones_like(undistorted[..., :1])), dim=-1), dim=-1)
        angle_x = torch.acos((rays[:, :, 1] * rays[:, :, 2]).sum(-1).clamp(-1, 1))
        angle_y = torch.acos((rays[:, :, 3] * rays[:, :, 4]).sum(-1).clamp(-1, 1))
        angular_radius = (angle_x + angle_y) / 4
        distance = self.ball_radius / angular_radius.sin().clamp_min(1e-6)
        camera_point = rays[:, :, 0] * distance.unsqueeze(-1)
        quaternion = F.normalize(torch.nan_to_num(camera_quat_xyzw.float()), dim=-1)
        q_vector = quaternion[:, None, :3].expand_as(camera_point)
        q_scalar = quaternion[:, None, 3:4]
        first = 2 * torch.cross(q_vector, camera_point, dim=-1)
        world_point = camera_point + q_scalar * first + torch.cross(q_vector, first, dim=-1)
        world_point += camera_position_w[:, None]
        good &= converged.all(-1) & torch.isfinite(world_point).all(-1)
        good &= (distance >= self.min_range) & (distance <= self.max_range)
        score = (1 - (fill - math.pi / 4).abs() / 0.25).clamp_min(0)
        score *= (1 - moment_error / 0.30).clamp_min(0)
        error = (world_point - prediction[:, None]).norm(dim=-1)
        gate = (self.max_speed * (self.age + self._dt) + 0.08)[:, None]
        good &= ~prior[:, None] | (error <= gate)
        score *= torch.where(prior[:, None], torch.exp(-error.square() / gate.square().clamp_min(1e-6)), 1.0)
        scores = torch.where(good, score, 0.0)
        positions = torch.nan_to_num(world_point)
        best_score, best = scores.max(dim=1)
        if scores.shape[1] > 1:
            second = scores.topk(2, dim=1).values[:, 1]
            unambiguous = second < best_score * 0.75
        else:
            unambiguous = torch.ones_like(prior)
        accepted = (best_score >= 0.15) & unambiguous
        point = positions.gather(1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
        return point, best_score, accepted

    @property
    def tracks(self) -> VisualBallTracks:
        """Return a current snapshot without advancing time or inventing a fresh image."""
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

    def _prediction(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Tracking needs a finite positive elapsed time.")
        self._dt = dt
        acceleration = self._acceleration
        elapsed = self.age + dt
        prior = self.valid & (elapsed <= self.max_age + 1e-6)
        prediction = self.position_w + self.velocity_w * dt + 0.5 * acceleration * dt**2
        predicted_velocity = self.velocity_w + acceleration * dt
        predicted_velocity *= (self.max_speed / predicted_velocity.norm(dim=-1).clamp_min(self.max_speed)).unsqueeze(-1)
        return acceleration, elapsed, prior, prediction, predicted_velocity

    def predict(self, dt: float) -> VisualBallTracks:
        """Advance only the world-frame track when the camera has no new exposure."""
        state = self._prediction(dt)
        return self._commit(torch.zeros_like(self.position_w), self.confidence * 0, self.visible & False, state)

    def advance(self, dt: float):
        """Advance without allocating a snapshot; use ``tracks`` when policy observations are read."""
        state = self._prediction(dt)
        self._commit(
            torch.zeros_like(self.position_w), self.confidence * 0, self.visible & False, state, snapshot=False
        )

    def _commit(self, measurement, confidence, accepted, state, snapshot=True):
        acceleration, elapsed, prior, prediction, predicted_velocity = state
        measured_velocity = (measurement - self._last_measurement) / elapsed.clamp_min(1e-6).unsqueeze(-1)
        measured_velocity += 0.5 * acceleration * elapsed.unsqueeze(-1)
        measured_velocity = (
            self.velocity_smoothing * measured_velocity + (1 - self.velocity_smoothing) * predicted_velocity
        )
        measured_velocity *= (self.max_speed / measured_velocity.norm(dim=-1).clamp_min(self.max_speed)).unsqueeze(-1)
        measured_velocity = torch.where(prior.unsqueeze(-1), measured_velocity, 0.0)
        self.valid = accepted | prior
        self.visible = accepted
        self.position_w = torch.where(accepted.unsqueeze(-1), measurement, prediction)
        self.velocity_w = torch.where(accepted.unsqueeze(-1), measured_velocity, predicted_velocity)
        self.age = torch.where(accepted, 0.0, elapsed)
        self._last_measurement = torch.where(accepted.unsqueeze(-1), measurement, self._last_measurement)
        self.last_seen_position_w = torch.where(accepted.unsqueeze(-1), measurement, self.last_seen_position_w)
        self.seen |= accepted
        self._measurement_confidence = torch.where(accepted, confidence, self._measurement_confidence)
        self.confidence = torch.where(
            accepted, confidence, self._measurement_confidence * (1 - self.age / self.max_age)
        )
        for value in (self.position_w, self.velocity_w):
            value.masked_fill_(~self.valid.unsqueeze(-1), 0.0)
        self.confidence = torch.where(self.valid, self.confidence.clamp_min(0), 0.0)
        self.age = torch.where(self.valid, self.age, 0.0)
        self._last_measurement.masked_fill_(~self.valid.unsqueeze(-1), 0.0)
        self._measurement_confidence.masked_fill_(~self.valid, 0.0)
        return self.tracks if snapshot else None

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
        """Accept one fresh RGB frame and its synchronized calibration/world pose.

        The caller must guard camera frame IDs: repeatedly passing the same
        frame is not a new measurement. Nonfinite calibration/poses reject
        measurements and use the ordinary short dropout path. Set ``snapshot``
        false in a physics tick loop, then read ``tracks`` once at the policy cadence.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("A fresh frame needs a finite positive elapsed time.")
        if not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8:
            raise ValueError("RGB frames must be uint8 torch tensors.")
        if rgb.ndim != 4 or rgb.shape[0] != self.num_envs or rgb.shape[-1] != 3 or min(rgb.shape[1:3]) < 2:
            raise ValueError("RGB frames must have shape (batch, height>=2, width>=2, 3).")
        if rgb.shape[1] * rgb.shape[2] > 2**24:
            raise ValueError("RGB frames exceed exact component-label precision.")
        if intrinsics.shape != (self.num_envs, 3, 3) or camera_position_w.shape != (self.num_envs, 3):
            raise ValueError("Camera calibration and world position shapes do not match the tracker batch.")
        if camera_quat_xyzw.shape != (self.num_envs, 4):
            raise ValueError("Camera world orientation must have shape (batch, 4) in XYZW order.")
        if any(value.device != self.device for value in (rgb, intrinsics, camera_position_w, camera_quat_xyzw)):
            raise ValueError("RGB, camera poses, calibration, and tracker must use the same device.")
        if measurement_mask is not None and (
            measurement_mask.shape != (self.num_envs,)
            or measurement_mask.dtype != torch.bool
            or measurement_mask.device != self.device
        ):
            raise ValueError("A fresh-measurement mask must be a boolean tracker-device batch vector.")
        state = self._prediction(dt)
        _, _, prior, prediction, _ = state
        measurement, confidence, accepted = self._measure(
            rgb, intrinsics, camera_position_w, camera_quat_xyzw, prediction, prior
        )
        if measurement_mask is not None:
            accepted &= measurement_mask
        return self._commit(measurement, confidence, accepted, state, snapshot=snapshot)
