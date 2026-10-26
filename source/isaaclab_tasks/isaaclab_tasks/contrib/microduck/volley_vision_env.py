# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual head-camera pixels replace all simulator ball position/velocity inputs to the actor."""

import math

import torch

from .ball_vision import BallVisualTracker, visual_actor_observations
from .volley_curriculum_env import MicroDuckVolleyCurriculumEnv
from .volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg


class MicroDuckVolleyVisionEnv(MicroDuckVolleyCurriculumEnv):
    """Train all joints using monocular ball tracking and a privileged value function.

    Each camera runs at its own sensor cadence. Tracking advances at physics
    resolution, so velocities use actual fresh-frame intervals even though the
    50 Hz policy sometimes sees a repeated frame. Physics resets clear selected
    trackers without initializing them from the simulated ball. Camera poses
    and root poses represent the calibrated kinematic/odometry reference; this
    task does not yet model estimation error in that reference.
    """

    cfg: MicroDuckVolleyVisionEnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.head_cameras = [self.scene[f"head_camera_{side}"] for side in ("a", "b")]
        self._visual_trackers = [
            BallVisualTracker(
                self.num_envs,
                self.device,
                ball_radius=cfg.ball_radius,
                max_age=cfg.vision_max_age,
                distortion=self._distortion(camera.cfg),
            )
            for camera in self.head_cameras
        ]
        self._last_camera_frames = [torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for _ in range(2)]
        self._camera_capture_steps = [
            max(1, math.ceil(camera.cfg.update_period / cfg.sim.dt - 1e-6)) for camera in self.head_cameras
        ]
        self.head_camera_images = [
            torch.zeros(self.num_envs, camera.cfg.height, camera.cfg.width, 3, dtype=torch.uint8, device=self.device)
            for camera in self.head_cameras
        ]
        self._camera_frames_seen = torch.zeros(2, device=self.device)
        self._visual_detections = torch.zeros_like(self._camera_frames_seen)
        self._visual_error_sum = torch.zeros_like(self._camera_frames_seen)
        self._visual_age_sum = torch.zeros_like(self._camera_frames_seen)
        self._vision_ticks = 0
        self._camera_panels = None
        if cfg.head_camera_panels:
            from .vision_viewer import HeadCameraPanels

            self._camera_panels = HeadCameraPanels(self.sim, max_fps=cfg.head_camera_panel_fps)

    @property
    def vision_tracks(self):
        """Current RGB measurements/predictions; reading this does not advance the tracker."""
        return [tracker.tracks for tracker in self._visual_trackers]

    @staticmethod
    def _distortion(camera_cfg):
        distortion = camera_cfg.spawn.distortion
        if distortion is None or not distortion.apply_lens_distortion:
            return (0.0,) * 5
        return tuple(getattr(distortion, name) for name in ("k1", "k2", "p1", "p2", "k3"))

    def _update_contacts(self):
        super()._update_contacts()
        if not hasattr(self, "_visual_trackers"):
            return
        self._vision_ticks += 1
        capture_due = [self._vision_ticks % interval == 0 for interval in self._camera_capture_steps]
        if any(capture_due):
            self.sim.render_context.update_scene_state(self.sim.get_physics_step_count())
        for player, (camera, tracker) in enumerate(zip(self.head_cameras, self._visual_trackers)):
            if capture_due[player]:
                # One global capture clock keeps all worlds in a camera batch synchronized,
                # even after independent resets. Contact sensors still run at every physics substep.
                data = camera.data
                frame = camera.frame.torch
                fresh = frame > self._last_camera_frames[player]
                tracker.update(
                    data.output["rgb"].torch,
                    data.intrinsic_matrices.torch,
                    data.pos_w.torch,
                    data.quat_w_ros.torch,
                    dt=self.cfg.sim.dt,
                    measurement_mask=fresh,
                    snapshot=False,
                )
                self.head_camera_images[player] = data.output["rgb"].torch
                self._last_camera_frames[player].copy_(frame)
                self._camera_frames_seen[player] += fresh.sum()
                self._visual_detections[player] += tracker.visible.sum()
                # Truth is diagnostic only; never initialize or correct a visual track with it.
                error = (tracker.position_w - self._ball.data.root_link_pos_w.torch).norm(dim=-1)
                self._visual_error_sum[player] += torch.where(tracker.visible, error, 0.0).sum()
            else:
                tracker.advance(self.cfg.sim.dt)
            self._visual_age_sum[player] += tracker.age.clamp_max(self.cfg.vision_max_age).sum()

    def _get_observations(self):
        privileged = super()._get_observations()["policy"]
        tracks = self.vision_tracks
        actor = visual_actor_observations(
            privileged,
            tracks,
            torch.stack([robot.data.root_link_pos_w.torch for robot in self._robots], dim=1),
            torch.stack([robot.data.root_link_quat_w.torch for robot in self._robots], dim=1),
            torch.stack([robot.data.root_link_lin_vel_w.torch for robot in self._robots], dim=1),
            self.cfg.vision_max_age,
        )
        if self._camera_panels is not None:
            self._camera_panels.update(self.head_cameras, tracks, images=self.head_camera_images)
        return {"policy": actor, "critic": privileged}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_visual_trackers"):
            for camera, tracker, frame in zip(self.head_cameras, self._visual_trackers, self._last_camera_frames):
                tracker.reset(env_ids)
                # Parent resets can eagerly capture intermediate robot/ball poses at dt=0.
                # Ignore every such image; the next positive-time camera exposure is the first usable one.
                frame[env_ids] = camera.frame.torch[env_ids]

    def vision_diagnostics(self):
        """Report runtime/perception quality separately from the physical volleyball success gate."""
        detections = self._visual_detections.clamp_min(1)
        return {
            "camera_frames_seen": self._camera_frames_seen.long().tolist(),
            "accepted_rgb_measurements": self._visual_detections.long().tolist(),
            "mean_position_error_m": (self._visual_error_sum / detections).tolist(),
            "position_error_samples": self._visual_detections.long().tolist(),
            "mean_track_age_s": (self._visual_age_sum / max(1, self._vision_ticks * self.num_envs)).tolist(),
            "physics_ticks": self._vision_ticks,
            "requested_capture_period_s": self.head_cameras[0].cfg.update_period,
            "capture_steps": self._camera_capture_steps,
            "physics_dt_s": self.cfg.sim.dt,
            "policy_dt_s": self.step_dt,
            "actor_uses_ball_truth": False,
            "critic_uses_ball_truth": True,
            "scope": "Cumulative RGB runtime/calibration diagnostics across all worlds, not a held-out skill gate.",
            "panels": self._camera_panels.stats() if self._camera_panels is not None else None,
        }

    def vision_contract(self):
        """Persist the camera and observation definition alongside optimizer/curriculum state."""
        tracker = self._visual_trackers[0]
        return {
            "version": 2,
            "actor_width": 146,
            "critic_width": 142,
            "ball_source": "rgb_monocular_orange_sphere",
            "cameras": [
                {
                    "prim_path": camera.cfg.prim_path,
                    "raw_image_size": [camera.cfg.width, camera.cfg.height],
                    "requested_capture_period_s": camera.cfg.update_period,
                    "position_in_head": list(camera.cfg.offset.pos),
                    "rotation_xyzw_in_head": list(camera.cfg.offset.rot),
                    "convention": camera.cfg.offset.convention,
                    "distortion": list(self._distortion(camera.cfg)),
                    "optics": {
                        name: getattr(camera.cfg.spawn, name)
                        for name in (
                            "focal_length",
                            "horizontal_aperture",
                            "vertical_aperture",
                            "horizontal_aperture_offset",
                            "vertical_aperture_offset",
                            "clipping_range",
                        )
                    },
                    "calibration": {
                        name: getattr(camera.cfg.spawn.distortion, name, None)
                        for name in ("fx", "fy", "cx", "cy", "image_size", "apply_lens_distortion")
                    },
                }
                for camera in self.head_cameras
            ],
            "physics_dt_s": self.cfg.sim.dt,
            "policy_dt_s": self.step_dt,
            "capture_steps": self._camera_capture_steps,
            "capture_clock": "global_physics_steps",
            "tracker": {
                name: getattr(tracker, name)
                for name in (
                    "ball_radius",
                    "max_age",
                    "min_radius_px",
                    "max_radius_fraction",
                    "min_range",
                    "max_range",
                    "max_speed",
                    "velocity_smoothing",
                    "max_candidates",
                )
            },
        }
