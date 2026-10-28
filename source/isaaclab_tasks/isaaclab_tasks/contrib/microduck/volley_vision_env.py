# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual head-camera pixels replace all simulator ball position/velocity inputs to the actor."""

import math

import torch

from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_mul

from .active_vision import MeasuredVisibilityWindow, active_gaze_reward, camera_gaze_geometry
from .ball_serves import frontal_serve_source_offset
from .ball_vision import BallVisualTracker, visual_actor_observations
from .vision_curriculum import VISION_STAGES
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
    stages = VISION_STAGES

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
        self._visibility = MeasuredVisibilityWindow(
            self.num_envs, self.device, 2 * max(self._camera_capture_steps) * cfg.sim.dt
        )
        self._gaze_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._visible_time = torch.zeros_like(self._gaze_time)
        self._alignment_time = torch.zeros_like(self._gaze_time)
        self._gaze_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._camera_offsets = torch.tensor([camera.cfg.offset.pos for camera in self.head_cameras], device=self.device)
        self._camera_rotations = torch.stack(
            [
                convert_camera_frame_orientation_convention(
                    torch.tensor(camera.cfg.offset.rot, device=self.device).unsqueeze(0),
                    origin=camera.cfg.offset.convention,
                    target="ros",
                )[0]
                for camera in self.head_cameras
            ]
        )
        calibration = [camera.cfg.spawn.distortion for camera in self.head_cameras]
        self._gaze_intrinsics = (
            torch.tensor(
                [[[d.fx, 0.0, d.cx], [0.0, d.fy, d.cy], [0.0, 0.0, 1.0]] for d in calibration], device=self.device
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1, -1)
        )
        self._gaze_distortion = torch.stack([tracker.distortion for tracker in self._visual_trackers])
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
        if self.stages[self.stage].objective == "watch_visible_ball":
            # The acquisition lesson graduates by sustained pixels before an impact.
            self._game_invalid |= (self._raw_hits > 0) & (self._episode.phase == 0)
        capture_due = [self._vision_ticks % interval == 0 for interval in self._camera_capture_steps]
        if any(capture_due):
            self.sim.render_context.update_scene_state(self.sim.get_physics_step_count())
        fresh_masks, matched_masks = [], []
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
                # Truth labels reward/qualification and diagnostics; it never
                # initializes or corrects a visual track or an actor observation.
                error = (tracker.position_w - self._ball.data.root_link_pos_w.torch).norm(dim=-1)
                self._visual_error_sum[player] += torch.where(tracker.visible, error, 0.0).sum()
                distance = (self._ball.data.root_link_pos_w.torch - data.pos_w.torch).norm(dim=-1)
                matched = fresh & tracker.visible & (error <= (0.15 * distance).clamp_min(0.05))
            else:
                tracker.advance(self.cfg.sim.dt)
                fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                matched = fresh
            fresh_masks.append(fresh)
            matched_masks.append(matched)
            self._visual_age_sum[player] += tracker.age.clamp_max(self.cfg.vision_max_age).sum()
        # A projected label alone cannot satisfy the lesson: each accepted interval
        # requires actual RGB evidence associated with the physical ball.
        matched = torch.stack(matched_masks, dim=1)
        if any(capture_due):
            matched &= self._gaze_geometry().in_frame
        matched &= (self._episode.phase == 0).unsqueeze(-1)
        self._visibility.update(matched, torch.stack(fresh_masks, dim=1), self.cfg.sim.dt)

    def _gaze_geometry(self):
        """Current physical optical poses for reward labels; never an actor ball input."""
        positions = torch.stack(
            [robot.data.body_link_pos_w.torch[:, head] for robot, head in zip(self._robots, self._head_ids)], dim=1
        )
        rotations = torch.stack(
            [robot.data.body_link_quat_w.torch[:, head] for robot, head in zip(self._robots, self._head_ids)], dim=1
        )
        positions = positions + quat_apply(rotations, self._camera_offsets.unsqueeze(0).expand(self.num_envs, -1, -1))
        rotations = quat_mul(rotations, self._camera_rotations.unsqueeze(0).expand(self.num_envs, -1, -1))
        camera = self.head_cameras[0].cfg
        return camera_gaze_geometry(
            self._ball.data.root_link_pos_w.torch,
            positions,
            rotations,
            self._gaze_intrinsics,
            (camera.width, camera.height),
            self._gaze_distortion,
        )

    def _stage_goal(self):
        objective = self.stages[self.stage].objective
        if objective == "watch_visible_ball":
            hold = self._visibility.hold.gather(1, self._server[:, None]).squeeze(1)
            recent = self._visibility.recent.gather(1, self._server[:, None]).squeeze(1)
            return recent & (hold + 1e-6 >= self.cfg.watch_duration) & (self._raw_hits == 0)
        if objective == "receive_forehead":
            return (self._hits + self._hit_event) > 0
        goal = super()._stage_goal()
        if self.stage == len(self.stages) - 1:
            # Qualified visual rallies must include measured RGB tracking by both
            # players, rather than only a memorized motor sequence for one serve.
            visible = self._visible_time / self._gaze_time.clamp_min(self.step_dt)
            goal &= (visible >= self.cfg.final_min_gaze_visibility).all(-1)
        return goal

    def _serve_height_range(self):
        return self.cfg.stage_serve_heights[self.stage]

    def _serve_source_offset(self, receiving, partner, lateral_fraction):
        limits = getattr(self.cfg, "stage_source_distance_limits", None)
        return frontal_serve_source_offset(
            receiving,
            partner,
            lateral_fraction,
            self.cfg.stage_source_distances[self.stage],
            distance_limits=limits[self.stage] if limits is not None else None,
        )

    def _get_rewards(self):
        reward = super()._get_rewards()
        geometry = self._gaze_geometry()
        accepted = self._visibility.recent & geometry.in_frame
        gaze = active_gaze_reward(
            geometry.alignment,
            accepted,
            (self._episode.phase == 0) & ~self._unsafe,
            self._contact,
            self._own_hit_age,
            self.step_dt,
            grace=self.cfg.gaze_strike_grace,
            alignment_rate=self.cfg.gaze_alignment_rate,
            visibility_rate=self.cfg.gaze_visibility_rate,
        )
        self._gaze_time += gaze.eligible * self.step_dt
        self._visible_time += (accepted & gaze.eligible) * self.step_dt
        self._alignment_time += geometry.alignment * gaze.eligible * self.step_dt
        self._gaze_reward_sum += gaze.reward
        return reward + gaze.reward

    def _episode_record_fields(self, env_ids):
        denominator = self._gaze_time[env_ids].clamp_min(self.step_dt)
        visible = self._visible_time[env_ids] / denominator
        alignment = self._alignment_time[env_ids] / denominator
        return {
            "gaze_visible_a": visible[:, 0],
            "gaze_visible_b": visible[:, 1],
            "gaze_alignment": alignment.mean(-1),
            "gaze_reward": self._gaze_reward_sum[env_ids],
            "watch_visible_s": self._visibility.peak[env_ids].gather(1, self._server[env_ids, None]).squeeze(1),
        }

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
            self._visibility.reset(env_ids)
            for buffer in (self._gaze_time, self._visible_time, self._alignment_time, self._gaze_reward_sum):
                buffer[env_ids] = 0
            records = self.extras.get("episode_records", [])
            if records:
                for name in ("gaze_visible_a", "gaze_visible_b", "gaze_alignment", "gaze_reward", "watch_visible_s"):
                    self.extras.setdefault("log", {})[f"Metrics/{name}"] = sum(r[name] for r in records) / len(records)

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
            "version": 3,
            "actor_width": 152,
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
            "curriculum_profile": self.cfg.curriculum_profile,
            "actor_search_memory": "last_seen_world_point_as_root_local_unit_bearing_until_episode_reset",
            "stage_serve_heights": [list(bounds) for bounds in self.cfg.stage_serve_heights],
            "watch_duration_s": self.cfg.watch_duration,
            "stage_source_distances_m": list(self.cfg.stage_source_distances),
            "final_min_gaze_visibility": self.cfg.final_min_gaze_visibility,
            "gaze": {
                "alignment_rate": self.cfg.gaze_alignment_rate,
                "visibility_rate": self.cfg.gaze_visibility_rate,
                "strike_grace_s": self.cfg.gaze_strike_grace,
                "maximum_measurement_error": "max(0.05m,0.15*true_camera_range)",
            },
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
