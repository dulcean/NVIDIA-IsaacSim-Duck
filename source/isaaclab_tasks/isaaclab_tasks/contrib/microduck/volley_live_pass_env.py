# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real head-to-head passes with camera-only ball/partner estimates for the actor."""

import torch

from isaaclab.utils.math import quat_apply

from .live_pass_curriculum import VISION_STAGES
from .live_pass_observations import live_pass_actor_observations
from .partner_vision import (
    PARTNER_CYAN_POSITION,
    PARTNER_MAGENTA_POSITION,
    PARTNER_MARKER_RADIUS,
    PartnerHeadVisualTracker,
)
from .passing import plan_ballistic_pass
from .volley_live_pass_env_cfg import MicroDuckLivePassEnvCfg
from .volley_vision_env import MicroDuckVolleyVisionEnv


class MicroDuckLivePassEnv(MicroDuckVolleyVisionEnv):
    """Physical truth labels rewards and the critic; both actor object states come from RGB."""

    cfg: MicroDuckLivePassEnvCfg
    stages = VISION_STAGES

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._partner_trackers = [
            PartnerHeadVisualTracker(
                self.num_envs, self.device, max_age=cfg.partner_max_age, distortion=self._distortion(camera.cfg)
            )
            for camera in self.head_cameras
        ]
        self._incoming_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_release = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._reference_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._partner_visible_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._partner_valid_time = torch.zeros_like(self._partner_visible_time)
        self._partner_active_time = torch.zeros(self.num_envs, device=self.device)
        self._partner_seen = torch.zeros_like(self._reference_valid)
        self._partner_rgb_pairs = torch.zeros(2, device=self.device)
        self._partner_error_sum = torch.zeros(2, device=self.device)
        self._release_samples = torch.zeros(self.num_envs, device=self.device)
        self._release_error_sum = torch.zeros_like(self._release_samples)
        self._release_speed_sum = torch.zeros_like(self._release_samples)
        self._required_speed_sum = torch.zeros_like(self._release_samples)

    @property
    def partner_tracks(self):
        return [tracker.tracks for tracker in self._partner_trackers]

    def _head_velocities(self):
        velocities = []
        for robot, body in zip(self._robots, self._head_ids):
            data = robot.data
            offset = quat_apply(
                data.body_link_quat_w.torch[:, body],
                self._actions.new_tensor(self.cfg.strike_offset).expand(self.num_envs, -1),
            )
            velocities.append(
                data.body_link_lin_vel_w.torch[:, body]
                + torch.cross(data.body_link_ang_vel_w.torch[:, body], offset, dim=-1)
            )
        return torch.stack(velocities, dim=1)

    def _refresh_targets(self):
        if hasattr(self, "_targets"):
            self._targets.copy_(self._head_positions().flip(1))
            self._targets[:, :, 2] += self.cfg.ball_radius

    def _arrange_players(self, env_ids):
        count = len(env_ids)
        for robot in self._robots:
            pose = robot.data.root_link_pose_w.torch[env_ids].clone()
            pose[:, 0] += self._uniform(count, -self.cfg.root_position_jitter, self.cfg.root_position_jitter)
            pose[:, 1] += self._uniform(count, -self.cfg.root_lateral_jitter, self.cfg.root_lateral_jitter)
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _new_strike_reference(self, player, hit):
        if not hasattr(self, "_desired_release"):
            return
        # Previous substep's incoming velocity precedes the collision impulse.
        selected = hit.nonzero(as_tuple=True)[0]
        incoming = (
            self._incoming_velocity[selected] + self._incoming_velocity.new_tensor((0.0, 0.0, -9.81)) * self.cfg.sim.dt
        )
        reference = plan_ballistic_pass(
            self._ball.data.root_link_pos_w.torch[selected],
            incoming,
            self._targets[selected, player],
            self._head_velocities()[selected, 1 - player],
            center_x=self.scene.env_origins[selected, 0],
            min_flight=max(0.2, self.cfg.minimum_flight_time),
            min_crossing_height=self.cfg.minimum_crossing_height,
            ball_mass=self.cfg.scene.ball.spawn.mass_props.mass,
        )
        self._desired_release[selected, player] = reference.outgoing_velocity
        self._reference_valid[selected, player] = reference.valid

    def _release_shaping(self, player, ball, quality):
        elapsed = self._launch_age[:, player]
        desired = self._desired_release[:, player] + ball.new_tensor((0.0, 0.0, -9.81)) * elapsed[:, None]
        error = (self._ball.data.root_link_lin_vel_w.torch - desired).square().sum(-1) / 0.75**2
        ready = (
            self._launch_pending[:, player]
            & ~self._contact[:, player]
            & (elapsed >= self.cfg.release_delay)
            & self._reference_valid[:, player]
            & (self._episode.phase == 0)
        )
        self._release_samples += ready
        self._release_error_sum += torch.where(ready, error.sqrt() * 0.75, 0.0)
        self._release_speed_sum += torch.where(ready, self._ball.data.root_link_lin_vel_w.torch.norm(dim=-1), 0.0)
        self._required_speed_sum += torch.where(ready, desired.norm(dim=-1), 0.0)
        return torch.nan_to_num(1 / (1 + error), nan=0.0) * self._reference_valid[:, player]

    def _update_contacts(self):
        previous_frames = (
            [frame.clone() for frame in self._last_camera_frames] if hasattr(self, "_partner_trackers") else []
        )
        super()._update_contacts()
        if not previous_frames:
            return
        for player, (camera, tracker) in enumerate(zip(self.head_cameras, self._partner_trackers)):
            if self._vision_ticks % self._camera_capture_steps[player] == 0:
                data = camera.data
                tracker.update(
                    self.head_camera_images[player],
                    data.intrinsic_matrices.torch,
                    data.pos_w.torch,
                    data.quat_w_ros.torch,
                    dt=self.cfg.sim.dt,
                    measurement_mask=camera.frame.torch > previous_frames[player],
                    snapshot=False,
                )
            else:
                tracker.advance(self.cfg.sim.dt)
        active = self._episode.phase == 0
        actual = self._head_positions().flip(1)
        for player, tracker in enumerate(self._partner_trackers):
            error = (tracker.position_w - actual[:, player]).norm(dim=-1)
            matched = error <= 0.06
            self._partner_rgb_pairs[player] += tracker.visible.sum()
            self._partner_error_sum[player] += torch.where(tracker.visible, error, 0.0).sum()
            recent = tracker.valid & (tracker.age <= 2 * self._camera_capture_steps[player] * self.cfg.sim.dt) & matched
            self._partner_visible_time[:, player] += (recent & active) * self.cfg.sim.dt
            self._partner_valid_time[:, player] += (tracker.valid & active) * self.cfg.sim.dt
            self._partner_seen[:, player] |= tracker.visible & matched & active
        self._partner_active_time += active * self.cfg.sim.dt
        self._incoming_velocity.copy_(self._ball.data.root_link_lin_vel_w.torch)

    def _get_observations(self):
        # Parent provides the physical critic and updates display panels. Its
        # temporary actor is discarded; this helper owns the RGB-only boundary.
        parent = super()._get_observations()
        actor = live_pass_actor_observations(
            parent["critic"],
            self.vision_tracks,
            self.partner_tracks,
            torch.stack([robot.data.root_link_pos_w.torch for robot in self._robots], dim=1),
            torch.stack([robot.data.root_link_quat_w.torch for robot in self._robots], dim=1),
            torch.stack([robot.data.root_link_lin_vel_w.torch for robot in self._robots], dim=1),
            self._head_positions(),
            self._head_velocities(),
            env_origins=self.scene.env_origins,
            ball_radius=self.cfg.ball_radius,
            max_ball_age=self.cfg.vision_max_age,
            max_partner_age=self.cfg.partner_max_age,
        )
        return {"policy": actor, "critic": parent["critic"]}

    def _stage_goal(self):
        goal = super()._stage_goal()
        if not hasattr(self, "_partner_seen"):
            return goal & False
        if self.stage < 2:
            goal &= self._partner_seen.gather(1, self._server[:, None]).squeeze(1)
        else:
            goal &= self._partner_seen.all(-1)
        if self.stage == len(self.stages) - 1:
            visibility = self._partner_visible_time / self._partner_active_time.clamp_min(self.step_dt)[:, None]
            goal &= self._partner_seen.all(-1) & (visibility >= self.cfg.final_min_partner_visibility).all(-1)
        return goal

    def _episode_record_fields(self, env_ids):
        fields = super()._episode_record_fields(env_ids)
        denominator = self._partner_active_time[env_ids].clamp_min(self.step_dt)
        for player, side in enumerate(("a", "b")):
            fields[f"partner_visible_{side}"] = self._partner_visible_time[env_ids, player] / denominator
            fields[f"partner_valid_{side}"] = self._partner_valid_time[env_ids, player] / denominator
            fields[f"partner_seen_{side}"] = self._partner_seen[env_ids, player].float()
        count = self._release_samples[env_ids].clamp_min(1)
        fields.update(
            pass_release_samples=self._release_samples[env_ids],
            pass_velocity_error_m_s=self._release_error_sum[env_ids] / count,
            pass_release_speed_m_s=self._release_speed_sum[env_ids] / count,
            pass_required_speed_m_s=self._required_speed_sum[env_ids] / count,
        )
        return fields

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_partner_trackers"):
            return
        for tracker in self._partner_trackers:
            tracker.reset(env_ids)
        for buffer in (
            self._desired_release,
            self._reference_valid,
            self._partner_visible_time,
            self._partner_valid_time,
            self._partner_active_time,
            self._partner_seen,
            self._release_samples,
            self._release_error_sum,
            self._release_speed_sum,
            self._required_speed_sum,
        ):
            buffer[env_ids] = 0
        self._incoming_velocity[env_ids] = self._ball.data.root_link_lin_vel_w.torch[env_ids]
        records = self.extras.get("episode_records", [])
        if records:
            self.extras.setdefault("log", {}).update(
                {
                    f"Metrics/{key}": sum(record[key] for record in records) / len(records)
                    for key in (
                        "partner_visible_a",
                        "partner_visible_b",
                        "partner_valid_a",
                        "partner_valid_b",
                        "pass_release_samples",
                        "pass_velocity_error_m_s",
                        "pass_release_speed_m_s",
                        "pass_required_speed_m_s",
                    )
                }
            )

    def vision_contract(self):
        contract = super().vision_contract()
        contract.update(
            actor_width=178,
            target_source="rgb_paired_head_badges",
            target_geometry="live_physical_forehead",
            stage_separations=list(self.cfg.stage_separations),
            root_jitter_m=[self.cfg.root_position_jitter, self.cfg.root_lateral_jitter],
            actor_information="independent89input14servoactors; no other-player proprioception",
            partner={
                "radius_m": PARTNER_MARKER_RADIUS,
                "cyan_position": list(PARTNER_CYAN_POSITION),
                "magenta_position": list(PARTNER_MAGENTA_POSITION),
                "reconstruction": "2*cyan-magenta",
                "max_age_s": self.cfg.partner_max_age,
                "final_min_visibility": self.cfg.final_min_partner_visibility,
            },
            planner={"min_flight_s": 0.2, "max_flight_s": 0.45, "max_speed_m_s": 3.5, "velocity_tolerance_m_s": 0.75},
        )
        return contract

    def vision_diagnostics(self):
        report = super().vision_diagnostics()
        report["partner"] = {
            "rgb_pairs": self._partner_rgb_pairs.detach().cpu().tolist(),
            "mean_localization_error_m": (self._partner_error_sum / self._partner_rgb_pairs.clamp_min(1))
            .detach()
            .cpu()
            .tolist(),
            "source": "two calibrated colored head badges in actual RGB frames",
            "truth_use": "Reward/validation labels only; never tracker initialization or actor fallback.",
        }
        return report

    def validate_source_contract(self, metadata):
        current = self.vision_contract()
        if metadata.get("expanded_visual_warm_start"):
            previous = metadata.get("source_vision_contract") or {}
            keys = ("cameras", "physics_dt_s", "policy_dt_s", "capture_steps", "capture_clock", "tracker")
            if any(previous.get(key) != current.get(key) for key in keys):
                raise ValueError("Source camera/ball-tracker calibration differs from the live-pass task.")
        elif metadata.get("vision_contract") != current:
            raise ValueError("Saved live-pass camera, partner or planning contract differs from this task.")
