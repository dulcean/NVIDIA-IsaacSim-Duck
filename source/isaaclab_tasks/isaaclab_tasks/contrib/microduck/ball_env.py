# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole-body heading with physical contacts and alternating cooperative returns."""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

from .ball_env_cfg import MicroDuckHeadingEnvCfg
from .ball_rules import PostRallyRecovery, RecoveryRewards, balance_errors, classify_head_contact, robot_collision_mask
from .heading_geometry import camera_alignment, front_strike_mask


class MicroDuckBallEnv(DirectRLEnv):
    """One centralized policy controls one robot, or both robots in a cooperative rally.

    Each 71-value block starts with the walking policy's 61-value layout, followed by
    ball position/relative velocity, receiver position (all in the root frame), and
    whether this robot should touch next. Observations use simulator state, not cameras.
    """

    cfg: MicroDuckHeadingEnvCfg

    def __init__(self, cfg: MicroDuckHeadingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Read contact history at every physics substep, including brief ball impacts.
        # Keep BAM running in Newton, but use the environment-owned decimation loop.
        self.sim.backend_manager.set_decimation(1)
        self._physics_handles_decimation = False
        self._robots = [self.scene[f"robot_{name}"] for name in ("a", "b")[: cfg.num_players]]
        self._sensors = [self.scene[f"ball_contacts_{name}"] for name in ("a", "b")[: cfg.num_players]]
        self._ball = self.scene["ball"]
        self._robot_contacts = self.scene["robot_contacts"] if cfg.num_players == 2 else None
        if self._robot_contacts is not None and self._robot_contacts.num_filter_objects == 0:
            raise ValueError("Robot collision sensor must match the other robot's rigid bodies.")
        self._joint_ids = []
        self._play_ids = []
        self._head_ids = []
        self._head_masks = []
        for robot, sensor in zip(self._robots, self._sensors):
            ids, _ = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
            play_ids, _ = robot.find_joints(
                [f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES], preserve_order=True
            )
            head_ids, head_names = robot.find_bodies(cfg.head_reference_body)
            contact_ids, contact_names = sensor.find_sensors(cfg.head_body_pattern)
            if len(ids) != 14 or len(play_ids) != 14 or len(head_ids) != 1 or len(contact_ids) != 1:
                raise ValueError("MicroDuck ball task requires 14 servos/play hinges and resolved head bodies.")
            if contact_names != head_names:
                raise ValueError("Strike contacts and camera geometry must use the same rigid head frame.")
            if sensor.num_filter_objects != 1:
                raise ValueError(f"Expected exactly one ball contact counterpart, got {sensor.filter_object_names}.")
            mask = torch.zeros(sensor.num_sensors, dtype=torch.bool, device=self.device)
            mask[contact_ids] = True
            self._joint_ids.append(ids)
            self._play_ids.append(play_ids)
            self._head_ids.append(head_ids[0])
            self._head_masks.append(mask)
            print(f"[MicroDuck ball] {robot.cfg.prim_path}: head={head_names}, hit colliders={contact_names}")
            forward = quat_apply(
                robot.data.body_link_quat_w.torch[:1, head_ids[0]],
                torch.tensor([cfg.camera_forward_axis], device=self.device),
            )[0].tolist()
            print(f"[MicroDuck ball] Initial camera axis in world: {forward}; up alignment={forward[2]:.3f}")

        shape = (self.num_envs, cfg.num_players)
        self._recovery = RecoveryRewards(self.num_envs, cfg.num_players, self.device)
        self._post_rally = PostRallyRecovery(self.num_envs, self.device)
        self._post_recovered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._fell_after_miss = torch.zeros_like(self._post_recovered)
        self._rally_invalid = torch.zeros_like(self._post_recovered)
        self._miss_event = torch.zeros_like(self._post_recovered)
        self._rally_steps = torch.zeros(self.num_envs, device=self.device)
        self._gaze_steps = torch.zeros_like(self._rally_steps)
        self._player_hit_event = torch.zeros(*shape, device=self.device)
        self._player_aim_event = torch.zeros_like(self._player_hit_event)
        self._player_return_event = torch.zeros_like(self._player_hit_event)
        self._recovered_hits = torch.zeros(self.num_envs, device=self.device)
        self._recovery_reward_sum = torch.zeros_like(self._recovered_hits)
        self._actions = torch.zeros(*shape, 14, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._contact = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self._last_hitter = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._server = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._since_hit = torch.zeros(self.num_envs, device=self.device)
        self._hits = torch.zeros(self.num_envs, device=self.device)
        self._raw_hits = torch.zeros_like(self._hits)
        self._gaze_sum = torch.zeros_like(self._hits)
        self._returns = torch.zeros_like(self._hits)
        self._hit_event = torch.zeros_like(self._hits)
        self._return_event = torch.zeros_like(self._hits)
        self._aim_event = torch.zeros_like(self._hits)
        self._invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._robot_collision = torch.zeros_like(self._invalid)
        self._bad_surface = torch.zeros_like(self._invalid)
        self._bad_turn = torch.zeros_like(self._invalid)
        self._bad_body = torch.zeros_like(self._invalid)
        self._ground = torch.zeros_like(self._invalid)
        self._outside = torch.zeros_like(self._invalid)
        self._ball_outside = torch.zeros_like(self._invalid)
        self._robot_outside = torch.zeros_like(self._invalid)
        self._crossed_center = torch.zeros_like(self._invalid)
        self._fast_returns = torch.zeros_like(self._hits)
        self._fast_event = torch.zeros_like(self._hits)
        self._fell = torch.zeros_like(self._invalid)
        self._success = torch.zeros_like(self._invalid)
        self._last_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target = torch.zeros_like(self._last_ball_pos)
        self._max_height = torch.zeros_like(self._hits)

    def _head_positions(self) -> torch.Tensor:
        return torch.stack(
            [
                robot.data.body_link_pos_w.torch[:, body]
                + quat_apply(
                    robot.data.body_link_quat_w.torch[:, body],
                    self._actions.new_tensor(self.cfg.strike_offset).expand(self.num_envs, -1),
                )
                for robot, body in zip(self._robots, self._head_ids)
            ],
            dim=1,
        )

    def _receivers(self) -> torch.Tensor:
        heads = self._head_positions()
        if self.cfg.num_players == 2:
            return heads.flip(1)
        return self._target.unsqueeze(1)

    def _expected(self) -> torch.Tensor:
        if self.cfg.num_players == 1:
            return self._server
        return torch.where(self._last_hitter < 0, self._server, 1 - self._last_hitter)

    def _stable_players(self) -> torch.Tensor:
        """Torso recovery and approximate landing envelope, without restricting head pose or yaw."""
        stable = []
        for robot in self._robots:
            data = robot.data
            height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
            initial_height = data.default_root_pose.torch[:, 2]
            stable.append(
                (-data.projected_gravity_b.torch[:, 2] >= self.cfg.recovery_min_upright)
                & (data.root_link_ang_vel_b.torch[:, :2].norm(dim=-1) <= self.cfg.recovery_max_tumbling)
                & (height >= initial_height * self.cfg.recovery_min_height_ratio)
                & (height <= initial_height + self.cfg.recovery_max_height_offset)
                & (data.root_link_lin_vel_w.torch[:, 2].abs() <= self.cfg.recovery_max_vertical_speed)
            )
        return torch.stack(stable, dim=1)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.reshape_as(self._actions))
        self._hit_event.zero_()
        self._return_event.zero_()
        self._aim_event.zero_()
        self._player_hit_event.zero_()
        self._player_aim_event.zero_()
        self._player_return_event.zero_()
        self._fast_event.zero_()
        self._invalid.zero_()
        self._robot_collision.zero_()
        self._skip_contact_sample = True
        self._since_hit += self.step_dt
        self._last_ball_pos.copy_(self._ball.data.root_link_pos_w.torch)

    def _apply_action(self):
        # Sample the preceding physics substep before advancing again. The first
        # call has no new state; the final substep is sampled in _get_dones.
        if self._skip_contact_sample:
            self._skip_contact_sample = False
        else:
            self._update_contacts()
        for i, (robot, ids) in enumerate(zip(self._robots, self._joint_ids)):
            target = self._actions[:, i] + robot.data.default_joint_pos.torch[:, ids]
            limits = robot.data.soft_joint_pos_limits.torch[:, ids]
            target = target.clamp(min=limits[..., 0], max=limits[..., 1])
            robot.actuators.target_command.set_position_index(value=target, joint_ids=ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = []
        ball_pos = self._ball.data.root_link_pos_w.torch
        ball_vel = self._ball.data.root_link_lin_vel_w.torch
        receivers = self._receivers()
        expected = self._expected()
        for i, (robot, ids, play_ids) in enumerate(zip(self._robots, self._joint_ids, self._play_ids)):
            data = robot.data
            quat = data.root_link_quat_w.torch
            # Preserve the 61-wide walking layout and servo order. No encoder bias or IMU corruption initially.
            proprio = torch.cat(
                (
                    data.root_ang_vel_b.torch,
                    data.projected_gravity_b.torch,
                    data.joint_pos.torch[:, ids]
                    + data.joint_pos.torch[:, play_ids]
                    - data.default_joint_pos.torch[:, ids],
                    data.joint_vel.torch[:, ids]
                    + data.joint_vel.torch[:, play_ids]
                    - data.default_joint_vel.torch[:, ids],
                    self._actions[:, i],
                    torch.zeros(self.num_envs, 13, device=self.device),
                ),
                dim=-1,
            )
            task = torch.cat(
                (
                    quat_apply_inverse(quat, ball_pos - data.root_link_pos_w.torch),
                    quat_apply_inverse(quat, ball_vel - data.root_link_lin_vel_w.torch),
                    quat_apply_inverse(quat, receivers[:, i] - data.root_link_pos_w.torch),
                    torch.where(self._post_rally.active, -1.0, (expected == i).float()).unsqueeze(-1),
                ),
                dim=-1,
            )
            observations.append(torch.cat((proprio, task), dim=-1))
        return {"policy": torch.cat(observations, dim=-1)}

    def _update_contacts(self):
        """Sample actual contact locations and head orientation at physics-step resolution."""
        if self._robot_contacts is not None:
            robot_forces = self._robot_contacts.data.normal_force_matrix_w
            if robot_forces is None:
                raise RuntimeError("Filtered robot-to-robot contact forces are required in the volley task.")
            self._robot_collision |= robot_collision_mask(robot_forces.torch, self.cfg.robot_contact_threshold)
        receivers = self._receivers()
        ball_pos = self._ball.data.root_link_pos_w.torch
        ball_vel = self._ball.data.root_link_lin_vel_w.torch
        for i, (robot, sensor, head_mask, body) in enumerate(
            zip(self._robots, self._sensors, self._head_masks, self._head_ids)
        ):
            forces = sensor.data.normal_force_matrix_w
            positions = sensor.data.contact_pos_w
            if forces is None or positions is None:
                raise RuntimeError("Filtered ball forces and contact positions are required for heading rewards.")
            touching = forces.torch.norm(dim=-1).amax(dim=2) > self.cfg.contact_threshold
            head_contact = touching[:, head_mask].any(dim=1)
            other_contact = touching[:, ~head_mask].any(dim=1)
            onset = head_contact & ~self._contact[:, i] & ~self._post_rally.active
            other_contact &= ~self._post_rally.active
            head_pos = robot.data.body_link_pos_w.torch[:, body]
            head_quat = robot.data.body_link_quat_w.torch[:, body]
            contact_b = quat_apply_inverse(head_quat, positions.torch[:, head_mask, 0].squeeze(1) - head_pos)
            ball_b = quat_apply_inverse(head_quat, ball_pos - head_pos) - ball_pos.new_tensor(self.cfg.camera_offset)
            forward_up = quat_apply(
                head_quat, ball_pos.new_tensor(self.cfg.camera_forward_axis).expand(self.num_envs, -1)
            )[:, 2]
            front_contact = front_strike_mask(
                contact_b,
                ball_b,
                forward_up,
                self.cfg.camera_forward_axis,
                self.cfg.strike_patch_min,
                self.cfg.strike_patch_max,
                self.cfg.min_camera_cosine,
                self.cfg.min_camera_up,
            )
            self._raw_hits += onset.float()
            valid, wrong_surface, wrong_turn = classify_head_contact(
                onset,
                front_contact,
                self._expected(),
                self._last_hitter,
                self._invalid | self._robot_collision,
                i,
                self.cfg.num_players,
            )
            previous = self._last_hitter >= 0
            self._bad_surface |= wrong_surface
            self._bad_turn |= wrong_turn
            self._bad_body |= other_contact
            self._invalid |= other_contact | (onset & ~valid)
            transfer = valid & previous
            self._fast_event += (transfer & (self._since_hit < self.cfg.min_transfer_time)).float()
            self._hit_event += valid.float()
            self._return_event += transfer.float()
            self._player_hit_event[:, i] += valid.float()
            self._player_return_event[:, i] += transfer.float()
            # Ballistic aim towards the receiver's current head height, on the descending arc.
            discriminant = ball_vel[:, 2].square() + 2 * 9.81 * (ball_pos[:, 2] - receivers[:, i, 2])
            flight = (ball_vel[:, 2] + discriminant.clamp_min(0).sqrt()) / 9.81
            landing = ball_pos[:, :2] + ball_vel[:, :2] * flight.unsqueeze(-1)
            error = (landing - receivers[:, i, :2]).square().sum(dim=-1)
            quality = torch.exp(-error / self.cfg.receiver_radius**2)
            quality *= (discriminant > 0) & (ball_vel[:, 2] > 0.1) & (flight > self.cfg.min_transfer_time)
            self._aim_event += valid.float() * quality
            self._player_aim_event[:, i] += valid.float() * quality
            self._last_hitter[valid] = i
            self._since_hit[valid] = 0
            self._contact[:, i] = head_contact

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_contacts()
        relative = self._ball.data.root_link_pos_w.torch - self.scene.env_origins
        self._max_height = torch.maximum(self._max_height, relative[:, 2])
        self._fell.zero_()
        nonfinite = ~torch.isfinite(self._ball.data.root_link_vel_w.torch).all(dim=-1)
        nonfinite |= ~torch.isfinite(relative).all(dim=-1)
        ball_outside = (relative[:, 0].abs() > self.cfg.court_half_length) | (
            relative[:, 1].abs() > self.cfg.court_half_width
        )
        robot_outside = torch.zeros_like(ball_outside)
        crossed_center = torch.zeros_like(ball_outside)
        for i, robot in enumerate(self._robots):
            self._fell |= robot.data.projected_gravity_b.torch[:, 2] > -0.35
            nonfinite |= ~torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
            nonfinite |= ~torch.isfinite(robot.data.joint_vel.torch).all(dim=-1)
            nonfinite |= ~torch.isfinite(robot.data.root_link_quat_w.torch).all(dim=-1)
            nonfinite |= ~torch.isfinite(robot.data.root_link_vel_w.torch).all(dim=-1)
            root = robot.data.root_link_pos_w.torch - self.scene.env_origins
            nonfinite |= ~torch.isfinite(root).all(dim=-1)
            robot_outside |= (root[:, 1].abs() > self.cfg.court_half_width) | (
                root[:, 0].abs() > self.cfg.court_half_length
            )
            if self.cfg.num_players == 2:
                # Separate half-courts stop robots from meeting under the ball.
                crossed_center |= root[:, 0] > -0.03 if i == 0 else root[:, 0] < 0.03
        outside = ball_outside | robot_outside | crossed_center
        ground = relative[:, 2] <= self.cfg.ball_radius + 0.001
        live = ~self._post_rally.active
        self._rally_steps += live.float()
        self._ground |= ground & live
        self._outside |= (ball_outside & live) | robot_outside | crossed_center
        self._ball_outside |= ball_outside & live
        self._robot_outside |= robot_outside
        self._crossed_center |= crossed_center
        self._rally_invalid |= self._invalid
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        post_done = torch.zeros_like(timeout)
        self._miss_event.zero_()
        self._post_recovered.zero_()
        if self.cfg.post_rally_duration > 0:
            unsafe = nonfinite | self._fell | self._robot_collision | robot_outside | crossed_center
            post_done, self._miss_event, self._post_recovered = self._post_rally.step(
                ground | ball_outside | self._invalid,
                unsafe,
                timeout,
                self._stable_players().all(-1),
                self.step_dt,
                self.cfg.post_rally_duration,
                self.cfg.post_rally_hold,
            )
            self._fell_after_miss |= self._fell & self._post_rally.active
        # An impact in a collapsing or robot-collision step must not earn a positive strike bonus.
        legal = (
            ~(
                self._invalid
                | self._fell
                | self._robot_collision
                | robot_outside
                | crossed_center
                | nonfinite
                | self._post_rally.active
            )
        ).float()
        self._hit_event *= legal
        self._return_event *= legal
        self._aim_event *= legal
        self._player_hit_event *= legal.unsqueeze(-1)
        self._player_aim_event *= legal.unsqueeze(-1)
        self._player_return_event *= legal.unsqueeze(-1)
        self._hits += self._hit_event
        self._returns += self._return_event
        self._fast_returns += self._fast_event * legal
        self._success.zero_()
        if self.cfg.num_players == 1:
            # Detect a swept crossing of the virtual receiver's head plane, avoiding control-step tunnelling.
            target = self._receivers()[:, 0]
            current = self._ball.data.root_link_pos_w.torch
            z_delta = self._last_ball_pos[:, 2] - current[:, 2]
            alpha = (self._last_ball_pos[:, 2] - target[:, 2]) / z_delta.clamp_min(1e-6)
            crossing = self._last_ball_pos + alpha.unsqueeze(-1) * (current - self._last_ball_pos)
            self._success = (
                (self._last_hitter >= 0)
                & (z_delta > 0)
                & (alpha >= 0)
                & (alpha <= 1)
                & ((crossing[:, :2] - target[:, :2]).norm(dim=-1) < self.cfg.receiver_radius)
            )
        failure = ground | outside | nonfinite | self._fell | self._invalid | self._robot_collision
        self._success &= ~failure
        died = post_done if self.cfg.post_rally_duration > 0 else failure | self._success
        return died, timeout

    def _get_rewards(self) -> torch.Tensor:
        head_pos = self._head_positions()
        ball_pos = self._ball.data.root_link_pos_w.torch
        index = self._expected().view(-1, 1, 1).expand(-1, 1, 3)
        receiving_head = head_pos.gather(1, index).squeeze(1)
        approach = torch.exp(-(receiving_head[:, :2] - ball_pos[:, :2]).square().sum(-1) / 0.05**2)
        gazes = []
        for robot, body in zip(self._robots, self._head_ids):
            ball_b = quat_apply_inverse(
                robot.data.body_link_quat_w.torch[:, body], ball_pos - robot.data.body_link_pos_w.torch[:, body]
            ) - ball_pos.new_tensor(self.cfg.camera_offset)
            gazes.append(camera_alignment(ball_b, self.cfg.camera_forward_axis))
        gaze = torch.stack(gazes, dim=1).gather(1, self._expected().unsqueeze(-1)).squeeze(-1).clamp_min(0)
        playing = (~self._post_rally.active).float()
        self._gaze_sum += gaze * playing
        self._gaze_steps += playing
        upright = torch.stack([-r.data.projected_gravity_b.torch[:, 2] for r in self._robots], dim=1).mean(1)
        tilt, tumbling = balance_errors(
            torch.stack([r.data.projected_gravity_b.torch for r in self._robots], dim=1),
            torch.stack([r.data.root_link_ang_vel_b.torch for r in self._robots], dim=1),
        )
        rate = (self._actions - self._previous_actions).square().sum(dim=(1, 2))
        limits_cost = torch.zeros(self.num_envs, device=self.device)
        for robot, ids in zip(self._robots, self._joint_ids):
            limits = robot.data.soft_joint_pos_limits.torch[:, ids]
            pos = robot.data.joint_pos.torch[:, ids]
            limits_cost += ((limits[..., 0] - pos).clamp_min(0) + (pos - limits[..., 1]).clamp_min(0)).sum(-1)
        dense = (
            (self.cfg.approach_reward_scale * approach + self.cfg.gaze_reward_scale * gaze) * playing
            + self.cfg.upright_reward_scale * upright
            - self.cfg.torso_tilt_cost * tilt
            - self.cfg.torso_tumbling_cost * tumbling
            - self.cfg.action_rate_cost * rate
            - self.cfg.joint_limit_cost * limits_cost
        ) * self.step_dt
        strike_reward = (
            self.cfg.hit_reward * self._hit_event
            + self.cfg.aim_reward * self._aim_event
            + self.cfg.transfer_reward * self._return_event
        )
        if self.cfg.defer_strike_rewards:
            credit = (
                self.cfg.hit_reward * (1 - self.cfg.immediate_hit_fraction) * self._player_hit_event
                + self.cfg.aim_reward * self._player_aim_event
                + self.cfg.transfer_reward * self._player_return_event
            )
            paid, recovered = self._recovery.step(
                credit,
                self._player_hit_event,
                self._stable_players(),
                self.reset_buf | self._post_rally.active,
                self.step_dt,
                self.cfg.recovery_delay,
                self.cfg.recovery_hold,
            )
            self._recovered_hits += recovered.sum(-1)
            self._recovery_reward_sum += paid.sum(-1)
            strike_reward = self.cfg.hit_reward * self.cfg.immediate_hit_fraction * self._hit_event + paid.sum(-1)
        reward = (
            dense
            + strike_reward
            + self.cfg.transfer_reward * self._success.float()
            - self.cfg.fall_cost
            * (self._fell | self._robot_collision | self._robot_outside | self._crossed_center).float()
            - self._invalid.float()
            - self.cfg.miss_cost * self._miss_event.float()
            + self.cfg.post_rally_reward * (self._post_recovered & (self._hits > 0)).float()
        )
        return torch.nan_to_num(reward, nan=-self.cfg.fall_cost, posinf=0.0, neginf=-self.cfg.fall_cost)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "_hits"):
            self.extras["log"] = {
                "Metrics/head_hits": self._hits[env_ids].mean(),
                "Metrics/head_contacts": self._raw_hits[env_ids].mean(),
                "Metrics/gaze_alignment": (self._gaze_sum[env_ids] / self._gaze_steps[env_ids].clamp_min(1)).mean(),
                "Metrics/rally_length_s": (self._rally_steps[env_ids] * self.step_dt).mean(),
                "Metrics/post_rally_phase": self._post_rally.active[env_ids].float().mean(),
                "Metrics/post_rally_recovered": self._post_recovered[env_ids].float().mean(),
                "Metrics/fell_after_miss": self._fell_after_miss[env_ids].float().mean(),
                "Metrics/returns": self._returns[env_ids].mean(),
                "Metrics/recovered_hits": self._recovered_hits[env_ids].mean(),
                "Metrics/recovery_reward": self._recovery_reward_sum[env_ids].mean(),
                "Metrics/rally_2_returns": (self._returns[env_ids] >= 2).float().mean(),
                "Metrics/rally_4_returns": (self._returns[env_ids] >= 4).float().mean(),
                "Metrics/rally_8_returns": (self._returns[env_ids] >= 8).float().mean(),
                "Metrics/fast_returns": self._fast_returns[env_ids].mean(),
                "Metrics/target_success": self._success[env_ids].float().mean(),
                "Metrics/fell": self._fell[env_ids].float().mean(),
                "Metrics/invalid_contact": self._rally_invalid[env_ids].float().mean(),
                "Metrics/invalid_head_surface": self._bad_surface[env_ids].float().mean(),
                "Metrics/invalid_turn": self._bad_turn[env_ids].float().mean(),
                "Metrics/invalid_body_contact": self._bad_body[env_ids].float().mean(),
                "Metrics/robot_collision": self._robot_collision[env_ids].float().mean(),
                "Metrics/ball_ground": self._ground[env_ids].float().mean(),
                "Metrics/out_of_bounds": self._outside[env_ids].float().mean(),
                "Metrics/ball_out_of_bounds": self._ball_outside[env_ids].float().mean(),
                "Metrics/robot_out_of_bounds": self._robot_outside[env_ids].float().mean(),
                "Metrics/crossed_center": self._crossed_center[env_ids].float().mean(),
                "Metrics/max_ball_height": self._max_height[env_ids].mean(),
            }
        super()._reset_idx(env_ids)
        self._recovery.reset(env_ids)
        self._post_rally.reset(env_ids)
        for i, robot in enumerate(self._robots):
            pose = robot.data.default_root_pose.torch[env_ids].clone()
            pose[:, :3] += self.scene.env_origins[env_ids]
            pose[:, 0] = self.scene.env_origins[env_ids, 0] + (-1 if i == 0 else 1) * self.cfg.separation / 2
            pose[:, 3:7] = torch.tensor((0.0, 0.0, float(i), float(1 - i)), device=self.device)
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
            )
            robot.write_joint_position_to_sim_index(
                position=robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(robot.data.default_joint_vel.torch[env_ids]), env_ids=env_ids
            )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        self._server[env_ids] = torch.randint(self.cfg.num_players, (len(env_ids),), device=self.device)
        head_pos = self._head_positions()[env_ids, self._server[env_ids]]
        self._target[env_ids] = head_pos
        self._target[env_ids, 0] = 2 * self.scene.env_origins[env_ids, 0] - head_pos[:, 0]
        self._target[env_ids, 1] = self.scene.env_origins[env_ids, 1]
        pose = self._ball.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = head_pos
        pose[:, 2] += self.cfg.serve_height
        pose[:, :2] += (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * self.cfg.serve_spread
        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        velocity[:, 2] = -self.cfg.serve_down_speed
        self._ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._ball.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        for buffer in (
            self._post_recovered,
            self._fell_after_miss,
            self._rally_invalid,
            self._miss_event,
            self._rally_steps,
            self._gaze_steps,
            self._player_hit_event,
            self._player_aim_event,
            self._player_return_event,
            self._recovered_hits,
            self._recovery_reward_sum,
            self._actions,
            self._previous_actions,
            self._contact,
            self._hits,
            self._raw_hits,
            self._gaze_sum,
            self._returns,
            self._hit_event,
            self._return_event,
            self._aim_event,
            self._invalid,
            self._robot_collision,
            self._bad_surface,
            self._bad_turn,
            self._bad_body,
            self._ground,
            self._outside,
            self._ball_outside,
            self._robot_outside,
            self._crossed_center,
            self._fast_returns,
            self._fast_event,
            self._fell,
            self._success,
            self._max_height,
        ):
            buffer[env_ids] = 0
        self._last_hitter[env_ids] = -1
        self._since_hit[env_ids] = 0
        self._last_ball_pos[env_ids] = pose[:, :3]
