# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Receive, return, recover, then rally: no artificial forces or scripted in-flight ball motion."""

import torch

from isaaclab.utils.math import quat_apply_inverse

from .ball_env import MicroDuckBallEnv
from .ball_rules import launch_progress, launch_quality
from .ball_serves import descending_target_crossing, incoming_ballistic_serve
from .volley_curriculum import STAGES
from .volley_curriculum_env_cfg import MicroDuckVolleyCurriculumEnvCfg
from .volley_episode import VolleyEpisode


class MicroDuckVolleyCurriculumEnv(MicroDuckBallEnv):
    """Keep the 142/28 interface; repurpose the 13 formerly zero channels of each player.

    Coach lessons exercise one receiver at a time with a virtual target. The other
    robot is parked farther away but remains under full policy control and must
    also recover. Pair stages use only actual, alternating, center-crossing returns.
    """

    cfg: MicroDuckVolleyCurriculumEnvCfg
    stages = STAGES

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.stage = cfg.curriculum_stage
        if not 0 <= self.stage < len(self.stages):
            raise ValueError("Unknown volleyball curriculum stage.")
        self.evaluation = cfg.evaluation
        self._train_generator = torch.Generator(device=self.device).manual_seed(cfg.seed or 0)
        self._eval_generator = torch.Generator(device=self.device).manual_seed(1000001)
        self._episode = VolleyEpisode(self.num_envs, self.device)
        self._home_roots = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._targets = torch.zeros_like(self._home_roots)
        self._previous_ball = torch.zeros(self.num_envs, 3, device=self.device)
        self._center_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flight_clock = torch.zeros(self.num_envs, device=self.device)
        self._own_hit_age = torch.full((self.num_envs, 2), 2.0, device=self.device)
        self._launch_pending = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._launch_age = torch.zeros(self.num_envs, 2, device=self.device)
        self._hold_time = torch.zeros_like(self._launch_age)
        self._low_time = torch.zeros_like(self._launch_age)
        self._qualified_returns = torch.zeros(self.num_envs, device=self.device)
        self._qualified_event = torch.zeros_like(self._qualified_returns)
        self._release_event = torch.zeros_like(self._qualified_returns)
        self._launch_total = torch.zeros_like(self._qualified_returns)
        self._progress_total = torch.zeros_like(self._qualified_returns)
        self._game_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._unsafe = torch.zeros_like(self._game_invalid)
        self._terminal_pending = torch.zeros_like(self._game_invalid)
        self._goal_started = torch.zeros_like(self._game_invalid)
        self._miss_started = torch.zeros_like(self._game_invalid)
        self._virtual_goal = torch.zeros_like(self._game_invalid)
        self._grounded_stable = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._potential = torch.zeros(self.num_envs, device=self.device)
        self._episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._unsafe_reasons = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in ("nonfinite", "tilt", "center", "excursion", "collapse", "robot_collision")
        }
        self._miss_reasons = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in (
                "invalid_head_surface",
                "invalid_body_contact",
                "invalid_turn",
                "game_invalid",
                "ball_ground",
                "ball_outside",
            )
        }
        self._feet = []
        for sensor in self._sensors:
            ids, _ = sensor.find_sensors("ankle_(left|right)")
            if len(ids) != 2:
                raise ValueError("Recovery requires both resolved ankle rigid-body contact sensors.")
            self._feet.append(ids)

    def set_stage(self, stage):
        """Select the next lesson; the caller resets worlds before stepping it."""
        if not 0 <= stage < len(self.stages):
            raise ValueError("Unknown volleyball curriculum stage.")
        self.stage = self.cfg.curriculum_stage = stage

    def set_evaluation(self, enabled, seed=None):
        """Use an independent serve generator and balanced receiver IDs during held-out evaluation."""
        self.evaluation = bool(enabled)
        if seed is not None:
            self._eval_generator.manual_seed(seed)

    def _uniform(self, count, low, high):
        generator = self._eval_generator if self.evaluation else self._train_generator
        return low + (high - low) * torch.rand(count, device=self.device, generator=generator)

    def _receivers(self):
        return self._targets if hasattr(self, "_targets") else super()._receivers()

    def _refresh_targets(self):
        """Optional live-target update; legacy coach targets remain fixed."""

    def _arrange_players(self, env_ids):
        """Optional reset-only distribution; no in-flight body placement."""

    def _new_strike_reference(self, player, hit):
        """Capture an optional physical pass reference without controlling the collision."""

    def _release_shaping(self, player, ball, quality):
        if not self.cfg.continuous_launch_shaping:
            return quality
        return launch_progress(
            ball,
            self._ball.data.root_link_lin_vel_w.torch,
            self._targets[:, player],
            self.cfg.target_radius,
            self.cfg.minimum_flight_time,
            self._launch_age[:, player],
        )

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.extras["episode_records"] = []
        self.extras.pop("log", None)
        self._qualified_event.zero_()
        self._release_event.zero_()
        self._own_hit_age += self.step_dt

    def _update_contacts(self):
        self._refresh_targets()
        dt = self.cfg.sim.dt
        active = self._episode.phase == 0
        self._post_rally.active.copy_(~active)
        ball = self._ball.data.root_link_pos_w.torch
        old_x = self._previous_ball[:, 0] - self.scene.env_origins[:, 0]
        new_x = ball[:, 0] - self.scene.env_origins[:, 0]
        direction = torch.where(self._last_hitter == 0, 1.0, -1.0)
        self._center_crossed |= (
            active
            & (self._last_hitter >= 0)
            & (old_x * direction < 0)
            & (new_x * direction >= 0)
            & (ball[:, 2] - self.scene.env_origins[:, 2] >= self.cfg.minimum_crossing_height)
        )
        self._flight_clock += dt
        self._launch_age += dt
        previous = self._player_hit_event.clone()
        old_hitter = self._last_hitter.clone()
        super()._update_contacts()
        new_hits = self._player_hit_event > previous
        spec = self.stages[self.stage]
        for player in range(2):
            hit = new_hits[:, player] & active
            reply = hit & (old_hitter >= 0)
            qualified = reply & self._center_crossed & (self._flight_clock >= self.cfg.minimum_flight_time)
            if spec.paired:
                self._game_invalid |= reply & ~qualified
                self._qualified_event += qualified.float()
                self._qualified_returns += qualified.float()
            else:
                self._game_invalid |= hit & ((self._server != player) | (old_hitter >= 0))
            self._new_strike_reference(player, hit)
            self._own_hit_age[hit, player] = 0
            self._launch_pending[hit, player] = True
            self._launch_age[hit, player] = 0
            self._flight_clock[hit] = 0
            self._center_crossed[hit] = False
        self._hold_time = torch.where(self._contact & active.unsqueeze(-1), self._hold_time + dt, 0.0)
        self._game_invalid |= (self._hold_time > self.cfg.maximum_ball_hold).any(-1)
        # Score outgoing flight after separation, rather than at the start of the impact impulse.
        ready = self._launch_pending & ~self._contact & (self._launch_age >= self.cfg.release_delay)
        for player in range(2):
            quality = launch_quality(
                ball,
                self._ball.data.root_link_lin_vel_w.torch,
                self._targets[:, player],
                self.cfg.target_radius,
                self.cfg.minimum_flight_time,
                smooth=True,
                elapsed_flight=self._launch_age[:, player],
            )
            credit = quality * (ready[:, player] & active).float()
            progress = self._release_shaping(player, ball, quality)
            self._release_event += progress * (ready[:, player] & active).float()
            self._launch_total += credit
        self._launch_pending &= ~ready
        if not spec.paired:
            target = self._targets.gather(1, self._server[:, None, None].expand(-1, 1, 3)).squeeze(1)
            self._virtual_goal |= (
                active
                & (self._last_hitter >= 0)
                & self._center_crossed
                & (self._flight_clock >= self.cfg.minimum_flight_time)
                & descending_target_crossing(self._previous_ball, ball, target, self.cfg.target_radius)
            )
        self._previous_ball.copy_(ball)

    def _balance(self):
        stable, potentials = [], []
        for player, (robot, sensor, feet) in enumerate(zip(self._robots, self._sensors, self._feet)):
            data = robot.data
            height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
            initial = data.default_root_pose.torch[:, 2]
            upright = -data.projected_gravity_b.torch[:, 2]
            speed = data.root_link_lin_vel_w.torch
            angular = data.root_link_ang_vel_b.torch[:, :2].norm(dim=-1)
            offset = (data.root_link_pos_w.torch[:, :2] - self._home_roots[:, player, :2]).norm(dim=-1)
            supported = (sensor.data.net_normal_forces_w.torch[:, feet, 2] > self.cfg.supported_force).any(-1)
            stable.append(
                supported
                & (height >= initial * self.cfg.recovered_height_ratio)
                & (height <= initial + 0.03)
                & (upright >= self.cfg.recovery_min_upright)
                & (speed[:, :2].norm(dim=-1) <= self.cfg.recovered_max_planar_speed)
                & (speed[:, 2].abs() <= 0.2)
                & (angular <= 2.0)
                & (offset <= 0.10)
            )
            error = (
                ((height - initial) / 0.05).square()
                + ((1 - upright) / 0.3).square()
                + (speed[:, :2] / 0.4).square().sum(-1)
                + (offset / 0.15).square()
            )
            potentials.append(torch.exp(-error))
        return torch.stack(stable, dim=1), torch.stack(potentials, dim=1).mean(-1)

    def _get_dones(self):
        self._update_contacts()
        self._grounded_stable[:], _ = self._balance()
        self._unsafe.zero_()
        for player, robot in enumerate(self._robots):
            data = robot.data
            root = data.root_link_pos_w.torch - self.scene.env_origins
            finite = torch.isfinite(data.root_link_vel_w.torch).all(-1)
            finite &= torch.isfinite(data.root_link_quat_w.torch).all(-1)
            finite &= torch.isfinite(root).all(-1) & torch.isfinite(data.joint_pos.torch).all(-1)
            finite &= torch.isfinite(data.joint_vel.torch).all(-1)
            self._low_time[:, player] = torch.where(
                root[:, 2] < self.cfg.collapse_height, self._low_time[:, player] + self.step_dt, 0.0
            )
            crossed = root[:, 0] > -0.03 if player == 0 else root[:, 0] < 0.03
            excursion = (data.root_link_pos_w.torch[:, :2] - self._home_roots[:, player, :2]).norm(dim=-1)
            reasons = {
                "nonfinite": ~finite,
                "tilt": data.projected_gravity_b.torch[:, 2] > -0.5,
                "center": crossed,
                "excursion": excursion > self.cfg.maximum_home_excursion,
                "collapse": self._low_time[:, player] >= self.cfg.collapse_duration,
            }
            for reason, occurred in reasons.items():
                self._unsafe_reasons[reason] |= occurred
                self._unsafe |= occurred
        ball = self._ball.data.root_link_pos_w.torch - self.scene.env_origins
        ball_nonfinite = ~torch.isfinite(ball).all(-1) | ~torch.isfinite(self._ball.data.root_link_vel_w.torch).all(-1)
        self._unsafe_reasons["nonfinite"] |= ball_nonfinite
        self._unsafe_reasons["robot_collision"] |= self._robot_collision
        self._unsafe |= self._robot_collision | ball_nonfinite
        playing = self._episode.phase == 0
        reasons = {
            "invalid_head_surface": self._bad_surface,
            "invalid_body_contact": self._bad_body,
            "invalid_turn": self._bad_turn,
            "game_invalid": self._game_invalid,
            "ball_ground": ball[:, 2] <= self.cfg.ball_radius + 0.001,
            "ball_outside": (ball[:, 0].abs() > 1.1) | (ball[:, 1].abs() > 0.45),
        }
        for reason, occurred in reasons.items():
            self._miss_reasons[reason] |= occurred & playing
        miss = (
            (ball[:, 2] <= self.cfg.ball_radius + 0.001)
            | (ball[:, 0].abs() > 1.1)
            | (ball[:, 1].abs() > 0.45)
            | self._invalid
            | self._game_invalid
        ) & playing
        goal = self._stage_goal()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        done, _, self._goal_started, self._miss_started = self._episode.step(
            goal,
            miss,
            self._unsafe,
            timeout,
            self._grounded_stable.all(-1),
            self.step_dt,
            self.cfg.recovery_duration,
            self.cfg.recovery_hold,
        )
        self._terminal_pending.copy_(done | timeout)
        legal = ~(self._unsafe | self._invalid | self._game_invalid)
        self._hit_event *= legal.float()
        self._qualified_event *= legal.float()
        self._release_event *= legal.float()
        self._hits += self._hit_event
        return done, timeout

    def _stage_goal(self):
        spec = self.stages[self.stage]
        required = self.cfg.play_returns if self.cfg.play_returns > 0 and not self.evaluation else spec.required_returns
        return self._qualified_returns >= required if spec.paired else self._virtual_goal

    def _serve_height_range(self):
        return (0.18, 0.25)

    def _serve_source_offset(self, receiving, partner):
        return None

    def _episode_record_fields(self, env_ids):
        return {}

    def _get_rewards(self):
        stable, potential = self._balance()
        self._progress_total += self._release_event
        terminal = self._terminal_pending
        # Bounded potential improvement cannot pay repeatedly for standing/crouching cycles.
        next_potential = torch.where(terminal, 0.0, potential)
        improvement = 0.2 * (0.995 * next_potential - self._potential)
        self._potential.copy_(potential)
        rate = (self._actions - self._previous_actions).square().sum(dim=(1, 2))
        reward = (
            self._qualified_event
            + 0.05 * self._hit_event
            + self.cfg.release_reward_scale * self._release_event
            + 3.0 * self._episode.success.float()
            + improvement
            + (0.1 * stable.float().mean(-1) - 0.01 * rate) * self.step_dt
            - 3.0 * self._unsafe.float()
            - 0.5 * self._miss_started.float()
            - (terminal & ~self._episode.success & ~self._unsafe).float()
        )
        return torch.nan_to_num(reward, nan=-3.0, posinf=-3.0, neginf=-3.0)

    def _get_observations(self):
        self._refresh_targets()
        blocks = super()._get_observations()["policy"].view(self.num_envs, 2, 71).clone()
        for player, robot in enumerate(self._robots):
            data = robot.data
            quat = data.root_link_quat_w.torch
            phase = self._episode.phase.float()
            if not self.stages[self.stage].paired:
                phase = torch.where((phase == 0) & (self._last_hitter >= 0), 0.5, phase)
            extras = torch.cat(
                (
                    data.root_link_lin_vel_b.torch,
                    quat_apply_inverse(quat, self._home_roots[:, player] - data.root_link_pos_w.torch),
                    (
                        data.root_link_pos_w.torch[:, 2]
                        - self.scene.env_origins[:, 2]
                        - data.default_root_pose.torch[:, 2]
                    ).unsqueeze(-1)
                    / 0.125,
                    self._own_hit_age[:, player : player + 1].clamp_max(2) / 2,
                    (self._episode.phase_time / self.cfg.recovery_duration).clamp_max(1).unsqueeze(-1),
                    (self._episode.stable_time / self.cfg.recovery_hold).clamp_max(1).unsqueeze(-1),
                    torch.full((self.num_envs, 1), float(not self.stages[self.stage].paired), device=self.device),
                    phase.unsqueeze(-1) / 2,
                    torch.full((self.num_envs, 1), self.stage / (len(self.stages) - 1), device=self.device),
                ),
                dim=-1,
            )
            blocks[:, player, 48:61] = extras
            waiting = self._episode.phase > 0
            if not self.stages[self.stage].paired:
                waiting |= self._last_hitter >= 0
            blocks[:, player, 70] = torch.where(waiting, -1.0, (self._expected() == player).float())
        return {"policy": blocks.flatten(1)}

    def _reset_idx(self, env_ids):
        records = []
        if hasattr(self, "_terminal_pending"):
            completed = env_ids[self._terminal_pending[env_ids]]
            if len(completed):
                timed_out = self.episode_length_buf[completed] >= self.max_episode_length - 1
                recovered_miss = (
                    (self._episode.phase[completed] == 2)
                    & (self._episode.phase_time[completed] + 1e-6 >= self.cfg.recovery_duration)
                    & (self._episode.stable_time[completed] + 1e-6 >= self.cfg.recovery_hold)
                    & self._grounded_stable[completed].all(-1)
                    & ~self._unsafe[completed]
                    & ~timed_out
                )
                fields = {
                    "env_id": completed,
                    "episode_id": self._episode_ids[completed],
                    "server": self._server[completed],
                    "success": self._episode.success[completed],
                    "unsafe": self._unsafe[completed],
                    "returns": self._qualified_returns[completed],
                    "goal_reached": self._episode.phase[completed] == 1,
                    "launch_quality": self._launch_total[completed] / self._hits[completed].clamp_min(1),
                    "launch_progress": self._progress_total[completed] / self._hits[completed].clamp_min(1),
                    "head_hits": self._hits[completed],
                    "head_contacts": self._raw_hits[completed],
                    "recovered_miss": recovered_miss,
                    "stable_a": self._grounded_stable[completed, 0],
                    "stable_b": self._grounded_stable[completed, 1],
                    "recovery_hold_s": torch.where(
                        self._grounded_stable[completed].all(-1), self._episode.stable_time[completed], 0.0
                    ),
                    "timed_out": timed_out,
                    "missed": self._episode.phase[completed] == 2,
                }
                fields.update({f"unsafe_{name}": flags[completed] for name, flags in self._unsafe_reasons.items()})
                fields.update({f"miss_{name}": flags[completed] for name, flags in self._miss_reasons.items()})
                additional = self._episode_record_fields(completed)
                fields.update(additional)
                columns = torch.stack(tuple(fields.values()), dim=1).detach().cpu().tolist()
                for values in columns:
                    row = dict(zip(fields, values))
                    if row["unsafe"]:
                        reason = "unsafe"
                    elif row["success"]:
                        reason = "success"
                    elif row["goal_reached"]:
                        reason = "target_unrecovered"
                    elif row["missed"]:
                        reason = "missed"
                    elif row["timed_out"]:
                        reason = "timeout"
                    else:
                        reason = "missed"
                    records.append(
                        {
                            "env_id": int(row["env_id"]),
                            "episode_id": int(row["episode_id"]),
                            "server": int(row["server"]),
                            "success": bool(row["success"]),
                            "unsafe": bool(row["unsafe"]),
                            "returns": int(row["returns"]),
                            "goal_reached": bool(row["goal_reached"]),
                            "launch_quality": row["launch_quality"],
                            "launch_progress": row["launch_progress"],
                            "head_hits": int(row["head_hits"]),
                            "head_contacts": int(row["head_contacts"]),
                            "recovered_miss": bool(row["recovered_miss"]),
                            "recovery_stable_by_player": [bool(row["stable_a"]), bool(row["stable_b"])],
                            "recovery_hold_s": row["recovery_hold_s"],
                            "unsafe_reasons": [name for name in self._unsafe_reasons if row[f"unsafe_{name}"]],
                            "miss_reasons": [name for name in self._miss_reasons if row[f"miss_{name}"]],
                            "stage": self.stage,
                            "reason": reason,
                            **{name: row[name] for name in additional},
                            **(
                                {"objective": self.stages[self.stage].objective}
                                if hasattr(self.stages[self.stage], "objective")
                                else {}
                            ),
                        }
                    )
        self.cfg.separation = self.cfg.stage_separations[self.stage]
        super()._reset_idx(env_ids)
        self._arrange_players(env_ids)
        self.extras.setdefault("episode_records", []).extend(records)
        if records:
            self.extras["log"] = {
                "Metrics/success": sum(r["success"] for r in records) / len(records),
                "Metrics/unsafe": sum(r["unsafe"] for r in records) / len(records),
                "Metrics/returns": sum(r["returns"] for r in records) / len(records),
                "Metrics/head_hits": sum(r["head_hits"] for r in records) / len(records),
                "Metrics/head_contacts": sum(r["head_contacts"] for r in records) / len(records),
                "Metrics/goal_reached": sum(r["goal_reached"] for r in records) / len(records),
                "Metrics/launch_quality": sum(r["launch_quality"] for r in records) / len(records),
                "Metrics/launch_progress": sum(r["launch_progress"] for r in records) / len(records),
                "Metrics/recovered_miss": sum(r["recovered_miss"] for r in records) / len(records),
                "Metrics/recovery_stable_a": sum(r["recovery_stable_by_player"][0] for r in records) / len(records),
                "Metrics/recovery_stable_b": sum(r["recovery_stable_by_player"][1] for r in records) / len(records),
                "Metrics/recovery_hold_s": sum(r["recovery_hold_s"] for r in records) / len(records),
                "Metrics/curriculum_stage": float(self.stage),
            }
            for reason in self._unsafe_reasons:
                self.extras["log"][f"Metrics/unsafe_{reason}"] = sum(
                    reason in r["unsafe_reasons"] for r in records
                ) / len(records)
            for reason in self._miss_reasons:
                self.extras["log"][f"Metrics/{reason}"] = sum(reason in r["miss_reasons"] for r in records) / len(
                    records
                )
        self._episode.reset(env_ids)
        for buffer in (
            self._flight_clock,
            self._center_crossed,
            self._launch_pending,
            self._launch_age,
            self._hold_time,
            self._low_time,
            self._qualified_returns,
            self._qualified_event,
            self._release_event,
            self._launch_total,
            self._progress_total,
            self._game_invalid,
            self._unsafe,
            self._terminal_pending,
            self._goal_started,
            self._miss_started,
            self._virtual_goal,
            *self._unsafe_reasons.values(),
            *self._miss_reasons.values(),
        ):
            buffer[env_ids] = 0
        self._own_hit_age[env_ids] = 2
        self._episode_ids[env_ids] += 1
        count = len(env_ids)
        self._server[env_ids] = env_ids.long() % 2 if self.evaluation else (self._uniform(count, 0, 1) >= 0.5).long()
        spec = self.stages[self.stage]
        if not spec.paired and not self.cfg.live_partner:
            for player, robot in enumerate(self._robots):
                pose = robot.data.root_link_pose_w.torch[env_ids].clone()
                parked = self._server[env_ids] != player
                pose[parked, 0] = (
                    self.scene.env_origins[env_ids[parked], 0]
                    + (-1 if player == 0 else 1) * self.cfg.coach_partner_distance
                )
                robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.0)
        self._home_roots[env_ids] = torch.stack([r.data.root_link_pos_w.torch[env_ids] for r in self._robots], dim=1)
        self._targets[env_ids] = self._head_positions()[env_ids].flip(1)
        self._targets[env_ids, :, 2] += self.cfg.ball_radius
        receiving = self._head_positions()[env_ids, self._server[env_ids]]
        partner = self._head_positions()[env_ids, 1 - self._server[env_ids]].clone()
        if not self.cfg.live_partner:
            partner = receiving.clone()
            partner[:, 0] = 2 * self.scene.env_origins[env_ids, 0] - receiving[:, 0]
        if not spec.paired and not self.cfg.live_partner:
            target = partner.clone()
            target[:, 2] += self.cfg.ball_radius
            self._targets[env_ids, self._server[env_ids]] = target
        offset = torch.zeros(count, 3, device=self.device)
        offset[:, :2] = torch.stack(
            [self._uniform(count, -spec.serve_spread, spec.serve_spread) for _ in range(2)], dim=1
        )
        offset[:, 2] = self.cfg.ball_radius
        source, velocity, _, valid = incoming_ballistic_serve(
            receiving - self.scene.env_origins[env_ids],
            partner - self.scene.env_origins[env_ids],
            self._uniform(count, *spec.flight_time_range),
            self._uniform(count, *self._serve_height_range()),
            self._uniform(count, *spec.lateral_range),
            offset,
            source_offset=self._serve_source_offset(receiving, partner),
        )
        if not valid.all():
            raise RuntimeError("Curriculum produced an invalid serve: refusing a nonphysical training batch.")
        pose = self._ball.data.root_link_pose_w.torch[env_ids].clone()
        pose[:, :3] = source + self.scene.env_origins[env_ids]
        speed = torch.zeros(count, 6, device=self.device)
        speed[:, :3] = velocity
        self._ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._ball.write_root_velocity_to_sim_index(root_velocity=speed, env_ids=env_ids)
        self._previous_ball[env_ids] = pose[:, :3]
        self._potential[env_ids] = self._balance()[1][env_ids]
