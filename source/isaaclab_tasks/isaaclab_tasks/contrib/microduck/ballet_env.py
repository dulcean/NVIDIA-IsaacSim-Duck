# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learn the entire one-legged pirouette and bow with unmodified physical dynamics."""

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

from .ballet import PHASE_NAMES, BalletCurriculum, BalletFrame, BalletRoutine, rotation_potential
from .ballet_env_cfg import MicroDuckBalletEnvCfg


class MicroDuckBalletEnv(DirectRLEnv):
    """A bespoke phase/reset controller qualifies every 5ms contact and root-yaw increment.

    The controller issues task references and rewards only. All 14 actuator
    targets are learned actions; no base impulses, fixed legs, or scripted
    joint trajectories are used. A direct environment keeps contact evidence
    and phase transitions on the same clock, including short toe touches.
    """

    cfg: MicroDuckBalletEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.support_foot not in ("left", "right"):
            raise ValueError("Support foot must be left or right.")
        self.sim.backend_manager.set_decimation(1)
        self._physics_handles_decimation = False
        self._robot = self.scene["robot"]
        self._contacts = self.scene["contacts"]
        if self._contacts.num_filter_objects < 1:
            raise RuntimeError(
                "Ballet ground shape filter matched no physical floor; refusing unfiltered support evidence."
            )
        self._joints, _ = self._robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
        self._play, _ = self._robot.find_joints(
            [f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES],
            preserve_order=True,
        )
        sides = (cfg.support_foot, "right" if cfg.support_foot == "left" else "left")
        names = [f"ankle_{side}" for side in sides]
        self._feet, _ = self._robot.find_bodies(names, preserve_order=True)
        self._foot_sensors, _ = self._contacts.find_sensors(names, preserve_order=True)
        head, _ = self._robot.find_bodies("jaw_soft")
        if (
            len(self._joints) != 14
            or len(self._play) != 14
            or len(self._feet) != 2
            or len(self._foot_sensors) != 2
            or len(head) != 1
        ):
            raise RuntimeError("Ballet requires all 14 servos/play hinges, both ankle contacts and the physical head.")
        self._head = head[0]
        self._nonfoot = torch.ones(self._contacts.num_sensors, dtype=torch.bool, device=self.device)
        self._nonfoot[self._foot_sensors] = False
        self.routine = BalletRoutine(self.num_envs, self.device, direction=cfg.turn_direction)
        self.curriculum = BalletCurriculum(self.num_envs, self.device, cfg.start_stage)
        self._actions = torch.zeros(self.num_envs, 14, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_progress = torch.zeros(self.num_envs, device=self.device)
        self._home = torch.zeros(self.num_envs, 3, device=self.device)
        self._anchor = torch.zeros_like(self._home)
        self._foot_zero = torch.zeros(self.num_envs, 2, device=self.device)
        self._action_nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._frame = None
        self._sample_before_step = False

    def _measure(self):
        data = self._robot.data
        quat = data.root_link_quat_w.torch
        position = data.root_link_pos_w.torch
        feet = data.body_link_pos_w.torch[:, self._feet]
        # The matrix contains floor counterparts only. Aggregate net forces
        # also contain self-contact, which must never masquerade as support.
        forces = self._contacts.data.force_matrix_w.torch.sum(dim=2)
        masses = data.body_mass.torch
        com = (data.body_com_pos_w.torch * masses[..., None]).sum(1) / masses.sum(1, keepdim=True).clamp_min(1e-6)
        _, pitch, yaw = euler_xyz_from_quat(quat)
        ang = data.root_link_ang_vel_b.torch
        velocity = data.root_link_lin_vel_w.torch
        head_forward = quat_apply(
            data.body_link_quat_w.torch[:, self._head],
            position.new_tensor((0.0, 0.0, -1.0)).expand(self.num_envs, -1),
        )
        finite = torch.isfinite(data.joint_pos.torch).all(-1) & torch.isfinite(data.joint_vel.torch).all(-1)
        finite &= torch.isfinite(torch.cat((position, quat, velocity, ang, com, head_forward), -1)).all(-1)
        finite &= torch.isfinite(forces).all(dim=(1, 2)) & ~self._action_nonfinite
        return BalletFrame(
            yaw=yaw,
            pitch=pitch,
            head_z=head_forward[:, 2],
            up=-data.projected_gravity_b.torch[:, 2],
            height=position[:, 2] - self.scene.env_origins[:, 2],
            planar_speed=velocity[:, :2].norm(dim=-1),
            tumbling=ang[:, :2].norm(dim=-1),
            yaw_rate=ang[:, 2],
            clearance=feet[:, :, 2] - self._foot_zero,
            forces=forces[:, self._foot_sensors],
            nonfoot_contact=(forces[:, self._nonfoot].norm(dim=-1) > 0.2).any(-1),
            displacement=(position[:, :2] - self._home[:, :2]).norm(dim=-1),
            anchor_drift=(feet[:, 0, :2] - self._anchor[:, :2]).norm(dim=-1),
            com_error=(com[:, :2] - feet[:, 0, :2]).norm(dim=-1),
            finite=finite,
        )

    def _sample(self):
        self._frame = self._measure()
        self.routine.update(self._frame, self.physics_dt)

    def _pre_physics_step(self, actions):
        self._previous_progress.copy_(self.routine.progress)
        self._previous_actions.copy_(self._actions)
        self._action_nonfinite.copy_(~torch.isfinite(actions).all(-1))
        self._actions.copy_(torch.nan_to_num(actions).clamp(-1.0, 1.0))
        self.routine.begin_control_step()
        self._sample_before_step = False

    def _apply_action(self):
        if self._sample_before_step:
            self._sample()
        self._sample_before_step = True
        data = self._robot.data
        target = data.default_joint_pos.torch[:, self._joints] + self._actions
        limits = data.soft_joint_pos_limits.torch[:, self._joints]
        target = target.clamp(min=limits[..., 0], max=limits[..., 1])
        self._robot.actuators.target_command.set_position_index(value=target, joint_ids=self._joints)

    def _get_dones(self):
        self._sample()
        if not bool(self._frame.finite.all()):
            raise RuntimeError("Nonfinite MicroDuck ballet physics or actions; refusing to train on reset-masked NaNs.")
        self.last_success.copy_(self.routine.success)
        return self.routine.failed | self.routine.success, self.episode_length_buf >= self.max_episode_length

    def _references(self):
        phase = self.routine.phase
        remaining = (self.routine.goal - self.routine.progress).clamp_min(0.0)
        yaw_rate = remaining.mul(3).clamp(0.15, 1.2) * self.cfg.turn_direction
        yaw_rate *= (phase == 1) & (self.routine.goal > 0)
        pitch = torch.where(phase == 3, 0.30, 0.0)
        lift = torch.where(phase == 0, (self.routine.age / 0.6).clamp(0.0, 1.0), (phase == 1).float()) * 0.04
        return remaining, yaw_rate, pitch, lift

    def _get_observations(self):
        frame = self._measure()
        data = self._robot.data
        quat = data.root_link_quat_w.torch
        position = data.root_link_pos_w.torch
        remaining, yaw_rate, pitch, lift = self._references()
        commands = torch.zeros(self.num_envs, 13, device=self.device)
        commands[:, 2] = yaw_rate
        prefix = torch.cat(
            (
                data.root_ang_vel_b.torch,
                data.projected_gravity_b.torch,
                data.joint_pos.torch[:, self._joints]
                + data.joint_pos.torch[:, self._play]
                - data.default_joint_pos.torch[:, self._joints],
                data.joint_vel.torch[:, self._joints] + data.joint_vel.torch[:, self._play],
                self._actions,
                commands,
            ),
            -1,
        )
        feet = data.body_link_pos_w.torch[:, self._feet]
        foot_local = torch.stack([quat_apply_inverse(quat, feet[:, i] - position) for i in range(2)], 1)
        masses = data.body_mass.torch
        com = (data.body_com_pos_w.torch * masses[..., None]).sum(1) / masses.sum(1, keepdim=True).clamp_min(1e-6)
        com_local = quat_apply_inverse(quat, com - feet[:, 0])[:, :2]
        home_local = quat_apply_inverse(quat, position - self._home)[:, :2]
        task = torch.cat(
            (
                F.one_hot(self.routine.phase, len(PHASE_NAMES)).float(),
                (remaining / (2 * torch.pi))[:, None],
                (self.routine.age / 12).clamp_max(1)[:, None],
                yaw_rate[:, None],
                pitch[:, None],
                lift[:, None],
                frame.clearance,
                (frame.forces[..., 2] > 0.05).float(),
                foot_local.flatten(1),
                com_local,
                home_local,
                data.root_link_lin_vel_b.torch,
            ),
            -1,
        )
        return {"policy": torch.nan_to_num(torch.cat((prefix, task), -1), nan=0.0, posinf=0.0, neginf=0.0)}

    def _get_rewards(self):
        frame = self._frame
        phase = self.routine.phase
        _, rate, pitch, lift = self._references()
        upright = torch.exp(-((frame.pitch - pitch) / 0.20).square()) * frame.up.clamp(0, 1)
        balance = torch.exp(-(frame.com_error / 0.035).square())
        lifted = torch.exp(-((frame.clearance[:, 1] - lift) / 0.020).square())
        single, both = frame.support()
        rotating = torch.exp(-((frame.yaw_rate - rate) / 0.6).square()) * single
        head_target = torch.where(phase == 3, -0.45, 0.0)
        head = torch.exp(-((frame.head_z - head_target) / 0.20).square())
        dense = 0.2 * upright
        dense += (phase <= 1) * (0.5 * balance + 2.0 * lifted + (phase == 1) * 1.5 * rotating)
        dense += (phase >= 2) * (0.5 * both + 1.5 * upright + head)
        dense -= 2.0 * frame.planar_speed.square() + 0.05 * frame.tumbling.square()
        dense -= 0.2 * (frame.yaw_rate - rate).square()
        dense -= 0.01 * (self._actions - self._previous_actions).square().sum(-1)
        limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joints]
        positions = self._robot.data.joint_pos.torch[:, self._joints]
        dense -= 0.5 * ((limits[..., 0] - positions).clamp_min(0) + (positions - limits[..., 1]).clamp_min(0)).sum(-1)
        shaping = rotation_potential(self._previous_progress, self.routine.progress, self.reset_buf)
        reward = dense * self.step_dt + shaping + 0.5 * self.routine.transitions
        reward += 15.0 * self.routine.success - 8.0 * self.routine.failed
        reward -= 2.0 * (self.reset_time_outs & ~self.routine.success)
        return torch.nan_to_num(reward, nan=-8.0, posinf=-8.0, neginf=-8.0)

    def _reset_idx(self, env_ids):
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(completed):
            stage = self.curriculum.stage[completed]
            success = self.routine.success[completed]
            self.extras["log"] = {
                "Metrics/success": success.float().mean(),
                "Metrics/full_success": (success & (stage == 4)).float().mean(),
                "Metrics/failure": self.routine.failed[completed].float().mean(),
                "Metrics/stage": stage.float().mean(),
                "Metrics/phase": self.routine.phase[completed].float().mean(),
                "Metrics/turn_radians": self.routine.peak[completed].mean(),
                "Metrics/turn_completed": ((self.routine.phase[completed] >= 2) & (stage > 0)).float().mean(),
                "Metrics/bow_completed": (self.routine.phase[completed] >= 4).float().mean(),
            }
            for level in range(5):
                self.extras["log"][f"Metrics/stage_{level}_fraction"] = (stage == level).float().mean()
                self.extras["log"][f"Metrics/stage_{level}_success"] = (success & (stage == level)).sum() / (
                    stage == level
                ).sum().clamp_min(1)
            if self.cfg.curriculum:
                self.curriculum.record(completed, success)
        super()._reset_idx(env_ids)
        data = self._robot.data
        pose = data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        yaw = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * torch.pi
        pose[:, 3:7] = 0
        pose[:, 5] = torch.sin(yaw / 2)
        pose[:, 6] = torch.cos(yaw / 2)
        self._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        self._robot.write_joint_position_to_sim_index(position=data.default_joint_pos.torch[env_ids], env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(data.default_joint_vel.torch[env_ids]), env_ids=env_ids
        )
        self._actions[env_ids] = 0
        self._previous_actions[env_ids] = 0
        self._previous_progress[env_ids] = 0
        self._action_nonfinite[env_ids] = False
        self._home[env_ids] = pose[:, :3]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        feet = self._robot.data.body_link_pos_w.torch[env_ids][:, self._feet]
        self._anchor[env_ids] = feet[:, 0]
        self._foot_zero[env_ids] = feet[:, :, 2]
        self.routine.reset(env_ids, yaw, self.curriculum.angles[env_ids])
