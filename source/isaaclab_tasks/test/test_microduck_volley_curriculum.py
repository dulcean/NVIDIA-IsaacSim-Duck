# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVIDIA GPU integration contracts for the receive/recover/rally environment.

Run on the CUDA training host. These checks instantiate the real Newton/MJWarp
scene and use its contacts, actuation and automatic resets. They do not establish
that a trained policy succeeds at a lesson or at unassisted volleyball.
"""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.volley_curriculum import STAGES
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES


def _assert_physical_reset_serve(env):
    """Check the actual reset state against an independent point-mass arrival calculation."""
    spec = STAGES[env.stage]
    relative = env._ball.data.root_link_pos_w.torch - env.scene.env_origins
    velocity = env._ball.data.root_link_lin_vel_w.torch
    heads = env._head_positions() - env.scene.env_origins.unsqueeze(1)
    receiver = heads[torch.arange(env.num_envs, device=env.device), env._server]
    arrival_z = receiver[:, 2] + env.cfg.ball_radius
    discriminant = velocity[:, 2].square() + 19.62 * (relative[:, 2] - arrival_z)
    assert (discriminant > 0).all()
    time = (velocity[:, 2] + discriminant.sqrt()) / 9.81
    assert (time >= spec.flight_time_range[0] - 1e-5).all()
    assert (time <= spec.flight_time_range[1] + 1e-5).all()
    arrival_xy = relative[:, :2] + velocity[:, :2] * time.unsqueeze(-1)
    assert ((arrival_xy - receiver[:, :2]).abs() <= spec.serve_spread + 1e-5).all()
    assert (velocity[:, 2] - 9.81 * time < -0.5).all()
    source_height = relative[:, 2] - arrival_z
    assert ((source_height >= 0.18 - 1e-5) & (source_height <= 0.25 + 1e-5)).all()
    assert torch.isfinite(relative).all() and torch.isfinite(velocity).all()


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_curriculum_physical_serve_support_control_and_partial_reset(device):
    """One real scene exercises all lessons, both receivers, foot support and isolated failure resets."""
    cfg = parse_env_cfg("IsaacContrib-Volley-Curriculum-MicroDuck", device=device, num_envs=2)
    cfg.evaluation = True
    cfg.sim.visualizer_cfgs = []
    sim_utils.create_new_stage()
    env = gym.make("IsaacContrib-Volley-Curriculum-MicroDuck", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 142)
        assert env.single_action_space.shape == (28,)
        assert torch.isfinite(obs["policy"]).all()
        assert env._server.tolist() == [0, 1]
        _assert_physical_reset_serve(env)
        for player, robot in enumerate(env._robots):
            root_x = robot.data.root_link_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
            distances = torch.where(env._server == player, cfg.stage_separations[0] / 2, cfg.coach_partner_distance)
            torch.testing.assert_close(root_x, (-1 if player == 0 else 1) * distances)

        # Feet must be actual rigid ankle bodies, and the real ground must load
        # them even though the same sensors also filter their matrix for the ball.
        supported = torch.zeros(2, 2, dtype=torch.bool, device=device)
        actions = torch.zeros(2, 28, device=device)
        for sensor, feet in zip(env._sensors, env._feet):
            assert sorted(sensor.sensor_names[index] for index in feet) == ["ankle_left", "ankle_right"]
        for _ in range(8):
            obs, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            for player, (sensor, feet) in enumerate(zip(env._sensors, env._feet)):
                supported[:, player] |= (sensor.data.net_normal_forces_w.torch[:, feet, 2] > cfg.supported_force).any(
                    -1
                )
        assert supported.all(), "An active or parked robot never developed upward ankle support from the floor."

        # A small yaw command must physically reach both robots, including each
        # world's parked partner. There is no scripted or frozen coach robot.
        yaw = MICRODUCK_JOINT_NAMES.index("head_yaw")
        before = torch.stack(
            [robot.data.joint_pos.torch[:, ids[yaw]].clone() for robot, ids in zip(env._robots, env._joint_ids)], dim=1
        )
        actions[:, yaw] = actions[:, 14 + yaw] = 0.2
        for _ in range(6):
            obs, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        after = torch.stack(
            [robot.data.joint_pos.torch[:, ids[yaw]] for robot, ids in zip(env._robots, env._joint_ids)], dim=1
        )
        assert ((after - before).abs() > 1e-4).all(), (
            "A coach robot's physical servo did not respond to its policy action."
        )

        # Physical out-of-court balls create recovery phases in both worlds. A subsequent
        # physical inversion resets only world 0 and must retain world 1's phase.
        env.reset()
        episode_ids = env._episode_ids.clone()
        pose = env._ball.data.root_link_pose_w.torch.clone()
        pose[:, :3] = env.scene.env_origins + torch.tensor((1.2, 0.3, 0.5), device=device)
        env._ball.write_root_pose_to_sim_index(root_pose=pose)
        env._ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6, device=device))
        actions.zero_()
        _, _, terminated, truncated, _ = env.step(actions)
        assert not terminated.any() and not truncated.any()
        assert env._episode.phase.tolist() == [2, 2]
        robot = env._robots[0]
        root = robot.data.root_link_pose_w.torch[:1].clone()
        root[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)  # 180-degree roll, XYZW.
        robot.write_root_pose_to_sim_index(root_pose=root, env_ids=torch.tensor([0], device=device))
        obs, reward, terminated, truncated, extras = env.step(actions)
        assert terminated.tolist() == [True, False] and not truncated.any()
        assert env._episode.phase.tolist() == [0, 2]
        assert env.episode_length_buf.tolist() == [0, 2]
        assert env._episode_ids.tolist() == [int(episode_ids[0]) + 1, int(episode_ids[1])]
        records = extras["episode_records"]
        assert len(records) == 1
        expected = {
            "env_id": 0,
            "episode_id": int(episode_ids[0]),
            "server": 0,
            "success": False,
            "unsafe": True,
            "returns": 0,
            "stage": 0,
            "reason": "unsafe",
        }
        assert all(records[0][key] == value for key, value in expected.items())
        assert records[0]["goal_reached"] is False
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()

        # Reuse this one physical fixture: later lessons change the reset
        # distribution and then restore two real partners at their pair spacing.
        for stage in range(1, len(STAGES)):
            env.set_stage(stage)
            obs, _ = env.reset()
            assert env._server.tolist() == [0, 1]
            assert torch.isfinite(obs["policy"]).all()
            _assert_physical_reset_serve(env)
            if STAGES[stage].paired:
                for player, robot in enumerate(env._robots):
                    root_x = robot.data.root_link_pos_w.torch[:, 0] - env.scene.env_origins[:, 0]
                    expected = torch.full_like(root_x, (-1 if player == 0 else 1) * cfg.stage_separations[stage] / 2)
                    torch.testing.assert_close(root_x, expected)
            obs, reward, _, _, _ = env.step(actions)
            assert obs["policy"].shape == (2, 142)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    finally:
        env.close()
        SimulationContext.clear_instance()
