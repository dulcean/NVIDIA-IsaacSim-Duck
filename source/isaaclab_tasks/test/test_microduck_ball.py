# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU scene contracts for heading and the two-robot rally; run on the NVIDIA host."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("task", ["IsaacContrib-Heading-MicroDuck", "IsaacContrib-Volley-MicroDuck"])
def test_microduck_ball_physical_contact_and_floor_reset(device, task):
    """A real dropped ball reaches a head sensor, and a missed ball resets only its own world."""
    cfg = parse_env_cfg(task, device=device, num_envs=2)
    sim_utils.create_new_stage()
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, cfg.observation_space)
        robot = env.scene["robot_a"]
        head_id = robot.find_bodies(cfg.head_reference_body)[0][0]
        ball = env.scene["ball"]
        pose = ball.data.root_link_pose_w.torch.clone()
        pose[:, :3] = robot.data.body_link_pos_w.torch[:, head_id]
        pose[:, 2] += 0.07
        velocity = torch.zeros(2, 6, device=device)
        velocity[:, 2] = -0.2
        ball.write_root_pose_to_sim_index(root_pose=pose)
        ball.write_root_velocity_to_sim_index(root_velocity=velocity)
        sensor = env.scene["ball_contacts_a"]
        head_ids = sensor.find_sensors(cfg.head_body_pattern)[0]
        saw_head_contact = False
        for _ in range(30):
            obs, reward, _, _, extras = env.step(torch.zeros(2, cfg.action_space, device=device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            forces = sensor.data.normal_force_matrix_w_history.torch[:, :, head_ids]
            saw_head_contact |= bool((forces.norm(dim=-1) > cfg.contact_threshold).any())
            # A terminal impact can reset the sensor history before step() returns.
            saw_head_contact |= float(extras.get("log", {}).get("Metrics/head_hits", 0)) > 0
        assert saw_head_contact, "A dropped ball never reached a head collider: inspect USD and sensor bindings."

        env.reset()
        # Force a miss in world 0 while world 1's ball remains safely airborne.
        pose = ball.data.root_link_pose_w.torch.clone()
        pose[0, :3] = env.scene.env_origins[0] + torch.tensor((0.5, 0.2, cfg.ball_radius), device=device)
        pose[1, 2] = env.scene.env_origins[1, 2] + 0.8
        ball.write_root_pose_to_sim_index(root_pose=pose)
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6, device=device))
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, cfg.action_space, device=device))
        assert terminated[0] and not truncated[0]
        assert not terminated[1] and not truncated[1]
        assert ball.data.root_link_pos_w.torch[0, 2] > env.scene.env_origins[0, 2] + cfg.ball_radius + 0.1
        assert env.episode_length_buf[0] == 0 and env.episode_length_buf[1] == 1
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    finally:
        env.close()
        SimulationContext.clear_instance()
