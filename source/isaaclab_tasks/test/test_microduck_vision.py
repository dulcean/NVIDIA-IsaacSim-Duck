# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real Newton RGB/physical-head integration; run on the NVIDIA training host.

The one scene checks actual rendered pixels, calibrated estimates, articulated
camera motion and isolated reset behavior. It does not claim that an untrained
policy plays volleyball or that alpha-family optics calibrate a particular robot.
"""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.head_camera_cfg import RAW_IMAGE_CLOCKWISE_ROTATIONS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES


def _assert_finite_step(env, actions):
    observations, reward, terminated, truncated, extras = env.step(actions)
    assert observations["policy"].shape == (2, 146)
    assert observations["critic"].shape == (2, 142)
    assert torch.isfinite(observations["policy"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
    return observations, terminated, truncated, extras


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_real_head_rgb_tracking_motion_and_partial_reset(device):
    """Pixels locate two airborne balls and survive head motion and an unrelated world's reset."""
    task = "IsaacContrib-Volley-Vision-MicroDuck"
    cfg = parse_env_cfg(task, device=device, num_envs=2)
    cfg.evaluation = True
    cfg.head_camera_panels = False
    cfg.sim.visualizer_cfgs = []
    sim_utils.create_new_stage()
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        observations, _ = env.reset()
        assert observations["policy"].shape == (2, 146)
        assert observations["critic"].shape == (2, 142)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert all(not track.valid.any() for track in env.vision_tracks), (
            "Reset must not initialize a visual track from simulator ball coordinates."
        )
        for camera, consumed_frame in zip(env.head_cameras, env._last_camera_frames):
            torch.testing.assert_close(consumed_frame, camera.frame.torch.long())
        actions = torch.zeros(2, 28, device=device)
        for _ in range(8):
            _assert_finite_step(env, actions)

        # Both real head sensors must produce nonconstant RGB, not a guide
        # marker, a backwards view of the lens, or the external Viser camera.
        forwards = []
        frames = []
        for camera, robot, head in zip(env.head_cameras, env._robots, env._head_ids):
            # Synchronize this frame with the current physical pose rather
            # than comparing a 30 Hz exposure with a newer 200 Hz body pose.
            camera.reset()
            data = camera.data
            rgb = data.output["rgb"].torch
            assert rgb.shape == (2, 90, 160, 3) and rgb.dtype == torch.uint8
            assert (rgb.float().std(dim=(1, 2, 3)) > 5.0).all()
            assert (camera.frame.torch > 0).all()
            optical_forward = quat_apply(
                data.quat_w_ros.torch,
                torch.tensor((0.0, 0.0, 1.0), device=device).expand(2, -1),
            )
            expected_forward = quat_apply(
                robot.data.body_link_quat_w.torch[:, head],
                torch.tensor((0.0, 0.0, -1.0), device=device).expand(2, -1),
            )
            torch.testing.assert_close(optical_forward, expected_forward, atol=2e-4, rtol=0)
            # Image-coordinate rotation, independently compared with the
            # physical head's upright axis. Detector inputs stay unrotated.
            raw_display_up = ((0, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0))[RAW_IMAGE_CLOCKWISE_ROTATIONS % 4]
            displayed_up = quat_apply(
                data.quat_w_ros.torch,
                torch.tensor(raw_display_up, dtype=torch.float32, device=device).expand(2, -1),
            )
            expected_up = quat_apply(
                robot.data.body_link_quat_w.torch[:, head],
                torch.tensor((1.0, 0.0, 0.0), device=device).expand(2, -1),
            )
            torch.testing.assert_close(displayed_up, expected_up, atol=2e-4, rtol=0)
            forwards.append(optical_forward.clone())
            frames.append(camera.frame.torch.clone())

        # A real servo action must turn each optical frame, including the
        # parked coach. A camera attached to the root would fail this check.
        yaw = MICRODUCK_JOINT_NAMES.index("head_yaw")
        actions[:, yaw] = actions[:, 14 + yaw] = 0.2
        for _ in range(6):
            _assert_finite_step(env, actions)
        for camera, before, old_frame in zip(env.head_cameras, forwards, frames):
            after = quat_apply(
                camera.data.quat_w_ros.torch,
                torch.tensor((0.0, 0.0, 1.0), device=device).expand(2, -1),
            )
            assert ((after - before).norm(dim=-1) > 1e-4).all()
            assert (camera.frame.torch > old_frame).all()

        # Controlled initial conditions put a real sphere 25 cm in front of
        # A/world 0 and B/world 1. Thereafter gravity and contacts act normally;
        # no position, velocity or ground-truth hint is fed to either tracker.
        env.reset()
        assert all(not track.valid.any() for track in env.vision_tracks)
        pose = env._ball.data.root_link_pose_w.torch.clone()
        for player, camera in enumerate(env.head_cameras):
            data = camera.data
            position = data.pos_w.torch[player]
            rotation = data.quat_w_ros.torch[player]
            pose[player, :3] = (
                position + quat_apply(rotation.unsqueeze(0), torch.tensor(((0.0, 0.0, 0.25),), device=device))[0]
            )
        assert (pose[:, 2] - env.scene.env_origins[:, 2] > cfg.ball_radius + 0.1).all()
        env._ball.write_root_pose_to_sim_index(root_pose=pose)
        env._ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(2, 6, device=device))
        for camera in env.head_cameras:
            camera.reset()
        measurements_before = env.vision_diagnostics()["accepted_rgb_measurements"]
        actions.zero_()
        located = torch.zeros(2, dtype=torch.bool, device=device)
        for _ in range(5):
            _, terminated, truncated, _ = _assert_finite_step(env, actions)
            assert not terminated.any() and not truncated.any()
            for player, track in enumerate(env.vision_tracks):
                error = (track.position_w[player] - env._ball.data.root_link_pos_w.torch[player]).norm()
                located[player] |= track.valid[player] & (error < 0.04)
            if located.all():
                break
        measurements_after = env.vision_diagnostics()["accepted_rgb_measurements"]
        assert all(after > before for before, after in zip(measurements_before, measurements_after)), (
            "A native RGB camera never produced an accepted orange-sphere measurement."
        )
        assert located.all(), "A rendered sphere's monocular estimate missed its physical position by 4 cm."
        assert env.vision_tracks[1].valid[1]

        # A physical inversion terminates only world 0. Its selected tracking
        # state and stale-frame cache must clear; B/world 1 must retain vision.
        episode_ids = env._episode_ids.clone()
        retained_frame = env._last_camera_frames[1][1].clone()
        robot = env._robots[0]
        root = robot.data.root_link_pose_w.torch[:1].clone()
        root[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)  # XYZW roll by pi.
        robot.write_root_pose_to_sim_index(root_pose=root, env_ids=torch.tensor([0], device=device))
        observations, terminated, truncated, _ = _assert_finite_step(env, actions)
        assert terminated.tolist() == [True, False] and not truncated.any()
        assert env._episode_ids.tolist() == [int(episode_ids[0]) + 1, int(episode_ids[1])]
        assert all(not track.valid[0] for track in env.vision_tracks)
        assert env.vision_tracks[1].valid[1]
        assert all(
            consumed_frame[0] == camera.frame.torch[0]
            for camera, consumed_frame in zip(env.head_cameras, env._last_camera_frames)
        ), "A capture from the intermediate reset state must not become a new visual measurement."
        assert env._last_camera_frames[1][1] >= retained_frame > 0
        assert not observations["policy"][0, 61:67].any()
        assert not observations["policy"][0, 134:140].any()
    finally:
        env.close()
        SimulationContext.clear_instance()
