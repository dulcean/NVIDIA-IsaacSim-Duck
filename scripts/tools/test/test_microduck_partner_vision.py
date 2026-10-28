# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Partner forehead localization from independently rasterized camera pixels."""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_DIRECTORY = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck"
_PACKAGE = "microduck_live_perception"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_DIRECTORY)]
    sys.modules[_PACKAGE] = package
vision = importlib.import_module(f"{_PACKAGE}.partner_vision")


@pytest.fixture(autouse=True)
def small_image_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def render_badges(cyan, magenta, *, orange=True, marker_radius=0.008):
    """Use independent pinhole ray/sphere intersections at the native 160x90 scale."""
    intrinsic = torch.tensor([[[132.72575, 0.0, 74.59721], [0.0, 132.77421, 59.31544], [0.0, 0.0, 1.0]]])
    yy, xx = torch.meshgrid(torch.arange(90), torch.arange(160), indexing="ij")
    rays = torch.stack(
        ((xx - intrinsic[0, 0, 2]) / intrinsic[0, 0, 0], (yy - intrinsic[0, 1, 2]) / intrinsic[0, 1, 1], xx * 0 + 1),
        -1,
    )
    rgb = torch.zeros(1, 90, 160, 3, dtype=torch.uint8)
    rgb[:, -10:] = torch.tensor((255, 230, 10), dtype=torch.uint8)
    spheres = [(cyan, marker_radius, (15, 185, 220)), (magenta, marker_radius, (220, 15, 185))]
    if orange:
        spheres.append((torch.tensor((0.055, -0.025, 0.40)), 0.025, (255, 115, 15)))
    for point, radius, color in spheres:
        if point is None:
            continue
        projected = (rays * point).sum(-1)
        mask = projected.square() - rays.square().sum(-1) * (point.square().sum() - radius**2) >= 0
        rgb[0, mask & (projected > 0)] = torch.tensor(color, dtype=torch.uint8)
    return rgb, intrinsic


def camera_pose():
    return torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]])


def test_rgb_pair_recovers_tilted_forehead_without_confusing_ball_or_yellow_body():
    """Known sphere size + a rigid color baseline recover the target, independently of orange pixels."""
    cyan = torch.tensor((-0.035, -0.010, 0.30))
    # A tilted 20 mm head baseline; the forehead is one baseline below cyan.
    magenta = cyan + torch.tensor((0.016, -0.012, 0.0))
    rgb, intrinsic = render_badges(cyan, magenta)
    result = vision.PartnerHeadVisualTracker(1, "cpu").update(rgb, intrinsic, *camera_pose())
    assert result.visible.item() and result.valid.item()
    torch.testing.assert_close(result.position_w[0], 2 * cyan - magenta, atol=0.025, rtol=0)
    assert result.confidence.item() > 0.3
    assert not result.velocity_w.any()
    ball = vision.BallVisualTracker(1, "cpu").update(rgb, intrinsic, *camera_pose())
    assert ball.visible.item()
    torch.testing.assert_close(ball.position_w[0], torch.tensor((0.055, -0.025, 0.40)), atol=0.025, rtol=0)


def test_incomplete_pair_or_inconsistent_baseline_cannot_initialize_partner():
    """A single badge and unrelated color blobs cannot invent a receiving point."""
    cyan = torch.tensor((-0.035, -0.010, 0.30))
    tracker = vision.PartnerHeadVisualTracker(1, "cpu")
    rgb, intrinsic = render_badges(cyan, None)
    result = tracker.update(rgb, intrinsic, *camera_pose())
    assert not result.valid.item() and not result.position_w.any()
    rgb, intrinsic = render_badges(cyan, torch.tensor((0.025, -0.010, 0.30)), orange=False)
    result = tracker.update(rgb, intrinsic, *camera_pose())
    assert not result.valid.item() and not result.position_w.any()


def test_camera_motion_and_pair_dropout_use_finite_head_memory_without_gravity():
    """Own camera motion is removed, a missing badge does not correct state, and expiry/reset clear it."""
    cyan = torch.tensor((-0.035, -0.010, 0.30))
    magenta = cyan + torch.tensor((0.020, 0.0, 0.0))
    rgb, intrinsic = render_badges(cyan, magenta)
    tracker = vision.PartnerHeadVisualTracker(1, "cpu", max_age=0.25)
    initial = tracker.update(rgb, intrinsic, *camera_pose())
    assert initial.valid.item()
    # Shift the observing camera while keeping the partner fixed in world space.
    camera_position = torch.tensor([[0.008, 0.0, 0.0]])
    rgb, intrinsic = render_badges(cyan - camera_position[0], magenta - camera_position[0])
    corrected = tracker.update(rgb, intrinsic, camera_position, camera_pose()[1], dt=0.04)
    assert corrected.visible.item()
    torch.testing.assert_close(corrected.position_w, initial.position_w, atol=0.020, rtol=0)
    rgb, intrinsic = render_badges(cyan, None)
    missed = tracker.update(rgb, intrinsic, *camera_pose(), dt=0.04)
    assert not missed.visible.item() and missed.valid.item()
    torch.testing.assert_close(missed.position_w, corrected.position_w + corrected.velocity_w * 0.04, atol=1e-7, rtol=0)
    expired = tracker.predict(0.22)
    assert not expired.valid.item() and not expired.position_w.any() and not expired.velocity_w.any()
    assert expired.seen.item() and torch.isfinite(expired.last_seen_position_w).all()
    tracker.reset()
    assert not tracker.tracks.seen.item() and not tracker.tracks.last_seen_position_w.any()
    stale = tracker.update(rgb, intrinsic, *camera_pose(), measurement_mask=torch.tensor([False]))
    assert not stale.valid.item()


def test_unseen_badge_memory_never_creates_a_partner_or_a_search_bearing():
    tracker = vision.PartnerHeadVisualTracker(1, "cpu")
    result = tracker.predict(0.10)
    assert not result.valid.item() and not result.seen.item()
    assert not result.position_w.any() and not result.last_seen_position_w.any()


def test_selected_reset_clears_only_that_world_and_rejects_its_stale_frame():
    cyan = torch.tensor((-0.035, -0.010, 0.30))
    rgb, intrinsic = render_badges(cyan, cyan + torch.tensor((0.020, 0.0, 0.0)))
    rgb, intrinsic = rgb.expand(2, -1, -1, -1), intrinsic.expand(2, -1, -1)
    position = camera_pose()[0].expand(2, -1)
    quaternion = camera_pose()[1].expand(2, -1)
    tracker = vision.PartnerHeadVisualTracker(2, "cpu")
    initial = tracker.update(rgb, intrinsic, position, quaternion)
    assert initial.valid.all()
    tracker.reset(torch.tensor([0]))
    result = tracker.update(rgb, intrinsic, position, quaternion, measurement_mask=torch.tensor([False, True]))
    assert result.valid.tolist() == [False, True] and result.seen.tolist() == [False, True]
    assert not result.position_w[0].any() and not result.last_seen_position_w[0].any()
    torch.testing.assert_close(result.position_w[1], initial.position_w[1])
