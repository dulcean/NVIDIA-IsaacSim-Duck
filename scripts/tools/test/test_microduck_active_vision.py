# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active-gaze labels and reward integration independent of the simulator."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/active_vision.py"
_SPEC = importlib.util.spec_from_file_location("microduck_active_vision", _PATH)
gaze = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gaze
_SPEC.loader.exec_module(gaze)


def cameras(batch=1):
    position = torch.zeros(batch, 2, 3)
    quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2] * batch)
    intrinsic = torch.tensor([[[[100.0, 0.0, 80.0], [0.0, 100.0, 45.0], [0.0, 0.0, 1.0]]] * 2] * batch)
    return position, quat, intrinsic


def test_optical_rotation_backside_and_off_frame_acquisition_geometry():
    """Acquisition improves before visibility; turning the optical axis changes the correct camera's score."""
    position, quat, intrinsic = cameras(3)
    quat[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    quat[1, 1] = torch.tensor([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    result = gaze.camera_gaze_geometry(points, position, quat, intrinsic, (160, 90))
    torch.testing.assert_close(result.alignment[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result.alignment[1], torch.tensor([0.5, 1.0]))
    assert result.in_frame.tolist() == [[True, False], [False, True], [False, False]]
    torch.testing.assert_close(result.pixel_xy[0, 0], torch.tensor([80.0, 45.0]))
    torch.testing.assert_close(result.pixel_xy[1, 1], torch.tensor([80.0, 45.0]))
    assert 0.5 < result.alignment[2, 0] < 1  # There is useful feedback before the ball reaches the narrow image.
    assert not result.pixel_xy[0, 1].any()


def test_calibrated_projection_uses_raw_distortion_and_rejects_invalid_pose():
    """A known radial/tangential projection uses raw pixels; invalid camera state yields finite empty labels."""
    position, quat, intrinsic = cameras()
    points = torch.tensor([[0.2, 0.1, 1.0]])
    # For x=.2,y=.1,r²=.05, k1=.1,p1=.01,p2=.02 give xd=.204, yd=.102.
    quat[0, 1] = float("nan")
    result = gaze.camera_gaze_geometry(
        points, position, quat, intrinsic, (160, 90), distortion=(0.1, 0.0, 0.01, 0.02, 0.0)
    )
    torch.testing.assert_close(result.pixel_xy[0, 0], torch.tensor([100.4, 55.2]), atol=1e-5, rtol=0)
    assert result.in_frame.tolist() == [[True, False]]
    assert result.alignment[0, 1] == 0 and not result.pixel_xy[0, 1].any()
    assert torch.isfinite(result.alignment).all() and torch.isfinite(result.pixel_xy).all()


def test_only_actual_strike_grace_masks_gaze_and_dropouts_cannot_earn_pixel_credit():
    """Missing pixels retain acquisition reward; contact/recent own strikes and ended rallies suppress it."""
    alignment = torch.ones(4, 2)
    accepted = torch.tensor([[True, True], [False, False], [True, True], [True, True]])
    playing = torch.tensor([True, True, True, False])
    contact = torch.tensor([[True, False], [False, False], [False, False], [False, False]])
    hit_age = torch.tensor([[2.0, 2.0], [2.0, 2.0], [0.079, 0.080], [2.0, 2.0]])
    result = gaze.active_gaze_reward(alignment, accepted, playing, contact, hit_age, 0.02)
    assert result.eligible.tolist() == [[False, True], [True, True], [False, True], [False, False]]
    torch.testing.assert_close(result.reward, torch.tensor([0.003, 0.003, 0.003, 0.0]))
    torch.testing.assert_close(result.visible_mean, torch.tensor([0.5, 0.0, 0.5, 0.0]))
    torch.testing.assert_close(result.alignment_mean, torch.tensor([0.5, 1.0, 0.5, 0.0]))


def test_gaze_reward_is_a_bounded_per_second_term_with_finite_failure_outputs():
    """Changing policy timestep does not change the reward earned over a physical second."""
    alignment = torch.tensor([[1.0, 1.0], [float("nan"), float("inf")]])
    accepted = torch.ones(2, 2, dtype=torch.bool)
    playing = torch.ones(2, dtype=torch.bool)
    contact = torch.zeros(2, 2, dtype=torch.bool)
    age = torch.full((2, 2), 2.0)
    coarse = gaze.active_gaze_reward(alignment, accepted, playing, contact, age, 0.1)
    fine = gaze.active_gaze_reward(alignment, accepted, playing, contact, age, 0.02)
    torch.testing.assert_close(coarse.reward * 10, fine.reward * 50)
    assert coarse.reward[0].item() == pytest.approx(0.03)
    assert torch.isfinite(coarse.reward).all() and (coarse.reward <= 0.03 + 1e-7).all()
    with pytest.raises(ValueError, match="positive"):
        gaze.active_gaze_reward(alignment, accepted, playing, contact, age, 0)


def test_watch_requires_real_continuous_exposures_and_independent_reset():
    """Cached/predicted frames cannot graduate acquisition; one missing exposure breaks the streak."""
    window = gaze.MeasuredVisibilityWindow(2, "cpu", max_gap=0.07)
    matched = torch.ones(2, 2, dtype=torch.bool)
    cached = torch.zeros_like(matched)
    window.update(matched, matched, 0.035)
    assert not window.hold.any()  # No unseen interval before the first exposure earns credit.
    for _ in range(8):
        window.update(matched, cached, 0.015)
        window.update(matched, matched, 0.020)
    torch.testing.assert_close(window.hold, torch.full((2, 2), 0.28))
    assert not (window.hold >= 0.3).any()
    window.update(matched, matched, 0.035)
    assert (window.hold > 0.3).all()
    missed = matched.clone()
    missed[0, 0] = False
    window.update(missed, matched, 0.035)
    assert window.hold[0, 0] == 0 and not window.recent[0, 0]
    assert window.peak[0, 0] > 0.3
    window.reset(torch.tensor([0]))
    assert not window.hold[0].any() and not window.peak[0].any() and not window.recent[0].any()
    assert (window.hold[1] > 0.3).all()
    # A stalled capture stream never keeps old evidence eligible indefinitely.
    window.update(matched, cached, 0.08)
    assert not window.recent.any()
    window.update(matched, matched, 0.005)
    assert not window.hold.any()
