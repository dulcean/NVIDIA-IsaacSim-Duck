# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-only tracking against independently rendered analytic sphere silhouettes."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

_PATH = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/ball_vision.py"
_SPEC = importlib.util.spec_from_file_location("microduck_ball_vision", _PATH)
vision = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vision
_SPEC.loader.exec_module(vision)


@pytest.fixture(autouse=True)
def small_image_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def calibration(batch=1, height=96, width=128):
    intrinsic = torch.tensor([[140.0, 0.0, width / 2], [0.0, 140.0, height / 2], [0.0, 0.0, 1.0]])
    return intrinsic.expand(batch, -1, -1).clone()


def render_spheres(centers, intrinsic, height=96, width=128):
    """Exact pinhole ray/sphere intersection, rather than copying the tracker's size inversion."""
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    frames = []
    for point, matrix in zip(centers, intrinsic):
        rays = torch.stack(((xx - matrix[0, 2]) / matrix[0, 0], (yy - matrix[1, 2]) / matrix[1, 1], xx * 0 + 1), -1)
        projected = (rays * point).sum(-1)
        mask = projected.square() - rays.square().sum(-1) * (point.square().sum() - 0.025**2) >= 0
        frame = torch.zeros(height, width, 3, dtype=torch.uint8)
        frame[-12:] = torch.tensor((255, 230, 10), dtype=torch.uint8)  # Yellow robot body is not the ball.
        frame[mask & (projected > 0)] = torch.tensor((255, 115, 15), dtype=torch.uint8)
        frames.append(frame)
    return torch.stack(frames)


def poses(batch=1):
    return torch.zeros(batch, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(batch, -1).clone()


def test_rgb_sphere_range_and_background_rejection():
    """Known spheres are localized from pixels; yellow bodies and a clipped orange blob cannot seed a track."""
    points = torch.tensor([[0.0, 0.0, 0.5], [0.08, -0.04, 0.65], [-0.225, 0.0, 0.5]])
    intrinsic = calibration(3)
    rgb = render_spheres(points, intrinsic)
    # An orange beak touching the image boundary must not hide the valid ball component.
    rgb[:, -4:, 55:73] = torch.tensor((255, 115, 15), dtype=torch.uint8)
    tracker = vision.BallVisualTracker(3, "cpu")
    result = tracker.update(rgb, intrinsic, *poses(3))
    assert result.visible.tolist() == [True, True, False]
    torch.testing.assert_close(result.position_w[:2], points[:2], atol=0.015, rtol=0)
    assert not result.velocity_w.any() and not result.position_w[2].any()
    assert (result.confidence[:2] > 0.5).all()


def test_head_motion_is_removed_before_world_velocity_estimation():
    """Moving/rotating the camera and moving the ball produce the known world displacement."""
    tracker = vision.BallVisualTracker(1, "cpu", velocity_smoothing=1.0)
    intrinsic = calibration()
    first = torch.tensor([[0.02, 0.0, 0.60]])
    tracker.update(render_spheres(first, intrinsic), intrinsic, *poses())
    # Three prediction-only physics ticks precede the next camera exposure.
    for _ in range(3):
        assert tracker.advance(0.005) is None
    angle = 0.04
    rotation = torch.tensor(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
    )
    camera_position = torch.tensor([[0.01, 0.0, 0.0]])
    camera_quaternion = torch.tensor([[0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]])
    second = torch.tensor([[0.04, 0.0, 0.60 - 0.5 * 9.81 * 0.02**2]])
    camera_point = (second - camera_position) @ rotation
    assert (
        tracker.update(
            render_spheres(camera_point, intrinsic),
            intrinsic,
            camera_position,
            camera_quaternion,
            dt=0.005,
            snapshot=False,
        )
        is None
    )
    result = tracker.tracks
    assert result.visible.item() and result.age.item() == 0
    torch.testing.assert_close(result.position_w, second, atol=0.015, rtol=0)
    assert result.velocity_w[0, 0].item() == pytest.approx(1.0, abs=0.3)
    assert result.velocity_w[0, 1].item() == pytest.approx(0.0, abs=0.05)


def test_occlusion_prediction_expiry_and_partial_reset():
    """Partial silhouettes use bounded prediction; expiry and selected resets erase old measurements."""
    tracker = vision.BallVisualTracker(2, "cpu")
    intrinsic = calibration(2)
    rgb = render_spheres(torch.tensor([[0.0, 0.0, 0.5]] * 2), intrinsic)
    initial = tracker.update(rgb, intrinsic, *poses(2))
    occluded = rgb.clone()
    occluded[:, :, 64:] = 0  # A half disk is not a complete spherical measurement.
    result = tracker.update(occluded, intrinsic, *poses(2), dt=0.04)
    assert not result.visible.any() and result.valid.all()
    torch.testing.assert_close(result.position_w[:, 2], initial.position_w[:, 2] - 0.5 * 9.81 * 0.04**2)
    tracker.reset(torch.tensor([0]))
    result = tracker.tracks
    assert result.valid.tolist() == [False, True]
    assert not result.position_w[0].any()
    result = tracker.predict(0.12)
    assert not result.valid.any() and not result.position_w.any() and not result.velocity_w.any()
    assert not result.confidence.any()
    assert result.seen.tolist() == [False, True]
    torch.testing.assert_close(result.last_seen_position_w[1], initial.position_w[1])
    # The search bearing survives expiry without reintroducing a usable ball pose.
    root_positions = torch.zeros(2, 2, 3)
    root_quats = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2] * 2)
    actor = vision.visual_actor_observations(
        torch.zeros(2, 142), [result, result], root_positions, root_quats, root_positions
    ).view(2, 2, 76)
    assert not actor[:, :, 61:67].any()
    assert not actor[0, :, 73:76].any()
    torch.testing.assert_close(actor[1, :, 73:76], torch.tensor([[0.0, 0.0, 1.0]] * 2), atol=0.01, rtol=0)
    # A stale frame after reset cannot initialize the selected world.
    result = tracker.update(rgb, intrinsic, *poses(2), measurement_mask=torch.tensor([False, True]))
    assert result.visible.tolist() == [False, True]
    assert not result.velocity_w[1].any()


def test_invalid_calibration_and_empty_images_cannot_invent_observations():
    tracker = vision.BallVisualTracker(2, "cpu")
    intrinsic = calibration(2)
    intrinsic[0, 0, 0] = float("nan")
    position, quaternion = poses(2)
    quaternion[1] = 0
    rgb = render_spheres(torch.tensor([[0.0, 0.0, 0.5]] * 2), calibration(2))
    result = tracker.update(rgb, intrinsic, position, quaternion)
    assert not result.valid.any() and torch.isfinite(result.position_w).all()
    with pytest.raises(ValueError, match="height"):
        tracker.update(torch.empty(2, 0, 128, 3, dtype=torch.uint8), intrinsic, position, quaternion)
    with pytest.raises(ValueError, match="uint8"):
        tracker.update(rgb.float() * float("nan"), intrinsic, position, quaternion)


def test_equally_plausible_orange_objects_require_an_existing_association():
    """Color alone cannot identify one of two indistinguishable spheres; a prior world track can."""
    tracker = vision.BallVisualTracker(1, "cpu")
    intrinsic = calibration()
    first = render_spheres(torch.tensor([[-0.10, 0.0, 0.5]]), intrinsic)
    second = render_spheres(torch.tensor([[0.10, 0.0, 0.5]]), intrinsic)
    combined = torch.maximum(first, second)
    result = tracker.update(combined, intrinsic, *poses())
    assert not result.valid.item()
    result = tracker.update(first, intrinsic, *poses())
    assert result.visible.item()
    result = tracker.update(combined, intrinsic, *poses())
    assert result.visible.item() and result.position_w[0, 0] < -0.08


def test_calibrated_distortion_is_inverted_for_an_off_axis_sphere():
    """Forward-project a sphere's tangent polygon with published OpenCV coefficients, then recover its range."""
    height, width = 90, 160
    intrinsic = torch.tensor([[[132.72575, 0.0, 74.59721], [0.0, 132.77421, 59.31544], [0.0, 0.0, 1.0]]])
    distortion = (-0.3469538869, 0.1461586615, -0.0007449158, -0.0022167171, -0.0164372661)
    point = torch.tensor([0.16, -0.04, 0.55])
    direction = point / point.norm()
    horizontal = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), direction)
    horizontal /= horizontal.norm()
    vertical = torch.linalg.cross(direction, horizontal)
    angle = math.asin(0.025 / point.norm().item())
    azimuth = torch.arange(256) * (2 * math.pi / 256)
    rays = direction * math.cos(angle) + math.sin(angle) * (
        azimuth.cos()[:, None] * horizontal + azimuth.sin()[:, None] * vertical
    )
    x, y = (rays[:, :2] / rays[:, 2:]).unbind(-1)
    k1, k2, p1, p2, k3 = distortion
    radial_squared = x.square() + y.square()
    radial = 1 + k1 * radial_squared + k2 * radial_squared.square() + k3 * radial_squared.pow(3)
    u = (x * radial + 2 * p1 * x * y + p2 * (radial_squared + 2 * x.square())) * intrinsic[0, 0, 0] + intrinsic[0, 0, 2]
    v = (y * radial + p1 * (radial_squared + 2 * y.square()) + 2 * p2 * x * y) * intrinsic[0, 1, 1] + intrinsic[0, 1, 2]
    # Standard even/odd polygon rasterization is independent of inverse distortion and sphere-size inversion.
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    inside = torch.zeros(height, width, dtype=torch.bool)
    for index in range(len(u)):
        previous = (index - 1) % len(u)
        if abs((v[previous] - v[index]).item()) > 1e-8:
            edge = (v[index] > yy) != (v[previous] > yy)
            crossing = (u[previous] - u[index]) * (yy - v[index]) / (v[previous] - v[index]) + u[index]
            inside ^= edge & (xx < crossing)
    rgb = torch.zeros(1, height, width, 3, dtype=torch.uint8)
    rgb[0, inside] = torch.tensor((255, 115, 15), dtype=torch.uint8)
    result = vision.BallVisualTracker(1, "cpu", distortion=distortion).update(rgb, intrinsic, *poses())
    assert result.visible.item()
    # Quantized 5-6 pixel silhouettes limit depth precision; this is a monocular approximation.
    torch.testing.assert_close(result.position_w[0], point, atol=0.025, rtol=0)


def test_actor_ball_features_depend_only_on_visual_tracks_and_each_robots_frame():
    """Changing all simulator ball channels cannot affect the actor, including absent and expired tracks."""
    privileged = torch.arange(284, dtype=torch.float32).reshape(2, 142)
    first = vision.VisualBallTracks(
        torch.tensor([[0.3, 0.2, 0.5], [float("nan")] * 3]),
        torch.tensor([[0.1, 0.2, 0.3], [float("nan")] * 3]),
        torch.tensor([True, False]),
        torch.tensor([0.0, 0.1]),
        torch.tensor([True, False]),
        torch.tensor([1.0, 0.0]),
    )
    second = vision.VisualBallTracks(
        torch.tensor([[0.4, 0.6, 0.8]] * 2),
        torch.tensor([[0.11, 0.22, 0.33]] * 2),
        torch.tensor([False, True]),
        torch.tensor([0.05, 0.20]),
        torch.tensor([True, True]),
        torch.tensor([1.0, 1.0]),
    )
    root_position = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]] * 2)
    root_velocity = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]]] * 2)
    root_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]] * 2)
    actor = vision.visual_actor_observations(privileged, [first, second], root_position, root_quat, root_velocity)
    changed_truth = privileged.clone()
    changed_truth[:, 61:67] = -999
    changed_truth[:, 132:138] = 999
    independent = vision.visual_actor_observations(
        changed_truth, [first, second], root_position, root_quat, root_velocity
    )
    torch.testing.assert_close(actor, independent, rtol=0, atol=0)
    result = actor.view(2, 2, 76)
    original = privileged.view(2, 2, 71)
    torch.testing.assert_close(result[:, :, :61], original[:, :, :61])
    torch.testing.assert_close(result[:, :, 67:71], original[:, :, 67:71])
    torch.testing.assert_close(result[0, 0, 61:67], torch.tensor([0.3, 0.2, 0.5, 0.1, 0.2, 0.3]))
    torch.testing.assert_close(result[0, 1, 61:67], torch.tensor([-0.3, -0.4, 0.5, -0.1, -0.2, 0.3]))
    assert not result[1, :, 61:67].any()
    assert result[1, :, 71].tolist() == [0, 0]
    assert result[1, :, 72].tolist() == [1, 1]
    assert torch.isfinite(actor).all()


def test_uploaded_gpu_camera_frame_preserves_the_shaded_ball_silhouette():
    """The actual shaded orange ball is detected; the other real camera's robot parts are rejected.

    Fixtures are unchanged GPU preview PNGs from 2026-10-26, saved upright by
    the preview. Restore native orientation before using the camera's raw K.
    Only the observed ball pixel centroid is known; this test asserts no
    invented world position or ground-truth ball distance.
    """
    frames = []
    for name in ("a", "b"):
        image = Image.open(Path(__file__).parent / "data" / f"microduck_head_{name}.png")
        image = image.transpose(Image.Transpose.ROTATE_90)
        frames.append(torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(90, 160, 3))
    intrinsic = torch.tensor([[[132.72575, 0.0, 74.59721], [0.0, 132.77421, 59.31544], [0.0, 0.0, 1.0]]] * 2)
    distortion = (-0.3469538869, 0.1461586615, -0.0007449158, -0.0022167171, -0.0164372661)
    tracker = vision.BallVisualTracker(2, "cpu", distortion=distortion)
    result = tracker.update(torch.stack(frames), intrinsic, *poses(2))
    assert result.visible.tolist() == [False, True]
    assert result.confidence[1] > 0.5
    x, y = result.position_w[1, :2] / result.position_w[1, 2]
    k1, k2, p1, p2, k3 = distortion
    radius_squared = x.square() + y.square()
    radial = 1 + k1 * radius_squared + k2 * radius_squared.square() + k3 * radius_squared.pow(3)
    u = (x * radial + 2 * p1 * x * y + p2 * (radius_squared + 2 * x.square())) * intrinsic[1, 0, 0]
    v = (y * radial + p1 * (radius_squared + 2 * y.square()) + 2 * p2 * x * y) * intrinsic[1, 1, 1]
    assert (u + intrinsic[1, 0, 2]).item() == pytest.approx(29.19, abs=0.5)
    assert (v + intrinsic[1, 1, 2]).item() == pytest.approx(49.93, abs=0.5)
