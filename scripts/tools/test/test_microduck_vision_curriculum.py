# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU gates keeping RGB acquisition, physical strikes and teacher evidence distinct."""

import importlib
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck"
_PACKAGE = "microduck_vision_curriculum_tests"
_namespace = ModuleType(_PACKAGE)
_namespace.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = _namespace
teacher = importlib.import_module(f"{_PACKAGE}.volley_curriculum")
vision = importlib.import_module(f"{_PACKAGE}.vision_curriculum")


def report(state, evaluation_id):
    """Balanced independent counts identify the actual completed objective."""
    return vision.VisionEvaluationReport(
        evaluation_id=evaluation_id,
        stage=state.stage,
        required_returns=state.spec.required_returns,
        attempts_by_side=(64, 64),
        successes_by_side=(60, 60),
        unsafe_episodes=0,
        objective=state.spec.objective,
    )


def test_visual_objectives_cannot_substitute_for_each_other_or_real_returns():
    """Equal zero-return counts cannot graduate watch, strike and target-crossing lessons interchangeably."""
    state = vision.VisionCurriculumState()
    teacher_report = teacher.EvaluationReport(0, 0, 0, (64, 64), (60, 60), 0)
    with pytest.raises(ValueError, match="objective"):
        state.update_evaluation(teacher_report)
    assert state.last_evaluation_id == -1 and state.stage == 0
    evaluation_id = 0
    expected = ("watch_visible_ball", "receive_forehead", "aim_return", "rally", "rally", "rally")
    for stage, objective in enumerate(expected):
        assert state.stage == stage and state.spec.objective == objective
        wrong = replace(report(state, evaluation_id), objective="watch_visible_ball" if stage else "receive_forehead")
        if stage < 3:
            before = state.state_dict()
            with pytest.raises(ValueError, match="objective"):
                state.update_evaluation(wrong)
            assert state.state_dict() == before
        else:
            with pytest.raises(ValueError, match="return requirement"):
                state.update_evaluation(replace(report(state, evaluation_id), required_returns=0))
        assert state.update_evaluation(report(state, evaluation_id)).status == "passed"
        evaluation_id += 1
        decision = state.update_evaluation(report(state, evaluation_id))
        assert decision.status == ("complete" if stage == 5 else "advanced")
        evaluation_id += 1
    assert state.complete and state.spec.required_returns == 8


def test_visual_resume_keeps_its_pass_streak_and_rejects_teacher_or_changed_objectives():
    """Persisted visual evidence cannot load into a teacher contract or a modified objective."""
    state = vision.VisionCurriculumState(stage=1)
    state.update_evaluation(report(state, 20))
    saved = json.loads(json.dumps(state.state_dict()))
    restored = vision.VisionCurriculumState()
    restored.load_state_dict(saved)
    assert restored.stage == 1 and restored.spec.objective == "receive_forehead"
    assert restored.consecutive_passes == 1 and restored.last_evaluation_id == 20
    with pytest.raises(ValueError, match="fresh"):
        restored.update_evaluation(report(restored, 20))
    assert restored.update_evaluation(report(restored, 21)).stage == 2
    original = restored.state_dict()
    with pytest.raises(ValueError, match="stage definitions"):
        restored.load_state_dict(teacher.CurriculumState().state_dict())
    assert restored.state_dict() == original
    with pytest.raises(ValueError, match="stage definitions"):
        teacher.CurriculumState().load_state_dict(saved)
    saved["stages"][1]["objective"] = "watch_visible_ball"
    with pytest.raises(ValueError, match="stage definitions"):
        restored.load_state_dict(saved)
    assert restored.state_dict() == original


def test_incoming_visual_serves_are_physical_and_fully_visible_at_first_exposure():
    """A frontal source must survive gravity until the first real image, with its whole silhouette in view.

    The independent static FK fixture was calculated from the complete MicroDuck
    all-collisions/backlash USD on 2026-10-26: root z=.125, neck_pitch=head_pitch
    =.3491rad, other head/backlash angles zero, camera pupil (.0155,-.0000913778,
    -.083) and forehead (.035,0,-.07) in jaw_soft. Root-relative world offsets
    below are measured from that FK, not the serve/projection helpers. Receiver
    roots use the current lesson separations .35/.40/.45m, with B rotated180°.

    This proves geometric startup at the worst .035s capture delay for the
    static reset pose. The real GPU sensor preflight separately verifies that
    the rendered ball is acquired while the complete robot physics advances.
    """
    serves = importlib.import_module(f"{_PACKAGE}.ball_serves")
    gaze = importlib.import_module(f"{_PACKAGE}.active_vision")
    receives, partners, pupils, quaternions, times, heights, fractions, offsets, source_offsets = ([] for _ in range(9))
    root_half_gaps = (0.175, 0.200, 0.225, 0.225, 0.225, 0.225)
    for spec, half_gap in zip(vision.VISION_STAGES, root_half_gaps):
        extrema = itertools.product(
            spec.flight_time_range,
            (spec.source_height_min, spec.source_height_max),
            spec.lateral_range,
            (-spec.serve_spread, spec.serve_spread),
            (-spec.serve_spread, spec.serve_spread),
            (-1, 1),
        )
        for flight, height, lateral, offset_x, offset_y, side in extrema:
            receiver = (side * (half_gap - 0.060997388), 0.0, 0.272598627)
            receives.append(receiver)
            partners.append((-receiver[0], 0.0, receiver[2]))
            pupils.append((side * (half_gap - 0.073997382), side * 0.0000913795455, 0.253098626))
            quaternions.append(
                (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)) if side < 0 else (-math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
            )
            times.append(flight)
            heights.append(height)
            fractions.append(lateral)
            offsets.append((offset_x, offset_y, 0.025))
            source_offsets.append((-side * spec.source_distance, 0.0, 0.0))
    receiver, partner, pupil, quaternion, flight, height, lateral, offset, source_offset = (
        torch.tensor(value, dtype=torch.float64)
        for value in (receives, partners, pupils, quaternions, times, heights, fractions, offsets, source_offsets)
    )
    source, velocity, _, valid = serves.incoming_ballistic_serve(
        receiver, partner, flight, height, lateral, offset, source_offset=source_offset
    )
    gravity = receiver.new_tensor((0.0, 0.0, -9.81))
    arrival = source + velocity * flight[:, None] + 0.5 * gravity * flight[:, None].square()
    arrival_velocity = velocity + gravity * flight[:, None]
    assert valid.all()
    torch.testing.assert_close(arrival, receiver + offset, atol=1e-12, rtol=0)
    assert (arrival_velocity.norm(dim=-1) <= 3.5).all() and (arrival_velocity[:, 2] <= -0.5).all()

    first = source + velocity * 0.035 + 0.5 * gravity * 0.035**2
    delta = first - pupil
    distance = delta.norm(dim=-1)
    direction = delta / distance[:, None]
    tangent = torch.nn.functional.normalize(
        torch.cross(direction, direction.new_tensor((0.0, 1.0, 0.0)).expand_as(direction), dim=-1), dim=-1
    )
    second = torch.cross(direction, tangent, dim=-1)
    angles = torch.arange(64, dtype=torch.float64) * (2 * math.pi / 64)
    cosine = (1 - (0.025 / distance).square()).sqrt()
    # Tangent points on a physical radius=.025m sphere, independent of the
    # tracker or projection's point-label implementation.
    silhouette = pupil[:, None] + direction[:, None] * (distance * cosine.square())[:, None, None]
    silhouette = silhouette + 0.025 * cosine[:, None, None] * (
        tangent[:, None] * angles.cos()[None, :, None] + second[:, None] * angles.sin()[None, :, None]
    )
    points = torch.cat((first[:, None], silhouette), dim=1)
    count, samples = points.shape[:2]
    intrinsic = first.new_tensor(
        ((132.72575031275218, 0.0, 74.59721060271258), (0.0, 132.77421424462887, 59.315435053810795), (0.0, 0.0, 1.0))
    )
    geometry = gaze.camera_gaze_geometry(
        points.reshape(-1, 3),
        pupil[:, None, None].expand(count, samples, 2, 3).reshape(-1, 2, 3),
        quaternion[:, None, None].expand(count, samples, 2, 4).reshape(-1, 2, 4),
        intrinsic[None, None].expand(count * samples, 2, 3, 3),
        (160, 90),
        (-0.34695388691888, 0.14615866153945595, -0.0007449157712424215, -0.0022167170528743837, -0.016437266141048814),
    )
    pixels = geometry.pixel_xy[:, 0]
    border = torch.stack((pixels[:, 0], 159 - pixels[:, 0], pixels[:, 1], 89 - pixels[:, 1]), dim=-1)
    assert geometry.in_frame[:, 0].all() and border.min() >= 1.0
