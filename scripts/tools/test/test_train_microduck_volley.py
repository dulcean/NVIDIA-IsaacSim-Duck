# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Driver boundaries keep evaluation selection and full-resume claims honest."""

import copy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from scripts.tools import train_microduck_volley as driver


@pytest.fixture
def resumable_checkpoint():
    """Use genuine RSL actor tensors and Torch optimizer structure, without a simulator."""
    source = MLPModel(
        TensorDict({"policy": torch.zeros(1, 61)}, batch_size=[1]),
        {"actor": ["policy"]},
        "actor",
        14,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
    )
    prepare = driver._tool("prepare_microduck_curriculum_checkpoint")
    checkpoint = prepare.prepare_curriculum_checkpoint({"actor_state_dict": source.state_dict()})
    parameters = [
        torch.nn.Parameter(value)
        for owner in ("actor_state_dict", "critic_state_dict")
        for name, value in checkpoint[owner].items()
        if name.startswith("mlp.") or name == "distribution.std_param"
    ]
    checkpoint["optimizer_state_dict"] = torch.optim.Adam(parameters).state_dict()
    checkpoint["iter"] = 8
    curriculum = driver._tool("evaluate_microduck_volley")._curriculum_module()
    checkpoint["infos"]["microduck_curriculum"].update(
        curriculum_state=curriculum.CurriculumState().state_dict(),
        completed_iterations=8,
        best_by_stage={},
        test_trials=0,
        certified=False,
        certification=None,
    )
    return checkpoint


def test_progress_selection_uses_physical_outcomes_and_the_weaker_receiving_side():
    """Higher training reward or one excellent player cannot replace balanced held-out success."""
    baseline = {
        "complete": True,
        "success_rate_by_side": [0.7, 0.7],
        "goal_reached_rate": 0.8,
        "unsafe_rate": 0.01,
        "mean_reward": -500.0,
    }
    score = driver.progress_score(baseline)
    inflated_reward = {**baseline, "mean_reward": 1000000.0, "value_loss": 0.0}
    assert driver.progress_score(inflated_reward) == score
    assert driver.progress_score({**baseline, "success_rate_by_side": [1.0, 0.4]}) < score
    assert driver.progress_score({**baseline, "unsafe_rate": 0.5}) < score
    assert driver.progress_score({**baseline, "complete": False}) is None
    assert driver.progress_score({**baseline, "watch_fraction_by_server": [1.0, 1.0]}) == score


def test_watch_selection_bins_partial_progress_without_overriding_one_real_success():
    """A tiny watch tie-break tolerates sub-bin noise and stays below one balanced 64-episode success increment."""
    baseline = {
        "complete": True,
        "success_rate_by_side": [0.0, 0.0],
        "goal_reached_rate": 0.0,
        "unsafe_rate": 0.0,
        "objective": "watch_visible_ball",
        "watch_fraction_by_server": [0.551, 0.7],
    }
    score = driver.progress_score(baseline)
    assert driver.progress_score({**baseline, "watch_fraction_by_server": [0.559, 0.7]}) == score
    assert driver.progress_score({**baseline, "watch_fraction_by_server": [1.0, 1.0]}) < driver.progress_score(
        {**baseline, "watch_fraction_by_server": [0.0, 0.0], "success_rate_by_side": [1 / 64, 1 / 64]}
    )


def test_full_resume_accepts_complete_state_without_mutating_it(resumable_checkpoint):
    """Loading progress validates the stored contract while retaining its metadata."""
    metadata = resumable_checkpoint["infos"]["microduck_curriculum"]
    before = copy.deepcopy(metadata)
    actual = driver.resume_metadata(resumable_checkpoint, "microduck_volley_curriculum_v1")
    assert actual == before and metadata == before


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_profile",
        "missing_curriculum",
        "partial_curriculum",
        "missing_optimizer",
        "partial_optimizer",
        "missing_actor",
        "missing_critic",
        "missing_iteration",
    ],
)
def test_full_resume_rejects_legacy_or_partial_checkpoints(resumable_checkpoint, damage):
    """A tagged warm start or partially written progress must not masquerade as a full resume."""
    checkpoint = resumable_checkpoint
    metadata = checkpoint["infos"]["microduck_curriculum"]
    if damage == "wrong_profile":
        metadata["profile"] = "legacy_volley"
    elif damage == "missing_curriculum":
        del metadata["curriculum_state"]
    elif damage == "partial_curriculum":
        metadata["curriculum_state"] = {"stage": 3}
    elif damage == "missing_optimizer":
        del checkpoint["optimizer_state_dict"]
    elif damage == "partial_optimizer":
        checkpoint["optimizer_state_dict"] = None
    elif damage == "missing_actor":
        del checkpoint["actor_state_dict"]
    elif damage == "missing_critic":
        del checkpoint["critic_state_dict"]
    else:
        del checkpoint["iter"]
    with pytest.raises(ValueError):
        driver.resume_metadata(checkpoint, "microduck_volley_curriculum_v1")


def test_neighbor_tool_loading_works_from_an_unrelated_working_directory(tmp_path):
    """The documented standalone CLI must find neighboring tools independently of cwd or PYTHONPATH."""
    path = Path(driver.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "driver = runpy.run_path(sys.argv[1]); "
                "tool = driver['_tool']('prepare_microduck_curriculum_checkpoint'); "
                "print(tool.CURRICULUM_PROFILE)"
            ),
            str(path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == "microduck_volley_curriculum_v1"


def test_visual_full_resume_uses_visual_objectives_and_rejects_teacher_progress(resumable_checkpoint):
    """A v2 tag needs v2 lesson evidence; a teacher stage number alone cannot resume the visual task."""
    evaluator = driver._tool("evaluate_microduck_volley")
    contract = evaluator._curriculum_module(evaluator.VISION_PROFILE)
    prepare = driver._tool("prepare_microduck_vision_checkpoint")
    checkpoint = prepare.prepare_vision_checkpoint(resumable_checkpoint)
    # The genuine empty Adam state remains valid: conversion changes tensor widths, not parameter count.
    checkpoint["optimizer_state_dict"] = resumable_checkpoint["optimizer_state_dict"]
    checkpoint["iter"] = 8
    state = contract.VisionCurriculumState(stage=1)
    metadata = checkpoint["infos"]["microduck_curriculum"]
    metadata.update(curriculum_state=state.state_dict(), completed_iterations=8, certified=False)
    assert driver.resume_metadata(checkpoint, evaluator.VISION_PROFILE) is metadata
    metadata["curriculum_state"] = resumable_checkpoint["infos"]["microduck_curriculum"]["curriculum_state"]
    with pytest.raises(ValueError, match="incompatible curriculum"):
        driver.resume_metadata(checkpoint, evaluator.VISION_PROFILE)


def test_visual_certification_cannot_reuse_another_lessons_passing_test(resumable_checkpoint):
    """A final visual checkpoint must identify the actual final rally objective in its independent evidence."""
    evaluator = driver._tool("evaluate_microduck_volley")
    contract = evaluator._curriculum_module(evaluator.VISION_PROFILE)
    checkpoint = driver._tool("prepare_microduck_vision_checkpoint").prepare_vision_checkpoint(resumable_checkpoint)
    checkpoint["optimizer_state_dict"] = resumable_checkpoint["optimizer_state_dict"]
    checkpoint["iter"] = 8
    state = contract.VisionCurriculumState(stage=5)
    state.complete = True
    report = contract.VisionEvaluationReport(1000001, 5, 8, (64, 64), (60, 60), 0, objective="rally")
    metadata = checkpoint["infos"]["microduck_curriculum"]
    metadata.update(
        curriculum_state=state.state_dict(),
        completed_iterations=8,
        certified=True,
        certification={**asdict(report), "gate_passed": True},
    )
    assert driver.resume_metadata(checkpoint, evaluator.VISION_PROFILE) is metadata
    metadata["certification"]["objective"] = "watch_visible_ball"
    with pytest.raises(ValueError, match="certified checkpoint"):
        driver.resume_metadata(checkpoint, evaluator.VISION_PROFILE)


def test_legacy_visual_resume_requires_explicit_warm_start(resumable_checkpoint):
    """Old visual stage evidence and optimizer cannot continue under the new acquisition-first objectives."""
    resumable_checkpoint["infos"]["microduck_curriculum"]["profile"] = "microduck_volley_vision_v1"
    with pytest.raises(ValueError, match="fresh v2 warm start"):
        driver.resume_metadata(resumable_checkpoint, "microduck_volley_vision_v2")
