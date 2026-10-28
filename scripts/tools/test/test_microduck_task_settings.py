# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint contact/reward contracts cannot silently change on a full resume."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/volley_task_settings.py"
)
_SPEC = importlib.util.spec_from_file_location("microduck_task_settings", _PATH)
settings = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(settings)


def test_contact_and_reward_settings_survive_serialization_and_guard_full_resumes():
    """Explicit warm-start overrides persist; omitting CLI flags later cannot restore old physics."""
    legacy = settings.resolve_task_settings({})
    changed = settings.resolve_task_settings({}, damping_ratio=0.5, continuous=True)
    saved = json.loads(json.dumps({"task_settings": changed}))
    assert settings.resolve_task_settings(saved, resume=True) == changed
    assert settings.resolve_task_settings({}, resume=True) == legacy
    assert legacy["ball_solref"] == [0.02, 0.15] and not legacy["continuous_launch_shaping"]
    assert changed["ball_solref"] == [0.02, 0.5] and changed["continuous_launch_shaping"]
    with pytest.raises(ValueError, match="fresh objective"):
        settings.resolve_task_settings(saved, damping_ratio=0.15, resume=True)
    with pytest.raises(ValueError, match="fresh objective"):
        settings.resolve_task_settings({}, continuous=True, resume=True)
    assert saved["task_settings"] == changed  # Overrides must not mutate the source checkpoint.
    saved["task_settings"]["ball_solref"][1] = float("nan")
    with pytest.raises(ValueError, match="solref"):
        settings.resolve_task_settings(saved)


def test_live_objective_migration_is_explicit_and_stays_live_after_resume():
    """New physical/reward settings cannot be dropped when CLI options are omitted later."""
    source = {"task_settings": settings.resolve_task_settings({}, continuous=True)}
    migrated = settings.resolve_task_settings(source, live_pass=True)
    saved = json.loads(json.dumps({"task_settings": migrated}))
    assert settings.resolve_task_settings(saved, resume=True, live_pass=True) == migrated
    assert settings.resolve_task_settings(saved) == migrated
    assert migrated["live_pass"] and not migrated["continuous_launch_shaping"]
    assert migrated["release_reward_scale"] > 0.2
    assert source["task_settings"]["continuous_launch_shaping"]
    with pytest.raises(ValueError, match="fresh objective"):
        settings.resolve_task_settings(source, live_pass=True, resume=True)
    with pytest.raises(ValueError, match="legacy release"):
        settings.resolve_task_settings(saved, continuous=True)
