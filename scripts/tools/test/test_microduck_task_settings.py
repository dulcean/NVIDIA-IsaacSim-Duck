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
