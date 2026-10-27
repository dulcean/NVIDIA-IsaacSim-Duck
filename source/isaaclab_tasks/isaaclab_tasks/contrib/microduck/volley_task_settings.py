# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persist the authored ball contact and release-reward settings across tool entry points."""

import math


def resolve_task_settings(metadata, *, damping_ratio=None, continuous=False, resume=False):
    """Read legacy defaults or a validated checkpoint contract; reject changed full resumes."""
    settings = metadata.get(
        "task_settings", {"version": 1, "ball_solref": [0.02, 0.15], "continuous_launch_shaping": False}
    )
    if (
        not isinstance(settings, dict)
        or set(settings) != {"version", "ball_solref", "continuous_launch_shaping"}
        or type(settings["version"]) is not int
        or settings["version"] != 1
        or type(settings["continuous_launch_shaping"]) is not bool
    ):
        raise ValueError("Invalid saved volleyball task settings.")
    solref = settings["ball_solref"]
    if (
        not isinstance(solref, (tuple, list))
        or len(solref) != 2
        or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in solref)
        or solref[1] > 1
    ):
        raise ValueError("Invalid saved ball solref.")
    saved = {**settings, "ball_solref": list(solref)}
    result = {**saved, "ball_solref": list(solref)}
    if damping_ratio is not None:
        if not math.isfinite(damping_ratio) or not 0 < damping_ratio <= 1:
            raise ValueError("Ball damping ratio must be finite and in (0, 1].")
        result["ball_solref"][1] = damping_ratio
    result["continuous_launch_shaping"] |= continuous
    if resume and result != saved:
        raise ValueError("Changed contacts or reward require --checkpoint for a fresh objective, not --resume.")
    return result


def apply_task_settings(cfg, settings):
    """Restore the same authored ball contact and reward in training, preview, evaluation and tracing."""
    settings = resolve_task_settings({"task_settings": settings})
    cfg.scene.ball.spawn.collision_props[1].solref = tuple(settings["ball_solref"])
    cfg.continuous_launch_shaping = settings["continuous_launch_shaping"]
