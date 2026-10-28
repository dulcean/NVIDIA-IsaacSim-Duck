# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-only ball observations, with an independent privileged training critic."""

from isaaclab.utils import configclass

from .ball_env_cfg import MicroDuckVolleySceneCfg
from .head_camera_cfg import microduck_head_camera_cfg
from .vision_curriculum import VISION_STAGES
from .volley_curriculum_env_cfg import MicroDuckVolleyCurriculumEnvCfg


@configclass
class MicroDuckVolleyVisionSceneCfg(MicroDuckVolleySceneCfg):
    """Native head-camera frames on both independently moving physical heads."""

    head_camera_a = microduck_head_camera_cfg("{ENV_REGEX_NS}/RobotA")
    head_camera_b = microduck_head_camera_cfg("{ENV_REGEX_NS}/RobotB")


@configclass
class MicroDuckVolleyVisionEnvCfg(MicroDuckVolleyCurriculumEnvCfg):
    """Visual acquisition, receiving and rally lessons with two 76-value actor blocks."""

    observation_space = 152
    state_space = 142
    scene = MicroDuckVolleyVisionSceneCfg(num_envs=32, env_spacing=2.0, lazy_sensor_update=True)
    vision_max_age: float = 0.15
    head_camera_panels: bool = True
    head_camera_panel_fps: float = 30.0
    curriculum_profile: str = "microduck_volley_vision_v2"
    stage_serve_heights: tuple = tuple((spec.source_height_min, spec.source_height_max) for spec in VISION_STAGES)
    stage_source_distances: tuple = tuple(spec.source_distance for spec in VISION_STAGES)
    watch_duration: float = 0.30
    gaze_alignment_rate: float = 0.10
    gaze_visibility_rate: float = 0.10
    gaze_strike_grace: float = 0.08
    final_min_gaze_visibility: float = 0.60
