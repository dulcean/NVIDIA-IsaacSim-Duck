# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-only ball observations, with an independent privileged training critic."""

from isaaclab.utils import configclass

from .ball_env_cfg import MicroDuckVolleySceneCfg
from .head_camera_cfg import microduck_head_camera_cfg
from .volley_curriculum_env_cfg import MicroDuckVolleyCurriculumEnvCfg


@configclass
class MicroDuckVolleyVisionSceneCfg(MicroDuckVolleySceneCfg):
    """Native head-camera frames on both independently moving physical heads."""

    head_camera_a = microduck_head_camera_cfg("{ENV_REGEX_NS}/RobotA")
    head_camera_b = microduck_head_camera_cfg("{ENV_REGEX_NS}/RobotB")


@configclass
class MicroDuckVolleyVisionEnvCfg(MicroDuckVolleyCurriculumEnvCfg):
    """Keep ball truth in the critic; the actor receives two RGB-derived 73-value blocks."""

    observation_space = 146
    state_space = 142
    scene = MicroDuckVolleyVisionSceneCfg(num_envs=32, env_spacing=2.0, lazy_sensor_update=False)
    vision_max_age: float = 0.15
    head_camera_panels: bool = True
