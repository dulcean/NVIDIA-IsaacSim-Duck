# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real partner-head optics, unchanged physical joints, and visual ballistic references."""

from isaaclab.utils import configclass

from .live_pass_curriculum import LIVE_PASS_PROFILE, VISION_STAGES
from .partner_marker_cfg import microduck_partner_marker_cfgs
from .volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg, MicroDuckVolleyVisionSceneCfg


def _live_ball_cfg():
    cfg = MicroDuckVolleyVisionSceneCfg().ball
    cfg.spawn.collision_props[1].solref = (0.02, 0.5)
    return cfg


@configclass
class MicroDuckLivePassSceneCfg(MicroDuckVolleyVisionSceneCfg):
    """Colored optical landmarks move rigidly with the two physical heads."""

    ball = _live_ball_cfg()
    partner_a_cyan = microduck_partner_marker_cfgs("{ENV_REGEX_NS}/RobotA")["cyan"]
    partner_a_magenta = microduck_partner_marker_cfgs("{ENV_REGEX_NS}/RobotA")["magenta"]
    partner_b_cyan = microduck_partner_marker_cfgs("{ENV_REGEX_NS}/RobotB")["cyan"]
    partner_b_magenta = microduck_partner_marker_cfgs("{ENV_REGEX_NS}/RobotB")["magenta"]


@configclass
class MicroDuckLivePassEnvCfg(MicroDuckVolleyVisionEnvCfg):
    """Two 89-value actor blocks and the separate 142-value physical training critic."""

    observation_space = 178
    live_partner = True
    release_reward_scale = 2.0
    scene = MicroDuckLivePassSceneCfg(num_envs=128, env_spacing=2.0, lazy_sensor_update=True)
    curriculum_profile = LIVE_PASS_PROFILE
    stage_separations = (0.45, 0.45, 0.45, 0.5, 0.55, 0.6)
    stage_serve_heights = tuple((spec.source_height_min, spec.source_height_max) for spec in VISION_STAGES)
    stage_source_distances = tuple(spec.source_distance for spec in VISION_STAGES)
    partner_max_age: float = 0.75
    final_min_partner_visibility: float = 0.1
    root_position_jitter: float = 0.04
    root_lateral_jitter: float = 0.025
