# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small nonvisual actor with conservative fine tuning of the walking backbone."""

from isaaclab.utils import configclass

from ..ballet import REWARD_DISCOUNT
from .ball_ppo_cfg import MicroDuckHeadingPPORunnerCfg


@configclass
class MicroDuckBalletPPORunnerCfg(MicroDuckHeadingPPORunnerCfg):
    """Nonvisual single-robot PPO with a discount shared by the rotation potential."""

    experiment_name = "microduck_ballet"
    max_iterations = 20000
    save_interval = 250
    actor = MicroDuckHeadingPPORunnerCfg().actor.replace(
        distribution_cfg=MicroDuckHeadingPPORunnerCfg().actor.distribution_cfg.replace(init_std=0.25),
    )
    algorithm = MicroDuckHeadingPPORunnerCfg().algorithm.replace(
        learning_rate=0.0003,
        gamma=REWARD_DISCOUNT,
        desired_kl=0.01,
    )
