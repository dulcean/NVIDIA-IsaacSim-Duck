# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative fine-tuning of all MicroDuck joints for ball tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MicroDuckHeadingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Single-robot heading; the value function starts afresh on the ball observations."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 250
    experiment_name = "microduck_heading"
    init_at_random_ep_len = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.0001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckVolleyPPORunnerCfg(MicroDuckHeadingPPORunnerCfg):
    """Two initially independent copies, with trainable connections for cooperation."""

    experiment_name = "microduck_volley"
    actor = MicroDuckHeadingPPORunnerCfg().actor.replace(hidden_dims=[1024, 512, 256])


@configclass
class MicroDuckVolleyCurriculumPPORunnerCfg(MicroDuckVolleyPPORunnerCfg):
    """The same trainable actor across receive/recover lessons and unassisted pair stages."""

    experiment_name = "microduck_volley_curriculum"
    save_interval = 100
    algorithm = MicroDuckVolleyPPORunnerCfg().algorithm.replace(
        gamma=0.995,
        learning_rate=0.00005,
        desired_kl=0.003,
    )


@configclass
class MicroDuckVolleyVisionPPORunnerCfg(MicroDuckVolleyCurriculumPPORunnerCfg):
    """Train an RGB-tracking actor and an independent simulator-state value function."""

    experiment_name = "microduck_volley_vision"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
