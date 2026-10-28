# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One policy and one explicit environment contract throughout volleyball lessons and rallies."""

from isaaclab.utils import configclass

from .ball_env_cfg import MicroDuckVolleyEnvCfg


@configclass
class MicroDuckVolleyCurriculumEnvCfg(MicroDuckVolleyEnvCfg):
    """Full joint control; only reset distributions and measured goals change by curriculum stage."""

    curriculum_stage: int = 0
    evaluation: bool = False
    episode_length_s = 8.0
    stage_separations: tuple[float, ...] = (0.35, 0.4, 0.45, 0.45, 0.45, 0.45)
    coach_partner_distance: float = 0.8
    target_radius: float = 0.065
    minimum_flight_time: float = 0.16
    minimum_crossing_height: float = 0.20
    release_delay: float = 0.02
    continuous_launch_shaping: bool = False
    live_partner: bool = False
    release_reward_scale: float = 0.2
    maximum_ball_hold: float = 0.08
    recovery_duration: float = 0.6
    recovery_hold = 0.25
    supported_force: float = 0.05
    recovered_height_ratio = 0.85
    recovered_max_planar_speed = 0.25
    maximum_home_excursion: float = 0.25
    collapse_height: float = 0.05
    collapse_duration: float = 0.12
    play_returns: int = 0
    # The new environment implements its own outcome objective. Legacy switches are inert here.
    enable_root_recovery = False
    enable_directed_returns = False

    def play_mode(self):
        """Preview the actual pair, allowing longer rallies than the eight-return qualification test."""
        super().play_mode()
        self.curriculum_stage = 5
        self.play_returns = 32
        self.episode_length_s = 30.0
