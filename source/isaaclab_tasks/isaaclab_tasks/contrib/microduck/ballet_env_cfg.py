# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One physical MicroDuck, full joint control, flat ground and contact sensing only."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG

from .ballet import OBSERVATION_WIDTH


@configclass
class MicroDuckBalletSceneCfg(InteractiveSceneCfg):
    """One robot and a static floor; shape filtering separates ground forces from self-contact."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot = MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*",
        history_length=1,
        update_period=0.005,
        filter_shape_prim_expr=["/World/ground(/.*)?"],
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class MicroDuckBalletEnvCfg(DirectRLEnvCfg):
    """Full-joint physical routine, with a measured curriculum during training."""

    decimation = 4
    episode_length_s = 20.0
    action_space = 14
    observation_space = OBSERVATION_WIDTH
    state_space = 0
    scene = MicroDuckBalletSceneCfg(num_envs=2048, env_spacing=2.0, lazy_sensor_update=True)
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=256,
                nconmax=96,
                iterations=100,
                ls_iterations=50,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2500000),
        ),
        visualizer_cfgs=VisualizerCfg(eye=(0.6, 0.6, 0.4), lookat=(0.0, 0.0, 0.14)),
    )
    support_foot: str = "left"
    turn_direction: float = 1.0
    start_stage: int = 0
    curriculum: bool = True

    def __post_init__(self):
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self):
        """Preview the complete pirouette and bow independently of training level mixture."""
        super().play_mode()
        self.scene.num_envs = 4
        self.start_stage = 4
        self.curriculum = False
