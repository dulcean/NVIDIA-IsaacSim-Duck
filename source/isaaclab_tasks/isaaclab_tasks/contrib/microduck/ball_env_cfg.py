# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton/MJWarp whole-body heading and cooperative volleyball for MicroDuck."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG


@configclass
class MicroDuckHeadingSceneCfg(InteractiveSceneCfg):
    """A small robot and a light, compliant ball on flat ground."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot_a = MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotA",
        init_state=MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.init_state.replace(pos=(-0.175, 0.0, 0.125)),
    )
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            activate_contact_sensors=True,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.005),
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(),
                MujocoCollisionCfg(priority=1, solref=(0.02, 0.15)),
            ],
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.4, dynamic_friction=0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    # Many sensing robot bodies, one filtered counterpart: contacts with the ball only.
    ball_contacts_a = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotA/Geometry/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
        history_length=4,
        update_period=0.005,
        track_contact_points=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class MicroDuckVolleySceneCfg(MicroDuckHeadingSceneCfg):
    """Two robots facing one another, without a net or wall."""

    robot_b = MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotB",
        init_state=MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.init_state.replace(
            pos=(0.175, 0.0, 0.125), rot=(0.0, 0.0, 1.0, 0.0)
        ),
    )
    ball_contacts_b = MicroDuckHeadingSceneCfg().ball_contacts_a.replace(prim_path="{ENV_REGEX_NS}/RobotB/Geometry/.*")
    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotA/Geometry/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RobotB/Geometry/.*"],
        history_length=1,
        update_period=0.005,
    )


@configclass
class MicroDuckHeadingEnvCfg(DirectRLEnvCfg):
    """One robot learns to head a convenient serve towards a virtual receiver."""

    decimation = 4
    episode_length_s = 6.0
    action_space = 14
    observation_space = 71
    state_space = 0
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
        visualizer_cfgs=VisualizerCfg(eye=(0.9, 0.9, 0.65), lookat=(0.0, 0.0, 0.2)),
    )
    scene = MicroDuckHeadingSceneCfg(num_envs=256, env_spacing=2.0, lazy_sensor_update=False)
    num_players: int = 1
    separation: float = 0.35
    ball_radius: float = 0.025
    serve_height: float = 0.25
    serve_spread: float = 0.01
    serve_down_speed: float = 0.2
    receiver_radius: float = 0.09
    court_half_width: float = 0.35
    court_half_length: float = 0.65
    # These are rigid-body names, not visual meshes. Verify the startup report on the GPU machine.
    head_body_pattern: str = "jaw_soft"
    head_reference_body: str = "jaw_soft"
    # Measured in the full-collision USD's rigid jaw_soft frame: +X is up, -Z is camera-forward.
    camera_forward_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    camera_offset: tuple[float, float, float] = (0.0155, -0.0000914, -0.05548)
    strike_offset: tuple[float, float, float] = (0.035, 0.0, -0.07)
    strike_patch_min: tuple[float, float, float] = (0.027, -0.04, -0.10)
    strike_patch_max: tuple[float, float, float] = (0.045, 0.04, -0.05)
    min_camera_cosine: float = 0.5
    min_camera_up: float = 0.25
    contact_threshold: float = 0.001
    robot_contact_threshold: float = 0.01
    # Minimum predicted flight horizon for the aiming bonus, not a ban on fast replies.
    min_transfer_time: float = 0.08
    approach_reward_scale: float = 0.2
    gaze_reward_scale: float = 0.2
    upright_reward_scale: float = 0.1
    torso_tilt_cost: float = 0.0
    torso_tumbling_cost: float = 0.0
    action_rate_cost: float = 0.01
    joint_limit_cost: float = 0.2
    fall_cost: float = 2.0
    hit_reward: float = 1.0
    aim_reward: float = 2.0
    transfer_reward: float = 5.0
    defer_strike_rewards: bool = False
    immediate_hit_fraction: float = 0.2
    recovery_delay: float = 0.12
    recovery_hold: float = 0.10
    recovery_min_upright: float = 0.9396926  # cos(20 degrees), measured on the torso, not the head.
    recovery_max_tumbling: float = 1.5  # Combined roll/pitch rate in rad/s; yaw is unrestricted.
    recovery_min_height_ratio: float = 0.6
    recovery_max_height_offset: float = 0.03
    recovery_max_vertical_speed: float = 0.25
    post_rally_duration: float = 0.0
    post_rally_hold: float = 0.2
    post_rally_reward: float = 0.5
    miss_cost: float = 0.5

    def __post_init__(self):
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class MicroDuckVolleyEnvCfg(MicroDuckHeadingEnvCfg):
    """Centralized cooperative RSL-RL policy; every joint of both robots is trainable."""

    scene = MicroDuckVolleySceneCfg(num_envs=256, env_spacing=2.0, lazy_sensor_update=False)
    num_players = 2
    action_space = 28
    observation_space = 142
    episode_length_s = 12.0
    torso_tilt_cost = 10.0
    torso_tumbling_cost = 0.05
    defer_strike_rewards = True
    post_rally_duration = 0.6
