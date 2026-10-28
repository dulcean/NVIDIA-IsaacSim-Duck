# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Noncolliding optical reference badges carried by MicroDuck's physical head."""

from .partner_vision import PARTNER_CYAN_POSITION, PARTNER_MAGENTA_POSITION, PARTNER_MARKER_RADIUS


def microduck_partner_marker_cfgs(robot_prim_path: str):
    """Create two colored sphere children, without mass, colliders, or independent bodies.

    The pair's known rigid baseline extrapolates to the forehead strike point.
    Both spheres are behind the observing head's own optical plane, so the
    camera sees its partner's badges without detecting its own pair. Their
    parent articulation owns Newton cloning; these authoring-only assets must
    not be imported again as independent static Newton clone roots.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from .head_camera_cfg import HEAD_BODY_PRIM_PATH, HEAD_CAMERA_POSITION

    markers = {}
    for name, position, color in (
        ("cyan", PARTNER_CYAN_POSITION, (0.0, 0.8, 1.0)),
        ("magenta", PARTNER_MAGENTA_POSITION, (1.0, 0.0, 0.8)),
    ):
        if position[2] - PARTNER_MARKER_RADIUS <= HEAD_CAMERA_POSITION[2]:
            raise ValueError("A partner badge must stay entirely behind its own head camera's optical plane.")
        markers[name] = AssetBaseCfg(
            prim_path=f"{robot_prim_path.rstrip('/')}/{HEAD_BODY_PRIM_PATH}/PartnerBadge_{name}",
            spawn=sim_utils.SphereCfg(
                radius=PARTNER_MARKER_RADIUS,
                collision_props=None,
                rigid_props=None,
                mass_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            cloning_contexts=(),
        )
    return markers
