# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Head-mounted RGB camera using MicroDuck's published alpha-family optics.

The USD ``head_camera`` is a guide sphere, not an image sensor. Its exported MJCF camera
also faces backwards (upstream issue 66), so neither can be used as a rendered camera as-is.
The rigid head frame has optical forward ``-Z`` and upright ``+X``. We render the native,
unrotated sensor frame and rotate its image clockwise once only for an upright display.

Sources, inspected 2026-10-26:

* https://github.com/pollen-robotics/microduck/blob/main/robotd-params/src/lib.rs
  (``CameraIntrinsics::alpha`` and ``Quality``).
* https://github.com/pollen-robotics/microduck/blob/main/mediad/src/pipeline.rs
  (``Rotation::Cw90`` and raw-frame capture).
* https://github.com/pollen-robotics/microduck_rl/issues/66
  (the corrected upright OpenGL frame).

The calibration is a family reference solved on the *graphite* alpha unit, not a measured
calibration of the user's robot. Capture is 30 Hz, independent of the 50 Hz controller.
160 x 90 is a simulation downsample of the published 1280 x 720 frame; it keeps its aspect
ratio, principal point and lens distortion. Exposure, motion blur and hardware latency are
not calibrated here.
"""

from isaaclab_newton.renderers import NewtonWarpRendererCfg

from isaaclab.sensors import CameraCfg
from isaaclab.sim import OpenCvPinholeDistortionCfg, PinholeCameraCfg

HEAD_BODY_PRIM_PATH = "Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft"
"""Rigid head path relative to the MicroDuck articulation prim in the supplied USD."""

HEAD_CAMERA_POSITION = (0.0155, -0.0000913778, -0.083)
"""Render origin in the rigid head frame [m].

The authored guide centre is at z=-0.0733, inside the opaque lens mesh, whose front extends
to approximately -0.0818. This origin clears the mesh by 1.2 mm. It is a render-origin
correction for this asset, not a measurement of the physical entrance pupil.
"""

HEAD_CAMERA_ROTATION_XYZW = (0.0, 0.0, 1.0, 0.0)
"""Native sensor's OpenGL frame in the head: right -X, up -Y, optical forward -Z.

Rotating this raw image clockwise 90 degrees gives right -Y and up +X, matching the
corrected upright OpenGL quaternion XYZW=(0, 0, -sqrt(0.5), sqrt(0.5)) from issue 66.
The same clockwise display turn is specified by the hardware's media pipeline.
"""

RAW_IMAGE_CLOCKWISE_ROTATIONS = 1
"""Number of clockwise quarter turns for display; detector coordinates remain raw."""

HEAD_CAMERA_CALIBRATION_SIZE = (1280, 720)
HEAD_CAMERA_INTRINSICS = (1061.8060025020175, 1062.193713957031, 596.7776848217006, 474.52348043048636)
"""Published alpha-family native ``fx, fy, cx, cy`` [px], before display rotation."""

HEAD_CAMERA_DISTORTION = (
    -0.34695388691888,
    0.14615866153945595,
    -0.0007449157712424215,
    -0.0022167170528743837,
    -0.016437266141048814,
)
"""OpenCV Brown-Conrady ``k1, k2, p1, p2, k3`` from the same family calibration."""


def head_camera_intrinsics(width: int = 160, height: int = 90) -> tuple[float, float, float, float]:
    """Scale native ``fx, fy, cx, cy`` without cropping or changing the aspect ratio."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width * HEAD_CAMERA_CALIBRATION_SIZE[1] != height * HEAD_CAMERA_CALIBRATION_SIZE[0]
    ):
        raise ValueError("MicroDuck raw camera dimensions must be positive integers with a 16:9 aspect ratio.")
    scale = width / HEAD_CAMERA_CALIBRATION_SIZE[0]
    return tuple(value * scale for value in HEAD_CAMERA_INTRINSICS)


def microduck_head_camera_cfg(
    robot_prim_path: str,
    *,
    width: int = 160,
    height: int = 90,
    update_period: float = 1.0 / 30.0,
    apply_lens_distortion: bool = True,
) -> CameraCfg:
    """Attach a Newton RGB sensor to the physical head, using native sensor coordinates.

    Args:
        robot_prim_path: Articulation prim, e.g. ``{ENV_REGEX_NS}/RobotA``.
        width: Raw image width; the default is an eightfold linear downsample.
        height: Raw image height. Must preserve the native 16:9 aspect ratio.
        update_period: Capture period in seconds, independent of policy steps.
        apply_lens_distortion: Apply the published Brown-Conrady model. When disabled,
            the same intrinsic matrix renders a rectified pinhole reference image.

    Returns:
        Camera configuration whose ``rgb`` output is ``(N, height, width, 3)`` uint8.
        ``data.quat_w_ros`` describes the unrotated output's OpenCV frame.
    """
    fx, fy, cx, cy = head_camera_intrinsics(width, height)
    spawn = PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        width=width,
        height=height,
        clipping_range=(0.001, 3.0),
    )
    k1, k2, p1, p2, k3 = HEAD_CAMERA_DISTORTION
    spawn.distortion = OpenCvPinholeDistortionCfg(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_size=(width, height),
        apply_lens_distortion=apply_lens_distortion,
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
    )
    return CameraCfg(
        prim_path=f"{robot_prim_path.rstrip('/')}/{HEAD_BODY_PRIM_PATH}/HeadCamera",
        width=width,
        height=height,
        update_period=update_period,
        data_types=["rgb"],
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=spawn,
        offset=CameraCfg.OffsetCfg(pos=HEAD_CAMERA_POSITION, rot=HEAD_CAMERA_ROTATION_XYZW, convention="opengl"),
        update_latest_camera_pose=True,
    )
