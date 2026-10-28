# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate both physical head RGB sensors before spending time on visual-policy training.

The probe places an actual sphere in front of each receiving head at reset and
then lets the complete robot and ball physics run. A second probe puts it outside
the receiving camera's view to reject false positive tracks from orange robot parts.
Each curriculum stage must also acquire its actual reset serve on the first
positive-time camera exposure, without replacing the serve with a test fixture.
Ball truth is compared with completed RGB estimates only; it never initializes a
tracker or an actor observation. These checks establish sensor operation, not skill.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def check_vision_sensors(env, *, steps_per_probe: int = 16, tolerance: float = 0.04) -> dict:
    """Check actual RGB localization and background rejection in an existing vision environment.

    Args:
        env: An existing MicroDuck vision environment or its RSL-RL wrapper. At
            least two worlds are required: receiver A in world 0 and B in world 1.
        steps_per_probe: Number of control steps for the visible-ball probe. The
            background probe lasts at most 0.2 seconds to avoid episode resets.
        tolerance: Maximum mean localization error [m] for accepted RGB measurements.

    Returns:
        JSON-compatible diagnostic report with a ``passed`` boolean. Failure of a
        physical check is reported rather than raising a training decision here.
        The original curriculum stage and evaluation flag are restored, and all
        worlds are reset through the supplied wrapper before returning. Zero
        actions hold normal actuator targets; no joint or body is frozen.
    """
    import torch

    from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_mul

    if type(steps_per_probe) is not int or steps_per_probe < 2:
        raise ValueError("Vision preflight needs at least two control steps per probe.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Vision localization tolerance must be finite and positive.")
    raw = env.unwrapped
    report = {
        "passed": False,
        "tolerance_m": tolerance,
        "minimum_rgb_measurements_per_side": 2,
        "minimum_captured_frames_per_side": 2,
        "probes": [],
        "serve_reset_probes": [],
        "ground_truth_use": "Reset placement and validation errors only; never passed to tracker or actor.",
        "scope": "Real RGB sensor/runtime check with controlled initial conditions; no volleyball skill assessed.",
    }
    if raw.num_envs < 2:
        report["error"] = "Both receivers require at least two physical worlds."
        return report
    if len(raw.head_cameras) != 2:
        report["error"] = "Two physical head cameras are required."
        return report

    started = time.perf_counter()
    original_stage, original_evaluation = raw.stage, raw.evaluation
    selected = torch.tensor((0, 1), dtype=torch.long, device=raw.device)
    actions = torch.zeros(raw.num_envs, raw.single_action_space.shape[0], device=raw.device)
    live_partner = getattr(raw.cfg, "live_partner", False)
    # The paired layout has a physical head ~.235m ahead at its closest reset.
    # Keep the radius-.025m probe clear of it and inside the detector's size range.
    visible_distance = 0.18 if live_partner else 0.25
    body_ids = [raw.scene[f"robot_{side}"].find_bodies(raw.cfg.head_reference_body)[0][0] for side in ("a", "b")]

    def camera_pose(player):
        """Use measured physical head FK, not a potentially stale cached exposure pose."""
        robot = raw.scene[f"robot_{('a', 'b')[player]}"]
        head = body_ids[player]
        position = robot.data.body_link_pos_w.torch[player, head]
        orientation = robot.data.body_link_quat_w.torch[player, head]
        offset = raw.head_cameras[player].cfg.offset
        offset_position = position.new_tensor(offset.pos)
        offset_orientation = orientation.new_tensor(offset.rot).unsqueeze(0)
        optical = convert_camera_frame_orientation_convention(
            offset_orientation, origin=offset.convention, target="ros"
        )[0]
        position = position + quat_apply(orientation.unsqueeze(0), offset_position.unsqueeze(0))[0]
        orientation = quat_mul(orientation.unsqueeze(0), optical.unsqueeze(0))[0]
        forward = quat_apply(orientation.unsqueeze(0), position.new_tensor(((0.0, 0.0, 1.0),)))[0]
        return position, forward

    def tracks_cleared():
        return all(
            not bool(track.valid[selected].any()) and (track.seen is None or not bool(track.seen[selected].any()))
            for track in raw.vision_tracks
        )

    # Rollout paths rebind tracker tensors. Keep their in-place resets inside
    # inference mode too, including when this check follows another rollout.
    with torch.inference_mode():
        try:
            # The coach layout keeps a second robot clear of the localization
            # target. Both receiver identities still use the real complete scene.
            raw.set_stage(0)
            raw.set_evaluation(True)
            for visible in (True, False):
                env.reset()
                identities_ok = raw._server[selected].tolist() == [0, 1]
                initially_clear = tracks_cleared()
                probe = {
                    "name": "visible_sphere" if visible else "background_rejection",
                    "passed": False,
                    "balanced_receivers": identities_ok,
                    "tracks_clear_after_reset": initially_clear,
                    "completed_steps": 0,
                    "premature_reset": False,
                    "sphere_distance_m": visible_distance if visible else None,
                    "sides": [],
                }
                report["probes"].append(probe)
                positions = []
                geometry_ok = True
                for player in range(2):
                    camera_position, forward = camera_pose(player)
                    if visible:
                        point = camera_position + visible_distance * forward
                    else:
                        # A lateral airborne point remains outside the narrow
                        # calibrated field of view even if the head starts
                        # looking up. "Behind" a raised camera could be under
                        # the floor. A court miss is harmless during this short
                        # background probe's normal recovery interval.
                        lateral = torch.stack((-forward[1], forward[0], forward.new_zeros(())))
                        if lateral.norm() < 1e-6:
                            lateral = forward.new_tensor((1.0, 0.0, 0.0))
                        lateral = lateral / lateral.norm()
                        point = camera_position + 2.0 * lateral
                        point[2] = torch.maximum(point[2], raw.scene.env_origins[player, 2] + 0.35)
                    height = point[2] - raw.scene.env_origins[player, 2]
                    in_front = torch.dot(point - camera_position, forward)
                    geometry_ok &= bool(torch.isfinite(point).all() and height > raw.cfg.ball_radius + 0.1)
                    angle_cosine = in_front / (point - camera_position).norm().clamp_min(1e-6)
                    geometry_ok &= bool(in_front > visible_distance - 0.05 if visible else angle_cosine.abs() < 0.25)
                    positions.append(point)
                probe["valid_reset_geometry"] = geometry_ok
                if not (geometry_ok and initially_clear and identities_ok):
                    probe["error"] = (
                        "Reset did not establish clear tracks, balanced receivers and valid optical geometry."
                    )
                    continue

                # All subsequent motion is unmodified simulation: no per-step
                # ball positioning and no privileged tracker initialization.
                pose = raw._ball.data.root_link_pose_w.torch[selected].clone()
                pose[:, :3] = torch.stack(positions)
                raw._ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=selected)
                velocity = torch.zeros(2, 6, device=raw.device)
                if visible and live_partner:
                    # A small physical toss keeps the closer sphere in frame
                    # through two exposures; gravity and all collisions remain active.
                    velocity[:, 2] = 9.81 * max(raw._camera_capture_steps) * raw.cfg.sim.dt
                raw._ball.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=selected)
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                raw.scene.update(dt=0.0)
                previous_frames = [int(camera.frame.torch[player]) for player, camera in enumerate(raw.head_cameras)]
                capture_counts, false_tracks = [0, 0], [0, 0]
                accepted_counts = [0, 0]
                errors = [[], []]
                finite = True
                probe_started = time.perf_counter()
                count = steps_per_probe if visible else min(steps_per_probe, max(2, int(0.2 / raw.step_dt)))
                for step in range(count):
                    result = env.step(actions)
                    observations, reward = result[:2]
                    done = result[2] | result[3] if len(result) == 5 else result[2]
                    finite &= bool(torch.isfinite(reward).all())
                    if hasattr(observations, "values"):
                        finite &= all(bool(torch.isfinite(value).all()) for value in observations.values())
                    probe["completed_steps"] = step + 1
                    if done[selected].any():
                        probe["premature_reset"] = True
                        break
                    tracks = raw.vision_tracks
                    for player, camera in enumerate(raw.head_cameras):
                        track = tracks[player]
                        frame = int(camera.frame.torch[player])
                        fresh = frame > previous_frames[player]
                        capture_counts[player] += max(0, frame - previous_frames[player])
                        previous_frames[player] = frame
                        valid = bool(track.valid[player])
                        if not visible:
                            false_tracks[player] += int(valid)
                        # Captures can occur between control boundaries. A
                        # recent valid track identifies an accepted exposure in
                        # this step even if the final substep predicted it forward.
                        recent = bool(track.age[player] < raw.step_dt - 1e-6)
                        accepted = fresh and valid and recent
                        accepted_counts[player] += int(accepted)
                        if visible and accepted:
                            error = (track.position_w[player] - raw._ball.data.root_link_pos_w.torch[player]).norm()
                            if math.isfinite(float(error)):
                                errors[player].append(float(error))
                            else:
                                finite = False
                    # Optical validation is complete after two real exposures
                    # per selected camera. Do not turn it into a long, untrained
                    # zero-action balance test after enough evidence exists.
                    enough_frames = all(captures >= 2 for captures in capture_counts)
                    enough_measurements = all(len(values) >= 2 for values in errors) if visible else True
                    if finite and enough_frames and enough_measurements:
                        break
                probe["wall_time_s"] = time.perf_counter() - probe_started
                probe["simulated_time_s"] = probe["completed_steps"] * raw.step_dt
                probe["finite"] = finite

                for player, camera in enumerate(raw.head_cameras):
                    rgb = raw.head_camera_images[player][player]
                    nonconstant = bool(rgb.float().std() > 5.0)
                    # Report actual OpenCV K read back by the camera, rather
                    # than the ordinary USD pinhole approximation in warnings.
                    # Camera has no public calibration-only getter: .data can
                    # submit another render. Read its existing metadata cache
                    # here so diagnostics never cause an off-clock exposure.
                    actual_k = camera._data.intrinsic_matrices.torch[player]
                    calibration = camera.cfg.spawn.distortion
                    expected_k = actual_k.new_tensor(
                        ((calibration.fx, 0.0, calibration.cx), (0.0, calibration.fy, calibration.cy), (0.0, 0.0, 1.0))
                    )
                    calibrated = bool(torch.allclose(actual_k, expected_k, atol=1e-4, rtol=0))
                    expected_distortion = tuple(
                        getattr(calibration, name) if calibration.apply_lens_distortion else 0.0
                        for name in ("k1", "k2", "p1", "p2", "k3")
                    )
                    tracker_distortion = raw._visual_trackers[player].distortion
                    same_distortion = bool(
                        torch.allclose(
                            tracker_distortion, tracker_distortion.new_tensor(expected_distortion), atol=1e-7, rtol=0
                        )
                    )
                    mean_error = sum(errors[player]) / len(errors[player]) if errors[player] else None
                    side_passed = capture_counts[player] >= 2 and nonconstant and calibrated and same_distortion
                    if visible:
                        side_passed &= len(errors[player]) >= 2 and mean_error <= tolerance
                    else:
                        side_passed &= not false_tracks[player]
                    probe["sides"].append(
                        {
                            "player": ("a", "b")[player],
                            "world": player,
                            "passed": bool(side_passed),
                            "captured_frames": capture_counts[player],
                            "accepted_rgb_measurements": accepted_counts[player],
                            "false_track_control_steps": false_tracks[player],
                            "mean_position_error_m": mean_error,
                            "maximum_position_error_m": max(errors[player]) if errors[player] else None,
                            "nonconstant_rgb": nonconstant,
                            "runtime_intrinsics_match_opencv": calibrated,
                            "tracker_distortion_matches_camera": same_distortion,
                            "opencv_distortion": list(expected_distortion),
                            "runtime_intrinsic_matrix": [
                                [value if math.isfinite(value) else None for value in row] for row in actual_k.tolist()
                            ],
                        }
                    )
                probe["passed"] = (
                    finite and not probe["premature_reset"] and all(side["passed"] for side in probe["sides"])
                )
            # These are the production reset distributions, not another
            # easy camera-forward sphere fixture. Their first frames must
            # contain usable pixels before a policy has learned to search.
            for stage, stage_spec in enumerate(raw.stages):
                raw.set_stage(stage)
                env.reset()
                stage_probe = {
                    "stage": stage,
                    "name": stage_spec.name,
                    "objective": stage_spec.objective,
                    "passed": False,
                    "balanced_receivers": raw._server[selected].tolist() == [0, 1],
                    "tracks_clear_after_reset": tracks_cleared(),
                    "completed_steps": 0,
                    "premature_reset": False,
                    "finite": True,
                    "sides": [
                        {"player": player, "world": world, "passed": False, "first_frame": None}
                        for world, player in enumerate(("a", "b"))
                    ],
                }
                report["serve_reset_probes"].append(stage_probe)
                initial_frames = [int(camera.frame.torch[player]) for player, camera in enumerate(raw.head_cameras)]
                stage_started = time.perf_counter()
                for step in range(4):
                    result = env.step(actions)
                    observations, reward = result[:2]
                    done = result[2] | result[3] if len(result) == 5 else result[2]
                    stage_probe["completed_steps"] = step + 1
                    stage_probe["finite"] &= bool(torch.isfinite(reward).all())
                    if hasattr(observations, "values"):
                        stage_probe["finite"] &= all(
                            bool(torch.isfinite(value).all()) for value in observations.values()
                        )
                    if done[selected].any():
                        stage_probe["premature_reset"] = True
                        break
                    tracks = raw.vision_tracks
                    for player, camera in enumerate(raw.head_cameras):
                        side = stage_probe["sides"][player]
                        frame = int(camera.frame.torch[player])
                        if side["first_frame"] is not None or frame <= initial_frames[player]:
                            continue
                        track = tracks[player]
                        # Reset cleared all prediction/history. Therefore a
                        # valid recent estimate at the first physical exposure
                        # can only originate in that exposure's actual pixels.
                        accepted = bool(track.valid[player] & (track.age[player] < raw.step_dt + 1e-6))
                        camera_position, _ = camera_pose(player)
                        ball = raw._ball.data.root_link_pos_w.torch[player]
                        distance = float((ball - camera_position).norm())
                        error = float((track.position_w[player] - ball).norm()) if accepted else None
                        allowed_error = max(0.05, 0.15 * distance) if math.isfinite(distance) else None
                        finite = bool(torch.isfinite(ball).all() & torch.isfinite(camera_position).all())
                        finite &= math.isfinite(distance) and (error is None or math.isfinite(error))
                        side.update(
                            first_frame=frame,
                            observed_at_control_step=step + 1,
                            accepted_rgb_measurement=accepted,
                            position_error_m=error if error is not None and math.isfinite(error) else None,
                            camera_range_m=distance if math.isfinite(distance) else None,
                            allowed_error_m=allowed_error,
                            finite=finite,
                            passed=bool(accepted and finite and error <= allowed_error),
                        )
                    if all(side["first_frame"] is not None for side in stage_probe["sides"]):
                        break
                stage_probe["wall_time_s"] = time.perf_counter() - stage_started
                stage_probe["simulated_time_s"] = stage_probe["completed_steps"] * raw.step_dt
                stage_probe["passed"] = (
                    stage_probe["balanced_receivers"]
                    and stage_probe["tracks_clear_after_reset"]
                    and stage_probe["finite"]
                    and not stage_probe["premature_reset"]
                    and all(side["passed"] for side in stage_probe["sides"])
                )
            report["passed"] = all(probe["passed"] for probe in (*report["probes"], *report["serve_reset_probes"]))
        finally:
            raw.set_stage(original_stage)
            raw.set_evaluation(original_evaluation)
            env.reset()
            report["restored_stage"] = raw.stage
            report["restored_evaluation"] = raw.evaluation
            report["worlds_reset_after_probes"] = True
            report["wall_time_s"] = time.perf_counter() - started
    return report


def main():
    """Run the same sensor checks in a small standalone Newton scene, without a learned actor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps-per-probe", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_vision_preflight.json"))
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps_per_probe < 2 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Use at least two worlds/control steps and a positive finite localization tolerance.")

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.microduck.volley_vision_env import MicroDuckVolleyVisionEnv
    from isaaclab_tasks.contrib.microduck.volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg

    cfg = MicroDuckVolleyVisionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    cfg.head_camera_panels = False
    with launch_simulation(cfg.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        env = MicroDuckVolleyVisionEnv(cfg)
        try:
            report = check_vision_sensors(env, steps_per_probe=args.steps_per_probe, tolerance=args.tolerance)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved sensor preflight: {args.output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
