# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check real ball/head RGB evidence before a live passing run on the NVIDIA host.

Positive head probes measure actual paired badges in stages 0, 2 and 5, including
the final separation and production reset jitter.
The negative probe turns only the observing robot away once at reset, then runs
ordinary complete-body physics and actuator targets. Ground truth validates
completed RGB estimates; it never corrects a tracker or an actor input.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path


def check_live_pass_sensors(
    env,
    *,
    steps_per_probe: int = 16,
    tolerance: float = 0.05,
    check_ball: bool = True,
    image_directory: Path | None = None,
) -> dict:
    """Require RGB head pairs at early/final ranges and reject behind-camera partners.

    The wrapped or raw environment must expose ``partner_tracks`` and contain
    two worlds. All probe resets clear RGB histories. No joints are frozen and
    no body is positioned after the first positive-time step of a probe. The
    original curriculum/evaluation mode is restored and the worlds reset before
    returning. These checks validate sensing and reset behavior, not playing skill.
    Optional raw-frame PNGs help diagnose badge clipping or occlusion if a probe fails.
    """
    import torch

    from isaaclab.utils.math import quat_mul

    if type(steps_per_probe) is not int or steps_per_probe < 2:
        raise ValueError("Live-pass preflight requires at least two control steps per probe.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Partner localization tolerance must be finite and positive.")
    raw = env.unwrapped
    report = {
        "passed": False,
        "tolerance_m": tolerance,
        "minimum_rgb_pairs_per_side": 2,
        "partner_probes": [],
        "ground_truth_use": "Controlled reset yaw and error labels only; never tracker/actor correction.",
        "scope": "Real head-camera pixels and complete-body simulation; no learned pass skill assessed.",
    }
    if raw.num_envs < 2 or len(raw.head_cameras) != 2 or not hasattr(raw, "partner_tracks"):
        report["error"] = "Two physical worlds, both head cameras and RGB partner trackers are required."
        return report
    if check_ball:
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        checker = importlib.import_module("scripts.tools.check_microduck_vision")
        report["ball"] = checker.check_vision_sensors(env, steps_per_probe=steps_per_probe)
        if report["ball"].get("passed") is not True:
            report["error"] = "The existing real-RGB ball sensor preflight failed."
            return report
    started = time.perf_counter()
    original_stage, original_evaluation = raw.stage, raw.evaluation
    selected = torch.tensor((0, 1), dtype=torch.long, device=raw.device)
    actions = torch.zeros(raw.num_envs, raw.single_action_space.shape[0], device=raw.device)

    def histories_clear():
        return all(
            not bool(track.valid[selected].any()) and (track.seen is None or not bool(track.seen[selected].any()))
            for track in raw.partner_tracks
        )

    with torch.inference_mode():
        try:
            raw.set_evaluation(True)
            for stage, visible in ((0, True), (2, True), (5, True), (2, False)):
                raw.set_stage(stage)
                env.reset()
                probe = {
                    "name": f"stage_{stage}_" + ("physical_partner_badges" if visible else "partner_behind_camera"),
                    "stage": stage,
                    "nominal_separation_m": raw.cfg.stage_separations[stage],
                    "root_separation_by_world_m": [
                        float(
                            (
                                raw._robots[0].data.root_link_pos_w.torch[world]
                                - raw._robots[1].data.root_link_pos_w.torch[world]
                            ).norm()
                        )
                        for world in (0, 1)
                    ],
                    "passed": False,
                    "tracks_clear_after_reset": histories_clear(),
                    "balanced_receivers": raw._server[selected].tolist() == [0, 1],
                    "completed_steps": 0,
                    "premature_reset": False,
                    "finite": True,
                    "sides": [],
                }
                report["partner_probes"].append(probe)
                if not (probe["tracks_clear_after_reset"] and probe["balanced_receivers"]):
                    probe["error"] = "Reset failed to clear partner histories or establish balanced receivers."
                    continue
                if not visible:
                    for player, robot in enumerate(raw._robots):
                        world = selected[player : player + 1]
                        pose = robot.data.root_link_pose_w.torch[world].clone()
                        yaw = pose.new_tensor(((0.0, 0.0, 1.0, 0.0),))
                        pose[:, 3:7] = quat_mul(yaw, pose[:, 3:7])
                        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=world)
                        robot.write_root_velocity_to_sim_index(
                            root_velocity=torch.zeros(1, 6, device=raw.device), env_ids=world
                        )
                    raw.scene.write_data_to_sim()
                    raw.sim.forward()
                    raw.scene.update(dt=0.0)
                    # An eager dt=0 exposure is never fresh evidence. Clear the
                    # pair history and ignore all such intermediate reset frames.
                    for player, tracker in enumerate(raw._partner_trackers):
                        tracker.reset(selected)
                        raw._last_camera_frames[player][selected] = raw.head_cameras[player].frame.torch[selected]
                previous_frames = [int(camera.frame.torch[player]) for player, camera in enumerate(raw.head_cameras)]
                captures, accepted, false_tracks, errors = [0, 0], [0, 0], [0, 0], [[], []]
                count = steps_per_probe if visible else min(steps_per_probe, max(2, int(0.2 / raw.step_dt)))
                for step in range(count):
                    result = env.step(actions)
                    observations, reward = result[:2]
                    done = result[2] | result[3] if len(result) == 5 else result[2]
                    probe["completed_steps"] = step + 1
                    probe["finite"] &= bool(torch.isfinite(reward).all())
                    if hasattr(observations, "values"):
                        probe["finite"] &= all(bool(torch.isfinite(value).all()) for value in observations.values())
                    if done[selected].any():
                        probe["premature_reset"] = True
                        break
                    tracks = raw.partner_tracks
                    true_receivers = raw._head_positions().flip(1)
                    for player, camera in enumerate(raw.head_cameras):
                        frame = int(camera.frame.torch[player])
                        fresh = frame > previous_frames[player]
                        captures[player] += max(0, frame - previous_frames[player])
                        previous_frames[player] = frame
                        track = tracks[player]
                        recent_pair = bool(track.valid[player] & (track.age[player] < raw.step_dt - 1e-6))
                        if not visible:
                            false_tracks[player] += int(bool(track.valid[player]))
                        if fresh and recent_pair:
                            accepted[player] += 1
                            error = float((track.position_w[player] - true_receivers[player, player]).norm())
                            if math.isfinite(error):
                                errors[player].append(error)
                            else:
                                probe["finite"] = False
                    enough = all(value >= 2 for value in captures)
                    if visible:
                        enough &= all(len(values) >= 2 for values in errors)
                    if enough and probe["finite"]:
                        break
                for player in range(2):
                    mean_error = sum(errors[player]) / len(errors[player]) if errors[player] else None
                    side_passed = captures[player] >= 2
                    side_passed &= (
                        accepted[player] >= 2 and mean_error <= tolerance if visible else false_tracks[player] == 0
                    )
                    probe["sides"].append(
                        {
                            "player": ("a", "b")[player],
                            "world": player,
                            "passed": bool(side_passed),
                            "captured_frames": captures[player],
                            "accepted_rgb_pairs": accepted[player],
                            "false_partner_control_steps": false_tracks[player],
                            "mean_forehead_error_m": mean_error,
                            "maximum_forehead_error_m": max(errors[player]) if errors[player] else None,
                        }
                    )
                    if image_directory is not None:
                        from PIL import Image

                        image_directory.mkdir(parents=True, exist_ok=True)
                        path = image_directory / f"{probe['name']}_{('a', 'b')[player]}.png"
                        Image.fromarray(raw.head_camera_images[player][player].cpu().numpy()).save(path)
                        probe["sides"][-1]["raw_rgb_frame"] = str(path)
                probe["simulated_time_s"] = probe["completed_steps"] * raw.step_dt
                probe["passed"] = (
                    probe["finite"] and not probe["premature_reset"] and all(side["passed"] for side in probe["sides"])
                )
            report["passed"] = all(probe["passed"] for probe in report["partner_probes"])
        finally:
            raw.set_stage(original_stage)
            raw.set_evaluation(original_evaluation)
            env.reset()
            report["restored_stage"] = raw.stage
            report["restored_evaluation"] = raw.evaluation
            report["partner_histories_clear_after_final_reset"] = histories_clear()
            report["passed"] &= report["partner_histories_clear_after_final_reset"]
            report["wall_time_s"] = time.perf_counter() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps-per-probe", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_live_pass_preflight.json"))
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps_per_probe < 2 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Use at least two worlds/control steps and finite positive localization tolerance.")

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.microduck.volley_live_pass_env import MicroDuckLivePassEnv
    from isaaclab_tasks.contrib.microduck.volley_live_pass_env_cfg import MicroDuckLivePassEnvCfg

    cfg = MicroDuckLivePassEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = []
    cfg.head_camera_panels = False
    with launch_simulation(cfg.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        env = MicroDuckLivePassEnv(cfg)
        try:
            report = check_live_pass_sensors(
                env,
                steps_per_probe=args.steps_per_probe,
                tolerance=args.tolerance,
                image_directory=args.output.parent / f"{args.output.stem}_frames",
            )
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved live-pass sensor preflight: {args.output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
