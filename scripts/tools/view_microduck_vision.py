# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview a MicroDuck actor using actual head-camera ball estimates, without training.

Viser keeps its normal 3D scene and adds both raw head views in the same UI.
Legacy weights are adapted in memory; source checkpoints are never changed.
PNG images and tracking/truth-error diagnostics are written for runtime review.
Ground truth in the report is diagnostic only and is not fed to the actor.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


def _tool(name):
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"scripts.tools.{name}")


def preview(args) -> dict:
    """Run physical camera inference; require frames, without asserting that the policy can play."""
    started = time.perf_counter()
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image
    from rsl_rl.runners import OnPolicyRunner

    wp.config.enable_backward = False

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.contrib.microduck.agents.ball_ppo_cfg import MicroDuckVolleyVisionPPORunnerCfg
    from isaaclab_tasks.contrib.microduck.volley_task_settings import apply_task_settings, resolve_task_settings
    from isaaclab_tasks.contrib.microduck.volley_vision_env import MicroDuckVolleyVisionEnv
    from isaaclab_tasks.contrib.microduck.volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg

    converter = _tool("prepare_microduck_vision_checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = (checkpoint.get("infos") or {}).get("microduck_curriculum", {})
    source_profile = metadata.get("profile", "legacy_state_actor")
    converted = metadata.get("profile") != converter.VISION_PROFILE
    if converted:
        checkpoint = converter.prepare_vision_checkpoint(checkpoint)
    vision_contract = _tool("evaluate_microduck_volley")._curriculum_module(converter.VISION_PROFILE)
    lesson = vision_contract.VISION_STAGES[args.stage]
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor.values()
    ):
        raise ValueError("The preview requires a finite actor state dictionary.")
    torch.manual_seed(args.seed)
    cfg = MicroDuckVolleyVisionEnvCfg()
    task_settings = resolve_task_settings(metadata)
    apply_task_settings(cfg, task_settings)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.curriculum_stage = args.stage
    cfg.seed = args.seed
    cfg.head_camera_panels = not args.no_head_panels
    cfg.head_camera_panel_fps = args.panel_fps
    if args.viz == "none":
        cfg.sim.visualizer_cfgs = []
    agent = MicroDuckVolleyVisionPPORunnerCfg()
    agent.device = args.device
    agent.seed = args.seed
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    saved = []
    report = {
        "task_settings": task_settings,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_profile": checkpoint["infos"]["microduck_curriculum"]["profile"],
        "source_checkpoint_profile": source_profile,
        "converted_legacy_actor": converted,
        "migration_scope": "Warm-start adaptation only; new visual curriculum performance is unassessed."
        if converted
        else "No checkpoint conversion.",
        "stage": args.stage,
        "stage_name": lesson.name,
        "objective": lesson.objective,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "real_time": args.real_time and args.viz == "viser",
        "head_panels_enabled": cfg.head_camera_panels,
        "head_panel_fps": args.panel_fps,
        "status": "initializing",
        "actor_observation_width": converter.ACTOR_WIDTH,
        "critic_observation_width": 142,
        "display_rotation": "90 degrees clockwise; original camera pixels feed the tracker",
        "runtime_passed": False,
        "skill_assessed": False,
        "ground_truth_use": (
            "Critic, task/reward labels and diagnostics; actor ball state and search memory come only from RGB."
        ),
    }
    phases = {name: 0.0 for name in ("policy", "environment_step", "diagnostics", "artifacts", "pacing_wait")}
    completed_steps = 0
    latest_images = None
    last_image_step = None
    loop_started = loop_finished = warmup_finished = None
    with launch_simulation(
        cfg.sim,
        {"device": args.device, "visualizer": [args.viz], "max_visible_envs": min(args.num_envs, 4)},
    ):
        sim_utils.create_new_stage()
        env = MicroDuckVolleyVisionEnv(cfg)
        try:
            contract = env.vision_contract()
            if not converted and "vision_contract" in metadata and metadata["vision_contract"] != contract:
                raise ValueError("Saved visual observation/camera contract differs from this environment.")
            report["vision_contract"] = contract
            env.set_evaluation(True, seed=args.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=args.device)
            policy = runner.get_inference_policy(device=args.device)
            policy.load_state_dict(actor, strict=True)

            def save_images(label, images):
                for name, image in zip(("a", "b"), images):
                    raw = image.cpu().numpy()
                    path = args.output / f"{label}_head_{name}.png"
                    Image.fromarray(np.ascontiguousarray(np.rot90(raw, k=-1))).save(path)
                    saved.append(str(path.resolve()))

            first_saved = False
            with torch.inference_mode():
                obs, _ = wrapped.reset()
                loop_started = time.perf_counter()
                next_step_at = loop_started
                report["status"] = "running"
                for step in range(args.steps):
                    if not env.sim.is_running():
                        break
                    if report["real_time"]:
                        phase_started = time.perf_counter()
                        remaining = next_step_at - phase_started
                        while remaining > 0:
                            time.sleep(min(remaining, 0.02))
                            remaining = next_step_at - time.perf_counter()
                        # A slow render starts a new schedule instead of
                        # producing a burst of catch-up physics steps.
                        next_step_at = time.perf_counter() + env.step_dt
                        phases["pacing_wait"] += time.perf_counter() - phase_started
                    phase_started = time.perf_counter()
                    actions = policy(obs)
                    phases["policy"] += time.perf_counter() - phase_started
                    phase_started = time.perf_counter()
                    obs, reward, dones, _ = wrapped.step(actions)
                    policy.reset(dones)
                    phases["environment_step"] += time.perf_counter() - phase_started
                    completed_steps = step + 1
                    if completed_steps % 10 == 0 or step == args.steps - 1:
                        phase_started = time.perf_counter()
                        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                            raise RuntimeError("Vision preview produced nonfinite actor observations or rewards.")
                        diagnostics = env.vision_diagnostics()
                        report["diagnostics"] = diagnostics
                        samples.append(
                            {
                                "step": completed_steps,
                                "loop_wall_s": time.perf_counter() - loop_started,
                                **diagnostics,
                            }
                        )
                        phases["diagnostics"] += time.perf_counter() - phase_started
                        phase_started = time.perf_counter()
                        frames = diagnostics["camera_frames_seen"]
                        frame_ids = torch.stack([camera.frame.torch[0] for camera in env.head_cameras]).cpu().tolist()
                        # Ignore frame-one intermediate reset captures and keep
                        # the last valid snapshots if world 0 resets at the end.
                        if all(frame >= 2 for frame in frame_ids) and len(frames) == 2 and all(frames):
                            image_sources = getattr(env, "head_camera_images", None)
                            if image_sources is None:
                                image_sources = [camera.data.output["rgb"].torch for camera in env.head_cameras]
                            latest_images = [image[0, ..., :3].detach().clone() for image in image_sources]
                            last_image_step = completed_steps
                            if not first_saved:
                                save_images("first", latest_images)
                                report["first_image_step"] = completed_steps
                                first_saved = True
                        phases["artifacts"] += time.perf_counter() - phase_started
                    if completed_steps == 10:
                        warmup_finished = time.perf_counter()
                loop_finished = time.perf_counter()
                if not first_saved:
                    raise RuntimeError("Both head cameras must produce physical frames before preview can pass.")
                report.update(
                    status="completed" if completed_steps == args.steps else "viewer_closed",
                    runtime_passed=True,
                    physics_dt_s=cfg.sim.dt,
                    policy_dt_s=env.step_dt,
                    camera_configs=[to_dict(camera.cfg) for camera in env.head_cameras],
                )
        except KeyboardInterrupt:
            report["status"] = "interrupted"
            raise
        except BaseException as error:
            report.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            loop_finished = loop_finished or time.perf_counter()
            cleanup_error = None
            if latest_images is not None:
                phase_started = time.perf_counter()
                try:
                    save_images("final", latest_images)
                except Exception as error:
                    cleanup_error = error
                    report["artifacts_error"] = f"{type(error).__name__}: {error}"
                finally:
                    phases["artifacts"] += time.perf_counter() - phase_started
            try:
                phase_started = time.perf_counter()
                report["diagnostics"] = env.vision_diagnostics()
            except Exception as error:
                cleanup_error = cleanup_error or error
                report["diagnostics_error"] = f"{type(error).__name__}: {error}"
            finally:
                phases["diagnostics"] += time.perf_counter() - phase_started
            if cleanup_error is not None and report["status"] in ("completed", "viewer_closed"):
                report.update(status="error", runtime_passed=False)
            loop_wall = loop_finished - loop_started if loop_started is not None else 0.0
            warmup_wall = (warmup_finished or loop_finished) - loop_started if loop_started is not None else 0.0
            measured_steps = max(0, completed_steps - 10)
            measured_wall = loop_finished - warmup_finished if warmup_finished is not None else 0.0
            report.update(
                completed_steps=completed_steps,
                final_image_step=last_image_step,
                wall_time_s=time.perf_counter() - started,
                timing={
                    "setup_wall_s": (loop_started or loop_finished) - started,
                    "loop_wall_s": loop_wall,
                    "warmup_steps": min(10, completed_steps),
                    "warmup_wall_s": warmup_wall,
                    "measured_steps": measured_steps,
                    "measured_wall_s": measured_wall,
                    "control_steps_per_second": measured_steps / measured_wall if measured_wall > 0 else None,
                    "real_time_factor": measured_steps * env.step_dt / measured_wall if measured_wall > 0 else None,
                    "environment_transitions_per_second": measured_steps * env.num_envs / measured_wall
                    if measured_wall > 0
                    else None,
                    "host_phases_wall_s": phases,
                    "scope": (
                        "Host wall timings; loop excludes setup, rates exclude first 10 steps, not a CUDA profile."
                    ),
                },
                images=saved,
                samples=samples,
            )
            try:
                (args.output / "preview.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
            finally:
                env.close()
            if cleanup_error is not None and sys.exc_info()[0] is None:
                raise RuntimeError("The preview could not save complete runtime evidence.") from cleanup_error
    return report


def main():
    """Inspect camera outputs and actor wiring with a small environment count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(6), default=0)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_vision_preview"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000042)
    parser.add_argument("--viz", choices=("viser", "none"), default="viser")
    parser.add_argument(
        "--no-head-panels", action="store_true", help="Keep the 3D scene but disable head image panels."
    )
    parser.add_argument("--panel-fps", type=float, default=30.0, help="Head-panel update ceiling, between 1 and 30 Hz.")
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace Viser playback to physical time; --no-real-time captures artifacts at full speed.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file() or min(args.num_envs, args.steps) < 1 or args.seed < 0:
        parser.error("Supply an existing checkpoint, positive world/step counts and a non-negative seed.")
    if not 1 <= args.panel_fps <= 30:
        parser.error("--panel-fps must be between 1 and 30.")
    report = preview(args)
    summary = {key: value for key, value in report.items() if key not in ("samples", "camera_configs")}
    print(json.dumps(summary, indent=2))
    print(f"Saved camera images and diagnostics: {args.output / 'preview.json'}")


if __name__ == "__main__":
    main()
