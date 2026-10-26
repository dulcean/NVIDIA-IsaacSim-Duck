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
    from isaaclab_tasks.contrib.microduck.volley_vision_env import MicroDuckVolleyVisionEnv
    from isaaclab_tasks.contrib.microduck.volley_vision_env_cfg import MicroDuckVolleyVisionEnvCfg

    converter = _tool("prepare_microduck_vision_checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = (checkpoint.get("infos") or {}).get("microduck_curriculum", {})
    converted = metadata.get("profile") != converter.VISION_PROFILE
    if converted:
        checkpoint = converter.prepare_vision_checkpoint(checkpoint)
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in actor.values()
    ):
        raise ValueError("The preview requires a finite actor state dictionary.")
    torch.manual_seed(args.seed)
    cfg = MicroDuckVolleyVisionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.curriculum_stage = args.stage
    cfg.seed = args.seed
    if args.viz == "none":
        cfg.sim.visualizer_cfgs = []
    agent = MicroDuckVolleyVisionPPORunnerCfg()
    agent.device = args.device
    agent.seed = args.seed
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    saved = []
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_profile": checkpoint["infos"]["microduck_curriculum"]["profile"],
        "converted_legacy_actor": converted,
        "stage": args.stage,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "real_time": args.real_time and args.viz == "viser",
        "actor_observation_width": 146,
        "critic_observation_width": 142,
        "display_rotation": "90 degrees clockwise; original camera pixels feed the tracker",
        "runtime_passed": False,
        "skill_assessed": False,
        "ground_truth_use": "diagnostic errors only; actor ball state comes from RGB",
    }
    started = time.monotonic()
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
            latest_images = None
            last_image_step = None
            completed_steps = 0
            with torch.inference_mode():
                obs, _ = wrapped.reset()
                next_step_at = time.monotonic()
                for step in range(args.steps):
                    if not env.sim.is_running():
                        break
                    if report["real_time"]:
                        remaining = next_step_at - time.monotonic()
                        while remaining > 0:
                            time.sleep(min(remaining, 0.02))
                            remaining = next_step_at - time.monotonic()
                        # A slow render starts a new schedule instead of
                        # producing a burst of catch-up physics steps.
                        next_step_at = time.monotonic() + env.step_dt
                    obs, reward, dones, _ = wrapped.step(policy(obs))
                    policy.reset(dones)
                    completed_steps = step + 1
                    if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                        raise RuntimeError("Vision preview produced nonfinite actor observations or rewards.")
                    diagnostics = env.vision_diagnostics()
                    frames = diagnostics["camera_frames_seen"]
                    # A reset may eagerly render its intermediate scene at frame
                    # one. Keep snapshots only after a later physical capture,
                    # and retain them if the final control step resets world 0.
                    ready = all(camera.frame.torch[0].item() >= 2 for camera in env.head_cameras)
                    if ready and len(frames) == 2 and all(value > 0 for value in frames):
                        latest_images = [
                            camera.data.output["rgb"].torch[0, ..., :3].detach().clone() for camera in env.head_cameras
                        ]
                        last_image_step = completed_steps
                        if not first_saved:
                            save_images("first", latest_images)
                            report["first_image_step"] = completed_steps
                            first_saved = True
                    if step % 10 == 0 or step == args.steps - 1:
                        samples.append({"step": completed_steps, **diagnostics})
                diagnostics = env.vision_diagnostics()
                if not first_saved:
                    raise RuntimeError("Both head cameras must produce physical frames before preview can pass.")
                save_images("final", latest_images)
                report.update(
                    runtime_passed=True,
                    completed_steps=completed_steps,
                    final_image_step=last_image_step,
                    physics_dt_s=cfg.sim.dt,
                    policy_dt_s=env.step_dt,
                    diagnostics=diagnostics,
                    camera_configs=[to_dict(camera.cfg) for camera in env.head_cameras],
                )
        finally:
            report.update(wall_time_s=time.monotonic() - started, images=saved, samples=samples)
            (args.output / "preview.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
            env.close()
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
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace Viser playback to physical time; --no-real-time captures artifacts at full speed.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file() or min(args.num_envs, args.steps) < 1 or args.seed < 0:
        parser.error("Supply an existing checkpoint, positive world/step counts and a non-negative seed.")
    report = preview(args)
    summary = {key: value for key, value in report.items() if key not in ("samples", "camera_configs")}
    print(json.dumps(summary, indent=2))
    print(f"Saved camera images and diagnostics: {args.output / 'preview.json'}")


if __name__ == "__main__":
    main()
