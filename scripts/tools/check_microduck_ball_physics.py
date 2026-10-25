# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the task ball's free fall and passive floor bounce without robots or PPO.

Run on the NVIDIA host: uv run python scripts/tools/check_microduck_ball_physics.py
The JSON report and CSV trace distinguish solver behavior from policy-driven strikes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def analyze_trace(rows: list[tuple[float, float, float]], radius: float) -> dict:
    """Measure acceleration, ballistic position error, and the first passive rebound.

    Each row is (elapsed seconds, center height in metres, vertical velocity in m/s).
    Incoming and outgoing speeds are sampled extrema around the first impact, so the
    reported restitution is an estimate affected by contact compliance and timestep.
    """
    if len(rows) < 3 or not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("The ball trace must contain at least three finite state samples.")
    t0, z0, v0 = rows[0]
    accelerations = []
    errors = []
    incoming = 0.0
    outgoing = 0.0
    rebound_height = 0.0
    bounced = False
    apex = False
    approaching_floor = False
    for previous, current in zip(rows, rows[1:]):
        time, z, vz = current
        dt = time - previous[0]
        if dt <= 0:
            raise ValueError("Trace timestamps must be strictly increasing.")
        if not approaching_floor and z > radius + 0.06:
            accelerations.append((vz - previous[2]) / dt)
            elapsed = time - t0
            errors.append((z - (z0 + v0 * elapsed - 0.5 * 9.81 * elapsed**2)) ** 2)
        if not bounced:
            incoming = max(incoming, -vz)
            approaching_floor |= z < radius + 0.06
            if approaching_floor and vz > 0:
                bounced = True
        if bounced:
            outgoing = max(outgoing, vz)
            rebound_height = max(rebound_height, z - radius)
            if previous[2] > 0 and vz <= 0:
                apex = True
                break
    if not accelerations:
        raise ValueError("No free-flight samples above the floor-contact region.")
    acceleration = sum(accelerations) / len(accelerations)
    rmse = math.sqrt(sum(errors) / len(errors))
    restitution = outgoing / incoming if bounced and incoming > 0 else None
    checks = {
        "gravity_within_5_percent": abs(acceleration + 9.81) < 0.05 * 9.81,
        "free_fall_rmse_below_2_cm": rmse < 0.02,
        "complete_first_rebound": apex,
        "no_passive_energy_gain": restitution is not None
        and restitution <= 1.05
        and rebound_height <= 1.05 * (z0 - radius),
    }
    return {
        "measured_acceleration_m_s2": acceleration,
        "free_fall_rmse_m": rmse,
        "incoming_speed_m_s": incoming,
        "outgoing_speed_m_s": outgoing,
        "estimated_restitution": restitution,
        "first_rebound_height_above_floor_m": rebound_height,
        "checks": checks,
    }


def measure(args: argparse.Namespace) -> tuple[list[tuple[float, float, float]], dict]:
    """Run the actual task's Newton solver, ball collider/material, and floor with no policy."""
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.utils import use_stage
    from isaaclab.utils import configclass

    from isaaclab_tasks.contrib.microduck.ball_env_cfg import MicroDuckHeadingEnvCfg

    task = MicroDuckHeadingEnvCfg()
    task.sim.device = args.device
    task.sim.dt = args.dt
    task.sim.visualizer_cfgs = []

    @configclass
    class BallSceneCfg(InteractiveSceneCfg):
        terrain = task.scene.terrain
        ball = task.scene.ball

    cfg = BallSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.ball.init_state.pos = (0.0, 0.0, args.height)
    cfg.ball.init_state.lin_vel = (0.0, 0.0, 0.0)
    rows = []
    with launch_simulation(task.sim, {"device": args.device, "visualizer": ["none"]}):
        sim_utils.create_new_stage()
        sim = sim_utils.SimulationContext(task.sim)
        with use_stage(sim.stage):
            scene = InteractiveScene(cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        sim.backend_manager.set_decimation(1)
        scene.update(0.0)
        ball = scene["ball"]
        mass = ball.data.body_mass.torch.flatten()[0].item()
        inertia = ball.data.body_inertia.torch.reshape(-1, 3, 3)[0]
        metadata = {
            "dt_s": args.dt,
            "drop_center_height_m": args.height,
            "radius_m": cfg.ball.spawn.radius,
            "declared_mass_kg": cfg.ball.spawn.mass_props.mass,
            "measured_mass_kg": mass,
            "inertia_kg_m2": inertia.tolist(),
            "uses_mujoco_contacts": task.sim.physics.solver_cfg.use_mujoco_contacts,
            "solref": list(cfg.ball.spawn.collision_props[1].solref),
        }
        metadata["mass_and_inertia_valid"] = (
            abs(mass - cfg.ball.spawn.mass_props.mass) < 1e-5
            and bool(torch.isfinite(inertia).all())
            and bool((torch.linalg.eigvalsh(inertia) > 0).all())
        )
        for step in range(math.ceil(args.seconds / args.dt) + 1):
            if step:
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(args.dt)
            z = ball.data.root_link_pos_w.torch[0, 2].item()
            vz = ball.data.root_link_lin_vel_w.torch[0, 2].item()
            rows.append((step * args.dt, z, vz))
    return rows, metadata


def main() -> int:
    """Save measured physical consistency checks; no training or automatic episode resets are used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--height", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck_ball_physics.json"))
    args = parser.parse_args()
    if not (0 < args.dt <= 0.01 and args.height > 0.1 and args.seconds >= 1.0):
        parser.error("Use 0 < dt <= 0.01, height > 0.1 m and seconds >= 1.0.")
    rows, metadata = measure(args)
    report = metadata | analyze_trace(rows, metadata["radius_m"])
    report["checks"]["mass_and_inertia_valid"] = report["mass_and_inertia_valid"]
    report["passed"] = all(report["checks"].values())
    report["scope"] = "Physical consistency of passive ball/floor interaction; not calibration to a real ball."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "center_z_m", "velocity_z_m_s"))
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report and trace: {args.output}, {args.output.with_suffix('.csv')}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
