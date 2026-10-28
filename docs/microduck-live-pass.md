# Camera-guided live MicroDuck passing

This profile replaces the coach's fixed virtual target with the actual moving
partner's forehead. Stage 2 needs one real legal transfer; later stages need
2, 4 and 8 alternating transfers. Predictions and reward cannot satisfy these
gates. All joints, including the legs, remain controlled by the learned policy.
The environment is registered as `IsaacContrib-Volley-LivePass-MicroDuck`; use
the curriculum driver below for automatic stages, gates and certified checkpoints.

## What each robot observes

Each robot uses its own calibrated head RGB camera. Orange sphere extraction
estimates ball position and velocity. Two small cyan/magenta spherical badges
are visual children of each physical head, without colliders, mass or extra
rigid bodies. Known-size image geometry estimates the badge centres, and
`2 * cyan - magenta` reconstructs the forehead strike point, including head tilt.
This is calibrated landmark tracking, not a general learned duck detector.
The spheres sit above the opaque shell and behind their own camera's optical
plane, so an observing robot detects the other head's landmarks.

Partner estimates have bounded 0.75 s constant-velocity memory. Missing or
expired measurements invalidate the target and pass plan; no simulator pose
initializes or repairs them. Ball memory retains its existing 0.15 s limit.
Camera poses use the observing robot's calibrated kinematics/odometry reference;
noise in that reference is not modeled. It is not a direct simulator-object input.

The actor has two independent 89-input/14-output paths. All linear layers
exclude cross-player connections, so one robot cannot reconstruct its partner
from the other robot's privileged proprioception. Each path contains its own
IMU/encoder/recovery state, RGB ball state/history, RGB target position/velocity,
measurement validity/age and the visual pass reference. The critic and physical
reward labels may use simulation state, but these are separate from actor inputs.

## Passing reference and physical feedback

From the observed ball position/velocity and own head kinematics, predict the
incoming descending contact. Advance the observed partner point to that time.
Choose a feasible descending outgoing arc over 0.20--0.45 s, with bounded speed
and a sufficient centre-court crossing height. For a selected flight time `T`:

```text
v_out = (partner_position + partner_velocity*T - ball_position)/T - gravity*T/2
ball_impulse = ball_mass * (v_out - incoming_velocity)
```

The policy receives the desired velocity and change in velocity in its own
frame. A physically measured release is compared with the reference captured
at impact, correcting its vertical velocity for gravity over the sampling delay.
The bounded velocity score has weight 2.0 and distinguishes weak contact from
a useful pass and excessive speed. No force or velocity command is applied to
the ball. Kinematic arc feasibility does not prove that a head posture or motor
can execute it; actual collision and balance learning still determine that.

Robot positions vary at reset by up to 4 cm longitudinally and 1 cm laterally.
Both robots stay on court in every lesson. Targets follow actual head motion;
the coach no longer parks the partner elsewhere. New physics uses authored ball
`solref=(0.02,0.5)`, preserving the measured damped contact experiment rather than
the previous high-speed rebound setting. This is not calibration to a real ball.

In the 2/4/8-return lessons the incoming serve starts 0.75--0.80 m horizontally
from the receiving forehead. This bounds the complete launch distance, including
the sampled head-to-head gap; it is not an additional offset beyond the partner.
The reset distribution must satisfy both the 3.5 m/s arrival-speed limit and
full-ball visibility at the first camera exposure. The live sensor check uses
a closer test sphere with a small initial toss to clear the physical partner
while retaining two RGB exposures. All subsequent motion follows the simulation.

## Run on the NVIDIA machine

Use the existing camera-trained checkpoint as a warm start. Migration retains
the learned within-player weights and normalization, adds zero-initialized pass
feature connections, and removes old cross-player connections. Consequently
the old centralized policy's complete actions are **not** claimed to be preserved.
Critic, optimizer, stage counters and certification start fresh. Source files
remain available for rollback. A native live-profile checkpoint can later resume
with its complete optimizer, sensor contract and qualification history.

After transferring the code, from `/home/user/test-duck`:

```bash
nohup uv run python scripts/tools/train_microduck_volley.py \
  --live-pass --player-symmetry \
  --checkpoint logs/rsl_rl/microduck_volley_vision/2026-10-28_02-33-39-078895/latest.pt \
  --start-stage 1 \
  --num-envs 256 --iterations 40000 \
  --eval-interval 500 --eval-episodes-per-side 64 \
  --keep-training-on-stall \
  --device cuda:0 --seed 42 --viz none \
  > volley_live_pass_40k.log 2>&1 &
```

Starting at stage 1 explicitly reuses the existing acquisition skill; it does
not mark stage 0 as newly passed. For all acquisition lessons from the beginning,
use `--start-stage 0`. The final gate still requires real RGB ball and partner
evidence in addition to the actual rally and recovery, regardless of the start.

This starts a new run under `logs/rsl_rl/microduck_volley_live_pass/`. Its GPU
preflight checks actual rendered ball measurements, paired head localisation,
behind-camera rejection and reset clearing before learning. Probe images and
JSON are saved inside the run. A failed sensor/runtime check stops immediately;
running 40,000 iterations without valid observations would waste the budget.
Successful preflight verifies sensing, not learned gameplay.

In this profile, stalled evaluations continue the same stage and optimizer
without automatic rollback. They remain recorded as failed gates, and never
promote a lesson. `--keep-training-on-stall` is saved and retained on resume.
The requested iterations are an upper learning budget; a certified policy may
finish earlier. Exceptions, interruption and invalid runtime state remain
visible. Budget exhaustion does not become a success certificate.

```bash
tail -f volley_live_pass_40k.log
```

Check `Metrics/returns`, `Metrics/success`, `Metrics/unsafe`, partner visibility,
`Metrics/pass_release_speed_m_s`, `Metrics/pass_required_speed_m_s` and
`Metrics/pass_velocity_error_m_s`. Read velocity errors together with
`Metrics/pass_release_samples`; zero samples are not perfect passes. The held-out
reports select best checkpoints by physical outcomes rather than training reward.

To resume, use the new run's `latest.pt`:

```bash
uv run python scripts/tools/train_microduck_volley.py \
  --live-pass --resume logs/rsl_rl/microduck_volley_live_pass/RUN/latest.pt \
  --num-envs 256 --iterations 40000 --viz none
```

`--iterations` adds that budget to the resumed run. Contact/reward parameters and
the long-run policy restore from metadata without repeating their original flags.

## Watch actual gameplay

`ready.pt` is written only after repeated balanced final gates and a separate
held-out seed pass. Each side must succeed in at least 80% of at least 64 episodes,
unsafe episode rate must be at most 5%, and the final physical rally needs 8
qualified returns plus both robots' recovery and visual evidence. Training may
not reach this standard within 30,000--40,000 iterations.

```bash
uv run --extra viser python scripts/tools/view_microduck_vision.py \
  --checkpoint logs/rsl_rl/microduck_volley_live_pass/RUN/ready.pt \
  --play --num-envs 2 --steps 100000 --viz viser --panel-fps 10
```

Viser prints its address, usually port 8080. For interim inspection substitute
`latest.pt`. `--play` uses the pair stage and keeps playing beyond the eight-return
training target, until a miss, unsafe outcome or 60 s episode timeout. It does
not confer certification. Without `--play`, `--stage 0` through `--stage 5` show
the exact held-out lesson contract. Local CPU tests cover perception geometry,
information isolation, ballistic references, checkpoint migration, actual RSL PPO
and long-run gate policy. Native rendering, collision behaviour and learning
convergence require the NVIDIA runtime and have not been verified locally.
