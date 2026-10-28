# MicroDuck: one-legged pirouette and bow

`IsaacContrib-Ballet-MicroDuck` uses one full-collision/backlash MicroDuck on
flat ground, with its original BAM actuators. All 14 joints remain trainable.
There are no cameras, balls, base impulses, scripted joint motions, or anchored
feet. Turning a planted foot must emerge from the physical contacts and motors.

The routine is: lift the right foot while balancing on the left, turn, land on
both feet, bow with the torso and head, then recover to a quiet upright stance.
The final target is one full revolution, within a two-degree tolerance. The bow
requires at least 0.35 seconds of measured body/head inclination; final recovery
requires 0.50 seconds with both feet supported and low body velocities.

## Automatic learning progression

Each world starts with balance and bow, then advances through 45, 90, 180 and
360 degrees after **two consecutive complete routines** at its current level.
A predicted angle, elapsed training iterations, head rotation, or reward total
cannot advance a lesson. Failure resets its pass streak. The driver keeps
training to the requested iteration budget; a difficult lesson does not stop it.

Root yaw is unwrapped at each 5 ms physics step. The same interval must show
an upward force on the support foot, at least 18 mm of free-foot clearance,
no free-foot contact, an upright torso, and controlled movement near the original
support location. A toe touch, flight, or loss of this support resets the current
turn. Oscillating left/right cannot accumulate a full rotation. Ground contacts
on the body/head invalidate support and sustained contact ends the attempt.

Dense posture, balance, foot-clearance and angular-velocity terms guide learning.
Discount-consistent rotation potential supplies intermediate feedback, including
its cancellation at success, failure and timeout. The large completion reward
requires the entire turn, landing, bow and recovery in that order.

The 89-input actor retains the walking policy's 61-input prefix, then adds the
routine phase, turn remainder/references, foot positions/clearances/contacts,
centre-of-mass offset, home displacement and body velocity. These are physical
simulation/proprioceptive quantities; this recipe does not model errors in
odometry, centre-of-mass estimation or contact sensing for a real robot.

## Train on the NVIDIA machine

Start with the existing **single-robot walking/standing** checkpoint, rather
than the two-player volleyball policy. The driver validates and expands its
61-input actor to 89 inputs in memory. Added input weights start at zero; all
connections train normally. Walking normalization statistics initialize the
new policy with a bounded prior count so they can adapt to ballet. The critic,
optimizer and curriculum start fresh. The source file is never overwritten.

From `/home/user/test-duck`, after pulling the code:

```bash
nohup uv run python scripts/tools/train_microduck_ballet.py \
  --checkpoint logs/rsl_rl/microduck_velocity_flat_backlash/2026-10-23_14-05-46/model_4000.pt \
  --num-envs 2048 \
  --iterations 20000 \
  --device cuda:0 \
  --viz none \
  > ballet_20k.log 2>&1 &
```

That checkpoint exists in the shared local logs. If its path differs on the GPU
host, substitute the corresponding 61-input walking checkpoint there. Omit
`--checkpoint` to train from scratch; it will require learning standing too.

The same invocation automatically checks the actual 89/14 interface, finite
physics and real upward forces on both named feet before learning. It writes
`preflight.json` in the run directory. This checks execution and sensors, not
whether the trick has been learned. For a small initial runtime check, use
`--num-envs 32 --iterations 2` instead of the large values.

```bash
tail -f ballet_20k.log
```

Watch `Metrics/stage` and the `stage_0_fraction` through `stage_4_fraction` metrics.
`Metrics/success` includes the current easier lessons; it alone does not prove a
full pirouette. `Metrics/full_success` counts only complete level-4 routines.
`Metrics/stage_4_success` reports success among completed level-4 episodes; read
it with `stage_4_fraction`. `turn_completed`, `bow_completed`, `phase`,
`turn_radians` and `failure` distinguish where a routine breaks down.

Weights, optimizer and curriculum levels are saved every 250 iterations under
`logs/rsl_rl/microduck_ballet/<run>/`, with an atomic `latest.pt` copy. On normal
completion the driver evaluates a fixed cohort of up to 128 robots once each
at the full-turn level, with deterministic actions and a separate reset seed.
`evaluation.json` reports their actual full-routine success rate, including
slow successes as well as early failures. It is a diagnostic, not a certificate
of broad reliability. Training reward does not determine this result.

Stop with Ctrl+C in the training terminal, or send `kill -INT <python_pid>` to
the Python training process from `pgrep -af train_microduck_ballet.py`.
An orderly interrupt writes `interrupted.pt` and updates `latest.pt`; SIGKILL
cannot save. To continue with an additional 20,000 iterations:

```bash
nohup uv run python scripts/tools/train_microduck_ballet.py \
  --resume latest --num-envs 2048 --iterations 20000 --viz none \
  > ballet_resume.log 2>&1 &
```

Resuming restores learned levels with the optimizer. Episodes restart from a
physical standing pose and interrupted pass streaks are cleared. With a new
environment count, saved levels repeat across the new world pool.

## Watch in Viser

```bash
checkpoint=$(find logs/rsl_rl/microduck_ballet -type f -name latest.pt | sort | tail -n 1)

uv run --extra viser isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Ballet-MicroDuck \
  --checkpoint "$checkpoint" \
  --num_envs 4 \
  --viz viser
```

Playback requests the full 360-degree routine and bow. To inspect an earlier
lesson, append `env.start_stage=0` through `env.start_stage=3`. The Viser address
appears in the terminal, normally on port 8080. Rendering during a separate
preview can compete with simultaneous training for the GPU.

CPU tests cover physical evidence rules, yaw wrapping, anti-exploit reward
accounting, curriculum restore, genuine RSL actor migration and a PPO update.
The local machine cannot run the target Newton/MJWarp CUDA simulation. Actual
balance, achievable pivot friction/torques, learning speed and final success
still require the NVIDIA runtime. The automatic preflight and final evaluation
make those results visible; no fixed iteration count guarantees convergence.
