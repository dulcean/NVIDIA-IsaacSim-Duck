# MicroDuck: heading and cooperative volleyball

Experimental tasks for the NVIDIA training host. Development and CPU checkpoint checks can run elsewhere;
the Newton/MJWarp scene, contact behavior, and learned performance must be validated on the GPU host.

| Task | Scene | Policy inputs | Actions | Actor hidden layers |
| --- | --- | --- | --- | --- |
| `IsaacContrib-Heading-MicroDuck` | One robot, ball, virtual receiving target | 71 | 14 | 512 / 256 / 128 |
| `IsaacContrib-Volley-MicroDuck` | Two robots, one ball, separate half-courts, no wall | 142 | 28 | 1024 / 512 / 256 |

Both tasks use Newton/MJWarp and the full-collision **backlash** MicroDuck asset. The backend and asset
are fixed; do not add `presets=newton_mjwarp` or `presets=backlash` to these commands.
The 5 g, 5 cm diameter ball has physical gravity and compliant contacts. This is a lightweight training
ball scaled to MicroDuck, rather than a regulation volleyball. Robot roots start 0.35 m apart.
The ball is repositioned only during an episode reset. Robots produce strikes through joint motion.

## First launch on the NVIDIA machine

Run from the root of this updated checkout, in the same uv environment used for MicroDuck walking.
The existing robot USDs, including `microduck_allcollisions_backlash.usd`, must be accessible.
No Isaac Sim installation is required for the chosen Newton backend and Viser display.

First check both physical scenes, filtered ball/head contacts, finite stepping, and missed-ball reset:

```bash
uv run python -m pytest source/isaaclab_tasks/test/test_microduck_ball.py -q
```

The startup report prints which bodies count as the head. The test drops a ball onto those bodies;
it must detect a real collision. Resolve sensor, asset, collision-capacity, or non-finite-state errors
before starting a long training run.

Prepare the standing actor for the heading task:

```bash
uv run python scripts/tools/prepare_microduck_ball_checkpoint.py \
  logs/rsl_rl/microduck_velocity_flat_backlash/2026-10-23_14-05-46/model_4000.pt \
  logs/microduck_initialization/heading.pt --players 1
```

The output is a new file; the utility refuses to overwrite an existing checkpoint. Start with a small
runner smoke test. This checks execution, not learning success:

```bash
uv run isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Heading-MicroDuck \
  --num_envs 8 --max_iterations 2 \
  --checkpoint logs/microduck_initialization/heading.pt \
  --no-resume_optimizer_state
```

Then train with Viser:

```bash
uv run --extra viser isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Heading-MicroDuck \
  --num_envs 256 --max_iterations 5000 --viz viser \
  --checkpoint logs/microduck_initialization/heading.pt \
  --no-resume_optimizer_state
```

For faster training, omit `--viz viser`. Inspect a resulting checkpoint with:

```bash
uv run --extra viser isaaclab play \
  --rl_library rsl_rl --task IsaacContrib-Heading-MicroDuck \
  --num_envs 4 --viz viser \
  --checkpoint logs/rsl_rl/microduck_heading/RUN_DIRECTORY/model_ITERATION.pt
```

Replace `RUN_DIRECTORY` and `ITERATION` with the actual saved run and checkpoint. Iteration count alone
does not establish success. Watch for a stable robot, intentional upward strikes, and arrivals at the
virtual target, rather than only a rising reward curve. Use `Metrics/target_success`, `head_hits`, `fell`,
and `invalid_contact` alongside playback. The virtual target is fixed at the neutral head height during
each episode, at the mirrored starting head position across the court center; it has no collider or visual marker.

## Progress to two robots

Once heading works, transfer its actor to two initially independent copies:

```bash
uv run python scripts/tools/prepare_microduck_ball_checkpoint.py \
  logs/rsl_rl/microduck_heading/RUN_DIRECTORY/model_ITERATION.pt \
  logs/microduck_initialization/volley.pt --players 2

uv run --extra viser isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Volley-MicroDuck \
  --num_envs 256 --max_iterations 10000 --viz viser \
  --checkpoint logs/microduck_initialization/volley.pt \
  --no-resume_optimizer_state
```

Run the same two-iteration/eight-environment check first when moving to this stage. You can also
initialize the pair directly from the standing checkpoint with `--players 2`, but it has not yet
learned a useful strike. A 28-action pair checkpoint is for ordinary continuation/play, not input to
the transfer utility. Resume an existing ball run using its own task and checkpoint, without converting
it again; the normal training command can restore that run's optimizer state.

`Metrics/returns` counts alternating head contacts after the initial serve. Repeated hits by the same
robot, simultaneous ambiguous hits, non-head ball contacts, falling, leaving a half-court, or dropping
the ball end a rally. The first head contact is a hit, not a completed transfer. Rewards include sparse
head-hit/return bonuses, a ballistic aiming bonus, small alignment/upright terms, and costs for abrupt
actions, joint limits, invalid contact, and falling. Aiming uses the partner's current head position;
successful subsequent contacts are the actual return metric.

## What is transferred

The first 61 entries per robot retain the walking observation order: angular velocity, gravity,
14 servo-plus-play position offsets, 14 servo-plus-play velocities, 14 previous actions, and 13 command
slots. Velocity/head/body commands are zero for these tasks. IMU mounting errors, encoder bias, delayed
observations, pushes, and the walking curricula are initially omitted. BAM actuator delays, battery
behavior, and mechanical backlash remain part of the asset. This preserves the input interface, while
changing its training distribution and the collision geometry; a stance check is still necessary.

Ten entries are appended: ball position, ball velocity relative to root translation, receiver position,
all expressed in the robot root frame, and a next-receiver indicator. These are **simulator-state
observations**, not camera vision. Camera perception/distillation is a later stage after physical
heading and rallies work.

The actor's weights and source normalization are copied. Additional ball channels start with identity
normalization and zero incoming weights. Pair layers start block diagonal, so both robots reproduce
the source actor before training. All actor parameters, including cross-robot connections, and all
leg/neck/head actions are trainable. The pair is a centralized cooperative controller; this is not
decentralized multi-agent competition. The source activation must be ELU with 512/256/128 hidden layers
and the source joint order must match the supplied MicroDuck walking/heading tasks.

The critic and optimizer start afresh; iteration resets to zero. The action standard deviation starts
at 0.15 radians and can be changed with the conversion utility's `--initial_std`. For converted
checkpoints, `--no-resume_optimizer_state` is required. The CPU regression tests load the results into
the real RSL-RL models and check inference preservation and that new connections can learn:

```bash
uv run python -m pytest scripts/tools/test/test_prepare_microduck_ball_checkpoint.py -q
```

For small steps of difficulty, increase `env.serve_spread` from `0.01` only after successful convenient
serves, and use `env.separation` to change robot spacing. Use one change per experiment. These are
configuration overrides, for example `env.serve_spread=0.02`; confirm playback before scaling further.
There is no automatic difficulty curriculum or pretrained ball-playing policy in this change.
