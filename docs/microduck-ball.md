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
uv run --extra test python -m pytest source/isaaclab_tasks/test/test_microduck_ball.py -q
```

The startup report prints the rigid head body and measured camera-forward axis in world coordinates.
The test drops a ball onto the head and must detect a real collision (a roof contact is physically
detected but is not a legal strike). Resolve sensor, asset, collision-capacity, or non-finite-state errors
before starting a long training run.

Prepare the standing actor for the heading task:

```bash
uv run python scripts/tools/prepare_microduck_ball_checkpoint.py \
  logs/rsl_rl/microduck_velocity_flat_backlash/2026-10-23_14-05-46/model_4000.pt \
  logs/microduck_initialization/heading.pt --players 1
```

This conversion is required: copying the original standing checkpoint alone does not create
`logs/microduck_initialization/heading.pt`. The output is a new file; the utility refuses to overwrite
an existing checkpoint. Start with a small
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
`invalid_contact`, and `gaze_alignment` alongside playback. `head_contacts` includes physical contacts
that were rejected as strikes. The virtual target is fixed at the neutral forehead height during
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

`Metrics/returns` counts alternating legal strikes after the initial serve. A strike must contact the
front forehead patch **above the lens**, with the camera raised towards the ball. The MicroDuck USD's
rigid `jaw_soft` frame has camera-forward along local **-Z**, and forehead-up along local **+X**.
The patch bounds are `(0.027, -0.04, -0.10)` to `(0.045, 0.04, -0.05)` metres in that frame, measured
against the asset geometry. A physical contact point, a camera-to-ball cosine of at least `0.5`, and
a camera-forward/world-up dot product of at least `0.25` are required. The geometric forward cone is
a camera-direction proxy, not an RGB sensor or an occlusion test. Onset geometry is sampled each
5 ms physics step, so a brief contact cannot be credited using a later head orientation.

There is no minimum flight time for a legal reply from the other robot: contact onset and alternating
turn order prevent repeated scoring. `min_transfer_time` limits only the predicted flight horizon for
the aiming bonus. Robot-to-robot collisions are sampled separately at every physics step and end the
rally. Strikes in a collision or fall step earn no positive strike/aim/return bonus; collision uses the
existing fall cost. These changes do not freeze any joints or change the policy's input/action sizes.

Standing with the camera pointed forward and bouncing the ball off the upper lid does not count.
The look-at-ball reward encourages the neck/head to aim at the approaching ball. All leg joints remain
trainable; stepping, crouching, jumping, and landing are allowed by the controller. There is no reward
for jumping alone, no airborne-foot penalty, and no prescribed jump animation. Whether the robot
learns a useful jump has to be established in GPU rollouts.

Repeated hits by the same robot, roof/lens contacts, simultaneous ambiguous hits, non-head ball contacts,
falling, leaving a half-court, or dropping the ball end a rally. In the pair task, ball misses
start a post-rally recovery phase before the world resets; robot safety failures reset immediately.
The first head contact is a hit, not a completed transfer. Rewards include sparse
head-hit/return bonuses, a ballistic aiming bonus, small positioning/gaze/upright terms, and costs for abrupt
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
uv run --extra test python -m pytest scripts/tools/test/test_prepare_microduck_ball_checkpoint.py -q
uv run --extra test python -m pytest scripts/tools/test/test_microduck_heading_geometry.py -q
```

For small steps of difficulty, increase `env.serve_spread` from `0.01` only after successful convenient
serves, and use `env.separation` to change robot spacing. Use one change per experiment. These are
configuration overrides, for example `env.serve_spread=0.02`; confirm playback before scaling further.
There is no automatic difficulty curriculum or pretrained ball-playing policy in this change.

## Diagnose short rallies

Use the following episode metrics alongside deterministic playback:

- `returns`: mean legal replies after the serve; two total strikes correspond to one return.
- `rally_2_returns`, `rally_4_returns`, `rally_8_returns`: fractions of episodes reaching that many replies.
- `invalid_head_surface`: the head touched the ball outside the raised, camera-facing forehead rule.
- `invalid_turn`: a duplicate hit or a hit by the wrong robot.
- `invalid_body_contact`: the ball hit another robot body.
- `robot_collision`: robots touched one another.
- `ball_ground`, `out_of_bounds`, `fell`: other physical ending conditions.
- `ball_out_of_bounds`, `robot_out_of_bounds`, `crossed_center`: the three components of `out_of_bounds`.
- `fast_returns`: legal replies arriving sooner than the aiming horizon, formerly rejected by the 80 ms gate.

Failure flags can overlap in the same physics step. A low `fell` value alone does not prove balance:
an invalid contact can end the episode before the original 70-degree tilt limit is reached.
Do not start another long run based only on total reward. After updating the rules, keep the old
checkpoint, run the GPU smoke test, then evaluate that checkpoint and inspect the failure breakdown.
The collision/turn changes are an exploratory rules correction, not a verified improvement to the
learned policy. Checkpoint tensors remain compatible; changing the rules still requires GPU rollout
and learning validation.

To try fine-tuning a saved pair under the corrected rules without overwriting its source run:

```bash
VOLLEY_CHECKPOINT="$(ls -t logs/rsl_rl/microduck_volley/*/model_*.pt | head -n 1)"

uv run --extra viser isaaclab play \
  --rl_library rsl_rl --task IsaacContrib-Volley-MicroDuck \
  --num_envs 4 --viz viser --checkpoint "$VOLLEY_CHECKPOINT"

# After the physical smoke/evaluation checks, try 2000 iterations before deciding on a longer run.
uv run --extra viser isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Volley-MicroDuck \
  --experiment_name microduck_volley_contact_fix \
  --num_envs 1024 --max_visible_envs 4 --max_iterations 2000 --viz viser \
  --checkpoint "$VOLLEY_CHECKPOINT" --no-resume_optimizer_state
```

This restores the pair's actor/critic weights with a fresh optimizer, rather than converting the
already-centralized pair again. Use an explicit checkpoint path from the new experiment on later
play/resume commands, since its experiment directory differs from the task's default.

## Check scene scale and passive ball physics before another long run

### Trace inaccurate second strikes and subsequent collapse

When video shows a repeatable first transfer but a poor second strike, collect one-world
inference traces before changing rewards again. This tool uses the task's current default
scene/rewards/reset rules, loads only the saved RSL-RL actor, and performs no learning.
It uses no visualizer and writes local reports without modifying the checkpoint:

```bash
CHECKPOINT="$(ls -t logs/rsl_rl/microduck_volley_post_rally_20k/*/model_*.pt | head -n 1)"
uv run python scripts/tools/trace_microduck_volley.py \
  --checkpoint "$CHECKPOINT" --episodes 30
```

Outputs are `logs/microduck_volley_trace.json` and the matching `.csv`. The CSV includes
5 ms physics states, candidate strike onsets, root positions/velocities, torso tilt,
head orientations/positions, initial forehead targets, phase flags, and pre-reset
terminal snapshots. The JSON compares first/second/later candidate strikes and includes
per-strike details and overlapping terminal-reason fractions. Default seed is 42;
record the checkout revision with the reports, since it selects the current environment rules.

The projected landing error is a diagnostic prediction towards the receiver's **initial**
forehead position, rather than its possibly displaced current head. Velocity is sampled
at least 20 ms after impact; an intervening head-contact onset, terminal, or reset makes
that estimate unavailable. Ongoing contact at the sample time and receiver motion can
invalidate a free-flight approximation. Candidate hits are sampled before control-step
safety checks, which may cancel their actual reward. Simultaneous candidate strikes are
reported as ambiguous and excluded from first-versus-second comparisons. Torso tilt/drift
within 250 ms after a strike includes later strikes but cannot cross an episode reset.
Use the raw CSV to locate the sequence of ball misdirection, torso acceleration, center
crossing and falling; the summary alone does not establish which event caused another.

### Fine-tune recovery on the original scene

The volley task pays only `0.2` immediately for each legal strike. The other `0.8`,
the aiming bonus (up to `2`), and the return bonus (`5` after the first strike) are
banked for the player who made that strike. They are paid once after at least
`0.12 s` since that player's latest strike and `0.10 s` of continuous recovery:

- Torso tilt at most 20 degrees, combined roll/pitch speed at most `1.5 rad/s`.
- Root height from `0.6 * initial height` to `initial height + 0.03 m`.
- Absolute vertical speed at most `0.25 m/s`.

The height/speed checks are a landing proxy, not a foot-to-floor contact measurement.
Jumping and head motion remain unrestricted; airborne motion simply delays payment.
A new own strike restarts the recovery clock and retains banked credit; a partner's
strike does not restart it. A ball miss or any terminal step cancels all unpaid
credit, taking priority over recovery on that same step. Reset
clears the bank only for the affected worlds. Previously paid rewards are not revoked
if the robot falls later; the recovery window is finite.

Two legal strikes with no recovery now receive only `0.4` in immediate strike bonuses.
If followed by a registered fall/collision/robot court exit (`-2`), their sparse subtotal
is `-1.6`, before dense terms. Two fully recovered strikes still receive `7` to `11`
in total strike/aim/return bonuses, before discounting. A ball exit alone has no `-2`
fall cost, but forfeits outstanding credit and incurs the separate miss cost described below.
This discourages the observed shortcut;
it does not prove that a learned policy will recover or that every short rally loses reward.
The single-player heading task keeps its immediate strike rewards.

The volley task also charges a continuous cost for the least balanced player's torso:
`10 * (1 - cos(tilt)) + 0.05 * (roll_rate**2 + pitch_rate**2)` per second.
Each term takes the maximum over both players. Head orientation, yaw, root height,
and vertical speed are not penalized by these terms; leg control and jumping remain available.
Robot court exits and center-line violations also incur the existing fall cost and suppress
positive strike bonuses on that step.
This is an experimental reward correction, not a GPU-verified recovery skill.

### Observe recovery after the rally ends

In the pair task, ball-ground contacts, ball exits, and invalid ball contacts now stop the
rally but keep simulation running for `0.6 s`. Ball strike scoring and ball-tracking rewards
stop, unpaid strike credit is cancelled immediately, and each player's existing turn flag
becomes `-1` to identify the recovery phase. Observation/action dimensions remain unchanged.
No serve or ball teleport occurs during the phase; the ball continues under normal physics.

Falls, robot collisions, robot court exits, and nonfinite state still terminate immediately,
including during recovery. The usual balance/action costs remain active. A miss costs `0.5`
once, including a miss before the first valid strike. At the phase deadline, both robots
must have satisfied the recovery envelope continuously during the final `0.2 s` to earn
`0.5`, and at least one legal strike must have occurred. The phase never ends early merely
because the robots briefly look stable. Timeout cancels the completion bonus.
Recovery without a legal strike earns no completion bonus; deliberate miss-and-recover
cannot earn a positive net miss/completion bonus. Previously paid strike credit is not revoked.

`Metrics/fell_after_miss` reveals falls that previously disappeared behind a ball-miss reset.
`Metrics/post_rally_phase` is the fraction of episodes that entered this phase;
`Metrics/post_rally_recovered` is the fraction ending it with both robots stable, even if
no valid strike occurred. It is not the completion reward itself.
`Metrics/rally_length_s` excludes the added phase: total episode length will increase
mechanically and must not be treated as longer ball rallies. Ball/invalid-contact failure
metrics retain their original causes across the phase; safety failure metrics can overlap.

Keep the original spacing and serve while testing this objective. Start with a short continuation
from the latest recovery pair, with a fresh optimizer in a separate experiment. First run
the GPU integration test (its volley case now checks miss-to-recovery-to-fall reset):

```bash
uv run --extra test python -m pytest source/isaaclab_tasks/test/test_microduck_ball.py -q
```

```bash
CHECKPOINT="$(ls -t logs/rsl_rl/microduck_volley_recovery/*/model_*.pt | head -n 1)"
nohup uv run --extra viser isaaclab train \
  --rl_library rsl_rl --task IsaacContrib-Volley-MicroDuck \
  --experiment_name microduck_volley_post_rally \
  --num_envs 1024 --max_visible_envs 4 --max_iterations 200 --viz viser \
  --checkpoint "$CHECKPOINT" --no-resume_optimizer_state \
  env.separation=0.35 env.serve_height=0.25 env.serve_down_speed=0.2 \
  > volley_post_rally.log 2>&1 < /dev/null &
```

Inspect a checkpoint from `logs/rsl_rl/microduck_volley_post_rally/` explicitly during playback.
`Metrics/recovered_hits` counts strikes whose banked credit was paid; compare it with
`Metrics/head_hits`. `Metrics/recovery_reward` is the mean sum of paid banked credit per episode.
Compare returns, rally counts, episode length, robot exits, center crossings, and falling,
alongside visible recovery after each strike. Standing still with fewer returns is not success.
Do not widen the scene or change ball physics in this comparison. To reproduce the old volley
reward for an ablation, set `env.defer_strike_rewards=false` and both torso costs to zero;
also set `env.post_rally_duration=0`. The boundary failure correction still applies.
Observation/action sizes are unchanged, so the old
pair checkpoint remains loadable; reward semantics changed, so use a fresh optimizer and a new run.

The original small-scene settings use 0.35 m root separation and a serve 0.25 m above the forehead,
already moving down at 0.2 m/s. Ignoring the small contact radius/forehead-shape correction, free fall
through 0.25 m takes about 0.206 s and arrives at about 2.22 m/s. This is only about ten policy updates
at 50 Hz. Close robot spacing also permits quick exchanges while lunging towards the center. A wider
court is an experiment requiring policy adaptation, not a guaranteed fix to an already learned lunge.

First measure the **same** ball mass, collider, contact parameters, solver, and floor without robots,
policy actions, reward logic, or automatic episode resets:

```bash
uv run python scripts/tools/check_microduck_ball_physics.py
```

The utility prints a JSON report and saves `logs/microduck_ball_physics.json` and a `.csv` trace.
It reports actual mass/inertia, acceleration, ballistic position error, sampled restitution, and first
rebound height. A passing report establishes basic passive physical consistency; it does not establish
that the ball is calibrated to a real object or that moving-head contacts are correct. A failed report
must be inspected before contact parameters are tuned. In particular, non-positive inertia, incorrect
gravity, or energy gain in a passive bounce cannot be attributed to the policy.

For a timestep comparison, keep all other parameters fixed:

```bash
uv run python scripts/tools/check_microduck_ball_physics.py \
  --dt 0.0025 --output logs/microduck_ball_physics_half_dt.json
```

After inspecting that report, use a single-environment playback to inspect more breathing room and
a higher, initially stationary serve:

```bash
CHECKPOINT="$(ls -t logs/rsl_rl/microduck_volley_contact_fix/*/model_*.pt | head -n 1)"

uv run --extra viser isaaclab play \
  --rl_library rsl_rl --task IsaacContrib-Volley-MicroDuck \
  --num_envs 1 --viz viser --checkpoint "$CHECKPOINT" \
  env.separation=0.5 env.serve_height=0.45 env.serve_down_speed=0.0
```

This changes scene geometry/serve timing together for visual inspection, so it is exploratory rather
than an isolated learning comparison. It does not change ball mass, contact damping, gravity, or fix
the robot legs. The stationary 0.45 m drop gives about 0.303 s before the nominal forehead height but
also increases impact speed to about 2.97 m/s. A higher serve is not automatically an easier strike.
Do not replace the existing checkpoint or start another long run solely because the new scene looks
more spacious. Use the physical report, boundary breakdown, and recovery behavior to select the next
single-variable training experiment.
