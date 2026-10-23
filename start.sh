uv run --extra viser isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-MicroDuck \
  --num_envs 256 \
  --max_visible_envs 16 \
  --viz viser \
  presets=backlash
