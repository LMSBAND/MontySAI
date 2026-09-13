#!/usr/bin/env bash
# Overnight battery: generalize the results across objects and sweep
# the knobs the Limitations section flagged. Robust to single failures.
set +e
cd /home/bryan/MontySAI/tbp.monty
export MUJOCO_GL=egl
OUT=/home/bryan/MontySAI/data/overnight
mkdir -p "$OUT"
run () {  # run <logname> <args...>
  local log="$OUT/$1.log"; shift
  echo "=== $(date +%H:%M) :: $* ===" | tee -a "$OUT/driver.log"
  uv run python run.py "$@" > "$log" 2>&1
  echo "    exit $? -> $log" | tee -a "$OUT/driver.log"
}

# ---- 1. object generalization: 12 objects, retina + metric eyes ----
run r_pretrain experiment=batch_retina_pretrain
for t in upright scaled rotated tilted; do
  run "r_$t" experiment=batch_retina_eval env_interface=batch_eval_$t \
      experiment.config.logging.run_name=batch_retina_$t
done
run c_pretrain experiment=batch_cam_pretrain
for t in upright scaled rotated tilted; do
  run "c_$t" experiment=batch_cam_eval env_interface=batch_eval_$t \
      experiment.config.logging.run_name=batch_cam_$t
done

# ---- 2. veto-fraction sweep (foils must stay 0 false) ----
for f in 0.10 0.15 0.20 0.25 0.30 0.40 0.50; do
  MONTY_VETO_FRAC=$f uv run python run.py experiment=triad_eval_consensus \
    env_interface=retina_eval_foils \
    experiment.config.logging.run_name=veto_$f > "$OUT/veto_$f.log" 2>&1
  echo "    veto $f exit $?" | tee -a "$OUT/driver.log"
done

# ---- 3. stretch-ratio sweep (find the breaking ratio) ----
for s in 13 22 28 35; do
  run "stretch_$s" experiment=retina_eval_scaled \
      env_interface=retina_eval_stretch$s \
      experiment.config.logging.run_name=stretch_$s
done

# ---- 4. flight-radius sweep (rooms recognized off the trained path) ----
for rad in 0.6 0.8 1.0 1.3; do
  uv run python run.py experiment=rooms_eval \
    +experiment.config.environment.env_init_args.patrol_radius=$rad \
    experiment.config.logging.run_name=radius_$rad > "$OUT/radius_$rad.log" 2>&1
  echo "    radius $rad exit $?" | tee -a "$OUT/driver.log"
done

echo "=== BATTERY DONE $(date +%H:%M) ===" | tee -a "$OUT/driver.log"
