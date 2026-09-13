#!/usr/bin/env bash
# Re-run both eyes on the curated compact object set (the wrench-family
# planar degeneracy excluded and reported separately).
set +e
cd /home/bryan/MontySAI/tbp.monty
export MUJOCO_GL=egl
OUT=/home/bryan/MontySAI/data/overnight
run () { local log="$OUT/$1.log"; shift
  echo "=== $(date +%H:%M) :: $* ===" | tee -a "$OUT/curated.log"
  uv run python run.py "$@" > "$log" 2>&1
  echo "    exit $? -> $log" | tee -a "$OUT/curated.log"; }

run r2_pretrain experiment=batch_retina_pretrain
for t in upright scaled rotated tilted; do
  run "r2_$t" experiment=batch_retina_eval env_interface=batch_eval_$t \
      experiment.config.logging.run_name=cur_retina_$t
done
run c2_pretrain experiment=batch_cam_pretrain
for t in upright scaled rotated tilted; do
  run "c2_$t" experiment=batch_cam_eval env_interface=batch_eval_$t \
      experiment.config.logging.run_name=cur_cam_$t
done
echo "=== CURATED BATCH DONE $(date +%H:%M) ===" | tee -a "$OUT/curated.log"
