#!/usr/bin/env bash
# Shuttle, direct rotor velocity (shuttle/direct_rotor/baseline/raptor_tracking): deterministic evaluation of
# intermediate checkpoints, for the training-evolution figures of results/shuttle/rotor_velocities.
#
# Same protocol as the final evaluations of run_ppo2_variant.sh (play.py, 500 episodes, training initial states,
# seed EVAL_SEED). The final checkpoints are not repeated: export_results.py reads them from the study folders.
#   PPO2 variants ppo2_default_600, ppo2_tanh_600, ppo2_tanh_600_std0p2, seeds 1-3: iterations PPO_ITERS
#   SAC sac_vs_ppo2/sac_s01: environment steps SAC_STEPS
# Each evaluation keeps metrics.csv (summarize_direct_rotor_eval.py), episodes.csv and eval_config.json; steps.npz
# (17 MB each) is deleted once the metrics are written.
# Layout: logs/eval_checkpoints/<study>/<run>/<it|step><N>/
#
# Usage (inside tmux):
#   tmux new -s ckpt_eval 'examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_checkpoints.sh 2>&1 | tee ckpt_eval.out'
#   examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_checkpoints.sh --dry-run
# Environment: PPO_ITERS, SAC_STEPS, N_EVAL_ENVS (500), EVAL_SEED (1000).
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/baseline/raptor_tracking"
LOGS="$REPO/examples/rl/tasks/$TASK/logs"
OUT="$LOGS/eval_checkpoints"
PPO_ITERS="${PPO_ITERS:-25 50 100 200 300 400}"
SAC_STEPS="${SAC_STEPS:-50000 100000 200000 300000 400000 500000 600000 700000 800000 900000}"
N_EVAL_ENVS="${N_EVAL_ENVS:-500}"
EVAL_SEED="${EVAL_SEED:-1000}"
SUMMARIZE="$REPO/examples/rl/scripts/shuttle/direct_rotor/summarize_direct_rotor_eval.py"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

export PYTHONUNBUFFERED=1
cd "$REPO"
[ "$DRY" = 0 ] && mkdir -p "$OUT"
echo "commit: $(git rev-parse --short HEAD); protocol: $N_EVAL_ENVS episodes, seed $EVAL_SEED, training initial states"

failed=0
# evaluate <algo> <preset> <run_dir> <checkpoint> <eval_dir>
evaluate() {
  local algo="$1" preset="$2" run_dir="$3" ckpt="$4" eval_dir="$5"
  if [ -f "$eval_dir/.done" ]; then echo "[skip ] ${eval_dir#$OUT/}"; return; fi
  echo "[eval ] ${eval_dir#$OUT/}: $(basename "$ckpt")"
  [ "$DRY" = 1 ] && return
  if [ ! -f "$ckpt" ]; then echo "[FAIL] missing $ckpt"; failed=$((failed+1)); return; fi
  rm -rf "$eval_dir"; mkdir -p "$eval_dir"
  "$PY" examples/rl/play.py --task "$TASK" --algo "$algo" --preset "$preset" --checkpoint "$ckpt" \
      --run-metadata "$run_dir/run_metadata.json" --headless --n_envs "$N_EVAL_ENVS" --num_episodes_per_env 1 \
      --seed "$EVAL_SEED" --train-init-state --metrics-dir "$eval_dir" > "$eval_dir/stdout.log" 2>&1
  local code=$?
  if [ $code -eq 0 ] && [ -f "$eval_dir/steps.npz" ] \
     && python3 "$SUMMARIZE" "$eval_dir" --csv "$eval_dir/metrics.csv" > /dev/null; then
    rm -f "$eval_dir/steps.npz"; touch "$eval_dir/.done"
  else echo "[FAIL] ${eval_dir#$OUT/} (exit $code)"; failed=$((failed+1)); fi
}

# PPO2: one checkpoint every 25 iterations; skrl names it after the timestep (24 per iteration)
for study in ppo2_default_600 ppo2_tanh_600 ppo2_tanh_600_std0p2; do
  preset=default; [[ "$study" == ppo2_tanh* ]] && preset=tanh
  for seed in 1 2 3; do
    run="ppo2_s0$seed"
    run_dir="$(ls -d "$LOGS/$study/ppo2/train/$run"/*_PPO2 2>/dev/null | tail -1)"
    for it in $PPO_ITERS; do
      evaluate ppo2 "$preset" "$run_dir" "$run_dir/checkpoints/agent_$((24 * it)).pt" "$OUT/$study/$run/it$it"
    done
  done
done

# SAC: one vehicle, one checkpoint every 10^4 environment steps
sac_run="$(ls -d "$LOGS"/sac_vs_ppo2/sac/train/sac_s01/*_SACDelayed* 2>/dev/null | tail -1)"
for step in $SAC_STEPS; do
  evaluate sac default "$sac_run" "$sac_run/checkpoints/agent_$step.pt" "$OUT/sac_vs_ppo2/sac_s01/step$step"
done

echo "checkpoint evaluation finished; failed: $failed"
[ $failed -eq 0 ]
