#!/usr/bin/env bash
# Shuttle, direct rotor velocity: re-evaluate final policies with the attitude quaternion recorded (steps.npz: quat),
# so that the heading (yaw) can be analysed. Same protocol as the original evaluations (500 episodes, training initial
# states, seed 1000), written to a separate folder (eval_yaw) to compare with the originals.
#   baseline seed 3  -> baseline task logs/ppo2_tanh_600_std0p2/ppo2/eval_yaw/ppo2_s03
#   ablation runs    -> ablation task logs/<run>/eval_yaw
# Usage: examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_yaw.sh [RUN ...]   (default: B0 R1 R5 R6_s03 R7_s03 R8_s03 R9_s03)
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
BASE_TASK="shuttle/direct_rotor/baseline/raptor_tracking"; ABL_TASK="shuttle/direct_rotor/ablation/raptor_tracking"
BASE_LOGS="$REPO/examples/rl/tasks/$BASE_TASK/logs"; ABL_LOGS="$REPO/examples/rl/tasks/$ABL_TASK/logs"
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(B0 R1 R5 R6_s03 R7_s03 R8_s03 R9_s03)
export PYTHONUNBUFFERED=1; cd "$REPO"; failed=0
for r in "${RUNS[@]}"; do
  if [ "$r" = B0 ]; then task="$BASE_TASK"; train="$BASE_LOGS/ppo2_tanh_600_std0p2/ppo2/train/ppo2_s03"; out="$BASE_LOGS/ppo2_tanh_600_std0p2/ppo2/eval_yaw/ppo2_s03"
  else task="$ABL_TASK"; train="$ABL_LOGS/$r/train"; out="$ABL_LOGS/$r/eval_yaw"; fi
  [ -f "$out/.done" ] && { echo "[skip ] $r"; continue; }
  run_dir="$(ls -d "$train"/*_PPO2 2>/dev/null | sort | tail -1)"
  [ -f "$run_dir/checkpoints/agent_14400.pt" ] || { echo "[FAIL] $r: no checkpoint"; failed=$((failed+1)); continue; }
  rm -rf "$out"; mkdir -p "$out"; echo "[eval ] $r"
  "$PY" examples/rl/play.py --task "$task" --algo ppo2 --preset tanh --checkpoint "$run_dir/checkpoints/agent_14400.pt" \
    --run-metadata "$run_dir/run_metadata.json" --headless --n_envs 500 --num_episodes_per_env 1 --seed 1000 \
    --train-init-state --metrics-dir "$out" > "$out/stdout.log" 2>&1
  if [ $? -eq 0 ] && [ -f "$out/steps.npz" ]; then touch "$out/.done"; else echo "[FAIL] $r"; failed=$((failed+1)); fi
done
echo "finished; failed: $failed"; [ $failed -eq 0 ]
