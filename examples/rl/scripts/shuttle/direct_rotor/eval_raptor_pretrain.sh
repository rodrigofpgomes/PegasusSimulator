#!/usr/bin/env bash
# Shuttle, direct rotor velocity: evaluate the RAPTOR pre-training actor (rl-tools SAC, Shuttle dynamics, 1e6 steps)
# with the protocol of the PPO2/SAC policies (row "RAPTOR pre-training" of the direct rotor results).
#
# play.py --controller pretrain feeds the task observation (26 inputs, rl-tools layout) to the actor of the HDF5
# checkpoint; the rotor order is matched to the dynamics json by position and spin (identity for the Shuttle).
# 500 episodes, training initial states, seed 1000, so the episodes are those of every other policy.
# Output: baseline task logs/eval_reference/raptor_pretrain (steps.npz, episodes.csv, eval_config.json, stdout.log),
# read by results/shuttle/rotor_velocities/export_results.py. The actor is trained with rl-tools (RAPTOR_DIR, default
# ~/raptor; see REPRODUCE.md): foundation_policy_pre_training with examples/rl/configs/raptor/shuttle.json.
#
# Usage: examples/rl/scripts/shuttle/direct_rotor/eval_raptor_pretrain.sh
# Environment: CKPT (default: the 1e6-step checkpoint of experiments/shuttle_pretrain), DYN, OUT.
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/baseline/raptor_tracking"
LOGS="$REPO/examples/rl/tasks/$TASK/logs"
RLTOOLS="${RAPTOR_DIR:-$HOME/raptor}/rl-tools"
CKPT="${CKPT:-$(ls "$RLTOOLS"/experiments/shuttle_pretrain/*/shuttle/0000/steps/000000001000000/checkpoint.h5 2>/dev/null | head -1)}"
DYN="${DYN:-$REPO/examples/rl/configs/raptor/shuttle.json}"
OUT="${OUT:-$LOGS/eval_reference/raptor_pretrain}"
[ -f "$CKPT" ] || { echo "pre-training checkpoint not found: '$CKPT'"; exit 1; }
[ -f "$DYN" ] || { echo "dynamics json not found: $DYN"; exit 1; }

export PYTHONUNBUFFERED=1; cd "$REPO"
mkdir -p "$OUT"
# --algo/--preset only build the task's agent interface; the actions come from the pre-training actor
"$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset tanh --controller pretrain \
  --checkpoint "$CKPT" --pretrain-dynamics "$DYN" --headless --n_envs 500 --num_episodes_per_env 1 --seed 1000 \
  --train-init-state --metrics-dir "$OUT" > "$OUT/stdout.log" 2>&1
code=$?
if [ $code -eq 0 ] && [ -f "$OUT/steps.npz" ]; then
  python3 examples/rl/scripts/shuttle/direct_rotor/summarize_direct_rotor_eval.py "$OUT"
else
  echo "[FAIL] evaluation ($code), see $OUT/stdout.log"; exit 1
fi
