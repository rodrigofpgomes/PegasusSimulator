#!/usr/bin/env bash
# Shuttle, direct rotor velocity: generalisation to a figure-eight (lemniscate) reference with increasing acceleration.
#
# Task shuttle/direct_rotor/ablation/raptor_tracking with eval_reference=lemniscate: p(s) = c + A (sin s, 0.5 sin 2s, 0),
# A = 1 m, the speed ramped up from rest over 2 s, 20 s per episode, peak reference acceleration a_peak (sets the
# frequency). 100 vehicles start from the training initial states (seed 1000) and the same reference. The policies
# were trained on references with at most 1.4 m/s^2 (fixed point or Langevin), so a_peak above that is generalisation.
# Policies (all with the 26-component observation of the baseline unless noted):
#   PPO baseline seeds 3, 1, 2 (B0, B0_s1, B0_s2), R5, R6, R7, R8, R9, O1 (+ reference acceleration), O2 (R_d^T R),
#   A1 (B0 reward + reference acceleration), A2 (R6 reward + reference acceleration)
#   SAC (skrl, seed 1) and the RAPTOR pre-training actor (rl-tools checkpoint, play.py --controller pretrain)
# Output: $OUT/<policy>/a<a_peak>/ (steps.npz, episodes.csv, .done) and $OUT/summary.csv (summarize_tracking_eval.py).
# Finished evaluations (.done) are skipped on restart; about 45 s per evaluation.
#
# Usage (inside tmux, from the repository root):
#   tmux new -s lemniscate 'examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_lemniscate.sh 2>&1 | tee lemniscate.out'
#   POLICIES="B0 SAC PRETRAIN" LEVELS="1 4" examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_lemniscate.sh
#   examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_lemniscate.sh --dry-run
# Environment: POLICIES, LEVELS (default "0.5 1 2 3 4 5"), N_EVAL_ENVS (100), DURATION (20), EVAL_SEED (1000), AMPLITUDE (1.0),
#   MAX_LIN_VEL (default: the training bound, 2 m/s per axis; with a value the results go to eval_lemniscate_v<value>), OUT_NAME.
# The training bound of 2 m/s per axis ends an episode when the vehicle overshoots the speed of a fast reference
# (a_peak = 5: reference 1.5 m/s per axis), which is not a loss of control; for higher speeds use MAX_LIN_VEL, e.g.
#   MAX_LIN_VEL=5 LEVELS="5 6 8 10" examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_lemniscate.sh
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/ablation/raptor_tracking"
BASE_LOGS="$REPO/examples/rl/tasks/shuttle/direct_rotor/baseline/raptor_tracking/logs"
ABL_LOGS="$REPO/examples/rl/tasks/$TASK/logs"
MAX_VEL="${MAX_LIN_VEL:-}"   # per-axis inertial speed bound of the task [m/s]; the training value (2) when empty
OUT="$ABL_LOGS/${OUT_NAME:-eval_lemniscate${MAX_VEL:+_v$MAX_VEL}}"
VEL_ARGS=(); [ -n "$MAX_VEL" ] && VEL_ARGS=(--env-cfg "max_lin_vel_per_axis=$MAX_VEL")
POLICIES="${POLICIES:-B0 B0_s1 B0_s2 R5 R6 R7 R8 R9 O1 O2 A1 A2 SAC PRETRAIN}"
LEVELS="${LEVELS:-0.5 1 2 3 4 5}"
N="${N_EVAL_ENVS:-100}"; DUR="${DURATION:-20}"; SEED="${EVAL_SEED:-1000}"; AMP="${AMPLITUDE:-1.0}"
DYN="$HOME/raptor/rl-tools/src/foundation_policy/dynamics_parameters_shuttle/shuttle.json"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
export PYTHONUNBUFFERED=1; cd "$REPO"; failed=0

latest() { ls -d "$1"/*_"$2" 2>/dev/null | sort | tail -1; }

# policy_args <name> -> sets ARGS (play.py arguments selecting the policy), or returns 1 when it is not available
policy_args() {
  local rd ck
  case "$1" in
    B0)    rd="$(latest "$BASE_LOGS/ppo2_tanh_600_std0p2/ppo2/train/ppo2_s03" PPO2)" ;;
    B0_s1) rd="$(latest "$BASE_LOGS/ppo2_tanh_600_std0p2/ppo2/train/ppo2_s01" PPO2)" ;;
    B0_s2) rd="$(latest "$BASE_LOGS/ppo2_tanh_600_std0p2/ppo2/train/ppo2_s02" PPO2)" ;;
    R5|R6|R7|R8|R9|O1|O2|A1|A2) v="$1"; case "$1" in R6|R7|R8|R9|A1|A2) v="${1}_s03";; esac
                         rd="$(latest "$ABL_LOGS/$v/train" PPO2)" ;;
    SAC)   rd="$(latest "$BASE_LOGS/sac_vs_ppo2/sac/train/sac_s01" SACDelayed_delay_2)"
           ck="$(ls "$rd"/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)"
           [ -n "$ck" ] || return 1
           ARGS=(--algo sac --preset default --checkpoint "$ck" --run-metadata "$rd/run_metadata.json"); return 0 ;;
    PRETRAIN) ck="$(ls "$HOME"/raptor/rl-tools/experiments/shuttle_pretrain/*/shuttle/0000/steps/000000001000000/checkpoint.h5 2>/dev/null | head -1)"
           [ -n "$ck" ] || return 1
           ARGS=(--algo ppo2 --preset tanh --controller pretrain --checkpoint "$ck" --pretrain-dynamics "$DYN"); return 0 ;;
    *) return 1 ;;
  esac
  [ -f "$rd/checkpoints/agent_14400.pt" ] || return 1
  ARGS=(--algo ppo2 --preset tanh --checkpoint "$rd/checkpoints/agent_14400.pt" --run-metadata "$rd/run_metadata.json")
}

for pol in $POLICIES; do
  if ! policy_args "$pol"; then echo "[FAIL] $pol: policy not found"; failed=$((failed+1)); continue; fi
  for a in $LEVELS; do
    out="$OUT/$pol/a$a"
    [ -f "$out/.done" ] && { echo "[skip ] $pol a=$a"; continue; }
    echo "[eval ] $pol a_peak=$a"; [ "$DRY" = 1 ] && continue
    rm -rf "$out"; mkdir -p "$out"
    "$PY" examples/rl/play.py --task "$TASK" "${ARGS[@]}" --headless --n_envs "$N" --num_episodes_per_env 1 --seed "$SEED" \
      --train-init-state --env-cfg eval_reference=lemniscate --env-cfg "episode_length_s=$DUR" --env-cfg "lem_peak_acc=$a" \
      --env-cfg "lem_amplitude=$AMP" "${VEL_ARGS[@]}" --metrics-dir "$out" > "$out/stdout.log" 2>&1
    if [ $? -eq 0 ] && [ -f "$out/steps.npz" ]; then touch "$out/.done"; else echo "[FAIL] $pol a=$a"; failed=$((failed+1)); fi
  done
done
[ "$DRY" = 0 ] && python3 "$REPO/examples/rl/scripts/shuttle/direct_rotor/summarize_tracking_eval.py" "$OUT" --csv "$OUT/summary.csv"
echo "finished; failed: $failed (results: $OUT)"; [ $failed -eq 0 ]
