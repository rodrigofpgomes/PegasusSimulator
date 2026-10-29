#!/usr/bin/env bash
# Shuttle, direct thrust and moments (shuttle/thrust_moment/isaac_lab): generalisation of the policies of the
# final study to the figure-eight reference of the direct rotor section.
#
# Configuration EXPERIMENT (default ms012_den08_yaw101: c_M = 0.12 N m, denominator 0.8 m, yaw-rate weight 1.01), ten
# seeds, final checkpoint (iteration 200). The goal buffers of the task are moved along the same figure eight as in
# eval_direct_rotor_lemniscate.sh (play.py --goal-lemniscate): A = 1 m, speed ramped from rest over 2 s, 20 s per
# episode, peak acceleration a_peak, around the spawn point (altitude 1 m). The policies start in the nominal state, as
# in their own evaluation; the policy step is 20 ms (50 Hz). The goal velocity is the velocity of the reference, which
# the observation subtracts from the vehicle velocity. FM_s03_v0 is seed 3 with the goal velocity kept at zero, the
# value of its training. The task ends an episode only when the altitude leaves [0.1, 2] m.
# Output: $OUT/FM_sNN/a<a_peak>/ and $OUT/summary.csv (summarize_tracking_eval.py); finished evaluations (.done) are skipped.
#
# Usage (from the repository root, inside tmux or nohup):
#   examples/rl/scripts/shuttle/direct_rotor/eval_force_moment_lemniscate.sh
#   SEEDS="3" LEVELS="2" examples/rl/scripts/shuttle/direct_rotor/eval_force_moment_lemniscate.sh --dry-run
# Environment: EXPERIMENT, STUDY (logs folder, default study_10seeds; study_10seeds_kt3p9 for k_T = 3.9), OUT (set it for any EXPERIMENT other than the default), SEEDS (default 1..10), LEVELS (default "0.5 1 2 3 4 5 6 8 10"), N_EVAL_ENVS (4), DURATION (20),
#              EVAL_SEED (1000), ZERO_VEL_SEEDS (default "3", empty for none).
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/thrust_moment/isaac_lab"
EXPERIMENT="${EXPERIMENT:-ms012_den08_yaw101}"
STUDY="${STUDY:-study_10seeds}"
SRC="$REPO/examples/rl/tasks/$TASK/logs/$STUDY/$EXPERIMENT/train"
OUT="${OUT:-$REPO/examples/rl/tasks/shuttle/direct_rotor/ablation/raptor_tracking/logs/eval_lemniscate_fm}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"; LEVELS="${LEVELS:-0.5 1 2 3 4 5 6 8 10}"
N="${N_EVAL_ENVS:-4}"; DUR="${DURATION:-20}"; SEED="${EVAL_SEED:-1000}"; ZERO="${ZERO_VEL_SEEDS-3}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
export PYTHONUNBUFFERED=1; cd "$REPO"; failed=0

run() {   # run <policy name> <seed> <extra play.py args...>
  local name="$1" seed="$2"; shift 2
  local rd; rd="$(ls -d "$SRC/${EXPERIMENT}_s$(printf '%02d' "$seed")"/*_PPO2 2>/dev/null | sort | tail -1)"
  [ -f "$rd/checkpoints/agent_4800.pt" ] || { echo "[FAIL] $name: no checkpoint"; failed=$((failed+1)); return; }
  for a in $LEVELS; do
    local out="$OUT/$name/a$a"
    [ -f "$out/.done" ] && { echo "[skip ] $name a=$a"; continue; }
    echo "[eval ] $name a_peak=$a"; [ "$DRY" = 1 ] && continue
    rm -rf "$out"; mkdir -p "$out"
    "$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset isaac_lab --checkpoint "$rd/checkpoints/agent_4800.pt" \
      --run-metadata "$rd/run_metadata.json" --headless --n_envs "$N" --num_episodes_per_env 1 --seed "$SEED" \
      --goal-lemniscate --lem-peak-acc "$a" --env-cfg "episode_length_s=$DUR" "$@" --metrics-dir "$out" > "$out/stdout.log" 2>&1
    if [ $? -eq 0 ] && [ -f "$out/steps.npz" ]; then touch "$out/.done"; else echo "[FAIL] $name a=$a"; failed=$((failed+1)); fi
  done
}

for s in $SEEDS; do run "FM_s$(printf '%02d' "$s")" "$s"; done
for s in $ZERO; do run "FM_s$(printf '%02d' "$s")_v0" "$s" --lem-zero-vel; done
[ "$DRY" = 0 ] && python3 "$REPO/examples/rl/scripts/shuttle/direct_rotor/summarize_tracking_eval.py" "$OUT" --csv "$OUT/summary.csv"
echo "finished; failed: $failed (results: $OUT)"; [ $failed -eq 0 ]
