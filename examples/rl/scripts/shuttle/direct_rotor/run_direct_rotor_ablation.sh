#!/usr/bin/env bash
# Shuttle, direct rotor velocity: reward and observation ablation of the PPO2 baseline (tanh + sigma_min 0.2).
#
# Task shuttle/direct_rotor/ablation/raptor_tracking (defaults = baseline). Every variant: preset "tanh",
# PPO2_STD_MIN=0.2, 600 iterations, one seed (SEED, default 3, the seed selected for the baseline), and the
# evaluation protocol of the baseline (play.py, 500 episodes, training initial states, seed 1000).
#
#   B0  baseline: q_z orientation cost 0.1     (already trained: baseline task, ppo2_tanh_600_std0p2, seed 3)
#   R1  velocity cost 0.1, no orientation cost
#   R2  velocity cost 0.3, no orientation cost
#   R3  0.5 tr(I - Rd^T R) cost 0.1             (Rd from the reference acceleration and yaw 0)
#   R4  0.5 tr(I - Rd^T R) cost 0.3
#   R5  best velocity weight (R1/R2) + best trace weight (R3/R4)
#   O1  best reward (B0, R1-R5) + reference acceleration in the observation
#   O2  best reward (B0, R1-R5) + attitude observed as Rd^T R
# "Best" = lowest final position error (err_final_cm). The choice is written to $ROOT/selection.txt and can be
# forced with R5_CFG / BEST_CFG (space-separated KEY=VALUE env-cfg items, e.g. BEST_CFG="ori_cost=trace w_ori=0.3").
#
# Layout ($ROOT = logs of the ablation task):
#   $ROOT/<variant>/train/   training run (run_metadata.json with the env-cfg snapshot, checkpoints, CSV export)
#   $ROOT/<variant>/eval/    play.py metrics (steps.npz, episodes.csv)
#   $ROOT/summary.csv        metrics of B0 and every variant;  $ROOT/progress.tsv  one line per finished step
# Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux, from the repository root):
#   tmux new -s ablation 'examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_ablation.sh 2>&1 | tee ablation.out'
#   examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_ablation.sh --dry-run
# Environment: SEED (3), ITERATIONS (600), STD_MIN (0.2), CKPT_ITERS (50), N_ENVS (4096), N_EVAL_ENVS (500),
#              EVAL_SEED (1000), VARIANTS (default "R1 R2 R3 R4 R5 O1 O2"), R5_CFG, BEST_CFG.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/ablation/raptor_tracking"
ROOT="$REPO/examples/rl/tasks/$TASK/logs"
B0_EVAL="$REPO/examples/rl/tasks/shuttle/direct_rotor/baseline/raptor_tracking/logs/ppo2_tanh_600_std0p2/ppo2/eval/ppo2_s03"
SUMMARIZE="$REPO/examples/rl/scripts/shuttle/direct_rotor/summarize_direct_rotor_eval.py"
SEED="${SEED:-3}"
ITERATIONS="${ITERATIONS:-600}"
STD_MIN="${STD_MIN:-0.2}"
N_ENVS="${N_ENVS:-4096}"
N_EVAL_ENVS="${N_EVAL_ENVS:-500}"
EVAL_SEED="${EVAL_SEED:-1000}"
VARIANTS="${VARIANTS:-R1 R2 R3 R4 R5 O1 O2}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

export PYTHONUNBUFFERED=1
export PPO2_STD_MIN="$STD_MIN"
export PPO2_CHECKPOINT_ITERS="${CKPT_ITERS:-50}"
PROGRESS="$ROOT/progress.tsv"
mkdir -p "$ROOT"
[ -f "$PROGRESS" ] || printf "variant\tstep\texit\tstart\tend\n" > "$PROGRESS"
cd "$REPO"
echo "commit: $(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files); seed=$SEED; iterations=$ITERATIONS; std_min=$STD_MIN; root=$ROOT"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$(date -Is)" >> "$PROGRESS"; }

# err_final_cm of one evaluation folder (empty if not evaluated)
err_final() {
  [ -f "$1/.done" ] || [ "$1" = "$B0_EVAL" ] || return 0
  python3 -c "import sys; sys.path.insert(0, '$(dirname "$SUMMARIZE")')
from pathlib import Path; from summarize_direct_rotor_eval import summarize
print(f\"{summarize(Path('$1'))['err_final_cm']:.3f}\")"
}

# best <variant>... : the variant with the lowest final error ("B0" = baseline)
best() {
  local best_v="" best_e="" v d e
  for v in "$@"; do
    if [ "$v" = B0 ]; then d="$B0_EVAL"; else d="$ROOT/$v/eval"; fi
    e="$(err_final "$d")"; [ -z "$e" ] && continue
    echo "  $v: err_final ${e} cm" >&2
    if [ -z "$best_e" ] || python3 -c "import sys; sys.exit(0 if $e < $best_e else 1)"; then best_v="$v"; best_e="$e"; fi
  done
  echo "$best_v"
}

# run_variant <variant> <env-cfg item>...
run_variant() {
  local v="$1"; shift
  local train_dir="$ROOT/$v/train" eval_dir="$ROOT/$v/eval" cfg=() item
  for item in "$@"; do cfg+=(--env-cfg "$item"); done
  echo "[$v] env-cfg: ${*:-(baseline)}"
  [ "$DRY" = 1 ] && return

  # A .done without the final checkpoint and a completed export is discarded
  if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1 \
       && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; }; then
    echo "[redo ] $v: incomplete training marked as done"; rm -f "$train_dir/.done" "$eval_dir/.done"
  fi
  if [ -f "$train_dir/.done" ]; then echo "[skip ] train $v"
  else
    rm -rf "$train_dir" "$eval_dir"; mkdir -p "$train_dir"
    printf "%s\n" "$@" > "$ROOT/$v/env_cfg.txt"
    local start; start=$(date -Is)
    "$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --preset tanh --headless --n_envs "$N_ENVS" \
      --seed "$SEED" --timesteps $((24 * ITERATIONS)) --log-dir "$train_dir" "${cfg[@]}" > "$train_dir/stdout.log" 2>&1
    local code=$?
    log_step "$v" train "$code" "$start"
    # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
    if [ $code -eq 0 ] && ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1 \
       && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; then
      touch "$train_dir/.done"
    else echo "[FAIL] train $v (exit $code)"; failed=$((failed+1)); return; fi
  fi

  if [ -f "$eval_dir/.done" ]; then echo "[skip ] eval $v"; return; fi
  local run_dir; run_dir="$(ls -d "$train_dir"/*_PPO2 | sort | tail -1)"
  rm -rf "$eval_dir"; mkdir -p "$eval_dir"
  local start; start=$(date -Is)
  # The run-metadata snapshot restores the env-cfg of the training run
  "$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset tanh \
    --checkpoint "$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt" --run-metadata "$run_dir/run_metadata.json" \
    --headless --n_envs "$N_EVAL_ENVS" --num_episodes_per_env 1 --seed "$EVAL_SEED" --train-init-state \
    --metrics-dir "$eval_dir" > "$eval_dir/stdout.log" 2>&1
  local code=$?
  log_step "$v" eval "$code" "$start"
  if [ $code -eq 0 ] && [ -f "$eval_dir/steps.npz" ]; then touch "$eval_dir/.done"
  else echo "[FAIL] eval $v ($code)"; failed=$((failed+1)); fi
}

declare -A CFG=(
  [B0]=""
  [R1]="ori_cost=none w_vel=0.1"
  [R2]="ori_cost=none w_vel=0.3"
  [R3]="ori_cost=trace w_ori=0.1"
  [R4]="ori_cost=trace w_ori=0.3"
)
wants() { [[ " $VARIANTS " == *" $1 "* ]]; }

# ---- 1. reward: velocity cost and trace orientation cost
for v in R1 R2 R3 R4; do wants "$v" && run_variant "$v" ${CFG[$v]}; done

# ---- 2. combination of the best weights
if [ "$DRY" = 1 ]; then CFG[R5]="ori_cost=trace w_ori=<R3|R4> w_vel=<R1|R2>"
elif [ -n "${R5_CFG:-}" ]; then CFG[R5]="$R5_CFG"
else
  echo "selection for R5:"
  bv="$(best R1 R2)"; bt="$(best R3 R4)"
  wv="$([ "$bv" = R2 ] && echo 0.3 || echo 0.1)"; wt="$([ "$bt" = R4 ] && echo 0.3 || echo 0.1)"
  CFG[R5]="ori_cost=trace w_ori=$wt w_vel=$wv"
fi
echo "R5: ${CFG[R5]}" | tee "$ROOT/selection.txt"
wants R5 && run_variant R5 ${CFG[R5]}

# ---- 3. observation, on the best reward
if [ "$DRY" = 1 ]; then BEST="<best of B0 R1-R5>"; BASE="<its env-cfg>"
elif [ -n "${BEST_CFG:-}" ]; then BEST="forced"; BASE="$BEST_CFG"
else
  echo "selection for O1/O2:"
  BEST="$(best B0 R1 R2 R3 R4 R5)"; BASE="${CFG[$BEST]}"
fi
echo "best reward: $BEST (${BASE:-baseline})" | tee -a "$ROOT/selection.txt"
wants O1 && run_variant O1 $BASE obs_ref_acc=True
wants O2 && run_variant O2 $BASE obs_attitude=R_err

# ---- summary: B0 first, then the variants
if [ "$DRY" = 0 ]; then
  dirs=("$B0_EVAL"); for v in R1 R2 R3 R4 R5 O1 O2; do [ -f "$ROOT/$v/eval/.done" ] && dirs+=("$ROOT/$v/eval"); done
  python3 "$SUMMARIZE" "${dirs[@]}" --csv "$ROOT/summary.csv"
  echo "(rows: B0 = ppo2_s03, then R1.. in order; env-cfg of each variant in $ROOT/<variant>/env_cfg.txt)"
fi
echo "ablation finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
