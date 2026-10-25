#!/usr/bin/env bash
# Shuttle, direct rotor velocity (shuttle/direct_rotor/baseline/raptor_tracking): PPO2 variants, trained and evaluated.
#
# Trains PRESET of agents/ppo2_cfg.py ("default": unbounded Gaussian mean, as RSL-RL; "tanh": mean = tanh(net)) for
# ITERATIONS from scratch, optionally with the exploration std bounded to [STD_MIN, 1] (PPO2_STD_MIN read by
# agents/ppo2_cfg.py), and evaluates every run with play.py on the training distribution. The SAC run and the PPO2
# runs of sac_vs_ppo2 (iteration 400) and ppo2_resume_600 (iteration 600) are evaluated with the same protocol, so
# the final table compares like with like.
# Evaluation: 500 episodes, training initial states (--train-init-state), seed EVAL_SEED, final checkpoint.
# Layout ($ROOT = logs/ppo2_<preset>_<iterations>[_std<min>]):
#   $ROOT/ppo2/train/ppo2_sNN/   training run (run_metadata.json, checkpoints, CSV export, stdout.log)
#   $ROOT/ppo2/eval/ppo2_sNN/    play.py metrics (steps.npz, episodes.csv)
#   $LOGS/eval_reference/<name>/ evaluations of the earlier runs (shared by all studies)
#   $ROOT/summary.csv            survival, final error, saturation and action change per run
#   $ROOT/progress.tsv           one line per finished step
# Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux):
#   tmux new -s std_min 'PRESET=default STD_MIN=0.2 examples/rl/scripts/shuttle/direct_rotor/run_ppo2_variant.sh 2>&1 | tee ppo2_std0p2.out'
#   PRESET=tanh examples/rl/scripts/shuttle/direct_rotor/run_ppo2_variant.sh                 # -> logs/ppo2_tanh_600
#   PRESET=tanh STD_MIN=0.2 examples/rl/scripts/shuttle/direct_rotor/run_ppo2_variant.sh     # -> logs/ppo2_tanh_600_std0p2
#   examples/rl/scripts/shuttle/direct_rotor/run_ppo2_variant.sh --dry-run
# Environment: PRESET (default "default"), STD_MIN (unset: no floor), ITERATIONS (600), PPO_SEEDS (default 1 2 3),
#              CKPT_ITERS (25), N_ENVS, N_EVAL_ENVS (500), EVAL_SEED (1000), ROOT.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/baseline/raptor_tracking"
LOGS="$REPO/examples/rl/tasks/$TASK/logs"
PRESET="${PRESET:-default}"
ITERATIONS="${ITERATIONS:-600}"
STD_MIN="${STD_MIN:-}"
TAG="ppo2_${PRESET}_${ITERATIONS}"; [ -n "$STD_MIN" ] && TAG="${TAG}_std${STD_MIN//./p}"
ROOT="${ROOT:-$LOGS/$TAG}"
PPO_SEEDS="${PPO_SEEDS:-1 2 3}"
N_ENVS="${N_ENVS:-4096}"
N_EVAL_ENVS="${N_EVAL_ENVS:-500}"
EVAL_SEED="${EVAL_SEED:-1000}"
CKPT_ITERS="${CKPT_ITERS:-25}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

export PYTHONUNBUFFERED=1
# The std floor is applied only when STD_MIN is given
if [ -n "$STD_MIN" ]; then export PPO2_STD_MIN="$STD_MIN"; else unset PPO2_STD_MIN; fi
export PPO2_CHECKPOINT_ITERS="$CKPT_ITERS"
PROGRESS="$ROOT/progress.tsv"
if [ "$DRY" = 0 ]; then
  mkdir -p "$ROOT" "$LOGS/eval_reference"
  [ -f "$PROGRESS" ] || printf "run\tseed\tstep\texit\tstart\tend\n" > "$PROGRESS"
fi
cd "$REPO"
echo "commit: $(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files); preset=$PRESET; iterations=$ITERATIONS; std_min=${STD_MIN:-none}; root=$ROOT"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >> "$PROGRESS"; }

# evaluate <name> <algo> <preset> <run_dir> <checkpoint> <eval_dir>
evaluate() {
  local name="$1" algo="$2" preset="$3" run_dir="$4" ckpt="$5" eval_dir="$6"
  if [ -f "$eval_dir/.done" ]; then echo "[skip ] eval $name"; return; fi
  local cmd=("$PY" examples/rl/play.py --task "$TASK" --algo "$algo" --preset "$preset" --checkpoint "$ckpt"
             --run-metadata "$run_dir/run_metadata.json" --headless --n_envs "$N_EVAL_ENVS"
             --num_episodes_per_env 1 --seed "$EVAL_SEED" --train-init-state --metrics-dir "$eval_dir")
  echo "[eval ] $name: $(basename "$ckpt")"
  [ "$DRY" = 1 ] && return
  if [ ! -f "$ckpt" ]; then echo "[FAIL] eval $name: no checkpoint $ckpt"; failed=$((failed+1)); return; fi
  rm -rf "$eval_dir"; mkdir -p "$eval_dir"; local start; start=$(date -Is)
  "${cmd[@]}" > "$eval_dir/stdout.log" 2>&1; local code=$?
  log_step "$name" - eval "$code" "$start"
  if [ $code -eq 0 ] && [ -f "$eval_dir/steps.npz" ]; then touch "$eval_dir/.done"
  else echo "[FAIL] eval $name ($code)"; failed=$((failed+1)); fi
}

# ---- training and evaluation of the selected variant
eval_dirs=()
for seed in $PPO_SEEDS; do
  name="ppo2_s$(printf "%02d" "$seed")"
  train_dir="$ROOT/ppo2/train/$name"

  # A .done without the final checkpoint and a completed export is discarded
  if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1 \
       && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; }; then
    echo "[redo ] $name: incomplete training marked as done"; rm -f "$train_dir/.done" "$ROOT/ppo2/eval/$name/.done"
  fi
  if [ ! -f "$train_dir/.done" ]; then
    cmd=("$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --preset "$PRESET" --headless
         --n_envs "$N_ENVS" --seed "$seed" --timesteps $((24 * ITERATIONS)) --log-dir "$train_dir")
    echo "[train] $name: ${cmd[*]}"
    if [ "$DRY" = 0 ]; then
      rm -rf "$train_dir" "$ROOT/ppo2/eval/$name"; mkdir -p "$train_dir"; start=$(date -Is)
      "${cmd[@]}" > "$train_dir/stdout.log" 2>&1; code=$?
      log_step "$name" "$seed" train "$code" "$start"
      # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
      last_ckpt="$(ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)"
      if [ $code -eq 0 ] && [ -n "$last_ckpt" ] && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; then
        touch "$train_dir/.done"
      else echo "[FAIL] train $name (exit $code, checkpoint or completed export missing)"; failed=$((failed+1)); continue; fi
    fi
  fi
  run_dir="$(ls -d "$train_dir"/*_PPO2 2>/dev/null | sort | tail -1)"
  evaluate "${TAG}_$name" ppo2 "$PRESET" "$run_dir" "$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt" "$ROOT/ppo2/eval/$name"
  eval_dirs+=("$ROOT/ppo2/eval/$name")
done

# ---- reference runs, same protocol
ref_dirs=()
sac_run="$(ls -d "$LOGS"/sac_vs_ppo2/sac/train/sac_s01/*_SACDelayed* 2>/dev/null | tail -1)"
if [ -n "$sac_run" ]; then
  sac_ckpt="$(ls "$sac_run"/checkpoints/agent_*.pt | sort -V | tail -1)"
  evaluate sac_s01 sac default "$sac_run" "$sac_ckpt" "$LOGS/eval_reference/sac_s01"
  ref_dirs+=("$LOGS/eval_reference/sac_s01")
fi
for seed in $PPO_SEEDS; do
  name="ppo2_s$(printf "%02d" "$seed")"
  run_400="$(ls -d "$LOGS"/sac_vs_ppo2/ppo2/train/"$name"/*_PPO2 2>/dev/null | tail -1)"
  run_600="$(ls -d "$LOGS"/ppo2_resume_600/ppo2/train/"$name"/*_PPO2 2>/dev/null | tail -1)"
  if [ -n "$run_400" ]; then
    evaluate "${name}_it400" ppo2 default "$run_400" "$run_400/checkpoints/agent_9600.pt" "$LOGS/eval_reference/${name}_it400"
    ref_dirs+=("$LOGS/eval_reference/${name}_it400")
  fi
  if [ -n "$run_600" ]; then
    evaluate "${name}_resume_it600" ppo2 default "$run_600" "$run_600/checkpoints/agent_4800.pt" "$LOGS/eval_reference/${name}_resume_it600"
    ref_dirs+=("$LOGS/eval_reference/${name}_resume_it600")
  fi
done

# ---- summary (this variant first, then the references)
if [ "$DRY" = 0 ]; then
  done_dirs=(); for d in "${eval_dirs[@]}" "${ref_dirs[@]}"; do [ -f "$d/.done" ] && done_dirs+=("$d"); done
  [ ${#done_dirs[@]} -gt 0 ] && python3 examples/rl/scripts/shuttle/direct_rotor/summarize_direct_rotor_eval.py "${done_dirs[@]}" --csv "$ROOT/summary.csv"
fi
echo "study finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
