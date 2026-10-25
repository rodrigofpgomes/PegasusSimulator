#!/usr/bin/env bash
# Shuttle, direct rotor velocity (shuttle/direct_rotor/baseline/raptor_tracking): SAC vs PPO2 training.
#
# Runs, in this order:
#   sac_s01            SAC, RAPTOR pre-training setup: one vehicle, 1e6 transitions, seed 1
#   ppo2_s01..s03      PPO2, 4096 vehicles, 400 iterations, seeds 1, 2 and 3
# Both use the task's "default" agent preset; seeds are passed explicitly.
# Layout:
#   $ROOT/<algo>/train/<algo>_sNN/   training run (run_metadata.json, checkpoints, CSV export, stdout.log)
#   $ROOT/progress.tsv               one line per finished run
# Finished runs (.done) are skipped on restart.
#
# Usage (inside tmux):
#   tmux new -s direct_rotor 'examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_baseline.sh 2>&1 | tee direct_rotor_baseline.out'
#   examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_baseline.sh --dry-run
# Environment: SAC_SEEDS (default 1), PPO_SEEDS (default 1 2 3), ROOT, N_ENVS (PPO2).
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/baseline/raptor_tracking"
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/sac_vs_ppo2}"
SAC_SEEDS="${SAC_SEEDS:-1}"
PPO_SEEDS="${PPO_SEEDS:-1 2 3}"
N_ENVS="${N_ENVS:-4096}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

export PYTHONUNBUFFERED=1
PROGRESS="$ROOT/progress.tsv"
if [ "$DRY" = 0 ]; then
  mkdir -p "$ROOT"
  [ -f "$PROGRESS" ] || printf "algo\tseed\tstep\texit\tstart\tend\n" > "$PROGRESS"
fi
cd "$REPO"
git_state="$(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files)"
echo "commit: $git_state"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >> "$PROGRESS"; }

# train <algo> <run suffix written by skrl> <n_envs> <seed>
train() {
  local algo="$1" suffix="$2" n_envs="$3" seed="$4"
  local name; name="${algo}_s$(printf "%02d" "$seed")"
  local train_dir="$ROOT/$algo/train/$name"

  # A .done without the final checkpoint and a completed export is discarded
  if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_"$suffix"*/checkpoints/agent_*.pt >/dev/null 2>&1 \
       && grep -qs '"status": "completed"' "$train_dir"/*_"$suffix"*/download_csvs/manifest.json; }; then
    echo "[redo ] $name: incomplete training marked as done"; rm -f "$train_dir/.done"
  fi
  if [ -f "$train_dir/.done" ]; then echo "[skip ] $name"; return; fi

  local cmd=("$PY" examples/rl/train.py --task "$TASK" --algo "$algo" --headless
             --n_envs "$n_envs" --seed "$seed" --log-dir "$train_dir")
  echo "[train] $name: ${cmd[*]}"
  [ "$DRY" = 1 ] && return

  rm -rf "$train_dir"; mkdir -p "$train_dir"; local start; start=$(date -Is)
  "${cmd[@]}" > "$train_dir/stdout.log" 2>&1; local code=$?
  log_step "$algo" "$seed" train "$code" "$start"
  # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
  local last_ckpt; last_ckpt="$(ls "$train_dir"/*_"$suffix"*/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)"
  if [ $code -eq 0 ] && [ -n "$last_ckpt" ] && grep -qs '"status": "completed"' "$train_dir"/*_"$suffix"*/download_csvs/manifest.json; then
    touch "$train_dir/.done"
  else
    echo "[FAIL] train $name (exit $code, checkpoint or completed export missing)"; failed=$((failed+1))
  fi
}

# SAC trains on one vehicle so that one timestep is one transition and one gradient step
# skrl names the run folder after the agent class (SACDelayed_delay_2 in algorithms/sac.py)
for seed in $SAC_SEEDS; do train sac SACDelayed 1 "$seed"; done
for seed in $PPO_SEEDS; do train ppo2 PPO2 "$N_ENVS" "$seed"; done

echo "training finished; failed runs: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
