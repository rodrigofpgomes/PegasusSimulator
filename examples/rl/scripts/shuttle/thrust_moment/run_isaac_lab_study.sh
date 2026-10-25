#!/usr/bin/env bash
# Shuttle, direct thrust and moments (shuttle/thrust_moment/isaac_lab): final study with 10 seeds.
#
# KT is the thrust-to-weight ratio k_T (task field thrust_to_weight):
#   KT=1.9 (default)  main study, logs/study_10seeds
#   KT=3.9            sensitivity test, logs/study_10seeds_kt3p9 (trained with the former copy 01_isaac_lab2,
#                     whose only difference was the default thrust_to_weight = 3.9)
#
# Experiments (one folder each):
#   ms001_den08 ms001_den15 ms006_den08 ms006_den15 ms012_den08 ms012_den15   moment_scale x denominator
#   ms012_den08_yaw011 ms012_den08_yaw101   extra yaw-rate penalty 0.1 and 1.0 on the best configuration,
#                                           i.e. effective wz weight 0.11 and 1.01 (0.01 is ms012_den08)
# Layout:
#   $ROOT/<experiment>/train/<experiment>_sNN/   training run (run_metadata.json, checkpoints, CSV export)
#   $ROOT/<experiment>/eval/<experiment>_sNN/    deterministic evaluation (steps.npz with actions, episodes.csv)
#   $ROOT/<experiment>/eval/summary_by_run.csv   per-seed metrics (written when its 10 seeds are evaluated)
#   $ROOT/progress.tsv                           one line per finished step
# Every (seed, experiment) is trained and immediately evaluated; seeds are the outer loop, so an
# interrupted study is complete for the first seeds. Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux):
#   tmux new -s study 'examples/rl/scripts/shuttle/thrust_moment/run_isaac_lab_study.sh 2>&1 | tee isaac_lab_study.out'
#   examples/rl/scripts/shuttle/thrust_moment/run_isaac_lab_study.sh --dry-run
#   KT=3.9 examples/rl/scripts/shuttle/thrust_moment/run_isaac_lab_study.sh
# Environment: KT, SEEDS (default 1..10), EXPERIMENTS (subset of the names above), ROOT, N_ENVS.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/thrust_moment/isaac_lab"
KT="${KT:-1.9}"
if [ "$KT" = "1.9" ]; then default_root="study_10seeds"; else default_root="study_10seeds_kt${KT/./p}"; fi
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/$default_root}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
N_ENVS="${N_ENVS:-4096}"
N_EVAL_ENVS=200
EVAL_SEED=1000            # goals of the evaluation, identical for every policy
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# experiment name -> task config overrides
declare -A CFG=(
  [ms001_den08]="moment_scale=0.01 distance_reward_denominator=0.8"
  [ms001_den15]="moment_scale=0.01 distance_reward_denominator=1.5"
  [ms006_den08]="moment_scale=0.06 distance_reward_denominator=0.8"
  [ms006_den15]="moment_scale=0.06 distance_reward_denominator=1.5"
  [ms012_den08]="moment_scale=0.12 distance_reward_denominator=0.8"
  [ms012_den15]="moment_scale=0.12 distance_reward_denominator=1.5"
  [ms012_den08_yaw011]="moment_scale=0.12 distance_reward_denominator=0.8 yaw_rate_reward_scale=-0.1"
  [ms012_den08_yaw101]="moment_scale=0.12 distance_reward_denominator=0.8 yaw_rate_reward_scale=-1.0"
)
for exp in "${!CFG[@]}"; do CFG[$exp]="${CFG[$exp]} thrust_to_weight=$KT"; done
ORDER="ms001_den08 ms001_den15 ms006_den08 ms006_den15 ms012_den08 ms012_den15 ms012_den08_yaw011 ms012_den08_yaw101"
EXPERIMENTS="${EXPERIMENTS:-$ORDER}"
for exp in $EXPERIMENTS; do
  [ -n "${CFG[$exp]+x}" ] || { echo "unknown experiment: $exp"; exit 2; }
done

export PYTHONUNBUFFERED=1
PROGRESS="$ROOT/progress.tsv"
if [ "$DRY" = 0 ]; then
  mkdir -p "$ROOT"
  [ -f "$PROGRESS" ] || printf "experiment\tseed\tstep\texit\tstart\tend\n" > "$PROGRESS"
fi
cd "$REPO"
git_state="$(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files)"
echo "commit: $git_state"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >> "$PROGRESS"; }

for seed in $SEEDS; do
  sid=$(printf "%02d" "$seed")
  for exp in $EXPERIMENTS; do
    name="${exp}_s${sid}"
    train_dir="$ROOT/$exp/train/$name"
    eval_dir="$ROOT/$exp/eval/$name"
    overrides=(); for kv in ${CFG[$exp]}; do overrides+=(--env-cfg "$kv"); done

    # ---- training (a .done without the final checkpoint and a completed export is discarded)
    if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1          && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; }; then
      echo "[redo ] $name: incomplete training marked as done"; rm -f "$train_dir/.done" "$eval_dir/.done"
    fi
    if [ ! -f "$train_dir/.done" ]; then
      cmd=("$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --preset isaac_lab --headless
           --n_envs "$N_ENVS" --seed "$seed" "${overrides[@]}" --log-dir "$train_dir")
      echo "[train] $name: ${cmd[*]}"
      if [ "$DRY" = 0 ]; then
        rm -rf "$train_dir"; mkdir -p "$train_dir"; start=$(date -Is)
        "${cmd[@]}" > "$train_dir/stdout.log" 2>&1; code=$?
        log_step "$exp" "$seed" train "$code" "$start"
        # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
        last_ckpt="$(ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)"
        if [ $code -eq 0 ] && [ -n "$last_ckpt" ] && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; then
          touch "$train_dir/.done"
        else echo "[FAIL] train $name (exit $code, checkpoint or completed export missing)"; failed=$((failed+1)); continue; fi
      fi
    fi

    # ---- evaluation (last checkpoint, task config restored from the run)
    if [ ! -f "$eval_dir/.done" ]; then
      run_dir="$(ls -d "$train_dir"/*_PPO2 2>/dev/null | sort | tail -1)"
      ckpt="$(ls "$run_dir"/checkpoints/agent_*.pt 2>/dev/null | sort -V | tail -1)"
      cmd=("$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset isaac_lab --checkpoint "$ckpt"
           --run-metadata "$run_dir/run_metadata.json" --headless --n_envs "$N_EVAL_ENVS"
           --num_episodes_per_env 1 --seed "$EVAL_SEED" --metrics-dir "$eval_dir")
      echo "[eval ] $name"
      if [ "$DRY" = 0 ]; then
        if [ -z "$ckpt" ]; then echo "[FAIL] eval $name: no checkpoint"; failed=$((failed+1)); continue; fi
        rm -rf "$eval_dir"; mkdir -p "$eval_dir"; start=$(date -Is)
        "${cmd[@]}" > "$eval_dir/stdout.log" 2>&1; code=$?
        log_step "$exp" "$seed" eval "$code" "$start"
        if [ $code -eq 0 ] && [ -f "$eval_dir/steps.npz" ]; then touch "$eval_dir/.done"
        else echo "[FAIL] eval $name ($code)"; failed=$((failed+1)); fi
      fi
    fi
  done
done

# per-experiment summaries (mean over the 200 goals for each seed)
if [ "$DRY" = 0 ]; then
  for exp in $EXPERIMENTS; do
    [ -d "$ROOT/$exp/eval" ] && python3 examples/rl/scripts/shuttle/thrust_moment/summarize_isaac_lab_eval.py "$ROOT/$exp/eval" > /dev/null
  done
fi
echo "study finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
