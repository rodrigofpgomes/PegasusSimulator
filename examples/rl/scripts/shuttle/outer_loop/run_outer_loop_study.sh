#!/usr/bin/env bash
# Shuttle, outer loop (acceleration level, shuttle/outer_loop/double_integrator): reward study with 10 seeds.
#
# A_MAX is the acceleration authority per axis [m/s^2] (task field max_acceleration):
#   A_MAX=7.0 (default)  the discounted LQR never saturates on the training goals (peak 6.29 m/s^2)
#   A_MAX=1.0            complementary study (LQR saturated above 0.32 m), logs/outer_loop_amax1_10seeds
# PRESET is the PPO2 agent preset: isaac_lab (default, initial std 1) or isaac_lab_std1_7 (initial std 1/7,
#   i.e. +-1 m/s^2 of initial exploration with A_MAX=7); a non-default preset is appended to ROOT.
#
# Experiments (one folder each), all with the one task, which differs only in the reward fields passed below
# (the published runs were trained with its former copies 02_quadratic, 03_exponential, 04_constant_death):
#   quad           r = -c                       (learns to terminate early)
#   alive          r = -c + 1
#   death          r = -c - 500 on termination
#   exp            r = exp(-c)
#   alive_death    r = -c + 1, -500 on termination
#   alive_death200 r = -c + 1, -200 on termination (extra; only with EXPERIMENTS=alive_death200)
#   lqr           the task's discounted DLQR u = K [ep, ev] (play.py --controller lqr), evaluated once
# Layout:
#   $ROOT/<experiment>/train/<experiment>_sNN/   training run (run_metadata.json, checkpoints, CSV export)
#   $ROOT/<experiment>/eval/<experiment>_sNN/    deterministic evaluation (steps.npz with actions, episodes.csv)
#   $ROOT/<experiment>/eval/summary_by_run.csv   per-seed metrics
#   $ROOT/lqr/eval/lqr/                          LQR evaluation on the same goals
#   $ROOT/code/<date>/                           git HEAD, diff and untracked sources at each launch
#   $ROOT/progress.tsv                           one line per finished step
# Every (seed, experiment) is trained and immediately evaluated; seeds are the outer loop, so an
# interrupted study is complete for the first seeds. Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux):
#   tmux new -s outer 'examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh 2>&1 | tee outer_loop_amax7.out'
#   A_MAX=1.0 EXPERIMENTS="quad alive death exp alive_death alive_death200" examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh
#   examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh --dry-run
# Environment: A_MAX, PRESET, SEEDS (default 1..10), EXPERIMENTS (subset of the names above), ROOT, N_ENVS.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/outer_loop/double_integrator"
A_MAX="${A_MAX:-7.0}"
default_root="outer_loop_amax${A_MAX%.0}_10seeds"
PRESET="${PRESET:-isaac_lab}"
[ "$PRESET" = "isaac_lab" ] || default_root="${default_root%_10seeds}_${PRESET#isaac_lab_}_10seeds"
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/$default_root}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
N_ENVS="${N_ENVS:-4096}"
N_EVAL_ENVS=200
EVAL_SEED=1000            # goals of the evaluation, identical for every policy
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# experiment name -> task config overrides (written explicitly, independent of task defaults)
declare -A CFG=(
  [quad]="reward_shape='quadratic' alive_bonus=0.0 death_penalty=0.0"
  [alive]="reward_shape='quadratic' alive_bonus=1.0 death_penalty=0.0"
  [death]="reward_shape='quadratic' alive_bonus=0.0 death_penalty=-500.0"
  [exp]="reward_shape='exponential' alive_bonus=0.0 death_penalty=0.0"
  [alive_death]="reward_shape='quadratic' alive_bonus=1.0 death_penalty=-500.0"
  [alive_death200]="reward_shape='quadratic' alive_bonus=1.0 death_penalty=-200.0"
)
for exp in "${!CFG[@]}"; do CFG[$exp]="${CFG[$exp]} max_acceleration=$A_MAX"; done
ORDER="quad alive death exp alive_death"
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
echo "commit: $git_state, A_MAX=$A_MAX, PRESET=$PRESET, ROOT=$ROOT"
if [ "$DRY" = 0 ]; then
  # The runs store only the commit hash: keep the exact working-tree sources of this launch
  snap="$ROOT/code/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$snap"
  git rev-parse HEAD > "$snap/HEAD"; git diff HEAD > "$snap/working_tree.diff"
  printf "A_MAX=%s\nPRESET=%s\n" "$A_MAX" "$PRESET" > "$snap/study_settings"
  git ls-files --others --exclude-standard -- examples/rl extensions | grep -E '\.(py|sh|json)$' \
    | tar -czf "$snap/untracked_sources.tgz" -T - 2>/dev/null
fi

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >> "$PROGRESS"; }

# ---- LQR reference (deterministic, independent of the seed and of the reward), same goals as the policies
lqr_dir="$ROOT/lqr/eval/lqr"
if [ ! -f "$lqr_dir/.done" ]; then
  overrides=(); for kv in ${CFG[quad]}; do overrides+=(--env-cfg "$kv"); done
  cmd=("$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset "$PRESET" --controller lqr
       "${overrides[@]}" --headless --n_envs "$N_EVAL_ENVS" --num_episodes_per_env 1 --seed "$EVAL_SEED"
       --metrics-dir "$lqr_dir")
  echo "[eval ] lqr: ${cmd[*]}"
  if [ "$DRY" = 0 ]; then
    rm -rf "$lqr_dir"; mkdir -p "$lqr_dir"; start=$(date -Is)
    "${cmd[@]}" > "$lqr_dir/stdout.log" 2>&1; code=$?
    log_step lqr 0 eval "$code" "$start"
    if [ $code -eq 0 ] && [ -f "$lqr_dir/steps.npz" ]; then touch "$lqr_dir/.done"
    else echo "[FAIL] eval lqr ($code)"; failed=$((failed+1)); fi
  fi
fi

for seed in $SEEDS; do
  sid=$(printf "%02d" "$seed")
  for exp in $EXPERIMENTS; do
    name="${exp}_s${sid}"
    train_dir="$ROOT/$exp/train/$name"
    eval_dir="$ROOT/$exp/eval/$name"
    overrides=(); for kv in ${CFG[$exp]}; do overrides+=(--env-cfg "$kv"); done

    # ---- training (a .done without the final checkpoint and a completed export is discarded)
    if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1 \
         && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; }; then
      echo "[redo ] $name: incomplete training marked as done"; rm -f "$train_dir/.done" "$eval_dir/.done"
    fi
    if [ ! -f "$train_dir/.done" ]; then
      cmd=("$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --preset "$PRESET" --headless
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
      cmd=("$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset "$PRESET" --checkpoint "$ckpt"
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

# summaries: per-experiment (as section 1) and the outer-loop table (return of r^Q, |u|^2, t20, LQR)
if [ "$DRY" = 0 ]; then
  for exp in $EXPERIMENTS; do
    [ -d "$ROOT/$exp/eval" ] && python3 examples/rl/scripts/shuttle/thrust_moment/summarize_isaac_lab_eval.py "$ROOT/$exp/eval" > /dev/null
  done
  python3 examples/rl/scripts/shuttle/outer_loop/summarize_outer_loop_eval.py "$ROOT"
fi
echo "study finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
