#!/usr/bin/env bash
# Shuttle, direct rotor velocity: control runs of the reward/observation ablation (task
# shuttle/direct_rotor/ablation/raptor_tracking), trained and evaluated one after the other.
#
# Same protocol as run_direct_rotor_ablation.sh: PPO2 "tanh" preset, PPO2_STD_MIN=0.2, 600 iterations, evaluation with
# play.py on 500 episodes (training initial states, seed 1000). Runs, in order:
#
#   R6_s03  q_z 0.1 + velocity cost 0.1, seed 3   separates the effect of the velocity cost from that of removing q_z
#                                                 (R1/R2 changed both); compare with B0 (q_z only) and R1 (ev only)
#   O2_s01  R5 + attitude observed as Rd^T R, seed 1  \  with seed 3 (O2): is the smooth O2 policy reproducible?
#   O2_s02  same, seed 2                              /
#   R7_s03  velocity cost 0.1, no attitude term, yaw-rate penalty 0.05|omega_z|   \  alternatives to the q_z term that
#   R8_s03  velocity cost 0.1, no attitude term, angular-rate penalty 0.05||omega||  >  remove the yaw oscillation of R1/R2
#   R9_s03  velocity cost 0.1, geometric attitude term with weight 10                /  (cf. the yaw-rate penalty of the force/moment section)
#   A1_s03  baseline reward (q_z) + reference acceleration in the observation   \  the controlled pairs for the effect of
#   A2_s03  R6 reward (q_z + velocity cost) + reference acceleration              /  the reference acceleration: B0 / R6 without it
#   R6_s01, R6_s02  R6 with seeds 1 and 2: confirm the reward q_z + velocity cost (RUNS="R6_s01 R6_s02")
# Optional (RUNS="R6_s03 O2_s01 O2_s02 R5_s01 R5_s02"):
#   R5_s01, R5_s02  R5 with seeds 1 and 2         R5 and O2 observe almost the same state, so the oscillation of R5
#                                                 (seed 3) against the smooth O2 (seed 3) is attributed to the observation
#                                                 only if R5 also oscillates, and O2 does not, in the other seeds
#
# Layout: $ROOT/<run>/{train,eval}, $ROOT/<run>/env_cfg.txt; $ROOT = logs of the ablation task (shared with the
# first study). Finished steps (.done) are skipped on restart. About 16 min of training + 1 min of evaluation per run.
#
# Usage (inside tmux, from the repository root):
#   tmux new -s ablation_extra 'examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_ablation_extra.sh 2>&1 | tee ablation_extra.out'
#   examples/rl/scripts/shuttle/direct_rotor/run_direct_rotor_ablation_extra.sh --dry-run
# Environment: RUNS (default "R6_s03 O2_s01 O2_s02"), ITERATIONS (600), STD_MIN (0.2), N_ENVS (4096),
#              N_EVAL_ENVS (500), EVAL_SEED (1000), CKPT_ITERS (50).
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle/direct_rotor/ablation/raptor_tracking"
ROOT="$REPO/examples/rl/tasks/$TASK/logs"
SUMMARIZE="$REPO/examples/rl/scripts/shuttle/direct_rotor/summarize_direct_rotor_eval.py"
RUNS="${RUNS:-R6_s03 O2_s01 O2_s02}"
ITERATIONS="${ITERATIONS:-600}"
STD_MIN="${STD_MIN:-0.2}"
N_ENVS="${N_ENVS:-4096}"
N_EVAL_ENVS="${N_EVAL_ENVS:-500}"
EVAL_SEED="${EVAL_SEED:-1000}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# name -> "seed|env-cfg items"
declare -A SPEC=(
  [R6_s03]="3|ori_cost=qz w_ori=0.1 w_vel=0.1"
  [O2_s01]="1|ori_cost=trace w_ori=0.3 w_vel=0.1 obs_attitude=R_err"
  [O2_s02]="2|ori_cost=trace w_ori=0.3 w_vel=0.1 obs_attitude=R_err"
  [R7_s03]="3|ori_cost=none w_vel=0.1 ang_cost=yaw w_ang=0.05"
  [R8_s03]="3|ori_cost=none w_vel=0.1 ang_cost=norm w_ang=0.05"
  [R9_s03]="3|ori_cost=trace w_ori=10 w_vel=0.1"
  [A1_s03]="3|ori_cost=qz w_ori=0.1 obs_ref_acc=True"
  [A2_s03]="3|ori_cost=qz w_ori=0.1 w_vel=0.1 obs_ref_acc=True"
  [R6_s01]="1|ori_cost=qz w_ori=0.1 w_vel=0.1"
  [R6_s02]="2|ori_cost=qz w_ori=0.1 w_vel=0.1"
  [R5_s01]="1|ori_cost=trace w_ori=0.3 w_vel=0.1"
  [R5_s02]="2|ori_cost=trace w_ori=0.3 w_vel=0.1"
)

export PYTHONUNBUFFERED=1
export PPO2_STD_MIN="$STD_MIN"
export PPO2_CHECKPOINT_ITERS="${CKPT_ITERS:-50}"
PROGRESS="$ROOT/progress_extra.tsv"
mkdir -p "$ROOT"
[ -f "$PROGRESS" ] || printf "run\tstep\texit\tstart\tend\n" > "$PROGRESS"
cd "$REPO"
echo "commit: $(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files); iterations=$ITERATIONS; std_min=$STD_MIN; runs: $RUNS"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$(date -Is)" >> "$PROGRESS"; }

run_one() {
  local name="$1" spec seed items cfg=() item
  spec="${SPEC[$name]:-}"
  if [ -z "$spec" ]; then echo "[FAIL] unknown run '$name' (known: ${!SPEC[*]})"; failed=$((failed+1)); return; fi
  seed="${spec%%|*}"; items="${spec#*|}"
  for item in $items; do cfg+=(--env-cfg "$item"); done
  local train_dir="$ROOT/$name/train" eval_dir="$ROOT/$name/eval"
  echo "[$name] seed $seed, env-cfg: $items"
  [ "$DRY" = 1 ] && return

  # A .done without the final checkpoint and a completed export is discarded
  if [ -f "$train_dir/.done" ] && ! { ls "$train_dir"/*_PPO2/checkpoints/agent_*.pt >/dev/null 2>&1 \
       && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; }; then
    echo "[redo ] $name: incomplete training marked as done"; rm -f "$train_dir/.done" "$eval_dir/.done"
  fi
  if [ -f "$train_dir/.done" ]; then echo "[skip ] train $name"
  else
    rm -rf "$train_dir" "$eval_dir"; mkdir -p "$train_dir"
    printf "%s\n" $items > "$ROOT/$name/env_cfg.txt"
    local start; start=$(date -Is)
    "$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --preset tanh --headless --n_envs "$N_ENVS" \
      --seed "$seed" --timesteps $((24 * ITERATIONS)) --log-dir "$train_dir" "${cfg[@]}" > "$train_dir/stdout.log" 2>&1
    local code=$?
    log_step "$name" train "$code" "$start"
    # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
    if [ $code -eq 0 ] && [ -f "$(ls -d "$train_dir"/*_PPO2 2>/dev/null | sort | tail -1)/checkpoints/agent_$((24 * ITERATIONS)).pt" ] \
       && grep -qs '"status": "completed"' "$train_dir"/*_PPO2/download_csvs/manifest.json; then
      touch "$train_dir/.done"
    else echo "[FAIL] train $name (exit $code, final checkpoint or completed export missing)"; failed=$((failed+1)); return; fi
  fi

  if [ -f "$eval_dir/.done" ]; then echo "[skip ] eval $name"; return; fi
  local run_dir; run_dir="$(ls -d "$train_dir"/*_PPO2 | sort | tail -1)"
  rm -rf "$eval_dir"; mkdir -p "$eval_dir"
  local start; start=$(date -Is)
  # The run-metadata snapshot restores the env-cfg of the training run
  "$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --preset tanh \
    --checkpoint "$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt" --run-metadata "$run_dir/run_metadata.json" \
    --headless --n_envs "$N_EVAL_ENVS" --num_episodes_per_env 1 --seed "$EVAL_SEED" --train-init-state \
    --metrics-dir "$eval_dir" > "$eval_dir/stdout.log" 2>&1
  local code=$?
  log_step "$name" eval "$code" "$start"
  if [ $code -eq 0 ] && [ -f "$eval_dir/steps.npz" ]; then touch "$eval_dir/.done"
  else echo "[FAIL] eval $name ($code)"; failed=$((failed+1)); fi
}

for name in $RUNS; do run_one "$name"; done

# ---- summary: baseline (seed 3), the first study and these runs
if [ "$DRY" = 0 ]; then
  dirs=("$REPO/examples/rl/tasks/shuttle/direct_rotor/baseline/raptor_tracking/logs/ppo2_tanh_600_std0p2/ppo2/eval/ppo2_s03")
  labels="B0"
  for v in R1 R2 R3 R4 R5 O1 O2 $RUNS; do
    [ -f "$ROOT/$v/eval/.done" ] && { dirs+=("$ROOT/$v/eval"); labels="$labels $v"; }
  done
  python3 "$SUMMARIZE" "${dirs[@]}" --csv "$ROOT/summary_extra.csv"
  echo "(rows in order: $labels)"
fi
echo "extra runs finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
