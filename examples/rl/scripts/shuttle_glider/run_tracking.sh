#!/usr/bin/env bash
# Shuttle-Glider tracking (shuttle_glider/tracking): training of the reward variants, without curriculum.
#
# Trains PPO2 (tanh mean, std floor 0.2, 2x256) once per variant and seed on the fixed mixture of flight regimes.
# A variant is NAME:KEY=VALUE,KEY=VALUE,... of env-cfg overrides (fields of QuadcopterEnvCfg). Named variants:
#   base         task defaults (w_power 0.05, w_heading 0.5, w_rate 0.05, w_d_action 0.5, 30 % large perturbations)
#   power01      w_power=0.1             power02      w_power=0.2   (power ablation: alpha near CLmax, degrades)
#   dact025      w_d_action=0.25         rate0        w_rate=0
#   nolarge      large_perturbation_prob=0 (trimmed starts with small perturbations only)
#   surfq        surface_low_q_gain=50 (A: surface cost x51 without airflow, ~x1 at cruise)
#   conflict     w_conflict=0.2 (B: surface and rotor moments of opposite sign)
#   surfq_conflict  A + B
#   conflict_alloc  B smoothed (conflict_eps=0.05 N m, against the chattering at the sign change) + C: rotor yaw
#                differential (front/rear roll and left/right pitch cancelling inside the lift rotors), w_rotor_cancel=0.2
#   alloc        conflict_alloc + D: lift-rotor thrust differentials (w_rotor_diff=0.1, moments move to the surfaces as
#                q grows) + E: low-q L1 surface cost (w_surface_l1=0.02, q_ref 5 Pa) + trim table "alloc" (optimum of B-E)
#                (conflict_alloc, alloc: rejected, the min-based costs B/C are biased by the exploration noise)
#   wls          convex allocation only: D (w_rotor_diff=0.1) + E (w_surface_l1=0.02) + trim table "wls"
#                (training degraded: D taxes the rotor pitch differential needed to pitch over at low q)
#   wls_q        wls with D scaled by q / (q + 30 Pa) + trim table "wls_q"
#   wls_ema      wls with D on a 2 s moving average of the differentials (observed: observation_space 41)
#   wls_ema_m    wls_ema + F: filtered roll/yaw moments of rotors and surfaces (w_moment_ema=0.05, tau 2 s; against
#                the crossed controls of wls_ema; observation_space 45)
#   wls_ema_m02  wls_ema_m with w_moment_ema=0.02
# Other useful keys: w_pos, w_vel, w_heading, surface_power_equivalent_w, episode_type_probs, speed_max, accel_max,
# termination_penalty, max_pos_error_per_axis, obs_pos_scale.
# Evaluation of the flight regimes: eval_tracking.sh (same VARIANTS/SEEDS/ROOT), e.g. run in series.
# Layout ($ROOT = tasks/<task>/logs/variants):
#   $ROOT/<variant>/train/sNN/   training run (run_metadata.json, checkpoints, CSV export, stdout.log)
#   $ROOT/progress.tsv           one line per finished step
# Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux, from the repository root):
#   tmux new -s glider 'S=examples/rl/scripts/shuttle_glider; $S/run_tracking.sh 2>&1 | tee glider_train.out; $S/eval_tracking.sh 2>&1 | tee glider_eval.out'
#   VARIANTS="dact025 rate0 nolarge" examples/rl/scripts/shuttle_glider/run_tracking.sh
#   VARIANTS="vel03:w_vel=0.3" examples/rl/scripts/shuttle_glider/run_tracking.sh
#   examples/rl/scripts/shuttle_glider/run_tracking.sh --dry-run
# Environment: VARIANTS (names above or NAME:KEY=VALUE,...), SEEDS (3), ITERATIONS (5000), CKPT_ITERS (250),
#              N_ENVS (4096), ROOT.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle_glider/tracking"
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/variants}"
VARIANTS="${VARIANTS:-base}"
SEEDS="${SEEDS:-3}"
ITERATIONS="${ITERATIONS:-5000}"
N_ENVS="${N_ENVS:-4096}"
CKPT_ITERS="${CKPT_ITERS:-250}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# variant_spec <entry> -> "NAME:overrides" (default names expanded)
variant_spec() {
  case "$1" in
    base)       echo "base:" ;;
    power01)    echo "power01:w_power=0.1" ;;
    power02)    echo "power02:w_power=0.2" ;;
    dact025)    echo "dact025:w_d_action=0.25" ;;
    rate0)      echo "rate0:w_rate=0" ;;
    nolarge)    echo "nolarge:large_perturbation_prob=0" ;;
    surfq)      echo "surfq:surface_low_q_gain=50" ;;
    conflict)   echo "conflict:w_conflict=0.2" ;;
    surfq_conflict) echo "surfq_conflict:surface_low_q_gain=50,w_conflict=0.2" ;;
    conflict_alloc) echo "conflict_alloc:w_conflict=0.2,conflict_eps=0.05,w_rotor_cancel=0.2" ;;
    alloc)      echo "alloc:w_conflict=0.2,conflict_eps=0.05,w_rotor_cancel=0.2,w_rotor_diff=0.1,w_surface_l1=0.02,trim_table=alloc" ;;
    wls)        echo "wls:w_rotor_diff=0.1,w_surface_l1=0.02,trim_table=wls" ;;
    wls_q)      echo "wls_q:w_rotor_diff=0.1,rotor_diff_q_ref=30.0,w_surface_l1=0.02,trim_table=wls_q" ;;
    wls_ema)    echo "wls_ema:w_rotor_diff=0.1,rotor_diff_tau=2.0,observation_space=41,w_surface_l1=0.02,trim_table=wls" ;;
    wls_ema_m)  echo "wls_ema_m:w_rotor_diff=0.1,rotor_diff_tau=2.0,w_moment_ema=0.05,observation_space=45,w_surface_l1=0.02,trim_table=wls" ;;
    wls_ema_m02) echo "wls_ema_m02:w_rotor_diff=0.1,rotor_diff_tau=2.0,w_moment_ema=0.02,observation_space=45,w_surface_l1=0.02,trim_table=wls" ;;
    *:*)        echo "$1" ;;
    *)          echo "unknown variant '$1' (use a default name or NAME:KEY=VALUE,...)" >&2; return 1 ;;
  esac
}

export PYTHONUNBUFFERED=1
export PPO2_CHECKPOINT_ITERS="$CKPT_ITERS"
PROGRESS="$ROOT/progress.tsv"
if [ "$DRY" = 0 ]; then
  mkdir -p "$ROOT"
  [ -f "$PROGRESS" ] || printf "run\tseed\tstep\texit\tstart\tend\n" > "$PROGRESS"
fi
cd "$REPO"
echo "commit: $(git rev-parse --short HEAD) ($(git status --porcelain --untracked-files=no | wc -l) modified files); variants=$VARIANTS; seeds=$SEEDS; iterations=$ITERATIONS; root=$ROOT"

failed=0
log_step() { printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >> "$PROGRESS"; }

for entry in $VARIANTS; do
  spec="$(variant_spec "$entry")" || { failed=$((failed+1)); continue; }
  variant="${spec%%:*}"; overrides="${spec#*:}"
  env_args=(); IFS=',' read -ra kvs <<< "$overrides"; for kv in "${kvs[@]}"; do [ -n "$kv" ] && env_args+=(--env-cfg "$kv"); done
  for seed in $SEEDS; do
    name="s$(printf "%02d" "$seed")"
    train_dir="$ROOT/$variant/train/$name"
    if [ -f "$train_dir/.done" ]; then echo "[skip ] train $variant/$name"; continue; fi
    cmd=("$PY" examples/rl/train.py --task "$TASK" --algo ppo2 --headless --n_envs "$N_ENVS" --seed "$seed"
         --timesteps $((24 * ITERATIONS)) --log-dir "$train_dir" "${env_args[@]}")
    echo "[train] $variant/$name: ${cmd[*]}"
    [ "$DRY" = 1 ] && continue
    rm -rf "$train_dir"; mkdir -p "$train_dir"; start=$(date -Is)
    "${cmd[@]}" > "$train_dir/stdout.log" 2>&1; code=$?
    log_step "$variant/$name" "$seed" train "$code" "$start"
    # Isaac Sim can exit with 0 after an interruption: require the final checkpoint and a completed export
    run_dir="$(ls -d "$train_dir"/*_PPO2 2>/dev/null | sort | tail -1)"
    if [ $code -eq 0 ] && [ -f "$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt" ] \
       && grep -qs '"status": "completed"' "$run_dir/download_csvs/manifest.json"; then
      touch "$train_dir/.done"
    else echo "[FAIL] train $variant/$name (exit $code, checkpoint or completed export missing)"; failed=$((failed+1)); fi
  done
done

echo "training finished; failed steps: $failed (progress: $PROGRESS)"
[ $failed -eq 0 ]
