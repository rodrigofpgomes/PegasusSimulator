#!/usr/bin/env bash
# Shuttle-Glider tracking (shuttle_glider/tracking): deterministic evaluation of the flight regimes.
#
# For every VARIANT/seed trained by run_tracking.sh, play.py runs N vehicles for one 12 s episode per regime (policy
# mean, eval_episode fixes the reference; trimmed starts with the training perturbations) and writes steps.npz and
# episodes.csv to $ROOT/<variant>/eval/sNN/<regime>/. summarize_tracking_eval.py then writes regimes_summary.csv.
# Regimes (eval_episode, speed [m/s], end speed / radius [m] / vertical speed [m/s]):
#   hover; mixed (training mixture); accel_vV (0 -> V from t = 1 s); decel_vV_E (V -> E within the deceleration
#   limits); cruise_vV; turn_vV_rR (on the arc from t = 0); climb_vV_zZ (vertical speed Z from t = 1 s);
#   helix_vV_rR_zZ (turn with vertical speed Z from t = 1 s; not in the training mixture, opt-in through REGIMES).
# The complete route (hover -> several speeds and turns -> hover) is evaluated with play_multi --trajectory mission.
#
# Usage (from the repository root): examples/rl/scripts/shuttle_glider/eval_tracking.sh [--dry-run]
# Environment: VARIANTS (base), SEEDS (3), ITERATIONS (5000), REGIMES, N_EVAL_ENVS (200), DURATION (12),
#              EVAL_SEED (1000), T_SKIP (3), ROOT, EXTRA_ENV_CFG (space-separated KEY=VALUE added to every regime,
#              e.g. surface_override=trim for the surface ablation), OUT_ROOT (if set, results go to
#              $OUT_ROOT/<variant>/all/<regime>, the layout read by results/shuttle_glider/tracking_regimes).
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
TASK="shuttle_glider/tracking"
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/variants}"
VARIANTS="${VARIANTS:-base}"
SEEDS="${SEEDS:-3}"
ITERATIONS="${ITERATIONS:-5000}"
REGIMES="${REGIMES:-hover mixed accel_v5 accel_v9 accel_v13 decel_v5_0 decel_v9_0 decel_v13_9
  cruise_v3 cruise_v7 cruise_v11 cruise_v14 turn_v8_r25 turn_v10_r40 turn_v12_r30 turn_v14_r40
  climb_v0_z1 climb_v10_z1 climb_v10_z-1}"
EXTRA=(); for kv in ${EXTRA_ENV_CFG:-}; do EXTRA+=(--env-cfg "$kv"); done
N="${N_EVAL_ENVS:-200}"; DUR="${DURATION:-12}"; SEED="${EVAL_SEED:-1000}"; T_SKIP="${T_SKIP:-3}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
export PYTHONUNBUFFERED=1; cd "$REPO"; failed=0

regime_args() {
  local r="$1" a b
  case "$r" in
    hover)      RARGS=(--env-cfg eval_episode=hover) ;;
    mixed)      RARGS=() ;;
    accel_v*)   RARGS=(--env-cfg eval_episode=accel --env-cfg "eval_speed=${r#accel_v}") ;;
    decel_v*_*) a="${r#decel_v}"; b="${a#*_}"; a="${a%%_*}"
                RARGS=(--env-cfg eval_episode=decel --env-cfg "eval_speed=$a" --env-cfg "eval_speed_end=$b") ;;
    cruise_v*)  RARGS=(--env-cfg eval_episode=cruise --env-cfg "eval_speed=${r#cruise_v}") ;;
    turn_v*_r*) a="${r#turn_v}"; b="${a#*_r}"; a="${a%%_r*}"
                RARGS=(--env-cfg eval_episode=turn --env-cfg "eval_speed=$a" --env-cfg "eval_turn_radius=$b") ;;
    helix_v*_r*_z*) a="${r#helix_v}"; b="${a#*_r}"; a="${a%%_r*}"
                RARGS=(--env-cfg eval_episode=turn --env-cfg "eval_speed=$a" --env-cfg "eval_turn_radius=${b%%_z*}"
                       --env-cfg "eval_vz=${b#*_z}") ;;
    climb_v*_z*) a="${r#climb_v}"; b="${a#*_z}"; a="${a%%_z*}"
                RARGS=(--env-cfg eval_episode=climb --env-cfg "eval_speed=$a" --env-cfg "eval_vz=$b") ;;
    *) return 1 ;;
  esac
}

for entry in $VARIANTS; do
  variant="${entry%%:*}"
  for seed in $SEEDS; do
    name="s$(printf "%02d" "$seed")"
    run_dir="$(ls -d "$ROOT/$variant/train/$name"/*_PPO2 2>/dev/null | sort | tail -1)"
    ckpt="$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt"
    if [ -z "$run_dir" ] || [ ! -f "$ckpt" ]; then echo "[FAIL] $variant/$name: no checkpoint agent_$((24 * ITERATIONS)).pt"; failed=$((failed+1)); continue; fi
    for regime in $REGIMES; do
      if ! regime_args "$regime"; then echo "[FAIL] unknown regime $regime"; failed=$((failed+1)); continue; fi
      out="$ROOT/$variant/eval/$name/$regime"
      [ -n "${OUT_ROOT:-}" ] && out="$OUT_ROOT/$variant/all/$regime"
      [ -f "$out/.done" ] && { echo "[skip ] $variant/$name $regime"; continue; }
      echo "[eval ] $variant/$name $regime"; [ "$DRY" = 1 ] && continue
      rm -rf "$out"; mkdir -p "$out"
      "$PY" examples/rl/play.py --task "$TASK" --algo ppo2 --checkpoint "$ckpt" --run-metadata "$run_dir/run_metadata.json" \
        --headless --n_envs "$N" --num_episodes_per_env 1 --seed "$SEED" --metrics-dir "$out" \
        --env-cfg test_mode=True --env-cfg "episode_length_s=$DUR" ${RARGS[@]+"${RARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} > "$out/stdout.log" 2>&1
      if [ $? -eq 0 ] && [ -f "$out/steps.npz" ]; then touch "$out/.done"; else echo "[FAIL] $variant/$name $regime"; failed=$((failed+1)); fi
    done
  done
done
[ "$DRY" = 0 ] && [ -z "${OUT_ROOT:-}" ] && python3 "$REPO/examples/rl/scripts/shuttle_glider/summarize_tracking_eval.py" "$ROOT" --t-skip "$T_SKIP" --csv "$ROOT/regimes_summary.csv"
echo "finished; failed: $failed (results: $ROOT)"; [ $failed -eq 0 ]
