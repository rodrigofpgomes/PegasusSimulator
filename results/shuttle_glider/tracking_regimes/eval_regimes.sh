#!/usr/bin/env bash
# Shuttle-Glider tracking: evaluation runs of the regime figures (play.py --metrics-dir with the task's eval_extras()).
#
# For every variant and regime: 200 vehicles from the training initial states (perturbed trim, evaluation seed 1000)
# -> <regime>/, and 4 vehicles from the trimmed start without perturbation -> <regime>_nominal/. Output:
# examples/rl/tasks/shuttle_glider/tracking/logs/regime_figures/<variant>/<regime>[_nominal]/steps.npz. Skips runs
# with .done. Regimes: decel_v9_0 (9 -> 0 m/s within the deceleration limit, from t = 1 s) and helix_v12_r30_z1
# (12 m/s, R = 30 m, vertical speed 1 m/s from t = 1 s; turn and climb are never combined in training).
#
# Usage: results/shuttle_glider/tracking_regimes/eval_regimes.sh     Environment: VARIANTS (base wls_ema wls_ema_m02)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
L=$REPO/examples/rl/tasks/shuttle_glider/tracking/logs
VARIANTS="${VARIANTS:-base wls_ema wls_ema_m02}"
cd "$REPO"; export PYTHONUNBUFFERED=1; failed=0
NOMINAL=(--env-cfg large_perturbation_prob=0.0 --env-cfg "small_perturbation=(0.0,0.0,0.0,0.0,0.0)")
for v in $VARIANTS; do
  run_dir=$(ls -d $L/variants/$v/train/s03/*_PPO2 | sort | tail -1)
  for spec in "decel_v9_0|eval_episode=decel eval_speed=9 eval_speed_end=0" \
              "helix_v12_r30_z1|eval_episode=turn eval_speed=12 eval_turn_radius=30 eval_vz=1"; do
    r=${spec%%|*}; args=(); for kv in ${spec#*|}; do args+=(--env-cfg "$kv"); done
    for mode in perturbed nominal; do
      if [ $mode = nominal ]; then out=$L/regime_figures/$v/${r}_nominal; n=4; extra=("${NOMINAL[@]}")
      else out=$L/regime_figures/$v/$r; n=200; extra=(); fi
      [ -f $out/.done ] && { echo "[skip ] $v $r $mode"; continue; }
      echo "[eval ] $v $r $mode ($(date +%T))"; rm -rf $out; mkdir -p $out
      $PY examples/rl/play.py --task shuttle_glider/tracking --algo ppo2 --checkpoint $run_dir/checkpoints/agent_120000.pt \
        --run-metadata $run_dir/run_metadata.json --headless --n_envs $n --num_episodes_per_env 1 --seed 1000 \
        --metrics-dir $out --env-cfg test_mode=True --env-cfg episode_length_s=12 "${args[@]}" ${extra[@]+"${extra[@]}"} \
        > $out/stdout.log 2>&1 && [ -f $out/steps.npz ] && touch $out/.done || { echo "[FAIL] $v $r $mode"; failed=$((failed+1)); }
    done
  done
done
exit $failed
