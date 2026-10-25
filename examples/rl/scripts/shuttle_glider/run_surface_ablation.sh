#!/usr/bin/env bash
# Shuttle-Glider tracking: surface ablation of the base policy. The 19 regimes with the policy surface commands replaced
# by the trim table at the measured airspeed (trim) or by zero deflection (zero); policy = the base evaluation.
# -> logs/surface_ablation/{policy,trim,zero}/eval/s03/<regime>/ and logs/surface_ablation/regimes_summary.csv
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; S="$REPO/examples/rl/scripts/shuttle_glider"
export ROOT="$REPO/examples/rl/tasks/shuttle_glider/tracking/logs/surface_ablation"
for mode in trim zero; do
  VARIANTS="$mode" EXTRA_ENV_CFG="surface_override=$mode" "$S/eval_tracking.sh"
done
