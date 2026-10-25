#!/usr/bin/env bash
# Shuttle, direct rotor velocity: finish the checkpoint evaluations, then export the data and draw the figures.
#
#   1. examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_checkpoints.sh  (skips finished evaluations, .done)
#   2. export_results.py   -> data/*.csv
#   3. plot_results.py     -> figures/*.pdf, *.png
# Each step runs only if the previous one succeeded. Safe to restart.
#
# Usage (inside tmux):
#   tmux new -s rotor_results 'results/shuttle/rotor_velocities/make_results.sh 2>&1 | tee rotor_results.out'
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

echo "[1/3] checkpoint evaluations ($(date -Is))"
"$REPO/examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_checkpoints.sh" || { echo "[FAIL] evaluations"; exit 1; }

echo "[2/3] export ($(date -Is))"
python3 "$HERE/export_results.py" || { echo "[FAIL] export"; exit 1; }

echo "[3/3] figures ($(date -Is))"
python3 "$HERE/plot_results.py" || { echo "[FAIL] figures"; exit 1; }

echo "done ($(date -Is)): $HERE/figures"
