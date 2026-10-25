#!/usr/bin/env bash
# Shuttle-Glider tracking regimes: evaluations, then the data export and the figures.
#
#   1. eval_regimes.sh     (skips finished runs, .done)
#   2. export_results.py   -> data/*.csv
#   3. plot_results.m (MATLAB, run by hand: plot_results) -> figures/*.pdf, *.png
# Each step runs only if the previous one succeeded. Safe to restart.
#
# Usage (inside tmux): results/shuttle_glider/tracking_regimes/make_results.sh 2>&1 | tee regimes_results.out
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] evaluations ($(date -Is))"
"$HERE/eval_regimes.sh" || { echo "[FAIL] evaluations"; exit 1; }
echo "[2/2] export ($(date -Is))"
python3 "$HERE/export_results.py" || { echo "[FAIL] export"; exit 1; }
echo "done ($(date -Is)): data/ updated; draw the figures in MATLAB: cd $HERE; plot_results"
