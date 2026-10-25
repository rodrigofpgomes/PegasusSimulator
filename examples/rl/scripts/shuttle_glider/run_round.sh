#!/usr/bin/env bash
# Shuttle-Glider tracking (shuttle_glider/tracking): unattended round - training, regime evaluation and missions.
#
#   1. run_tracking.sh   TRAIN_VARIANTS (default: dact025 rate0 nolarge, all with the default w_power 0.05)
#   2. eval_tracking.sh  EVAL_VARIANTS  (default: power01 power02 + TRAIN_VARIANTS); finished regimes (.done) are
#                        skipped, an interrupted regime is evaluated again
#   3. play_multi missions of MISSION_VARIANTS (default: base + EVAL_VARIANTS) with the limits of the training
#      references (peak acceleration 3 m/s^2, deceleration of the task table at the cruise speed):
#        v12: 12 m/s, R 50 m, deceleration 0.7 m/s^2 (~70 s)      v14: 14 m/s, R 70 m, deceleration 0.45 m/s^2 (~94 s)
#      -> $ROOT/<variant>/mission/<flight>/ (metrics.json, trajectories.csv) and $ROOT/missions_summary.csv
# Finished steps are skipped on restart (training .done, regime .done, mission metrics.json).
#
# Usage (inside tmux, from the repository root):
#   tmux send-keys -t glider 'examples/rl/scripts/shuttle_glider/run_round.sh 2>&1 | tee glider_round.out' Enter
# Environment: TRAIN_VARIANTS, EVAL_VARIANTS, MISSION_VARIANTS, SEEDS (3), ITERATIONS (5000), ROOT.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
S="$REPO/examples/rl/scripts/shuttle_glider"
TASK="shuttle_glider/tracking"
ROOT="${ROOT:-$REPO/examples/rl/tasks/$TASK/logs/variants}"
TRAIN_VARIANTS="${TRAIN_VARIANTS:-dact025 rate0 nolarge}"
EVAL_VARIANTS="${EVAL_VARIANTS:-power01 power02 $TRAIN_VARIANTS}"
MISSION_VARIANTS="${MISSION_VARIANTS:-base $EVAL_VARIANTS}"
SEEDS="${SEEDS:-3}"
ITERATIONS="${ITERATIONS:-5000}"
export ROOT SEEDS ITERATIONS PYTHONUNBUFFERED=1
cd "$REPO"

echo "=== [$(date -Is)] training: $TRAIN_VARIANTS"
VARIANTS="$TRAIN_VARIANTS" "$S/run_tracking.sh"

echo "=== [$(date -Is)] regime evaluation: $EVAL_VARIANTS"
VARIANTS="$EVAL_VARIANTS" "$S/eval_tracking.sh"

echo "=== [$(date -Is)] missions: $MISSION_VARIANTS"
failed=0
for variant in $MISSION_VARIANTS; do
  for seed in $SEEDS; do
    name="s$(printf "%02d" "$seed")"
    run_dir="$(ls -d "$ROOT/$variant/train/$name"/*_PPO2 2>/dev/null | sort | tail -1)"
    ckpt="$run_dir/checkpoints/agent_$((24 * ITERATIONS)).pt"
    if [ -z "$run_dir" ] || [ ! -f "$ckpt" ]; then echo "[FAIL] mission $variant/$name: no checkpoint"; failed=$((failed+1)); continue; fi
    out="$ROOT/$variant/mission"; mkdir -p "$out"
    manifest="$out/manifest_$name.json"
    python3 - "$manifest" "$variant" "$name" "$ckpt" "$run_dir/run_metadata.json" <<'EOF'
import json, sys
path, variant, name, ckpt, meta = sys.argv[1:]
json.dump({"vehicle": "shuttle_glider2", "vehicles": [{"name": f"{variant}_{name}", "type": "ppo2",
           "task": "shuttle_glider/tracking", "preset": "default", "checkpoint": ckpt, "run_metadata": meta}]},
          open(path, "w"), indent=2)
EOF
    for flight in "v12 12 50 0.7 75" "v14 14 70 0.45 100"; do
      set -- $flight
      dir="$out/mission_${1}_$name"
      [ -f "$dir/metrics.json" ] && { echo "[skip ] mission $variant/$name $1"; continue; }
      echo "[mission] $variant/$name $1"
      rm -rf "$dir"
      timeout 1800 "$PY" examples/rl/play_multi.py --config "$manifest" --trajectory mission --traj_speed "$2" \
        --traj_radius "$3" --traj_a_max 3 --mission_decel_amax "$4" --episode_duration "$5" --traj_z 1.5 \
        --n_envs 1 --headless --record --record_dir "$dir" --failure-distance 1000 > "$dir.log" 2>&1
      [ -f "$dir/metrics.json" ] || { echo "[FAIL] mission $variant/$name $1"; failed=$((failed+1)); }
    done
  done
done

# One line per mission: RMSE over the whole flight and per phase, failures, RMS command change per step [rad/s] of
# the lift rotors and puller
python3 - "$ROOT" <<'EOF'
import csv, json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); rows = []
for metrics in sorted(root.glob("*/mission/mission_*/metrics.json")):
    d = metrics.parent
    m = next(iter(json.load(open(metrics)).values()))
    tr = list(csv.DictReader(open(d / "trajectories.csv")))
    t = np.array([float(r["t"]) for r in tr]); e = np.array([float(r["pos_error"]) for r in tr])
    cmd = np.array([[float(r[f"cmd_w_{i}"]) for i in range(5)] for r in tr])
    # Skip the first second: play_multi primes the motors with a mid-range command before the first policy action
    after = cmd[t > 1.0]
    rate = np.sqrt((np.diff(after, axis=0) ** 2).mean(0)) if len(after) > 2 else np.full(5, np.nan)
    row = {"variant": d.parents[1].name, "flight": d.name, "failures": m["failures"],
           "rmse_cm": 100 * m["position_rmse_m"], "max_cm": 100 * float(e.max()),
           "lift_rate": float(rate[:4].mean()), "puller_rate": float(rate[4])}
    for p in json.load(open(d / "mission_phases.json")):
        sel = (t >= p["t_start"]) & (t < p["t_end"])
        if sel.any():
            row[f"{p['phase'].replace(' ', '_').replace('/', '_')}_cm"] = 100 * float(np.sqrt((e[sel] ** 2).mean()))
    rows.append(row)
if rows:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "missions_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['variant']:10s} {r['flight']:16s} failures {r['failures']}  RMSE {r['rmse_cm']:6.2f} cm  max {r['max_cm']:6.2f} cm")
    print(f"wrote {root / 'missions_summary.csv'}")
EOF
echo "=== [$(date -Is)] round finished; mission failures: $failed"
