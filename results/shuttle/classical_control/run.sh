#!/usr/bin/env bash
# Classical cascaded controller (position outer loop and geometric attitude inner loop) on the Shuttle in Pegasus, for
# the goal of the example episodes of the thrust-and-moment and acceleration studies (environment 69 of the 200
# evaluation goals, 1.59 m). Gains (manifest.json, force units): Kp = m 3.15, Kd = m 2.37 (the LQR of the acceleration
# study, m = 3.5 kg), Ki = 0, kR = 3.5, kw = 0.5, gyroscopic term w x Jw with the Shuttle inertia, desired attitude
# from y_c = [-sin psi, cos psi, 0].
# The fed-back position is that of the CoM, 0.2422 m above the /body origin of shuttle.usda (Pegasus reports the
# origin position and the CoM velocity), so the goal is the CoM goal: the same displacement (1.59 m) as in
# the reinforcement learning studies, where the goal refers to the origin (0.6592 = 0.4170 + 0.2422). Physics and control at 100 Hz; rotors
# start at the hover speed. Writes run/ (trajectories.csv of play_multi and controller_log.csv of the controller).
set -eu
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
OUT="results/shuttle/classical_control/run"
cd "$REPO"
mkdir -p "$OUT"
PYTHONUNBUFFERED=1 "$PY" examples/rl/play_multi.py --config results/shuttle/classical_control/manifest.json \
    --vehicle shuttle --n_envs 1 --physics-dt 0.01 --spawn-altitude 1.0 \
    --goal_offset -1.3273372650146484 0.7694244384765625 0.6592484542846680 \
    --episode_duration 10 --num_episodes_per_env 1 --headless --record --record_dir "$OUT" > "$OUT/run.log" 2>&1
echo "done: $OUT"
