#!/usr/bin/env bash
# Shuttle-Glider, retained policy wls_ema_m02 (seed 3): mission at 12 m/s (R = 50 m, hover at 10 m, climb to 20 m,
# 180 deg turn, descent, deceleration at 0.7 m/s^2, final hover; 101 s), flown with the true state and with the PX4
# EKF2 estimate. PX4 only estimates the state; the policy keeps the actuation.
#
# Cases (-> $OUT/<case>/mission_v12alt_s03, read by results/shuttle_glider/tracking_regimes/export_results.py):
#   true                 simulator state (configs/shuttle_glider_true_state.json), recorded to variants/wls_ema_m02/mission
#   gpsref_inair_gazebo  sensors as the PX4 gazebo-classic plugins (IMU noise and turn-on biases, GPS with gazebo's bias
#                        statistics and 120 ms delay, projected in double precision with the PX4 Earth radius), default
#                        EKF2 (GPS height reference), airframe 10055_none_shuttle_glider_inair
#   baroref_inair_gnss   standard GNSS receiver without RTK (Gauss-Markov bias 0.5 m horizontal and 1 m vertical, 300 s
#                        correlation), barometer as the EKF2 height reference, airframe 10058_none_shuttle_glider_baro_inair
# In both PX4 cases the backend force-arms PX4 after the EKF warmup (arm_after_warmup), so that the land detector
# reports flight and EKF2 fuses airspeed and sideslip; the airframes disable the auto-disarm on landing.
#
# Requirements: the airframes 10051, 10053, 10055 and 10058 of examples/rl/px4_vehicles installed in PX4-Autopilot
# (see REPRODUCE.md) and the trained policy in variants/wls_ema_m02/train/s03.
# Usage (from the repository root): examples/rl/scripts/shuttle_glider/run_px4_mission.sh [--dry-run]
# Environment: CASES (default "true gpsref_inair_gazebo baroref_inair_gnss"), ISAACSIM_PYTHON.
# The PX4 logs (ulog) of each case are kept in <case>/px4_log (PEGASUS_PX4_LOG_DIR).
set -u
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; cd "$REPO"
PY="${ISAACSIM_PYTHON:-$HOME/isaacsim_5.1.0/python.sh}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
V="examples/rl/tasks/shuttle_glider/tracking/logs/variants/wls_ema_m02"
C="examples/rl/configs"
CASES="${CASES:-true gpsref_inair_gazebo baroref_inair_gnss}"
declare -A CFG=([true]="$C/shuttle_glider_true_state.json"
                [gpsref_inair_gazebo]="$C/px4_shuttle_glider_estimator_inair_gazebo.json"
                [baroref_inair_gnss]="$C/px4_shuttle_glider_estimator_baro_inair_gnss.json")
declare -A DIR=([true]="$V/mission"
                [gpsref_inair_gazebo]="$V/mission_px4_height_ref/gpsref_inair_gazebo"
                [baroref_inair_gnss]="$V/mission_px4_height_ref/baroref_inair_gnss")
MISSION=(--seed 42 --physics-dt 0.01 --spawn-altitude 10.0 --failure-distance 1000.0 --n_envs 1 --device cuda:0
         --headless --episode_duration 101.0 --vehicle iris --trajectory mission --traj_amplitude 1.5 --traj_period 8.0
         --traj_z 10.0 --traj_speed 12.0 --traj_radius 50.0 --traj_z_amplitude 0.3 --traj_z_period 8.0
         --traj_climb_rate 0.15 --traj_ramp_time 2.5 --traj_a_max 3.0 --init_speed 0.0 --traj_lemniscate_ramp 0.0
         --traj_z_ramp_tau 2.5 --traj_turn_direction left --mission_turn_angle 180.0 --mission_hover 3.0
         --mission_cruise 5.0 --mission_climb 10.0 --mission_climb_rate 1.0 --mission_curvature_ramp 2.0
         --mission_final_hover 5.0 --mission_decel_amax 0.7 --mission_profile level --record --record_every 1
         --num_episodes_per_env 1)
for case in $CASES; do
  d="${DIR[$case]}"; [ -f "$d/.done" ] && { echo "[skip ] $case"; continue; }
  ARGS=("${MISSION[@]}" --config "${CFG[$case]}" --record_dir "$d/mission_v12alt_s03")
  echo "[run  ] $case: ${ARGS[*]}"
  [ $DRY = 1 ] && continue
  mkdir -p "$d"
  PEGASUS_PX4_LOG_DIR="$d/px4_log" "$PY" examples/rl/play_multi.py "${ARGS[@]}" > "$d/stdout.log" 2>&1
  code=$?; [ $code -eq 0 ] && [ -f "$d/mission_v12alt_s03/metrics.json" ] && touch "$d/.done" || echo "[FAIL ] $case ($code)"
done
