"""Shuttle - classical cascaded (geometric) controller in Pegasus: data and metrics of its simulation.

Reads run/ (written by run.sh): trajectories.csv of play_multi (state, rotor speeds) and controller_log.csv of the
controller (CoM position, goal, commanded thrust and torques, attitude error tr(I - Rd^T R), Euler angles).
Writes data/episode.csv (one row per 10 ms step) and data/metrics.json, read by plot_results.m.

    python3 export_results.py

Positions are given as in the example episodes of the reinforcement learning studies: x and y relative to the spawn point
and z as the altitude of the /body origin of the level vehicle, i.e. the CoM altitude minus COM_OFFSET_Z, so the
vehicle starts at 1.0 m and the goal is at 1.414 m. The controller itself feeds back the CoM position.
Metrics per axis, over the first EVAL_S seconds: 2 % settling time (error within 2 % of the initial error of that
axis and staying there), overshoot (beyond the goal, in % of the initial error), and the final error.
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
OUT = HERE / "data"
EVAL_S = 5.0                      # duration of the figures and of the metrics [s]
ROTOR_CONSTANT = 1.709716e-5      # k_f [N/(rad/s)^2], vehicle_physics.SHUTTLE_PHYSICS
OMEGA_MAX = 1400.0                # [rad/s]
T_MAX = 4 * ROTOR_CONSTANT * OMEGA_MAX ** 2
COM_OFFSET_Z = 0.2422             # CoM above the /body origin [m], manifest.json "com_offset"


def read(path):
    """CSV columns as float arrays."""
    rows = list(csv.DictReader(path.open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "controller"}


def settling_time(t, err, e0, band=0.02):
    """First time after which |err| stays within band * |e0| (nan if it never does)."""
    outside = np.abs(err) > band * abs(e0)
    if not outside.any():
        return 0.0
    last = np.nonzero(outside)[0][-1]
    return float(t[last + 1]) if last + 1 < len(t) else float("nan")


def main():
    traj, ctrl = read(RUN / "trajectories.csv"), read(RUN / "controller_log.csv")
    n = min(len(traj["t"]), len(ctrl["t"]))
    t = traj["t"][:n]
    p = np.stack([ctrl[k][:n] for k in ("x", "y", "z")], axis=1)
    goal = np.stack([ctrl[k][:n] for k in ("goal_x", "goal_y", "goal_z")], axis=1)
    p0 = p[0].copy()
    p0[2] = COM_OFFSET_Z                  # x, y relative to the spawn point; z as the altitude of the /body origin
    p, goal = p - p0, goal - p0
    v = np.stack([traj[k][:n] for k in ("vx", "vy", "vz")], axis=1)
    w_rotor = np.stack([traj[f"actual_w_{i}"][:n] for i in range(4)], axis=1)
    T_cmd = ctrl["T"][:n]
    tau = np.stack([ctrl[k][:n] for k in ("tau_x", "tau_y", "tau_z")], axis=1)
    T_applied = ROTOR_CONSTANT * np.sum(w_rotor ** 2, axis=1)

    OUT.mkdir(exist_ok=True)
    cols = ["t", "x", "y", "z", "goal_x", "goal_y", "goal_z", "vx", "vy", "vz", "T_cmd", "T_applied",
            "tau_x", "tau_y", "tau_z", "att_err_tr", "roll", "pitch", "yaw", "roll_d", "pitch_d", "yaw_d",
            "w1", "w2", "w3", "w4"]
    data = np.column_stack([t, p, goal, v, T_cmd, T_applied, tau, ctrl["att_err_tr"][:n],
                            *(ctrl[k][:n] for k in ("roll", "pitch", "yaw", "roll_d", "pitch_d", "yaw_d")), w_rotor])
    with (OUT / "episode.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(cols)
        writer.writerows(np.round(data, 9).tolist())

    m = t <= EVAL_S + 1e-9
    err = p - goal
    e0 = err[0]
    axes = {}
    for i, ax in enumerate("xyz"):
        overshoot = max(0.0, float(np.max(-np.sign(e0[i]) * err[m, i]))) / abs(e0[i]) * 100.0
        axes[ax] = {"initial_error_m": float(e0[i]), "settling_time_2pct_s": settling_time(t[m], err[m, i], e0[i]),
                    "overshoot_percent": overshoot, "final_error_m": float(err[m][-1, i])}
    metrics = {
        "duration_s": EVAL_S,
        "initial_distance_m": float(np.linalg.norm(e0)),
        "axes": axes,
        "settling_time_2pct_norm_s": settling_time(t[m], np.linalg.norm(err[m], axis=1), np.linalg.norm(e0)),
        "final_error_m": float(np.linalg.norm(err[m][-1])),
        "final_error_10s_m": float(np.linalg.norm(err[-1])),
        "peak_thrust_N": float(T_cmd[m].max()), "min_thrust_N": float(T_cmd[m].min()),
        "hover_thrust_N": float(T_cmd[-1]), "max_thrust_N": T_MAX,
        "peak_torque_Nm": {ax: float(np.abs(tau[m, i]).max()) for i, ax in enumerate("xyz")},
        "peak_rotor_speed_rad_s": float(w_rotor[m].max()), "max_rotor_speed_rad_s": OMEGA_MAX,
        "peak_tilt_deg": float(np.degrees(np.arccos(np.clip(np.cos(ctrl["roll"][:n][m]) * np.cos(ctrl["pitch"][:n][m]),
                                                            -1.0, 1.0))).max()),
        "peak_att_err_tr": float(ctrl["att_err_tr"][:n][m].max()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
