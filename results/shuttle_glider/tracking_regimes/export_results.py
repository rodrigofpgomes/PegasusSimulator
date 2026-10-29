#!/usr/bin/env python3
"""Export the regime evaluations of shuttle_glider/tracking (eval_regimes.sh) to data/*.csv.

Per regime, data/episode_<regime>.csv: the episode from the trimmed start without perturbation (vehicle 0 of
<regime>_nominal) of every variant, one row per control step (100 Hz):
  t; x, y, z and x_ref, y_ref, z_ref [m] (position and reference relative to the first reference sample, rotated so
  that the initial reference heading is +x); e_p, e_v (norm of the position [m] and velocity [m/s] errors);
  e_along, e_cross, e_vert [m] (position error in the path frame: reference heading, left, up);
  v_along, v_cross, v_vert and v_along_ref, v_vert_ref [m/s] (velocity in the same frame);
  roll, pitch, heading_err [deg] (ZYX angles of the FLU body frame: pitch < 0 nose-up; heading error = body x axis
  projected on the horizontal vs reference heading);
  p, q, r and p_ref, q_ref, r_ref [deg/s] (body rates; reference R^T (0, 0, psi_dot_ref) as in the reward);
  alpha, beta [deg] (airspeed > 3 m/s, beta as the FRD sideslip); lift, drag, side [N] (aerodynamic force in wind
  axes, side = body y component, airspeed > 1 m/s); Fx, Fy, Fz and Mx, My, Mz [N, N m] (total aerodynamic force and
  moment about the CoM, FLU body axes);
  Mx_surf, My_surf, Mz_surf, Mx_rot, My_rot, Mz_rot [N m] (control moments of the surfaces and of the lift rotors,
  FLU body axes about the CoM);
  T1..T4, Tp [N] (lift rotor and puller thrust, k_f Omega^2); T_lift [N] (sum of the lift rotors); T_pitch [N]
  (front pair 1, 3 minus rear pair 2, 4); T_roll [N] (left pair 2, 3 minus right pair 1, 4); power [W];
  de, da, dr [deg] (elevator, aileron, rudder deflection after the rate limit).
data/steady_alpha_power.csv: per variant, in level cruise at 1, 2, ..., 14 m/s (200 perturbed
vehicles, t >= 3 s, logs .../<variant>/all/<regime>): median angle of attack, power, lift-rotor thrust and Fz.
data/trim_curve.csv: minimum-power trim in level flight (angle of attack of the task's trim table, shaft power of the
trim table, linearly interpolated).
data/regime_metrics.csv: per variant and regime (deceleration, helix and the regimes in <variant>/all/), over the 200
perturbed vehicles and t >= 3 s: position RMSE (median and 90th percentile over vehicles), heading error, power, |beta|,
alpha, mean |delta|, and the control moment of the surfaces cancelled by the lift rotors (min(|M_surf|, |M_rot|) where
the signs are opposite) per axis.
data/mission_v12.csv (wls_ema_m02, retained policy): mission at 12 m/s (R = 50 m; hover at 10 m, cruise climb to 20 m
and descent back, play_multi --traj_z 10 --spawn-altitude 10 --mission_climb 10) of the policy with the true state and
with the PX4 EKF2 estimate (PX4 armed; examples/rl/scripts/shuttle_glider/run_px4_mission.sh) from gazebo-classic
sensors (px4, GPS height reference) and from a standard GNSS without RTK (px4_gnss, barometric height reference), at
20 Hz. data/mission_v12_base.csv (base, appendix): the earlier PX4 flights (Pegasus default sensors, PX4 disarmed), cases px4 and px4_nodrift (GPS without random walk). Columns: t, phase, V, V_ref; for each case
(true and the PX4 cases) the true position x_, y_, z_, the error e_true_ [m] of the true position and, for the PX4
cases, the estimated position xe_, ye_, ze_, the estimated speed Ve_ [m/s] and the error e_est_ [m] of the estimate with
respect to the reference (the error the policy observes); reference x_ref, y_ref, z_ref. data/mission_v12_phases.csv: phase, t_start, t_end.

    python3 export_results.py
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "examples/rl/tasks/shuttle_glider/tracking/logs/regime_figures"
DATA = HERE / "data"
VARIANTS = ["base", "wls_ema", "wls_ema_m02"]
REGIMES = ["decel_v9_0", "helix_v12_r30_z1"]
T_SKIP = 3.0                 # [s] start of the window of the regime metrics, as in the regime table

sys.path.insert(0, str(REPO / "examples/rl/diagnostics"))
import shuttle_glider_trim as sgt  # noqa: E402


def load(path):
    """Arrays indexed [vehicle, step]."""
    d = np.load(path)
    order = np.lexsort((d["step"], d["env"]))
    n_env = int(d["env"].max()) + 1
    return {k: d[k][order].reshape(n_env, -1, *d[k].shape[1:]) for k in d.files}


def rotation(q):
    """Body-to-world rotation matrices from (w, x, y, z)."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def derived(d, veh):
    """Physical quantities [vehicle, step]."""
    t = d["t"][0]
    dt = float(t[1] - t[0])
    R = rotation(d["quat"])
    gv = d["goal_vel"]
    V_ref = np.linalg.norm(gv[..., :2], axis=-1)
    # Reference heading: direction of the reference velocity, held while the reference is (nearly) at rest
    psi_ref = np.arctan2(gv[..., 1], gv[..., 0])
    body_heading = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    for i in range(psi_ref.shape[0]):
        valid = V_ref[i] > 0.3
        last = body_heading[i, 0] if not valid[0] else psi_ref[i, 0]
        for k in range(psi_ref.shape[1]):
            last = psi_ref[i, k] if valid[k] else last
            psi_ref[i, k] = last
    psi_ref = np.unwrap(psi_ref, axis=1)
    psi_dot = np.gradient(psi_ref, dt, axis=1)
    c, s = np.cos(psi_ref), np.sin(psi_ref)
    frame = lambda v: (c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2])
    out = {"t": np.broadcast_to(t, V_ref.shape)}
    c0, s0 = np.cos(psi_ref[:, :1]), np.sin(psi_ref[:, :1])
    for name, p in (("", d["pos"]), ("_ref", d["goal"])):
        dp = p - d["goal"][:, :1]
        out["x" + name] = c0 * dp[..., 0] + s0 * dp[..., 1]
        out["y" + name] = -s0 * dp[..., 0] + c0 * dp[..., 1]
        out["z" + name] = dp[..., 2]
    out["e_p"] = np.linalg.norm(d["pos"] - d["goal"], axis=-1)
    out["e_v"] = np.linalg.norm(d["vel"] - gv, axis=-1)
    out["e_along"], out["e_cross"], out["e_vert"] = frame(d["pos"] - d["goal"])
    out["v_along"], out["v_cross"], out["v_vert"] = frame(d["vel"])
    out["v_along_ref"], _, out["v_vert_ref"] = frame(gv)
    w, x, y, z = np.moveaxis(d["quat"], -1, 0)
    out["roll"] = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    out["pitch"] = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    out["heading_err"] = np.degrees(wrap(body_heading - psi_ref))
    for j, name in enumerate("pqr"):
        out[name] = np.degrees(d["w_b"][..., j])
        out[f"{name}_ref"] = np.degrees(R[..., 2, j] * psi_dot)
    va = d["v_air"]
    Va = np.linalg.norm(va, axis=-1)
    alpha = np.arctan2(-va[..., 2], va[..., 0])
    out["alpha"] = np.where(Va > 3.0, np.degrees(alpha), np.nan)
    out["beta"] = np.where(Va > 3.0, np.degrees(np.arcsin(np.clip(-va[..., 1] / np.maximum(Va, 1e-6), -1, 1))), np.nan)
    f = d["f_aero"]
    moving = Va > 1.0
    out["lift"] = np.where(moving, f[..., 0] * np.sin(alpha) + f[..., 2] * np.cos(alpha), np.nan)
    out["drag"] = np.where(moving, -(f * va).sum(-1) / np.maximum(Va, 1e-6), np.nan)
    out["side"] = np.where(moving, f[..., 1], np.nan)
    for j, axis in enumerate("xyz"):
        out[f"F{axis}"], out[f"M{axis}"] = f[..., j], d["m_aero"][..., j]
    for j, axis in enumerate("xyz"):
        out[f"M{axis}_surf"], out[f"M{axis}_rot"] = d["m_surf"][..., j], d["m_rot"][..., j]
    omega = d["omega"].astype(np.float64)
    for i in range(4):
        out[f"T{i + 1}"] = veh["kf"][i] * omega[..., i] ** 2
    out["Tp"] = veh["kf_p"] * omega[..., 4] ** 2
    front = np.asarray(veh["r_rotor"])[:, 0] > 0
    T = np.stack([out[f"T{i + 1}"] for i in range(4)], -1)
    out["T_lift"] = T.sum(-1)
    out["T_pitch"] = T[..., front].sum(-1) - T[..., ~front].sum(-1)
    left = np.asarray(veh["r_rotor"])[:, 1] > 0
    out["T_roll"] = T[..., left].sum(-1) - T[..., ~left].sum(-1)
    out["power"] = (veh["km"] * omega[..., :4] ** 3).sum(-1) + veh["km_p"] * omega[..., 4] ** 3
    for j, name in enumerate(("de", "da", "dr")):
        out[name] = np.degrees(d["surfaces"][..., j])
    # The first sample precedes the first physics step of the episode (no aerodynamic wrench yet)
    for key in ("lift", "drag", "side", "Fx", "Fy", "Fz", "Mx", "My", "Mz"):
        out[key][:, 0] = np.nan
    return out


def metrics(variant, regime, X):
    s = X["t"][0] >= T_SKIP
    rms = lambda a: np.sqrt(np.nanmean(a[:, s] ** 2, axis=1))
    e_p = rms(np.sqrt(X["e_along"] ** 2 + X["e_cross"] ** 2 + X["e_vert"] ** 2))
    row = {"variant": variant, "regime": regime, "vehicles": X["t"].shape[0],
           "pos_rmse_cm": 100 * np.median(e_p), "pos_rmse_p90_cm": 100 * np.percentile(e_p, 90),
           "heading_rms_deg": np.median(rms(X["heading_err"])), "power_W": np.nanmean(X["power"][:, s]),
           "beta_abs_deg": np.nanmean(np.abs(X["beta"][:, s])), "alpha_deg": np.nanmean(X["alpha"][:, s])}
    for name in ("de", "da", "dr"):
        row[f"{name}_abs_deg"] = np.mean(np.abs(X[name][:, s]))
    for axis in "xyz":
        a, b = X[f"M{axis}_surf"][:, s], X[f"M{axis}_rot"][:, s]
        row[f"M{axis}_cancelled_Nm"] = np.mean(np.where(a * b < 0, np.minimum(np.abs(a), np.abs(b)), 0.0))
    return row


VARIANT_LOGS = REPO / "examples/rl/tasks/shuttle_glider/tracking/logs/variants"
MISSION_CASES = (("true", "mission"),
                 ("px4", "mission_px4_height_ref/gpsref_inair_gazebo"),
                 ("px4_gnss", "mission_px4_height_ref/baroref_inair_gnss"))
MISSION_CASES_OLD = (("true", "mission"), ("px4", "mission_px4"), ("px4_nodrift", "mission_px4_gps_nodrift"))


def export_mission(variant="wls_ema_m02", suffix="", flight="mission_v12alt_s03", every=5, cases=MISSION_CASES):
    """Mission of one policy: true state vs PX4 estimate (play_multi records, px4_estimate.jsonl)."""
    MISSION = VARIANT_LOGS / variant
    cols = {}
    for case, sub in cases:
        d = MISSION / sub / flight
        with open(d / "trajectories.csv") as f:
            rows = list(csv.DictReader(f))
        A = lambda k: np.array([float(r[k]) for r in rows])
        if case == "true":
            cols["t"] = A("t")
            cols["V"] = A("speed")
            cols["V_ref"] = np.linalg.norm(np.stack([A("ref_vx"), A("ref_vy"), A("ref_vz")], 1), axis=1)
            for k in "xyz":
                cols[f"{k}_ref"] = A(f"goal_{k}")
            phases = json.load(open(d / "mission_phases.json"))
        pos, ref = np.stack([A(k) for k in "xyz"], 1), np.stack([A(f"goal_{k}") for k in "xyz"], 1)
        if case == "true":
            ref_true = ref
        elif ref.shape != ref_true.shape or np.abs(ref - ref_true).max() > 1e-4:
            raise ValueError(f"{d}: reference differs from the true-state flight")
        for i, k in enumerate("xyz"):
            cols[f"{k}_{case}"] = pos[:, i]
        cols[f"e_true_{case}"] = np.linalg.norm(pos - ref, axis=1)
        if case != "true":
            with open(d / "px4_estimate.jsonl") as f:
                est = [json.loads(line) for line in f][1:]
            n = min(len(est), len(rows))
            xe = np.full((len(rows), 3), np.nan)
            xe[:n] = np.array([e["estimated_state"][0][:3] for e in est[:n]])
            ve = np.full((len(rows), 3), np.nan)
            ve[:n] = np.array([e["estimated_state"][0][3:6] for e in est[:n]])
            cols[f"Ve_{case}"] = np.linalg.norm(ve, axis=1)
            for i, k in enumerate("xyz"):
                cols[f"{k}e_{case}"] = xe[:, i]
            cols[f"e_est_{case}"] = np.linalg.norm(xe - ref, axis=1)
    t = cols["t"]
    cols["phase"] = np.array([next((p["phase"] for p in phases if p["t_start"] <= x < p["t_end"]), phases[-1]["phase"])
                              for x in t])
    keys = ["t", "phase"] + [k for k in cols if k not in ("t", "phase")]
    with open(DATA / f"mission_v12{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(0, len(t), every):
            w.writerow([cols[k][i] if k == "phase" else f"{cols[k][i]:.6g}" for k in keys])
    with open(DATA / f"mission_v12{suffix}_phases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "t_start", "t_end"])
        w.writerows([p["phase"], f"{p['t_start']:.3f}", f"{p['t_end']:.3f}"] for p in phases)
    print(f"data/mission_v12{suffix}.csv")


STEADY = [(f"cruise_v{v}", float(v), "cruise") for v in range(1, 15)]     # level flight at every speed


def export_steady(veh):
    """Angle of attack and power in steady flight per variant (200 perturbed vehicles, t >= 3 s; logs .../all/)."""
    rows = []
    for regime, speed, kind in STEADY:
        for v in VARIANTS:
            X = derived(load(LOGS / v / "all" / regime / "steps.npz"), veh)
            s = X["t"][0] >= 3.0
            rows.append({"variant": v, "regime": regime, "kind": kind, "speed": speed,
                         "alpha_deg": f"{np.nanmedian(np.nanmean(X['alpha'][:, s], axis=1)):.3f}",
                         "power_W": f"{np.median(np.nanmean(X['power'][:, s], axis=1)):.2f}",
                         "T_lift_N": f"{np.median(np.nanmean(X['T_lift'][:, s], axis=1)):.3f}",
                         "Fz_N": f"{np.nanmedian(np.nanmean(X['Fz'][:, s], axis=1)):.3f}"})
    with open(DATA / "steady_alpha_power.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("data/steady_alpha_power.csv")
    # Minimum-power trim in level flight: angle of attack of the task's trim table and shaft power of Table
    # minimum-power trim table (examples/rl/diagnostics/shuttle_glider_trim.py --limits-table)
    spec = importlib.util.spec_from_file_location(
        "trim_table", REPO / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/shuttle_glider_trim.py")
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
    p_knots = (0, 2, 4, 6, 8, 10, 12, 13, 14, 16)
    p_trim = (663.6, 656.2, 631.3, 562.3, 425.5, 263.8, 120.0, 71.5, 45.8, 28.0)
    V = np.asarray(tt.TRIM_SPEED_KNOTS)
    with open(DATA / "trim_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed", "alpha_deg", "power_W"])
        w.writerows([f"{x:g}", f"{al:.2f}", f"{np.interp(x, p_knots, p_trim):.1f}"] for x, al in zip(V, tt.TRIM_ALPHA_DEG))
    print("data/trim_curve.csv")


def main():
    veh = sgt.load_vehicle()
    DATA.mkdir(exist_ok=True)
    export_steady(veh)
    export_mission("wls_ema_m02")            # retained policy (main text)
    export_mission("base", "_base", cases=MISSION_CASES_OLD)   # appendix (earlier PX4 flights only)
    rows = []
    for regime in REGIMES:
        columns, episode = None, []
        for v in VARIANTS:
            rows.append(metrics(v, regime, derived(load(LOGS / v / regime / "steps.npz"), veh)))
            N = derived(load(LOGS / v / f"{regime}_nominal" / "steps.npz"), veh)
            columns = list(N)
            episode += [[v] + [f"{N[k][0, i]:.6g}" for k in columns] for i in range(N["t"].shape[1])]
        with open(DATA / f"episode_{regime}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant"] + columns)
            w.writerows(episode)
        print(f"data/episode_{regime}.csv")
    # The other regimes of the regime table (examples/rl/scripts/shuttle_glider/eval_tracking.sh with OUT_ROOT: <variant>/all/<regime>)
    for regime in sorted(p.name for p in (LOGS / VARIANTS[0] / "all").iterdir() if (p / "steps.npz").exists()):
        if regime in REGIMES:
            continue
        for v in VARIANTS:
            rows.append(metrics(v, regime, derived(load(LOGS / v / "all" / regime / "steps.npz"), veh)))
    with open(DATA / "regime_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (f"{x:.4g}" if isinstance(x, float) else x) for k, x in r.items()} for r in rows)
    print("data/regime_metrics.csv")


if __name__ == "__main__":
    main()
