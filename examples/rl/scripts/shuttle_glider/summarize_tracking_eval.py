#!/usr/bin/env python3
"""
| File: summarize_tracking_eval.py
| Description: Metrics of the Shuttle-Glider flight-regime evaluations (eval_tracking.sh, task shuttle_glider/tracking).

For each ROOT/<variant>/eval/sNN/<regime>/ folder (play.py steps.npz + episodes.csv), first episode of every vehicle:
  survival        fraction of episodes not terminated
  rmse_cm         position RMSE [cm], samples with t >= T_SKIP of the surviving episodes; rmse_z_cm vertical part
  vel_rmse        velocity RMSE [m/s] (same samples)
  speed           mean inertial speed [m/s]
  heading_deg     mean |angle| between the horizontal body x axis and the reference velocity (reference >= 2 m/s)
  d_action        mean ||a_k - a_{k-1}|| per step
  act_std         mean over episodes and channels of the command standard deviation (chattering, bang-bang ~0.8)
  w_rms           RMS body angular rate [rad/s]
  yaw_net         mean |net yaw rate| per episode [deg/s] (heading drift; in a turn, the turn rate)
  lift_lo, lift_hi, puller_lo   fraction of lift-rotor commands at a <= -0.99 / a >= 0.99, puller at a <= -0.99
  lift_cmd, puller_cmd        mean normalised command (a + 1) / 2 of the lift rotors and of the puller
  elev, ail, rud  mean |surface command| (fraction of the deflection range)
The first T_SKIP seconds (recovery from the initial state, ramp of the reference) are excluded from all but survival.

Usage:
  summarize_tracking_eval.py ROOT [--t-skip 3] [--csv out.csv]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

COLUMNS = ("survival", "rmse_cm", "rmse_z_cm", "vel_rmse", "speed", "heading_deg", "d_action", "act_std",
           "w_rms", "yaw_net", "lift_lo", "lift_hi", "puller_lo", "lift_cmd", "puller_cmd", "elev", "ail", "rud")


def regime_metrics(eval_dir: Path, t_skip: float) -> dict:
    """Return the metrics of one evaluation folder."""
    episodes = [r for r in csv.DictReader(open(eval_dir / "episodes.csv")) if int(r["episode_id"]) == 0]
    survived = [int(r["env_id"]) for r in episodes if r["reason"] != "terminated"]
    z = np.load(eval_dir / "steps.npz")
    first = z["episode"] == 0
    order = np.lexsort((z["step"][first], z["env"][first]))
    get = lambda key: z[key][first][order]
    env, t = get("env"), get("t")
    act = np.clip(get("action"), -1.0, 1.0)
    keep = np.isin(env, survived) & (t >= t_skip - 1e-9)
    same = np.r_[False, env[1:] == env[:-1]]
    d_act = np.r_[0.0, np.linalg.norm(act[1:] - act[:-1], axis=1)]

    out = {"episodes": len(episodes), "survival": len(survived) / max(len(episodes), 1)}
    if not keep.any():
        return out | {k: float("nan") for k in COLUMNS[1:]}
    err = get("pos")[keep] - get("goal")[keep]
    vel, goal_vel = get("vel")[keep], get("goal_vel")[keep]

    # Body x axis in the world frame from the quaternion (w, x, y, z)
    w, x, y, zq = get("quat")[keep].T
    bx = np.stack((1 - 2 * (y * y + zq * zq), 2 * (x * y + w * zq)), axis=1)
    ref_xy = goal_vel[:, :2]
    moving = np.linalg.norm(ref_xy, axis=1) >= 2.0
    heading = np.degrees(np.abs(np.arctan2(ref_xy[:, 0] * bx[:, 1] - ref_xy[:, 1] * bx[:, 0],
                                           (ref_xy * bx).sum(axis=1))))
    a = act[keep]
    yaw = np.arctan2(bx[:, 1], bx[:, 0])
    env_k, t_k = env[keep], t[keep]
    act_std, yaw_net = [], []
    for e in np.unique(env_k):
        sel = env_k == e
        act_std.append(a[sel].std(axis=0).mean())
        if sel.sum() > 1:
            yaw_e = np.unwrap(yaw[sel])
            yaw_net.append(abs(np.degrees(yaw_e[-1] - yaw_e[0])) / max(t_k[sel][-1] - t_k[sel][0], 1e-6))
    out |= {
        "rmse_cm": 100 * float(np.sqrt((err ** 2).sum(axis=1).mean())),
        "rmse_z_cm": 100 * float(np.sqrt((err[:, 2] ** 2).mean())),
        "vel_rmse": float(np.sqrt(((vel - goal_vel) ** 2).sum(axis=1).mean())),
        "speed": float(np.linalg.norm(vel, axis=1).mean()),
        "heading_deg": float(heading[moving].mean()) if moving.any() else float("nan"),
        "d_action": float(d_act[keep & same].mean()),
        "act_std": float(np.mean(act_std)),
        "w_rms": float(np.sqrt((get("w_b")[keep] ** 2).sum(axis=1).mean())),
        "yaw_net": float(np.mean(yaw_net)) if yaw_net else float("nan"),
        "lift_lo": float((a[:, :4] <= -0.99).mean()), "lift_hi": float((a[:, :4] >= 0.99).mean()),
        "puller_lo": float((a[:, 4] <= -0.99).mean()),
        "lift_cmd": float(((a[:, :4] + 1) / 2).mean()),
        "puller_cmd": float(((a[:, 4] + 1) / 2).mean()),
        "elev": float(np.abs(a[:, 5]).mean()), "ail": float(np.abs(a[:, 6]).mean()), "rud": float(np.abs(a[:, 7]).mean()),
    }
    return out


def main():
    """Print the table of every <variant>/eval/<seed>/<regime> folder under ROOT and optionally write it to CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--t-skip", type=float, default=3.0)
    p.add_argument("--csv", type=Path)
    args = p.parse_args()

    rows = []
    for done in sorted(args.root.glob("*/eval/*/*/.done")):
        eval_dir = done.parent
        regime, seed, variant = eval_dir.name, eval_dir.parent.name, eval_dir.parents[2].name
        rows.append({"variant": variant, "seed": seed, "regime": regime} | regime_metrics(eval_dir, args.t_skip))
    if not rows:
        print(f"no finished evaluations under {args.root}")
        return

    header = ("variant", "seed", "regime", "episodes") + COLUMNS
    print(" ".join(f"{h:>11}" for h in header))
    for r in rows:
        print(" ".join(f"{r[h]:>11.3f}" if isinstance(r[h], float) else f"{str(r[h]):>11}" for h in header))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
