#!/usr/bin/env python3
"""
| File: summarize_tracking_eval.py
| Description: Metrics of reference-tracking evaluations (play.py --metrics-dir, lemniscate or any moving reference).

For one evaluation folder (steps.npz + episodes.csv), over the first episode of every vehicle:
  survival       fraction of episodes not terminated
  rmse_cm        position RMSE [cm] over all samples with t >= T_SKIP of the surviving episodes
  rmse_xy_cm     the same for the horizontal error;  rmse_z_cm  the same for the vertical error
  err_max_cm     median over surviving episodes of the largest position error after T_SKIP [cm]
  d_action       mean ||a_k - a_{k-1}|| per step after T_SKIP
  w_rms          RMSE of the body angular rate after T_SKIP [rad/s]
The first T_SKIP seconds (recovery from the initial state and ramp of the reference) are excluded.

Usage:
  summarize_tracking_eval.py ROOT [--t-skip 5] [--csv out.csv]
  with ROOT/<policy>/a<peak acceleration>/ evaluation folders (as written by eval_direct_rotor_lemniscate.sh).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

T_SKIP = 5.0


def tracking_metrics(eval_dir: Path, t_skip: float = T_SKIP) -> dict:
    """Return the tracking metrics of one evaluation folder."""
    episodes = [r for r in csv.DictReader(open(eval_dir / "episodes.csv")) if int(r["episode_id"]) == 0]
    survived = [int(r["env_id"]) for r in episodes if r["reason"] != "terminated"]
    z = np.load(eval_dir / "steps.npz")
    first = z["episode"] == 0
    env, step, t = z["env"][first], z["step"][first], z["t"][first]
    order = np.lexsort((step, env))
    env, t = env[order], t[order]
    err = (z["pos"][first] - z["goal"][first])[order]
    act = np.clip(z["action"][first], -1.0, 1.0)[order]
    w = z["w_b"][first][order]
    keep = np.isin(env, survived) & (t >= t_skip - 1e-9)
    same = np.r_[False, env[1:] == env[:-1]]
    d_act = np.zeros(len(env))
    d_act[1:] = np.linalg.norm(act[1:] - act[:-1], axis=1)
    ok = keep & same
    per_episode_max = [np.linalg.norm(err[keep & (env == e)], axis=1).max() for e in survived if (keep & (env == e)).any()]
    return {
        "episodes": len(episodes), "survival": len(survived) / len(episodes),
        "rmse_cm": 100 * float(np.sqrt((err[keep] ** 2).sum(axis=1).mean())),
        "rmse_xy_cm": 100 * float(np.sqrt((err[keep][:, :2] ** 2).sum(axis=1).mean())),
        "rmse_z_cm": 100 * float(np.sqrt((err[keep][:, 2] ** 2).mean())),
        "err_max_cm": 100 * float(np.median(per_episode_max)) if per_episode_max else float("nan"),
        "d_action": float(d_act[ok].mean()) if ok.any() else float("nan"),
        "w_rms": float(np.sqrt((w[keep] ** 2).sum(axis=1).mean())),
    }


def main():
    """Print the table of every <policy>/a<level> folder under ROOT and optionally write it to CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--t-skip", type=float, default=T_SKIP)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    rows = []
    for d in sorted(args.root.glob("*/a*")):
        if (d / ".done").exists():
            rows.append({"policy": d.parent.name, "peak_acc": float(d.name[1:]), **tracking_metrics(d, args.t_skip)})
    rows.sort(key=lambda r: (r["policy"], r["peak_acc"]))
    print(f"{'policy':10s} {'a_peak':>6s} {'surv':>6s} {'rmse':>7s} {'rmse_xy':>8s} {'rmse_z':>7s} {'err_max':>8s} {'|da|':>7s} {'w_rms':>6s}")
    for r in rows:
        print(f"{r['policy']:10s} {r['peak_acc']:6.1f} {r['survival']:6.3f} {r['rmse_cm']:7.1f} {r['rmse_xy_cm']:8.1f} "
              f"{r['rmse_z_cm']:7.1f} {r['err_max_cm']:8.1f} {r['d_action']:7.4f} {r['w_rms']:6.2f}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
