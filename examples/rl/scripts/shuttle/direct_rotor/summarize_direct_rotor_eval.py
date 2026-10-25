#!/usr/bin/env python3
"""
| File: summarize_direct_rotor_eval.py
| Description: Summarize play.py --metrics-dir evaluations of direct rotor policies.

For each evaluation folder (steps.npz + episodes.csv) prints and writes to CSV:
  survival       fraction of the first episodes (one per vehicle) that were not terminated; play.py also logs
                 the next episode of a vehicle that terminated early, which is not one of the evaluation episodes
  err_final_cm   position error averaged over the last 2 s of each surviving episode, then over episodes
  err_rms_cm     position RMSE over all steps of the surviving episodes
  sat_pct        share of rotor commands at the limits (|a| >= 0.99), last 2 s
  d_action       mean ||a_k - a_{k-1}|| per step (the reward's action-change term), last 2 s
  w_rms          RMSE of the body angular rate [rad/s], last 2 s

Usage:
  summarize_direct_rotor_eval.py EVAL_DIR [EVAL_DIR ...] [--csv out.csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

FINAL_WINDOW_S = 2.0


def first_episodes(eval_dir: Path) -> list:
    """Rows of episodes.csv for the first episode of each vehicle (the evaluation episodes)."""
    return [r for r in csv.DictReader(open(eval_dir / "episodes.csv")) if int(r["episode_id"]) == 0]


def survival(eval_dir: Path) -> float:
    """Fraction of the evaluation episodes that were not terminated."""
    episodes = first_episodes(eval_dir)
    return sum(r["reason"] != "terminated" for r in episodes) / len(episodes)


def summarize(eval_dir: Path) -> dict:
    """Return the metrics of one evaluation folder."""
    episodes = first_episodes(eval_dir)
    # A surviving vehicle runs the full episode, so all its samples belong to its first episode
    survived = [int(r["env_id"]) for r in episodes if r["reason"] != "terminated"]

    z = np.load(eval_dir / "steps.npz")
    # Order the samples by vehicle, then by step, so consecutive rows are consecutive steps
    order = np.lexsort((z["step"], z["env"]))
    env, t = z["env"][order], z["t"][order]
    action = np.clip(z["action"][order], -1.0, 1.0)
    err = np.linalg.norm(z["pos"][order] - z["goal"][order], axis=1)
    w_b = z["w_b"][order]

    alive = np.isin(env, survived)
    final = alive & (t >= t.max() - FINAL_WINDOW_S + 1e-9)
    consecutive = np.r_[False, env[1:] == env[:-1]] & final
    d_action = np.zeros(len(env))
    d_action[1:] = np.linalg.norm(action[1:] - action[:-1], axis=1)

    per_episode = [err[final & (env == k)].mean() for k in survived]
    return {
        "run": eval_dir.name,
        "survival": len(survived) / len(episodes),
        "err_final_cm": 100 * float(np.mean(per_episode)),
        "err_rms_cm": 100 * float(np.sqrt((err[alive] ** 2).mean())),
        "sat_pct": 100 * float((np.abs(action[final]) >= 0.99).mean()),
        "d_action": float(d_action[consecutive].mean()),
        "w_rms": float(np.sqrt((w_b[final] ** 2).sum(axis=1).mean())),
    }


def main():
    """Print the table and optionally write it to CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("eval_dirs", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    rows = [summarize(d) for d in args.eval_dirs]
    keys = list(rows[0])
    print(f"{'run':24s} {'surv':>6s} {'err_end':>8s} {'err_rms':>8s} {'sat%':>6s} {'|da|':>6s} {'w_rms':>6s}")
    for r in rows:
        print(f"{r['run']:24s} {r['survival']:6.3f} {r['err_final_cm']:8.1f} {r['err_rms_cm']:8.1f} "
              f"{r['sat_pct']:6.1f} {r['d_action']:6.3f} {r['w_rms']:6.2f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
