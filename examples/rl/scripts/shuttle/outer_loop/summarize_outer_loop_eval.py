#!/usr/bin/env python3
"""Summarise an outer-loop (acceleration level) study written by run_outer_loop_study.sh.

Usage: summarize_outer_loop_eval.py ROOT [--tail-s 2.0]

Reads ROOT/<experiment>/eval/<run>/{steps.npz, eval_config.json} (experiments include "lqr") and scores
every controller with the same quadratic return, whatever reward it was trained with:
    J = -sum_k gamma^k (q_pos |ep_{k+1}|^2 + q_vel |ev_{k+1}|^2 + r_u |u_k|^2) dt
where u_k = max_acceleration * clamp(action_k, -1, 1) (the acceleration applied during step k) and the
weights, dt and max_acceleration come from eval_config.json. Each row of steps.npz is (x_k, action_k).
Only the first episode of every environment is used (all runs see the same goals).

Per-goal metrics: survival, final error (mean of the last tail_s), position RMSE, t20 (first time the error
falls below 20 % of its initial value), J, mean |u|^2, fraction of steps with a saturated action.
Per run: means over goals (t20: median). Per experiment: mean and sample std over seeds.
Outputs ROOT/outer_loop_by_run.csv and ROOT/outer_loop_by_config.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ["survival", "final_err_m", "rmse_pos_m", "t20_s", "J_quad", "u2_mean", "sat_frac"]
ORDER = ["quad", "alive", "death", "exp", "alive_death", "alive_death200", "lqr"]


def episode_metrics(pos, vel, goal, action, cfg, dt, gamma, n_full, tail_s):
    """Metrics of one episode from its (x_k, action_k) rows."""
    ep = goal - pos
    ev = -vel
    err = np.linalg.norm(ep, axis=1)
    a = np.clip(action, -1.0, 1.0)
    u = cfg["max_acceleration"] * a
    cost = (cfg["q_pos"] * (ep[1:] ** 2).sum(1) + cfg["q_vel"] * (ev[1:] ** 2).sum(1)
            + cfg["r_u"] * (u[:-1] ** 2).sum(1)) * dt
    t = np.arange(len(pos)) * dt
    hit = np.where(err <= 0.2 * err[0])[0]
    return {
        "survival": float(len(pos) >= n_full),
        "final_err_m": float(err[t >= t[-1] - tail_s].mean()),
        "rmse_pos_m": float(np.sqrt((err ** 2).mean())),
        "t20_s": float(t[hit[0]]) if hit.size else np.nan,
        "J_quad": float(-(gamma ** np.arange(len(cost)) * cost).sum()),
        "u2_mean": float((u[:-1] ** 2).sum(1).mean()),
        "sat_frac": float((np.abs(action[:-1]) >= 1.0).any(1).mean()),
    }


def summarize_run(run_dir, gamma, tail_s):
    """Mean metrics over the goals of one evaluation folder."""
    cfg_all = json.loads((run_dir / "eval_config.json").read_text())
    cfg = cfg_all["env_cfg_effective"]
    cfg.setdefault("max_acceleration", 1.0)
    dt = cfg["sim_dt"] * cfg["decimation"]
    n_full = int(round(cfg["episode_length_s"] / dt)) - 1
    d = np.load(run_dir / "steps.npz")
    per_goal = []
    for env in np.unique(d["env"]):
        m = (d["env"] == env) & (d["episode"] == 0)
        per_goal.append(episode_metrics(d["pos"][m], d["vel"][m], d["goal"][m][0], d["action"][m],
                                        cfg, dt, gamma, n_full, tail_s))
    out = {k: float(np.nanmedian([g[k] for g in per_goal])) if k == "t20_s" and
           not all(np.isnan(g[k]) for g in per_goal) else
           (np.nan if all(np.isnan(g[k]) for g in per_goal) else float(np.nanmean([g[k] for g in per_goal])))
           for k in METRICS}
    out["goals"] = len(per_goal)
    out["max_acceleration"] = cfg["max_acceleration"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--tail-s", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.99)
    args = p.parse_args()

    runs, configs = [], []
    experiments = [e for e in ORDER if (args.root / e / "eval").is_dir()]
    for exp in experiments:
        rows = []
        for run_dir in sorted((args.root / exp / "eval").iterdir()):
            if (run_dir / "steps.npz").is_file() and (run_dir / "eval_config.json").is_file():
                row = {"experiment": exp, "run": run_dir.name, **summarize_run(run_dir, args.gamma, args.tail_s)}
                rows.append(row)
        runs += rows
        if rows:
            agg = {"experiment": exp, "seeds": len(rows)}
            for k in METRICS:
                v = np.array([r[k] for r in rows], dtype=float)
                v = v[~np.isnan(v)]
                agg[k + "_mean"] = float(v.mean()) if v.size else np.nan
                agg[k + "_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            configs.append(agg)

    for name, rows in (("outer_loop_by_run.csv", runs), ("outer_loop_by_config.csv", configs)):
        if rows:
            with (args.root / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"{'experiment':<12}{'n':>3}" + "".join(f"{k:>20}" for k in METRICS))
    for c in configs:
        print(f"{c['experiment']:<12}{c['seeds']:>3}"
              + "".join(f"{c[k + '_mean']:>11.3f} ±{c[k + '_std']:>6.3f}" for k in METRICS))


if __name__ == "__main__":
    main()
