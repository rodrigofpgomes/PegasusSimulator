#!/usr/bin/env python3
"""Aggregate the per-run evaluation arrays written by play.py --metrics-dir.

Usage: summarize_isaac_lab_eval.py EVAL_ROOT [--tail-s 2.0] [--settle-m 0.10]

Only the first episode of every environment is used (all checkpoints see the
same goals). Per-run metrics are means over environments; per-configuration
metrics are mean and sample standard deviation over training seeds.
Outputs summary_by_run.csv and summary_by_config.csv in EVAL_ROOT.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

METRICS = ["success", "length_s", "final_err_m", "rmse_pos_m", "settle_s", "settled_frac",
           "w_x_rms", "w_y_rms", "w_z_rms", "w_norm_rms", "w_norm_p95", "w_norm_max",
           "w_norm_rms_tail", "tilt_max_deg", "speed_max"]


def nanmean(values):
    """Mean ignoring NaN; NaN without a warning when no value is defined (e.g. no goal ever settled)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else np.nan


def episode_metrics(t, err, w, tilt, speed, tail_s, settle_m, died):
    """Return the metrics of one episode from its time series."""
    tail = t >= t.max() - tail_s
    wn = np.linalg.norm(w, axis=1)
    inside = err <= settle_m
    # first time after which the error never leaves the settle band
    outside = np.where(~inside)[0]
    if inside[-1]:
        k = 0 if outside.size == 0 else outside[-1] + 1
        settle = float(t[k])
    else:
        settle = np.nan
    return {
        "success": 0.0 if died else 1.0,
        "length_s": float(t.max()),
        "final_err_m": float(err[tail].mean()),
        "rmse_pos_m": float(np.sqrt(np.mean(err ** 2))),
        "settle_s": settle,
        "settled_frac": float(not np.isnan(settle)),
        "w_x_rms": float(np.sqrt(np.mean(w[:, 0] ** 2))),
        "w_y_rms": float(np.sqrt(np.mean(w[:, 1] ** 2))),
        "w_z_rms": float(np.sqrt(np.mean(w[:, 2] ** 2))),
        "w_norm_rms": float(np.sqrt(np.mean(wn ** 2))),
        "w_norm_p95": float(np.percentile(wn, 95)),
        "w_norm_max": float(wn.max()),
        "w_norm_rms_tail": float(np.sqrt(np.mean(wn[tail] ** 2))),
        "tilt_max_deg": float(tilt.max()),
        "speed_max": float(speed.max()),
    }


def summarize_run(run_dir, tail_s, settle_m):
    """Return the per-run mean of every metric and the number of episodes."""
    data = np.load(run_dir / "steps.npz")
    reasons = {}
    with (run_dir / "episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["episode_id"]) == 0:
                reasons[int(row["env_id"])] = row["reason"]
    rows = []
    first = data["episode"] == 0
    for env in sorted(reasons):
        sel = first & (data["env"] == env)
        if not sel.any():
            continue
        order = np.argsort(data["step"][sel])
        pos, vel = data["pos"][sel][order], data["vel"][sel][order]
        goal, t = data["goal"][sel][order], data["t"][sel][order]
        rows.append(episode_metrics(
            t, np.linalg.norm(goal - pos, axis=1), data["w_b"][sel][order], data["tilt_deg"][sel][order],
            np.linalg.norm(vel, axis=1), tail_s, settle_m, reasons[env] == "terminated"))
    out = {m: nanmean([r[m] for r in rows]) for m in METRICS}
    out["episodes"] = len(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root")
    parser.add_argument("--tail-s", type=float, default=2.0)
    parser.add_argument("--settle-m", type=float, default=0.10)
    args = parser.parse_args()
    root = Path(args.eval_root)

    runs = {}
    for run_dir in sorted(root.glob("*_s*")):
        if (run_dir / "steps.npz").exists() and (run_dir / "episodes.csv").exists():
            runs[run_dir.name] = summarize_run(run_dir, args.tail_s, args.settle_m)
    if not runs:
        sys.exit(f"no evaluated runs under {root}")

    with (root / "summary_by_run.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run"] + METRICS + ["episodes"])
        for name, r in runs.items():
            writer.writerow([name] + [f"{r[m]:.6g}" for m in METRICS] + [r["episodes"]])

    groups = {}
    for name, r in runs.items():
        groups.setdefault(re.sub(r"_s\d+$", "", name), []).append(r)
    with (root / "summary_by_config.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["config", "seeds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")])
        for cfg, rs in groups.items():
            row = [cfg, len(rs)]
            for m in METRICS:
                v = np.array([r[m] for r in rs])
                row += [f"{np.nanmean(v):.6g}", f"{np.nanstd(v, ddof=1) if len(v) > 1 else np.nan:.6g}"]
            writer.writerow(row)

    show = ["success", "final_err_m", "rmse_pos_m", "settle_s", "w_norm_rms", "w_norm_p95", "w_z_rms", "tilt_max_deg"]
    print(f"\n{'config':14s} n  " + "  ".join(f"{m:>17s}" for m in show))
    for cfg, rs in groups.items():
        cols = []
        for m in show:
            v = np.array([r[m] for r in rs])
            cols.append(f"{np.nanmean(v):7.3f}±{(np.nanstd(v, ddof=1) if len(v) > 1 else np.nan):6.3f}")
        print(f"{cfg:14s} {len(rs)}  " + "  ".join(f"{c:>17s}" for c in cols))
    print(f"\nwritten: {root/'summary_by_run.csv'}, {root/'summary_by_config.csv'}")


if __name__ == "__main__":
    main()
