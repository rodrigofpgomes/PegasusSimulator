#!/usr/bin/env python3
"""Compare the three vehicles of a video run (play_multi, tracking_wls_ema_m02_px4_video.json) with the separate runs
of the reference runs (logs variants/wls_ema_m02/{mission,mission_px4,mission_px4_gps_nodrift}).

Per vehicle: RMSE of the true position, of the estimated position w.r.t. the reference and of the estimate itself
[cm], dOmega (RMS change per step of the lift-rotor commands) [rad/s], maximum true error [m], survival; and the
distance between the video-run path and the reference-run path (max and RMS over the flight) [m].

    python3 check_against_runs.py RUN_DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_mission_video import load_run  # noqa: E402

VARIANT = Path(__file__).resolve().parents[3] / "tasks/shuttle_glider/tracking/logs/variants/wls_ema_m02"
REFERENCE = {"glider_true_state": "mission", "glider_px4": "mission_px4", "glider_px4_gps_nodrift": "mission_px4_gps_nodrift"}
FLIGHT = "mission_v12alt_s03"
TABLE = {"glider_true_state": (1.2, None, None, 0.5), "glider_px4": (246, 11.7, 243, 20.3),
         "glider_px4_gps_nodrift": (93, 5.3, 93, 10.2)}   # published values


def metrics(d):
    return dict(true=100 * d["rmse_e_true"][-1], est=100 * d["rmse_e_est"][-1] if d["px4"] else None,
                estimate=100 * d["rmse_e_nav"][-1] if d["px4"] else None, domega=float(d["domega"][-1]),
                max_true=float(d["e_true"].max()), max_dz=float(np.abs(d["p"][:, 2] - d["ref"][:, 2]).max()),
                n=len(d["t"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    new, *_ = load_run(args.run_dir)
    report = {}
    fmt = lambda v, w=7, p=1: f"{v:{w}.{p}f}" if v is not None else " " * (w - 2) + "--"
    print(f"{'vehicle':<26}{'source':<9}{'true':>8}{'est':>8}{'estim.':>8}{'dOmega':>8}{'max e':>8}{'max dz':>8}  samples")
    for name, d in new.items():
        reference_run = VARIANT / REFERENCE[name] / FLIGHT
        old = load_run(reference_run)[0]
        o = next(iter(old.values()))
        o["px4"] = d["px4"]
        mn, mo = metrics(d), metrics(o)
        n = min(len(d["t"]), len(o["t"]))
        gap = np.linalg.norm(d["p"][:n] - o["p"][:n], axis=1)
        report[name] = dict(video=mn, reference_run=mo, published_table=dict(zip(("true", "est", "estimate", "domega"), TABLE[name])),
                            path_gap_max_m=float(gap.max()), path_gap_rms_m=float(np.sqrt((gap ** 2).mean())))
        tt = TABLE[name]
        for src, m in (("video", mn), ("reference", mo)):
            print(f"{name:<26}{src:<9}{fmt(m['true'])}{fmt(m['est'])}{fmt(m['estimate'])}{fmt(m['domega'])}"
                  f"{fmt(m['max_true'], 8, 2)}{fmt(m['max_dz'], 8, 2)}  {m['n']}")
        print(f"{'':<26}{'table':<9}{fmt(tt[0])}{fmt(tt[1])}{fmt(tt[2])}{fmt(tt[3])}")
        print(f"{'':<26}path gap video vs reference run: max {gap.max():.2f} m, RMS {np.sqrt((gap ** 2).mean()):.2f} m")
    out = Path(args.run_dir) / "check_against_runs.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[check] {out}")


if __name__ == "__main__":
    main()
