#!/usr/bin/env python3
"""Compose the telemetry of a play_multi --video-dir run onto its frames and encode the video (system python, ffmpeg).

RUN_DIR is the play_multi --record_dir of the flight (trajectories.csv, metadata.json, mission_phases.json, the PX4
estimate logs px4_estimate[_<name>].jsonl) with the frames of --video-dir in RUN_DIR/frames (frames.csv: frame, step, t).
Each controller gets a panel at the bottom of the frame, named and coloured as its manifest 'label' and 'color':
  header: what the policy observes, and the running values of the mission metrics:
          RMSE of the true position, of the estimated position w.r.t. the reference (the error the policy observes) and
          of the estimate itself, and the RMS change per step of the lift-rotor commands (dOmega);
  rows (sliding window): speed, altitude and position error (true and estimated, reference dashed grey) and the four
          lift-rotor commands beside the control-surface deflections (elevator, aileron, rudder).
The estimate of a PX4 controller is aligned with its trajectory rows as in results/shuttle_glider/tracking_regimes/
export_results.py (row i <-> estimate record i). At the end of the flight the headers equal the RMSEs of the full run.
Top left: time and mission phase; top right: colour scale of the reference line (reference speed, parula).

    python3 compose_mission_video.py RUN_DIR                       # RUN_DIR/mission_video.mp4
    python3 compose_mission_video.py RUN_DIR --preview 12 36.5 80  # PNG previews of those times only
    python3 compose_mission_video.py RUN_DIR --panels glider_px4   # one panel (bottom left)
"""
import argparse
import csv
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from speed_colormap import parula  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
REF_RGB = (160, 160, 160)
PANEL_BG = (10, 12, 16, 205)
TEXT = (235, 235, 235)
DIM = (170, 175, 185)


# =============================================================================
# Data
# =============================================================================

def load_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    ctrls = meta["controllers"]
    with open(run_dir / "trajectories.csv") as f:
        rows = list(csv.DictReader(f))
    n_px4 = sum(1 for c in ctrls if c.get("px4"))
    data = {}
    for c in ctrls:
        name = c["name"]
        r = [x for x in rows if x["controller"] == name and x["env_id"] == "0"]
        A = lambda k: np.array([float(x[k]) for x in r])
        d = dict(name=name, label=c.get("label", name), color=c.get("color", "#ffffff"), px4=bool(c.get("px4")),
                 t=A("t"))
        d["p"] = np.stack([A(k) for k in "xyz"], 1)
        d["ref"] = np.stack([A(f"goal_{k}") for k in "xyz"], 1)
        d["v"] = np.stack([A(k) for k in ("vx", "vy", "vz")], 1)
        d["v_ref"] = np.stack([A(k) for k in ("ref_vx", "ref_vy", "ref_vz")], 1)
        d["cmd"] = np.stack([A(f"cmd_w_{i}") for i in range(4)], 1)
        d["surf"] = np.stack([A(f"{k}_deg") for k in ("elevator", "aileron", "rudder")], 1)
        n = len(r)
        if d["px4"]:
            path = run_dir / ("px4_estimate.jsonl" if n_px4 == 1 else f"px4_estimate_{name}.jsonl")
            with open(path) as f:
                est = [json.loads(line) for line in f][1:]   # first line: metadata
            m = min(len(est), n)
            d["p_est"], d["v_est"] = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
            d["p_est"][:m] = [e["estimated_state"][0][:3] for e in est[:m]]
            d["v_est"][:m] = [e["estimated_state"][0][3:6] for e in est[:m]]
        else:
            d["p_est"], d["v_est"] = d["p"], d["v"]
        d["e_true"] = np.linalg.norm(d["p"] - d["ref"], axis=1)
        d["e_est"] = np.linalg.norm(d["p_est"] - d["ref"], axis=1)
        d["e_nav"] = np.linalg.norm(d["p"] - d["p_est"], axis=1)
        d["speed"] = np.linalg.norm(d["v"], axis=1)
        d["speed_est"] = np.linalg.norm(d["v_est"], axis=1)
        d["speed_ref"] = np.linalg.norm(d["v_ref"], axis=1)
        k = np.arange(1, n + 1)
        # running values of the mission metrics up to each row (the last row gives the value of the whole flight)
        for key in ("e_true", "e_est", "e_nav"):
            d[f"rmse_{key}"] = np.sqrt(np.nancumsum(d[key] ** 2) / k)
        # dOmega as lift_rate of scripts/shuttle_glider/run_round.sh: changes per step after t = 1 s (the
        # first second holds play_multi's motor priming), RMS per lift rotor, mean over the four rotors
        dw2 = np.vstack([np.zeros((1, 4)), np.diff(d["cmd"], axis=0) ** 2])
        after = d["t"] > 1.0
        dw2[~after] = 0.0
        count = np.maximum(np.cumsum(after) - 1, 1)[:, None]
        dw2[np.argmax(after)] = 0.0          # the first change counted is between two samples after t = 1 s
        d["domega"] = np.where(d["t"] > 1.0 + 1e-9, np.sqrt(np.cumsum(dw2, axis=0) / count).mean(axis=1), np.nan)
        data[name] = d
    phases = json.loads((run_dir / "mission_phases.json").read_text()) if (run_dir / "mission_phases.json").is_file() else []
    dt = float(meta["arguments"]["physics_dt"])
    vmax = float(meta["arguments"].get("traj_speed") or 12.0)
    return data, phases, dt, vmax


def summary(data):
    """Whole-flight values (same definitions as run_round.sh and export_results.py)."""
    out = {}
    for name, d in data.items():
        out[name] = {"label": d["label"], "rmse_true_cm": 100 * d["rmse_e_true"][-1],
                     "rmse_est_cm": 100 * d["rmse_e_est"][-1] if d["px4"] else None,
                     "rmse_estimate_cm": 100 * d["rmse_e_nav"][-1] if d["px4"] else None,
                     "domega_rad_s": float(d["domega"][-1]), "samples": int(len(d["t"]))}
    return out


# =============================================================================
# Drawing
# =============================================================================

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lighten(rgb, a):
    return tuple(int(c + (255 - c) * a) for c in rgb)


class Fonts:
    def __init__(self, k):
        f = lambda w, s: ImageFont.truetype(str(FONT_DIR / f"Inter-{w}.otf"), max(8, int(round(s * k))))
        self.title, self.label = f("SemiBold", 34), f("Medium", 23)
        self.value, self.small, self.big = f("Regular", 22), f("Regular", 19), f("SemiBold", 40)


def fmt_len(x):
    """Length with a readable unit (cm below 1 m)."""
    return f"{100 * x:.1f} cm" if abs(x) < 1.0 else f"{x:.2f} m"


def polyline(draw, pts, rgb, width, dashed=False, phase=0, dash=(6, 4)):
    if len(pts) < 2:
        return
    if not dashed:
        draw.line(pts, fill=rgb + (255,), width=width, joint="curve")
        return
    on, off = dash
    run = []
    for j, q in enumerate(pts):
        if (phase + j) % (on + off) < on:
            run.append(q)
        else:
            if len(run) > 1:
                draw.line(run, fill=rgb + (255,), width=width)
            run = []
    if len(run) > 1:
        draw.line(run, fill=rgb + (255,), width=width)


def plot_row(draw, fonts, k, box, title, series, i_now, n_win, every, floor, lo_zero=False, fixed=None, symmetric=False):
    """box: (x0, y0, x1, y1) of the row; series: list of (name, values, rgb, dashed, value_fmt), dashed True for a
    reference (dashed), False for a main signal, None for a secondary one (thin solid)."""
    x0, y0, x1, y1 = box
    i0 = max(0, i_now - n_win)
    idx = np.arange(i0, i_now + 1, every)
    # title and legend (current values)
    draw.text((x0, y0), title, font=fonts.label, fill=TEXT)
    xl, ly, lw = x1, y0 + 14 * k, max(1, int(3 * k))
    for name, vals, rgb, dashed, vfmt in reversed(series):
        val = vfmt(vals[i_now])
        if not val:            # series without a legend entry
            continue
        txt = f"{name} {val}" if name else val
        xl -= draw.textlength(txt, font=fonts.value)
        draw.text((xl, y0 + 1 * k), txt, font=fonts.value, fill=rgb + (255,))
        if name:               # line sample: solid or dashed, as in the plot
            xl -= 34 * k
            for a, b in ([(0, 6), (10, 16), (20, 26)] if dashed else [(0, 26)]):
                draw.line([(xl + a * k, ly), (xl + b * k, ly)], fill=rgb + (255,), width=lw if dashed is not None else max(1, lw - 1))
        xl -= 22 * k
    # plot area
    py0, py1 = y0 + 42 * k, y1 - 8 * k
    px0, px1 = x0 + 64 * k, x1
    win = [v[idx] for _, v, *_ in series]
    finite = np.concatenate([w[np.isfinite(w)] for w in win]) if win else np.array([0.0])
    if fixed is not None:
        lo, hi = fixed
    elif lo_zero:
        lo, hi = 0.0, max(floor, 1.12 * float(finite.max(initial=0.0)))
    elif symmetric:
        hi = max(floor, 1.12 * float(np.abs(finite).max(initial=0.0)))
        lo = -hi
    else:
        mid = 0.5 * (finite.min() + finite.max())
        span = max(floor, 1.15 * (finite.max() - finite.min()))
        lo, hi = mid - span / 2, mid + span / 2
    span = max(hi - lo, 1e-9)
    # grid: 3 horizontal lines with labels
    for g in (0.0, 0.5, 1.0):
        yy = py1 - (py1 - py0) * g
        draw.line([(px0, yy), (px1, yy)], fill=(255, 255, 255, 38), width=1)
        val = lo + g * span
        lab = f"{val:.0f}" if span >= 6 else (f"{val:.1f}" if span >= 0.6 else f"{val:.2f}")
        lab = lab.lstrip("-") if float(lab) == 0 else lab
        tw = draw.textlength(lab, font=fonts.small)
        draw.text((px0 - 8 * k - tw, yy - 11 * k), lab, font=fonts.small, fill=DIM)
    # samples: x by time, the window always spans n_win steps (fixed time axis, newest on the right)
    xs = px1 - (px1 - px0) * (i_now - idx) / n_win
    for (name, vals, rgb, dashed, _), w in zip(series, win):
        ys = py1 - (py1 - py0) * (np.clip(w, lo - 0.05 * span, hi + 0.05 * span) - lo) / span
        pts = [(float(a), float(b)) for a, b in zip(xs, ys) if np.isfinite(b)]
        # dashed: True (reference, dashed), False (main signal), None (secondary signal: thin solid)
        polyline(draw, pts, rgb, max(1, int((3.0 if dashed is False else 2.0) * k)), bool(dashed), phase=int(idx[0] // every),
                 dash=(5, 4))


def domega_txt(d, i):
    v = d["domega"][i]
    return f"\u0394\u03a9 {v:.1f} rad/s" if np.isfinite(v) else "\u0394\u03a9 \u2013 (from t = 1 s)"


def draw_panel(draw, fonts, k, box, d, i_now, n_win, every):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=int(14 * k), fill=PANEL_BG)
    pad = 22 * k
    rgb = hex_rgb(d["color"])
    est_rgb = lighten(rgb, 0.45) if rgb != (255, 255, 255) else (150, 200, 255)
    # header: colour key, label, state source
    hx, hy = x0 + pad, y0 + 16 * k
    draw.rounded_rectangle((hx, hy + 8 * k, hx + 46 * k, hy + 26 * k), radius=int(5 * k), fill=rgb + (255,))
    draw.text((hx + 60 * k, hy), d["label"], font=fonts.title, fill=TEXT)
    src = "policy observes the PX4 EKF2 estimate  p̂" if d["px4"] else "policy observes the true state  p"
    tw = draw.textlength(src, font=fonts.value)
    draw.text((x1 - pad - tw, hy + 9 * k), src, font=fonts.value, fill=DIM)
    # running mission metrics
    if d["px4"]:
        m = (f"RMSE  true {fmt_len(d['rmse_e_true'][i_now])}   ·   estimate→ref {fmt_len(d['rmse_e_est'][i_now])}"
             f"   ·   estimate error {fmt_len(d['rmse_e_nav'][i_now])}   ·   {domega_txt(d, i_now)}")
    else:
        m = f"RMSE  true {fmt_len(d['rmse_e_true'][i_now])}   ·   {domega_txt(d, i_now)}"
    draw.text((hx, hy + 52 * k), m, font=fonts.value, fill=TEXT)
    # rows
    ry0 = y0 + 112 * k
    rows_h = (y1 - 14 * k - ry0) / 4
    f1, f2 = (lambda v: f"{v:5.1f}"), (lambda v: f"{v:5.2f}")
    flen = fmt_len
    rotor_rgb = [lighten(rgb, 0.18 * j) if rgb != (255, 255, 255) else (255 - 25 * j,) * 3 for j in range(4)]
    no_legend = lambda v: ""
    domega_now = lambda v: f"ΔΩ {d['domega'][i_now]:.1f}" if np.isfinite(d["domega"][i_now]) else "ΔΩ  –"
    surf_rgb = [lighten(rgb, a) if rgb != (255, 255, 255) else (255 - 45 * j,) * 3 for j, a in enumerate((0.0, 0.35, 0.6))]
    f_deg = lambda v: f"{v:4.1f}"
    rows = [
        ("speed [m/s]", [("true", d["speed"], rgb, False, f1)]
         + ([("est.", d["speed_est"], est_rgb, None, f1)] if d["px4"] else [])
         + [("ref.", d["speed_ref"], REF_RGB, True, f1)], dict(fixed=(0.0, 14.0))),
        ("altitude z [m]", [("true", d["p"][:, 2], rgb, False, f2)]
         + ([("est.", d["p_est"][:, 2], est_rgb, None, f2)] if d["px4"] else [])
         + [("ref.", d["ref"][:, 2], REF_RGB, True, f2)], dict(floor=1.0)),
        ("position error w.r.t. reference [m]", [("true ‖p−p_ref‖", d["e_true"], rgb, False, flen)]
         + ([("estimate ‖p̂−p_ref‖", d["e_est"], est_rgb, None, flen)] if d["px4"] else []),
         dict(floor=0.05, lo_zero=True)),
        [("lift rotors [rad/s]", [("", d["cmd"][:, j], rotor_rgb[j], False, domega_now if j == 3 else no_legend)
                                  for j in range(4)], dict(floor=60.0, lo_zero=True)),
         ("surfaces [deg]", [(n, d["surf"][:, j], surf_rgb[j], False, f_deg) for j, n in enumerate(("e", "a", "r"))],
          dict(floor=2.0, symmetric=True))],
    ]
    gap = 40 * k
    for r, row in enumerate(rows):
        cells = row if isinstance(row, list) else [row]        # a list: plots side by side in the row
        cw = (x1 - x0 - 2 * pad - gap * (len(cells) - 1)) / len(cells)
        for c, (title, series, kw) in enumerate(cells):
            cx0 = x0 + pad + c * (cw + gap)
            plot_row(draw, fonts, k, (cx0, ry0 + r * rows_h, cx0 + cw, ry0 + (r + 1) * rows_h - 6 * k), title,
                     series, i_now, n_win, every, **{"floor": 1.0, **kw})


def draw_header(draw, fonts, k, W, t, phases, vmax, title):
    phase = next((p["phase"] for p in phases if p["t_start"] <= t < p["t_end"]), phases[-1]["phase"] if phases else "")
    m = 28 * k
    lines = [(title, fonts.label, DIM), (f"t = {t:6.2f} s    {phase}", fonts.big, TEXT)]
    w = max(draw.textlength(s, font=f) for s, f, _ in lines) + 44 * k
    draw.rounded_rectangle((m, m, m + w, m + 104 * k), radius=int(14 * k), fill=PANEL_BG)
    draw.text((m + 22 * k, m + 14 * k), lines[0][0], font=fonts.label, fill=DIM)
    draw.text((m + 22 * k, m + 46 * k), lines[1][0], font=fonts.big, fill=TEXT)
    # colour scale of the reference line
    bw, bh = 520 * k, 18 * k
    bx1 = W - m - 22 * k
    bx0 = bx1 - bw
    draw.rounded_rectangle((bx0 - 22 * k, m, W - m, m + 104 * k), radius=int(14 * k), fill=PANEL_BG)
    draw.text((bx0, m + 12 * k), "reference path: speed [m/s]", font=fonts.label, fill=TEXT)
    by = m + 48 * k
    n = int(bw)
    for i in range(n):
        r, g, b, _ = parula(vmax * i / max(n - 1, 1), 0.0, vmax)
        draw.line([(bx0 + i, by), (bx0 + i, by + bh)], fill=(int(255 * r), int(255 * g), int(255 * b), 255))
    for v in np.linspace(0, vmax, 4):
        x = bx0 + bw * v / vmax
        lab = f"{v:.0f}"
        draw.text((x - draw.textlength(lab, font=fonts.small) / 2, by + bh + 4 * k), lab, font=fonts.small, fill=TEXT)


# =============================================================================
# Frames
# =============================================================================

CTX = {}


def _init(ctx):
    CTX.update(ctx)


def compose(frame_row):
    c = CTX
    img = Image.open(c["frames_dir"] / f"frame_{int(frame_row[0]):05d}.jpg").convert("RGBA")
    W, H = img.size
    k = H / 2160.0
    fonts = Fonts(k)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    t = float(frame_row[2])
    n_rows = min(len(d["t"]) for d in c["data"].values())
    i_now = int(min(round(t / c["dt"]), n_rows - 1))
    draw_header(draw, fonts, k, W, t, c["phases"], c["vmax"], c["title"])
    names = c["panels"]
    m = 28 * k
    ph = c["panel_h"] * H
    pw = (W - (len(names) + 1) * m) / len(names) if len(names) > 1 else (W - 4 * m) / 3
    n_win = int(round(c["window"] / c["dt"]))
    for j, name in enumerate(names):
        x0 = m + j * (pw + m)
        draw_panel(draw, fonts, k, (x0, H - m - ph, x0 + pw, H - m), c["data"][name], i_now, n_win, c["every"])
    img.alpha_composite(layer)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frames", default=None, help="Frames directory (default RUN_DIR/frames).")
    ap.add_argument("--out", default=None, help="Output video (default RUN_DIR/mission_video.mp4).")
    ap.add_argument("--panels", nargs="*", default=None, help="Controllers with a panel (default: all, manifest order).")
    ap.add_argument("--window", type=float, default=10.0, help="Time window of the plots [s].")
    ap.add_argument("--panel-height", type=float, default=0.36, help="Panel height, fraction of the frame height.")
    ap.add_argument("--plot-every", type=int, default=2, help="Plot every n-th physics sample.")
    ap.add_argument("--title", default="Shuttle-Glider  ·  retained policy  ·  12 m/s mission with PX4")
    ap.add_argument("--crf", type=int, default=16, help="libx264 quality (lower = better).")
    ap.add_argument("--codec", default="libx264", help="ffmpeg video encoder (libx264, hevc_nvenc, ...).")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--preview", type=float, nargs="*", default=None, help="Only write PNG previews at these times [s].")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    frames_dir = Path(args.frames).resolve() if args.frames else run_dir / "frames"
    data, phases, dt, vmax = load_run(run_dir)
    panels = args.panels or list(data)
    with open(frames_dir / "frames.csv") as f:
        frames = [(int(r["frame"]), int(r["step"]), float(r["t"])) for r in csv.DictReader(f)]
    fps = round(1.0 / (frames[1][2] - frames[0][2])) if len(frames) > 1 else 25

    s = summary(data)
    (run_dir / "video_summary.json").write_text(json.dumps(s, indent=2))
    for name, v in s.items():
        print(f"[summary] {v['label']:<36} RMSE true {v['rmse_true_cm']:7.2f} cm"
              + (f"  est->ref {v['rmse_est_cm']:6.2f} cm  estimate {v['rmse_estimate_cm']:7.2f} cm" if v["rmse_est_cm"] is not None else "")
              + f"  dOmega {v['domega_rad_s']:5.2f} rad/s  ({v['samples']} samples)")

    ctx = dict(frames_dir=frames_dir, data=data, phases=phases, dt=dt, vmax=vmax, panels=panels, window=args.window,
               panel_h=args.panel_height, every=args.plot_every, title=args.title)
    if args.preview is not None:
        _init(ctx)
        for t in args.preview:
            row = min(frames, key=lambda r: abs(r[2] - t))
            out = run_dir / f"preview_t{row[2]:06.2f}.png"
            compose(row).save(out)
            print(f"[preview] {out}")
        return

    first = Image.open(frames_dir / f"frame_{frames[0][0]:05d}.jpg")
    W, H = first.size
    out = Path(args.out).resolve() if args.out else run_dir / "mission_video.mp4"
    enc = ["-c:v", args.codec] + (["-preset", "slow", "-crf", str(args.crf)] if args.codec == "libx264"
                                  else ["-preset", "p7", "-cq", str(args.crf)])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"] + enc + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    print(f"[video] {len(frames)} frames {W}x{H} at {fps} fps -> {out}", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.jobs, initializer=_init, initargs=(ctx,)) as pool:
        for n, img in enumerate(pool.imap(compose, frames, chunksize=4)):
            proc.stdin.write(img.tobytes())
            if n % 250 == 0:
                print(f"[video] frame {n}/{len(frames)}", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"[video] done: {out}" if proc.returncode == 0 else f"[video] ffmpeg failed ({proc.returncode})", flush=True)


if __name__ == "__main__":
    main()
