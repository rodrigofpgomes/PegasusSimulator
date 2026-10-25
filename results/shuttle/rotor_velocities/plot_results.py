#!/usr/bin/env python3
"""Figures of the direct rotor velocity study (Shuttle, RAPTOR-style tracking task).

SAC (RAPTOR pre-training setup, seed 1) and three PPO2 variants (seeds 1-3, 600 iterations):
  initial  - RSL-RL configuration of the force/moment section (unbounded Gaussian mean)
  tanh     - policy mean bounded by tanh
  tanh_std - tanh and exploration standard deviation bounded to [0.2, 1]
Deterministic evaluation: 500 episodes with the training initial states and references, seed 1000.
Text is typeset by LaTeX (Computer Modern), at 9 pt.

Reads data/*.csv (export_results.py); writes figures/*.pdf and *.png.
    python3 plot_results.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import csv  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIGS = HERE / "data", HERE / "figures"
CM = 1 / 2.54

# Categorical slots 1-3 of the reference palette (validated for all pairs); SAC is the neutral reference
COLOR = {"initial": "#2a78d6", "tanh": "#eb6834", "tanh_std": "#1baf7a", "sac": "#404040"}
LIGHT = "#c8c8c8"
VARIANTS = ["initial", "tanh", "tanh_std"]
LABEL = {"initial": r"PPO, initial", "tanh": r"PPO, $\tanh$", "tanh_std": r"PPO, $\tanh$ + $\sigma_{\min}$",
         "sac": "SAC"}
SHORT = {"initial": "Initial", "tanh": r"$\tanh$", "tanh_std": r"\shortstack{$\tanh$\\$+\,\sigma_{\min}$}",
         "sac": "SAC"}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300,
})


def save(fig, name):
    """Write the figure as vector PDF and 300 dpi PNG."""
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"figures/{name}.pdf")


class Table(dict):
    """Columns of a CSV file as NumPy arrays (numeric where possible); rows selected with t[mask]."""

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, key):
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        return Table({k: v[key] for k, v in self.items()})


def read(name):
    """Load data/<name>.csv."""
    rows = list(csv.DictReader((DATA / name).open()))
    out = Table()
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(c) for c in col])
        except ValueError:
            out[k] = np.array(col)
    return out


def rolling(y, window, func):
    """Centred moving statistic, ignoring NaN, shorter at the ends."""
    h = window // 2
    return np.array([func(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def by_seed(df, x, y):
    """Common x values and one column per seed."""
    xs = np.unique(df[x][~np.isnan(df[x])])
    seeds = np.unique(df.seed)
    Y = np.full((len(xs), len(seeds)), np.nan)
    for j, s in enumerate(seeds):
        m = df.seed == s
        for i, xv in enumerate(xs):
            hit = np.flatnonzero(df[x][m] == xv)
            if hit.size:
                Y[i, j] = df[y][m][hit[0]]
    return xs, Y


def fig_sac(T, E):
    """SAC training: position error and survival versus transitions."""
    Ts = T[T.variant == "sac"]; Ts = Ts[np.argsort(Ts.transitions)]
    Es = E[E.variant == "sac"]; Es = Es[np.argsort(Es.transitions)]
    x, xe = Ts.transitions / 1e6, Es.transitions / 1e6
    fig, (a, b) = plt.subplots(1, 2, figsize=(16 * CM, 6 * CM), constrained_layout=True)

    a.plot(x, Ts.final_distance_m, ".", ms=2, color=LIGHT, label="Training episodes (end of episode)")
    a.plot(x, rolling(Ts.final_distance_m, 25, np.nanmedian), color=COLOR["sac"],
           lw=1.1, label="Training, moving median")
    a.plot(xe, Es.err_final_cm / 100, "-o", color=COLOR["tanh_std"], lw=1.3, ms=3.5,
           label=r"Evaluation, last 2\,s")
    a.set_yscale("log"); a.set_ylim(0.02, 2); a.set_xlim(0, 1.0)
    a.set_xlabel(r"Transitions [$\times 10^{6}$]"); a.set_ylabel("Position error [m]")
    a.set_title("(a) Position error"); a.legend(loc="upper right")

    b.plot(x, rolling(1 - Ts.died, 25, np.nanmean), color=COLOR["sac"], lw=1.1,
           label="Training, moving mean")
    b.plot(xe, Es.survival, "-o", color=COLOR["tanh_std"], lw=1.3, ms=3.5, label="Evaluation")
    b.set_ylim(0, 1.02); b.set_xlim(0, 1.0)
    b.set_xlabel(r"Transitions [$\times 10^{6}$]"); b.set_ylabel("Survival rate")
    b.set_title("(b) Survival"); b.legend(loc="lower right")
    save(fig, "fig01_sac_training")


def fig_ppo(T, E, sac):
    """PPO2 variants: training and evaluation versus iteration; band = range over the three seeds."""
    panels = [("final_distance_m", "Position error [m]", "(a) Training, end of episode", T),
              ("err_final_cm", "Position error [cm]", r"(b) Evaluation, last 2\,s", E),
              ("sat_pct", r"Commands at $\pm 1$ [\%]", "(c) Evaluation, saturated commands", E),
              ("survival", "Survival rate", "(d) Evaluation, survival", E)]
    fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 11.5 * CM), constrained_layout=True)
    for ax, (col, ylab, title, df) in zip(axes.flat, panels):
        for v in VARIANTS:
            it, Y = by_seed(df[df.variant == v], "iteration", col)
            if df is T:
                Y = np.column_stack([rolling(Y[:, j], 5, np.nanmean) for j in range(Y.shape[1])])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # checkpoints where every episode fails
                lo, hi, mean = np.nanmin(Y, 1), np.nanmax(Y, 1), np.nanmean(Y, 1)
            ax.fill_between(it, lo, hi, color=COLOR[v], alpha=0.18, lw=0)
            style = dict(color=COLOR[v], lw=1.3, label=LABEL[v])
            ax.plot(it, mean, "-" if df is T else "-o", ms=3, **style)
        if df is E:
            ax.axhline(sac[col], ls="--", color=COLOR["sac"], lw=1.0, label=r"SAC, final ($10^{6}$ transitions)")
        ax.set_xlim(0, 600); ax.set_xlabel("PPO iteration"); ax.set_ylabel(ylab); ax.set_title(title)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylim(0.05, 2)
    axes[0, 1].set_yscale("log"); axes[0, 1].set_ylim(3, 150)
    axes[1, 0].set_ylim(0, 100); axes[1, 1].set_ylim(0, 1.02)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    save(fig, "fig02_ppo2_training")


def fig_final(M):
    """Final policies: one marker per seed, short bar at the mean."""
    cols = [("err_final_cm", "Final position error [cm]", "(a) Accuracy"),
            ("sat_pct", r"Commands at $\pm 1$ [\%]", "(b) Saturation"),
            ("w_rms", "Angular rate RMSE [rad/s]", "(c) Angular rate")]
    names = VARIANTS + ["sac"]
    fig, axes = plt.subplots(1, 3, figsize=(16 * CM, 5.4 * CM), constrained_layout=True)
    for ax, (col, ylab, title) in zip(axes, cols):
        for i, v in enumerate(names, start=1):
            y = M[M.variant == v][col]
            ax.scatter(i + np.linspace(-0.08, 0.08, len(y)), y, s=14, color=COLOR[v], zorder=3,
                       edgecolors="white", linewidths=0.6)
            ax.plot([i - 0.22, i + 0.22], [y.mean()] * 2, color=COLOR[v], lw=1.6)
        ax.set_xticks(range(1, len(names) + 1), [SHORT[v] for v in names])
        ax.set_xlim(0.4, len(names) + 0.6)
        top = ax.get_ylim()[1]; ax.set_ylim(-0.04 * top, top)
        ax.grid(axis="x", visible=False); ax.set_ylabel(ylab); ax.set_title(title)
    save(fig, "fig03_final_per_seed")


def fig_episode(X):
    """One episode (same initial state and reference): the four rotor commands and the position error."""
    pols = VARIANTS + ["sac"]
    fig, axes = plt.subplots(len(pols) + 1, 1, figsize=(16 * CM, 13 * CM), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [1] * len(pols) + [1.4]})
    for ax, p in zip(axes, pols):
        D = X[X.policy == p]
        for r in range(4):
            ax.plot(D.t, D[f"a{r}"], color=COLOR[p], lw=0.6, alpha=0.8)
        ax.set_ylim(-1.15, 1.15); ax.set_yticks([-1, 0, 1]); ax.set_ylabel(r"$a_i$")
        ax.text(0.995, 0.93, LABEL[p], transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax = axes[-1]
    for p in pols:
        D = X[X.policy == p]
        ax.plot(D.t, D.err, color=COLOR[p], lw=1.2, label=LABEL[p])
    ax.set_yscale("log"); ax.set_ylim(0.01, 1); ax.set_xlim(0, 5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel(r"$\|\mathbf{p}-\mathbf{p}_r\|$ [m]")
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig04_example_episode")


def fig_ablation(A):
    """Reward and observation variants: final error against the command variation of every run.
    Colour: velocity cost; marker: attitude term; open marker: seed other than 3."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12 * CM, 7.4 * CM), constrained_layout=True)
    marker = {"qz": "o", "none": "^", "trace": "s"}
    color = {False: COLOR["tanh"], True: COLOR["initial"]}
    # label offsets in points (dx, dy, horizontal alignment), by run
    place = {"B0": (6, 0, "left"), "B0_s01": (0, 7, "center"), "B0_s02": (-6, 0, "right"), "R6_s03": (9, -9, "left"),
             "R1": (6, -11, "left"), "R2": (6, 3, "left"), "R5": (6, -4, "left"), "O2_s01": (6, -4, "left"),
             "O2_s02": (6, -4, "left"), "O1": (6, 3, "left"), "O2": (6, 3, "left"), "R3": (6, -3, "left"),
             "R4": (6, 3, "left"), "R7_s03": (9, 1, "left"), "R8_s03": (-6, -9, "right"), "R9_s03": (6, 3, "left")}
    for i, run in enumerate(A.run):
        tag = str(run).split("_")[0]
        with_vel, solid = A.w_vel[i] > 0, int(A.seed[i]) == 3
        c = color[bool(with_vel)]
        shape = "D" if str(A.ang_cost[i]) != "none" else marker[str(A.ori_cost[i])]
        ax.scatter(A.d_action[i], A.err_final_cm[i], s=34 if shape != "D" else 26, marker=shape, zorder=3,
                   linewidths=1.1, facecolors=c if solid else "white", edgecolors=c)
        label = tag if solid and tag != "R6" else (f"{tag} (s{int(A.seed[i])})" if not solid else tag)
        dx, dy, ha = place.get(str(run), (6, 3, "left"))
        ax.annotate(label, (A.d_action[i], A.err_final_cm[i]), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(8e-4, 0.4); ax.set_ylim(0, 12)
    ax.set_xlabel(r"Command variation $\lVert\Delta\mathbf{a}\rVert$ per step (last 2\,s)")
    ax.set_ylabel("Final position error [cm]")
    gray = dict(linestyle="", color="0.35")
    handles = [Line2D([], [], marker="o", linestyle="", color=color[True], label="With velocity cost"),
               Line2D([], [], marker="o", linestyle="", color=color[False], label="Without velocity cost"),
               Line2D([], [], marker="o", label=r"Attitude: $q_z$", **gray),
               Line2D([], [], marker="^", label="Attitude: none", **gray),
               Line2D([], [], marker="s", label="Attitude: geometric", **gray),
               Line2D([], [], marker="D", label="Angular-rate cost", **gray),
               Line2D([], [], marker="o", linestyle="", color="0.35", markerfacecolor="white", label="Other seed")]
    ax.legend(handles=handles, loc="upper right", fontsize=7, handletextpad=0.2, borderaxespad=0.2, ncol=2,
              columnspacing=0.8)
    ax.set_ylim(0, 13)
    save(fig, "fig05_ablation")


def lemniscate_points(L):
    """{(policy, peak acceleration): row}; the run with the relaxed speed bound replaces the one with the training
    bound, and a level where every vehicle was lost at the training speed bound is not shown."""
    pts = {}
    for i in range(len(L.policy)):
        key = (str(L.policy[i]), float(L.peak_acc[i]))
        row = {k: L[k][i] for k in L if k != "policy"}
        if key not in pts or row["speed_bound"] > pts[key]["speed_bound"]:
            pts[key] = row
    return {k: v for k, v in pts.items() if v["survival"] >= 0.5}


FM_COLOR = "#8e44ad"
LEM_STYLE = {"B0": ("#404040", "-", "o", r"PPO baseline"), "SAC": ("#8a8a8a", "--", "x", "SAC"),
             "PRETRAIN": ("#8a8a8a", ":", "+", "RAPTOR pre-training"),
             "A1": (COLOR["initial"], "-", "o", r"PPO + $\mathbf{a}_r$"),
             "A2": (COLOR["tanh_std"], "-", "s", r"PPO + $\mathbf{a}_r$ + $\lVert\mathbf{v}-\mathbf{v}_r\rVert$"),
             "O1": (COLOR["tanh"], "-", "^", r"PPO + $\mathbf{a}_r$, geometric")}


def fig_lemniscate(L):
    """RMSE and mean along-track error against the peak acceleration of the figure eight."""
    pts = lemniscate_points(L)
    levels = sorted({k[1] for k in pts})
    speed = {a: np.nanmean([v["ref_speed_max"] for k, v in pts.items() if k[1] == a]) for a in levels}
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM), constrained_layout=True)
    for ax, col, ylab, title in ((axes[0], "rmse_cm", "Position RMSE [cm]", "(a) Tracking error"),
                                 (axes[1], "along_mean_cm", "Mean error along the path [cm]", "(b) Lag")):
        ax.axvspan(0, 1.4, color="0.5", alpha=0.12, lw=0)
        seeds = [[pts.get((p, a)) for a in levels] for p in ("B0", "B0_s1", "B0_s2")]
        lo = [min(v[col] for v in col_ if v) if all(col_) else np.nan for col_ in zip(*seeds)]
        hi = [max(v[col] for v in col_ if v) if all(col_) else np.nan for col_ in zip(*seeds)]
        ok = ~np.isnan(lo)
        ax.fill_between(np.array(levels)[ok], np.array(lo)[ok], np.array(hi)[ok], color=LEM_STYLE["B0"][0], alpha=0.18, lw=0)
        # thrust/moment policy: median and interquartile range over its seeds, where most seeds were not lost
        fm = {a: [pts[(p, a)][col] for p in sorted({k[0] for k in pts if re.fullmatch(r"FM_s\d\d", k[0])})
                  if (p, a) in pts and pts[(p, a)][col] == pts[(p, a)][col]] for a in levels}
        fa = [a for a in levels if len(fm[a]) >= 6]
        if fa:
            q = np.array([np.percentile(fm[a], [25, 50, 75]) for a in fa])
            ax.fill_between(fa, q[:, 0], q[:, 2], color=FM_COLOR, alpha=0.18, lw=0)
            ax.plot(fa, q[:, 1], "-", color=FM_COLOR, marker="d", ms=3.2, lw=1.2, label="Thrust and moments")
        for pol, (c, ls, mk, label) in LEM_STYLE.items():
            xs = [a for a in levels if (pol, a) in pts]
            ax.plot(xs, [pts[(pol, a)][col] for a in xs], ls, color=c, marker=mk, ms=3.2, lw=1.2, label=label)
        ax.set_xlabel(r"Peak reference acceleration [m/s$^2$]"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_xlim(0, max(levels) * 1.03)
        sec = ax.secondary_xaxis("top", functions=(lambda a: np.interp(a, levels, [speed[x] for x in levels]),
                                                   lambda v: np.interp(v, [speed[x] for x in levels], levels)))
        sec.set_xlabel(r"Peak reference speed [m/s]", fontsize=8); sec.tick_params(labelsize=7)
        sec.set_xticks([round(speed[a], 1) for a in levels if a in (0.5, 2, 4, 6, 8, 10)])
    axes[1].axhline(0, color="0.3", lw=0.6)
    axes[0].text(0.03, 0.96, "training range", transform=axes[0].get_xaxis_transform(), fontsize=7, va="top", rotation=90)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    save(fig, "fig06_lemniscate")


def fig_lemniscate_paths(P):
    """Horizontal path of one episode, reference and vehicle, for the policies of the export. The reference is a wide
    grey band under the paths with a thin white dashed line on top, so it stays visible where the paths overlap it."""
    levels = sorted(set(P.peak_acc))
    fig, axes = plt.subplots(1, len(levels), figsize=(16 * CM, 6.4 * CM), constrained_layout=True, squeeze=False)
    legend = {}
    for ax, a in zip(axes[0], levels):
        m = P.peak_acc == a
        ref = P[m & (P.policy == "B0")]
        under, = ax.plot(ref.ref_x, ref.ref_y, "-", color="#bdbdbd", lw=5.0, solid_capstyle="round", zorder=1)
        over, = ax.plot(ref.ref_x, ref.ref_y, linestyle=(0, (3, 2)), color="w", lw=1.0, zorder=5)
        legend.setdefault("Reference", (under, over))
        for pol in ("FM_s03", "B0", "PRETRAIN", "A1"):
            d = P[m & (P.policy == pol)]
            if len(d.x) == 0:
                continue
            c, label = (FM_COLOR, "Thrust and moments") if pol == "FM_s03" else (LEM_STYLE[pol][0], LEM_STYLE[pol][3])
            line, = ax.plot(d.x, d.y, "-", color=c, lw=1.0, zorder=3)
            legend.setdefault(label, line)
        ax.set_aspect("equal"); ax.set_xlabel(r"$x - x_0$ [m]"); ax.set_ylabel(r"$y - y_0$ [m]")
        ax.set_title(rf"$a_{{\mathrm{{peak}}}} = {a:g}$ m/s$^2$")
    fig.legend(list(legend.values()), list(legend.keys()), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=7.5)
    save(fig, "fig07_lemniscate_paths")


def main():
    T, E, M, X = (read(n) for n in ("training_curves.csv", "eval_checkpoints.csv", "run_metrics.csv",
                                    "example_episode.csv"))
    sac = {k: v[0] for k, v in M[M.variant == "sac"].items()}
    fig_sac(T, E)
    fig_ppo(T, E, sac)
    fig_final(M)
    fig_episode(X)
    if (DATA / "lemniscate_metrics.csv").exists():
        fig_lemniscate(read("lemniscate_metrics.csv"))
    if (DATA / "lemniscate_paths.csv").exists():
        fig_lemniscate_paths(read("lemniscate_paths.csv"))
    if (DATA / "ablation_metrics.csv").exists():
        fig_ablation(read("ablation_metrics.csv"))
    cols = ("survival", "err_final_cm", "err_rms_cm", "sat_pct", "d_action", "w_rms")
    print("variant      " + " ".join(f"{c:>18s}" for c in cols))
    for v in VARIANTS + ["sac", "raptor_pretrain", "std_only"]:
        S = M[M.variant == v]
        sd = (lambda a: f" +- {np.std(a, ddof=1):.3f}" if len(a) > 1 else "")
        print(f"{v:12s} " + " ".join(f"{np.mean(S[c]):9.3f}{sd(S[c]):>9s}" for c in cols))


if __name__ == "__main__":
    main()
