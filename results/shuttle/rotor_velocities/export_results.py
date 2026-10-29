#!/usr/bin/env python3
"""Export the direct rotor velocity study (Shuttle, RAPTOR-style tracking) to CSV files (plotted by plot_results.py).

Task: examples/rl/tasks/shuttle/direct_rotor/baseline/raptor_tracking. Sources (logs/ of that task):
  sac_vs_ppo2/sac/train/sac_s01        SAC, RAPTOR pre-training setup, 1e6 transitions, seed 1
  ppo2_default_600                     PPO2 initial configuration (RSL-RL), 600 iterations, seeds 1-3
  ppo2_tanh_600                        PPO2 with the policy mean bounded by tanh, seeds 1-3
  ppo2_tanh_600_std0p2                 PPO2 with tanh and the exploration std bounded to [0.2, 1], seeds 1-3
  ppo2_default_600_std0p2              PPO2 with the std bound only (seed 1, ablation, not plotted)
  eval_checkpoints                     deterministic evaluation of intermediate checkpoints
                                       (examples/rl/scripts/shuttle/direct_rotor/eval_direct_rotor_checkpoints.sh)
  eval_reference/sac_s01               deterministic evaluation of the final SAC checkpoint
  eval_reference/raptor_pretrain       RAPTOR pre-training actor (rl-tools SAC, 1e6 steps, Shuttle dynamics json),
                                       play.py --controller pretrain, same episodes as every other policy
Every evaluation: play.py, 500 episodes, training initial states and references, seed 1000.

  ablation/raptor_tracking/logs/<run>   reward/observation ablation of the baseline (seed 3: R1-R5, O1, O2;
                                       control runs of run_direct_rotor_ablation_extra.sh: R6_s03, O2_s01, ...)

  ablation/raptor_tracking/logs/eval_lemniscate[_v5]/<policy>/a<peak acceleration>   lemniscate tracking
                                       (eval_direct_rotor_lemniscate.sh; _v5: speed bound relaxed from 2 to 5 m/s)
  ablation/raptor_tracking/logs/<FM_LEMNISCATE>/FM_sNN/a<peak acceleration>   the same figure eight for the ten seeds
                                       of the thrust/moment policy (eval_force_moment_lemniscate.sh): c_M 0.25, d_0 0.8,
                                       w_psi 1.01, k_T 3.9 with the hover at a0 = 0 (FM_sNN) and k_T 1.9 (FMk19_sNN)

Writes data/training_curves.csv, eval_checkpoints.csv, run_metrics.csv, survival_by_tilt.csv, ablation_metrics.csv,
example_episode.csv.
    python3 export_results.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = REPO / "examples/rl/tasks/shuttle/direct_rotor/baseline/raptor_tracking/logs"
sys.path.insert(0, str(REPO / "examples/rl/scripts/shuttle/direct_rotor"))
from summarize_direct_rotor_eval import summarize, survival  # noqa: E402
from summarize_tracking_eval import tracking_metrics  # noqa: E402

OUT = HERE / "data"
N_ENVS_PPO = 4096          # PPO2 vehicles; one skrl timestep is one step of every vehicle
STEPS_PER_ITER = 24        # PPO2 rollout length
PPO = {"initial": "ppo2_default_600", "tanh": "ppo2_tanh_600", "tanh_std": "ppo2_tanh_600_std0p2"}
ABLATION = {"std_only": "ppo2_default_600_std0p2"}
SEEDS = (1, 2, 3)
EXAMPLE_SEED = 3           # seed selected for the following tests (lowest rms and final error)
TRAIN_TAGS = {"final_distance_m": "Metrics_final_distance_to_goal", "died": "Episode_Termination_died",
              "reward_pos": "Episode_Reward_pos", "reward_d_action": "Episode_Reward_d_action",
              "policy_std": "Policy _ Standard deviation--"}
TILT_BINS = ((0.0, 30.0), (30.0, 57.3), (57.3, 90.1))   # initial rotation angle [deg]; rl-tools samples <= 57.3
METRICS = ("survival", "err_final_cm", "err_rms_cm", "sat_pct", "d_action", "w_rms")


def read_scalar(run_dir, tag):
    """Return {step: value} of one exported TensorBoard scalar, or {} when absent."""
    files = [f for f in (run_dir / "download_csvs").glob("*.csv") if tag in f.name]
    if not files:
        return {}
    rows = list(csv.reader(files[0].open()))[1:]
    return {int(float(r[1])): float(r[2]) for r in rows}


def run_dir(study, seed, algo="ppo2"):
    """Training run folder written by skrl inside the study layout."""
    pattern = "*_PPO2" if algo == "ppo2" else "*_SACDelayed*"
    return sorted((LOGS / study / algo / "train" / f"{algo}_s{seed:02d}").glob(pattern))[-1]


def training_rows(algo, variant, seed, rdir):
    """One row per logged step: training scalars of the stochastic policy."""
    series = {k: read_scalar(rdir, tag) for k, tag in TRAIN_TAGS.items()}
    rows = []
    for step in sorted(series["final_distance_m"]):
        it = step / STEPS_PER_ITER if algo == "ppo2" else np.nan
        transitions = step * N_ENVS_PPO if algo == "ppo2" else step
        rows.append({"algo": algo, "variant": variant, "seed": seed, "step": step, "iteration": it,
                     "transitions": transitions, **{k: s.get(step, np.nan) for k, s in series.items()}})
    return rows


def eval_row(algo, variant, seed, iteration, transitions, eval_dir):
    """Deterministic-evaluation metrics from metrics.csv, or computed from steps.npz; the survival rate is
    always recomputed from episodes.csv (first episode of each vehicle)."""
    metrics = eval_dir / "metrics.csv"
    m = next(csv.DictReader(metrics.open())) if metrics.exists() else summarize(eval_dir)
    m["survival"] = survival(eval_dir)
    return {"algo": algo, "variant": variant, "seed": seed, "iteration": iteration, "transitions": transitions,
            **{k: float(m[k]) for k in METRICS}}


ABLATION_LOGS = REPO / "examples/rl/tasks/shuttle/direct_rotor/ablation/raptor_tracking/logs"
ABLATION_ORDER = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "O1", "O2")
BASELINE_CFG = {"ori_cost": "qz", "w_ori": 0.1, "w_vel": 0.0, "ang_cost": "none", "w_ang": 0.0,
                "obs_ref_acc": False, "obs_attitude": "R"}
FINAL_WINDOW_S = 2.0


def yaw_metrics(eval_dir):
    """Heading of the surviving episodes from the recorded quaternion (re-evaluation, eval_yaw): median |yaw| of the
    mean over the last 2 s [deg] and median |yaw rate| fitted over the last 2 s [deg/s]; NaN when not recorded."""
    if not (eval_dir / "steps.npz").exists():
        return {"yaw_end_deg": float("nan"), "yaw_rate_deg_s": float("nan")}
    died = {int(r["env_id"]) for r in csv.DictReader((eval_dir / "episodes.csv").open())
            if int(r["episode_id"]) == 0 and r["reason"] == "terminated"}
    z = np.load(eval_dir / "steps.npz")
    keep = z["episode"] == 0
    env, t, step, q = z["env"][keep], z["t"][keep], z["step"][keep], z["quat"][keep]
    order = np.lexsort((step, env))
    env, t, q = env[order], t[order], q[order]
    w, x, y, zz = q.T
    yaw = np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
    bounds = np.flatnonzero(np.r_[True, env[1:] != env[:-1], True])
    end, rate = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if int(env[a]) in died:
            continue
        unwrapped, tt = np.unwrap(yaw[a:b]), t[a:b]
        last = tt >= tt.max() - FINAL_WINDOW_S + 1e-9
        end.append(abs((np.degrees(unwrapped[last]).mean() + 180.0) % 360.0 - 180.0))
        rate.append(abs(np.degrees(np.polyfit(tt[last], unwrapped[last], 1)[0])))
    return {"yaw_end_deg": float(np.median(end)), "yaw_rate_deg_s": float(np.median(rate))}


def ablation_row(name, eval_dir, seed, cfg, yaw_dir=None):
    """Metrics of one ablation evaluation: the standard ones plus the split by reference type and the spread of
    the rotor commands (first episode of every vehicle, surviving episodes, last 2 s)."""
    m = summarize(eval_dir)
    reasons = {int(r["env_id"]): r["reason"] for r in csv.DictReader((eval_dir / "episodes.csv").open())
               if int(r["episode_id"]) == 0}
    z = np.load(eval_dir / "steps.npz")
    keep = z["episode"] == 0
    env, t = z["env"][keep], z["t"][keep]
    err = np.linalg.norm(z["pos"][keep] - z["goal"][keep], axis=1)
    goal, act, w = z["goal"][keep], np.clip(z["action"][keep], -1.0, 1.0), z["w_b"][keep]
    final_err = {True: [], False: []}
    a_std, w_z = [], []
    for e in sorted(k for k, r in reasons.items() if r != "terminated"):
        i = np.flatnonzero(env == e)
        last = i[t[i] >= t.max() - FINAL_WINDOW_S + 1e-9]
        final_err[bool(np.abs(goal[i] - goal[i][0]).max() > 1e-4)].append(err[last].mean())
        a_std.append(act[last].std(axis=0).mean())
        w_z.append((w[last, 2] ** 2).mean())
    return {"run": name, "seed": seed, **cfg,
            "episodes": len(reasons), "failures": sum(r == "terminated" for r in reasons.values()),
            "survival": survival(eval_dir),
            "err_final_cm": m["err_final_cm"], "err_final_fixed_cm": 100 * float(np.mean(final_err[False])),
            "err_final_langevin_cm": 100 * float(np.mean(final_err[True])), "err_rms_cm": m["err_rms_cm"],
            "sat_pct": m["sat_pct"], "d_action": m["d_action"], "a_std": float(np.mean(a_std)),
            "w_rms": m["w_rms"], "w_rms_z": float(np.sqrt(np.mean(w_z))),
            **yaw_metrics(yaw_dir if yaw_dir is not None else eval_dir.parent / "eval_yaw")}


def ablation_rows():
    """Baseline (seed 3) first, then the variants of the first study, then the control runs in name order."""
    rows = [ablation_row(f"B0{'' if seed == 3 else f'_s{seed:02d}'}",
                         LOGS / PPO["tanh_std"] / "ppo2/eval" / f"ppo2_s{seed:02d}", seed, BASELINE_CFG,
                         LOGS / PPO["tanh_std"] / "ppo2/eval_yaw" / f"ppo2_s{seed:02d}")
            for seed in (3, 1, 2)]
    names = [d.parent.name for d in ABLATION_LOGS.glob("*/eval") if (d / ".done").exists()]
    for name in sorted(names, key=lambda n: (n.split("_")[0] not in ABLATION_ORDER,
                                              ABLATION_ORDER.index(n.split("_")[0]) if n.split("_")[0] in ABLATION_ORDER else 0, n)):
        eval_dir = ABLATION_LOGS / name / "eval"
        cfg = json.loads((eval_dir / "eval_config.json").read_text())
        eff = cfg["env_cfg_effective"]
        rows.append(ablation_row(name, eval_dir, cfg.get("training_seed") or 3, {k: eff.get(k, BASELINE_CFG[k]) for k in BASELINE_CFG}))
    return rows


FM_LEMNISCATE = "eval_lemniscate_fm_ms025_den08_yaw101_hc_kt3p9"   # policies FM_sNN (k_T 3.9, hover at a0 = 0)
FM_LEMNISCATE_KT19 = "eval_lemniscate_fm_ms025_den08_yaw101"       # same configuration with k_T 1.9, exported as FMk19_sNN
# (folder, per-axis speed bound of the task [m/s]; 0 = the task has no speed bound)
LEM_ROOTS = ((ABLATION_LOGS / "eval_lemniscate", 2.0, ""), (ABLATION_LOGS / "eval_lemniscate_v5", 5.0, ""),
             (ABLATION_LOGS / FM_LEMNISCATE, 0.0, ""))
PATH_POLICIES = ("B0", "A1", "PRETRAIN")
PATH_OPTIONAL = ("FM_s03",)       # drawn where it was not lost
PATH_FM_ROOT = ABLATION_LOGS / FM_LEMNISCATE_KT19   # thrust/moment path: k_T 1.9, the reference of its section
PATH_LEVELS = (2.0, 5.0)


def lag_metrics(eval_dir, t_skip=5.0, min_speed=0.2):
    """Mean error along the direction of the reference velocity (negative: behind the reference) and rms lateral
    error, both in the horizontal plane [cm], over the surviving episodes after t_skip, where the reference moves
    faster than min_speed; plus the largest reference speed [m/s]."""
    survived = [int(r["env_id"]) for r in csv.DictReader((eval_dir / "episodes.csv").open())
                if int(r["episode_id"]) == 0 and r["reason"] != "terminated"]
    z = np.load(eval_dir / "steps.npz")
    first = z["episode"] == 0
    keep = np.isin(z["env"][first], survived) & (z["t"][first] >= t_skip - 1e-9)
    err = (z["pos"][first] - z["goal"][first])[keep]
    vel = z["goal_vel"][first][keep]
    speed = np.linalg.norm(vel[:, :2], axis=1)
    moving = speed > min_speed
    if not moving.any():                                   # goal velocity kept at zero (FM_s03_v0): no direction
        return {"along_mean_cm": float("nan"), "lateral_rms_cm": float("nan"), "ref_speed_max": float("nan")}
    u = vel[moving, :2] / speed[moving, None]
    along = (err[moving, :2] * u).sum(axis=1)
    lateral = -err[moving, 0] * u[:, 1] + err[moving, 1] * u[:, 0]
    return {"along_mean_cm": 100 * float(along.mean()), "lateral_rms_cm": 100 * float(np.sqrt((lateral ** 2).mean())),
            "ref_speed_max": float(np.linalg.norm(z["goal_vel"], axis=1).max())}


def lemniscate_rows():
    """Tracking metrics of every finished lemniscate evaluation (both speed bounds)."""
    rows = []
    for root, bound, prefix in LEM_ROOTS + ((ABLATION_LOGS / FM_LEMNISCATE_KT19, 0.0, "FMk19_"),):
        for d in sorted(root.glob("*/a*")):
            if (d / ".done").exists() and (not prefix or re.fullmatch(r"FM_s\d\d", d.parent.name)):
                name = prefix + d.parent.name[3:] if prefix else d.parent.name
                rows.append({"policy": name, "peak_acc": float(d.name[1:]), "speed_bound": bound,
                             **tracking_metrics(d), **lag_metrics(d)})
    return sorted(rows, key=lambda r: (r["policy"], r["peak_acc"], r["speed_bound"]))


def lemniscate_paths():
    """Horizontal path of one episode (the median-RMSE episode of B0 that every plotted policy survives) of the
    policies in PATH_POLICIES (and PATH_OPTIONAL where available), with the reference, for the levels in PATH_LEVELS;
    seconds 5 to 20. Coordinates are relative to the centre of the figure eight (the reference at t = 0), because the
    vehicles start at different grid points; the thrust/moment policy, evaluated with 4 vehicles, uses vehicle 0."""
    rows = []
    for level in PATH_LEVELS:
        dirs = {}
        for root in (PATH_FM_ROOT,) + tuple(r for r, _, _ in reversed(LEM_ROOTS)):   # relaxed bound wins
            for pol in PATH_POLICIES + PATH_OPTIONAL:
                d = root / pol / f"a{level:g}"
                if pol not in dirs and (d / ".done").exists():
                    dirs[pol] = d
        if any(pol not in dirs for pol in PATH_POLICIES):
            continue
        data, alive = {}, None
        for pol, d in dirs.items():
            ok = {int(r["env_id"]) for r in csv.DictReader((d / "episodes.csv").open())
                  if int(r["episode_id"]) == 0 and r["reason"] != "terminated"}
            if pol in PATH_OPTIONAL:
                if not ok:
                    continue
            else:
                alive = ok if alive is None else alive & ok
            data[pol] = np.load(d / "steps.npz")
        if not alive:                                      # e.g. every vehicle of one policy lost at the speed bound
            continue
        ref = data["B0"]
        first = ref["episode"] == 0
        errs = {}
        for e in alive:
            m = first & (ref["env"] == e) & (ref["t"] >= 5.0 - 1e-9)
            errs[e] = float(np.sqrt((np.linalg.norm(ref["pos"][m] - ref["goal"][m], axis=1) ** 2).mean()))
        env = sorted(errs, key=errs.get)[len(errs) // 2]
        for pol, z in data.items():
            e = 0 if pol in PATH_OPTIONAL else env
            mine = (z["episode"] == 0) & (z["env"] == e)
            centre = z["goal"][mine & (z["step"] == 0)][0]
            m = mine & (z["t"] >= 5.0 - 1e-9)
            order = np.argsort(z["step"][m])
            for t, pos, g in zip(z["t"][m][order], z["pos"][m][order], z["goal"][m][order]):
                pos, g = pos - centre, g - centre
                rows.append({"peak_acc": level, "policy": pol, "env_id": e, "t": t, "x": pos[0], "y": pos[1], "z": pos[2],
                             "ref_x": g[0], "ref_y": g[1], "ref_z": g[2]})
    return rows


def survival_by_tilt(name, eval_dir):
    """Rows (policy, tilt bin, failures, episodes): terminations of the first episodes by initial rotation angle."""
    reasons = {int(r["env_id"]): r["reason"] for r in csv.DictReader((eval_dir / "episodes.csv").open())
               if int(r["episode_id"]) == 0}
    z = np.load(eval_dir / "steps.npz")
    first = (z["episode"] == 0) & (z["step"] == 0)
    tilt = dict(zip(z["env"][first].tolist(), z["tilt_deg"][first].tolist()))
    rows = []
    for lo, hi in TILT_BINS:
        ids = [e for e, t in tilt.items() if lo <= t < hi]
        rows.append({"policy": name, "tilt_from_deg": lo, "tilt_to_deg": min(hi, 90.0), "episodes": len(ids),
                     "failures": sum(reasons[e] == "terminated" for e in ids)})
    return rows


def example_episode(runs):
    """One episode (same initial state and reference for every policy): errors and rotor commands."""
    data = {}
    for name, eval_dir in runs.items():
        reasons = {int(r["env_id"]): r["reason"] for r in csv.DictReader((eval_dir / "episodes.csv").open())}
        z = np.load(eval_dir / "steps.npz")
        data[name] = (z, reasons)
    z0, _ = data["initial"]
    first = z0["step"] == z0["step"].min()
    n_env = int(z0["env"].max()) + 1
    init_err = np.zeros(n_env)
    moving = np.zeros(n_env, dtype=bool)
    for e in range(n_env):
        m = z0["env"] == e
        g = z0["goal"][m]
        init_err[e] = np.linalg.norm(z0["pos"][first & (z0["env"] == e)] - z0["goal"][first & (z0["env"] == e)])
        moving[e] = np.abs(g - g[0]).max() > 1e-4
    # Candidates: Langevin reference, every policy survives, same initial state in every evaluation
    ok = [e for e in range(n_env) if moving[e]
          and all(d[1].get(e) != "terminated" for d in data.values())
          and all(np.allclose(d[0]["pos"][(d[0]["env"] == e) & (d[0]["step"] == d[0]["step"].min())],
                              z0["pos"][first & (z0["env"] == e)], atol=1e-4) for d in data.values())]
    env = min(ok, key=lambda e: abs(init_err[e] - np.median(init_err[ok])))
    rows = []
    for name, (z, _) in data.items():
        m = z["env"] == env
        order = np.argsort(z["step"][m])
        t, pos, goal = z["t"][m][order], z["pos"][m][order], z["goal"][m][order]
        act = np.clip(z["action"][m][order], -1.0, 1.0)
        w = np.linalg.norm(z["w_b"][m][order], axis=1)
        for k in range(len(t)):
            e = pos[k] - goal[k]
            rows.append({"policy": name, "env_id": env, "t": t[k], "ex": e[0], "ey": e[1], "ez": e[2],
                         "err": float(np.linalg.norm(e)), "a0": act[k, 0], "a1": act[k, 1], "a2": act[k, 2],
                         "a3": act[k, 3], "w_norm": w[k]})
    print(f"example episode: env {env}, initial error {init_err[env]:.3f} m (median of {len(ok)} candidates)")
    return rows


def write(name, rows):
    """Write a list of dicts to data/<name>.csv."""
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"data/{name}: {len(rows)} rows")


def main():
    OUT.mkdir(exist_ok=True)
    training, evals, finals = [], [], []

    # SAC: training scalars, checkpoints every 1e5 steps and the final checkpoint
    sac_dir = run_dir("sac_vs_ppo2", 1, algo="sac")
    training += training_rows("sac", "sac", 1, sac_dir)
    for d in sorted((LOGS / "eval_checkpoints/sac_vs_ppo2/sac_s01").glob("step*")):
        if not (d / ".done").exists():
            continue
        step = int(d.name[4:])
        evals.append(eval_row("sac", "sac", 1, np.nan, step, d))
    final = eval_row("sac", "sac", 1, np.nan, 1_000_000, LOGS / "eval_reference/sac_s01")
    evals.append(final)
    finals.append(final)
    tilt_rows = survival_by_tilt("sac", LOGS / "eval_reference/sac_s01")

    # RAPTOR pre-training actor (rl-tools), evaluated with the same protocol; not part of the checkpoint curves
    finals.append(eval_row("rl-tools", "raptor_pretrain", 1, np.nan, 1_000_000, LOGS / "eval_reference/raptor_pretrain"))
    tilt_rows += survival_by_tilt("raptor_pretrain", LOGS / "eval_reference/raptor_pretrain")

    # PPO2 variants: training scalars, checkpoints and final evaluation (iteration 600)
    for variant, study in {**PPO, **ABLATION}.items():
        for seed in (SEEDS if variant in PPO else (1,)):
            rdir = run_dir(study, seed)
            if variant in PPO:
                training += training_rows("ppo2", variant, seed, rdir)
                for d in sorted((LOGS / "eval_checkpoints" / study / f"ppo2_s{seed:02d}").glob("it*")):
                    if not (d / ".done").exists():
                        continue
                    it = int(d.name[2:])
                    evals.append(eval_row("ppo2", variant, seed, it, it * STEPS_PER_ITER * N_ENVS_PPO, d))
            final = eval_row("ppo2", variant, seed, 600, 600 * STEPS_PER_ITER * N_ENVS_PPO,
                             LOGS / study / "ppo2/eval" / f"ppo2_s{seed:02d}")
            if variant in PPO:
                evals.append(final)
            finals.append(final)
            tilt_rows += survival_by_tilt(f"{variant}_s{seed}", LOGS / study / "ppo2/eval" / f"ppo2_s{seed:02d}")

    evals.sort(key=lambda r: (r["algo"], r["variant"], r["seed"], r["transitions"]))
    write("training_curves.csv", training)
    write("eval_checkpoints.csv", evals)
    write("run_metrics.csv", finals)
    write("survival_by_tilt.csv", tilt_rows)
    write("ablation_metrics.csv", ablation_rows())
    lem = lemniscate_rows()
    if lem:
        write("lemniscate_metrics.csv", lem)
        paths = lemniscate_paths()
        if paths:
            write("lemniscate_paths.csv", paths)

    s = f"ppo2_s{EXAMPLE_SEED:02d}"
    write("example_episode.csv", example_episode({
        "initial": LOGS / PPO["initial"] / "ppo2/eval" / s,
        "tanh": LOGS / PPO["tanh"] / "ppo2/eval" / s,
        "tanh_std": LOGS / PPO["tanh_std"] / "ppo2/eval" / s,
        "sac": LOGS / "eval_reference/sac_s01",
    }))


if __name__ == "__main__":
    main()
