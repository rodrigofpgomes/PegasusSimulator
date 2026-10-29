#!/usr/bin/env python3
"""Export the final study of the direct thrust/moment section (Shuttle) to CSV files for MATLAB.

Sources (10 seeds each, deterministic evaluation on the same 200 goals):
  logs/study_10seeds       thrust-to-weight 1.9 (main study)
  logs/study_10seeds_kt3p9 thrust-to-weight 3.9 (sensitivity test)
Writes data/run_metrics.csv, eval_curves.csv, training_curves.csv, example_episode.csv.
    python3 export_results.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[2] / "examples/rl/tasks/shuttle/thrust_moment/isaac_lab/logs"
STUDIES = {1.9: LOGS / "study_10seeds", 3.9: LOGS / "study_10seeds_kt3p9"}
OUT = HERE / "data"
BINS = 100
EXAMPLE_RUNS = ("ms012_den08", "ms025_den08", "ms012_den08_yaw011", "ms012_den08_yaw101", "ms012_den08_hc",
                "ms025_den08_yaw011", "ms025_den08_yaw101", "ms025_den08_yaw101_hc")   # all seeds exported
TRAIN_TAGS = {"final_distance_m": "Metrics_final_distance_to_goal", "died": "Episode_Termination_died"}


def run_config(eval_dir):
    """Task parameters of the evaluated policy (restored from its training run)."""
    env = json.loads((eval_dir / "eval_config.json").read_text())["env_cfg_effective"]
    return env["moment_scale"], env["distance_reward_denominator"], 0.01 - env.get("yaw_rate_reward_scale", 0.0), env


def read_scalar(run_dir, tag):
    files = [f for f in (run_dir / "download_csvs").glob("*.csv") if tag in f.name]
    if not files:
        return None
    rows = list(csv.reader(files[0].open()))[1:]
    return np.array([[float(r[1]), float(r[2])] for r in rows])


def main():
    OUT.mkdir(exist_ok=True)
    metrics, curves, training, episodes = [], [], [], []
    example_env = None
    for kt, root in STUDIES.items():
        for eval_dir in sorted(root.glob("*/eval/*_s??")):
            exp, name = eval_dir.parent.parent.name, eval_dir.name
            seed = int(name[-2:])
            cM, cD, w_psi, env_cfg = run_config(eval_dir)
            z = np.load(eval_dir / "steps.npz")
            sel = z["episode"] == 0
            env, t, step = z["env"][sel], z["t"][sel], z["step"][sel]
            pos, goal, vel, w = z["pos"][sel], z["goal"][sel], z["vel"][sel], z["w_b"][sel]
            act = np.clip(z["action"][sel], -1.0, 1.0) if "action" in z.files else np.full((len(t), 4), np.nan)
            err = np.linalg.norm(goal - pos, axis=1)
            n_env = env.max() + 1
            reasons = {int(r["env_id"]): r["reason"] for r in csv.DictReader((eval_dir / "episodes.csv").open())
                       if r["episode_id"] == "0"}

            fin, rmse, t20, wz_mean, tilt, vmax = [], [], [], [], [], []
            for e in range(n_env):
                m = env == e
                o = np.argsort(t[m])
                tt, ee = t[m][o], err[m][o]
                fin.append(ee[tt >= tt.max() - 2.0].mean())
                rmse.append(np.sqrt(np.mean(ee ** 2)))
                below = np.where(ee <= 0.2 * ee[0])[0]
                t20.append(tt[below[0]] if below.size else np.nan)
                wz_mean.append(w[m][o, 2].mean())
                tilt.append(z["tilt_deg"][sel][m].max())
                vmax.append(np.linalg.norm(vel[m], axis=1).max())
            per_goal_rms = lambda k: np.mean([np.sqrt(np.mean(w[env == e, k] ** 2)) for e in range(n_env)])
            wz_mean = np.array(wz_mean)
            metrics.append([kt, exp, name, cM, cD, w_psi, seed,
                            np.mean([reasons.get(e) != "terminated" for e in range(n_env)]),
                            np.mean(fin), np.mean(rmse), np.nanmedian(t20),
                            per_goal_rms(0), per_goal_rms(1), per_goal_rms(2),
                            wz_mean.mean(), np.mean(tilt), np.mean(vmax), np.nanmean(act[t > 8.0, 0]),
                            int(env_cfg.get("thrust_hover_centered", False))])

            if kt != 1.9:
                continue                                    # curves and episodes only for the main study
            edges = np.linspace(0.0, t.max() + 1e-9, BINS + 1)
            b = np.clip(np.digitize(t, edges) - 1, 0, BINS - 1)
            for k in range(BINS):
                m = b == k
                if m.any():
                    curves.append([kt, exp, name, cM, cD, w_psi, seed, 0.5 * (edges[k] + edges[k + 1]),
                                   err[m].mean(), np.abs(w[m, 2]).mean()])

            train_dir = sorted((root / exp / "train" / name).glob("*_PPO2"))[-1]
            series = {k: read_scalar(train_dir, tag) for k, tag in TRAIN_TAGS.items()}
            for i, (st, val) in enumerate(series["final_distance_m"]):
                died = series["died"][i, 1] if series["died"] is not None and i < len(series["died"]) else np.nan
                training.append([kt, exp, name, cM, cD, w_psi, seed, st / 24.0, val, died])

            if example_env is None:                         # goal at the median initial distance
                d0 = np.array([err[(env == e) & (step == 0)][0] for e in range(n_env)])
                example_env = int(np.argmin(np.abs(d0 - np.median(d0))))
        # example episodes (same goal for every policy), both thrust-to-weight ratios for the thrust input
        for exp in EXAMPLE_RUNS:
            if not (root / exp).is_dir():
                continue
            for eval_dir in sorted((root / exp / "eval").glob("*_s??")):
                cM, cD, w_psi, env_cfg = run_config(eval_dir)
                z = np.load(eval_dir / "steps.npz")
                sel = (z["episode"] == 0) & (z["env"] == example_env)
                o = np.argsort(z["t"][sel])
                t, pos, goal = z["t"][sel][o], z["pos"][sel][o], z["goal"][sel][o]
                vel, w = z["vel"][sel][o], z["w_b"][sel][o]
                a = np.clip(z["action"][sel][o], -1.0, 1.0)
                weight = env_cfg["drone_mass"] * env_cfg["gravity"]
                centered = env_cfg.get("thrust_hover_centered", False)
                if centered:                                # same mapping as QuadcopterEnv._apply_action
                    thrust = weight * (1.0 + np.where(a[:, 0] < 0.0, a[:, 0], (env_cfg["thrust_to_weight"] - 1.0) * a[:, 0]))
                    a0_hover = 0.0
                else:
                    thrust = env_cfg["thrust_to_weight"] * weight * (a[:, 0] + 1.0) / 2.0
                    a0_hover = 2.0 / env_cfg["thrust_to_weight"] - 1.0
                tau = env_cfg["moment_scale"] * a[:, 1:]
                spawn = pos[0, :2]
                for i in range(len(t)):
                    episodes.append([kt, exp, eval_dir.name, cM, cD, w_psi, int(eval_dir.name[-2:]), t[i],
                                     *(pos[i, :2] - spawn), pos[i, 2], *(goal[i, :2] - spawn), goal[i, 2],
                                     *vel[i], *w[i], np.linalg.norm(goal[i] - pos[i]),
                                     a[i, 0], thrust[i], *tau[i], weight, a0_hover])

    head = ["thrust_to_weight", "experiment", "run", "moment_scale", "denominator", "w_psi", "seed"]

    def write(fname, cols, rows):
        with (OUT / fname).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(head + cols)
            writer.writerows(rows)

    write("run_metrics.csv", ["survival", "final_err_m", "rmse_m", "t20_s", "w_x_rms", "w_y_rms", "w_z_rms",
                              "wz_mean_signed", "tilt_max_deg", "speed_max", "a0_hover", "hover_centered"], metrics)
    write("eval_curves.csv", ["t", "err_mean", "wz_abs_mean"], curves)
    write("training_curves.csv", ["iteration", "final_distance_m", "died"], training)
    write("example_episode.csv", ["t", "x", "y", "z", "goal_x", "goal_y", "goal_z", "vx", "vy", "vz",
                                  "wx", "wy", "wz", "err", "a0", "thrust_N", "tau_x_Nm", "tau_y_Nm", "tau_z_Nm",
                                  "weight_N", "a0_hover"], episodes)
    print(f"{len(metrics)} runs; example goal = environment {example_env}")


if __name__ == "__main__":
    main()
