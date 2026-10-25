#!/usr/bin/env python3
"""Export the outer-loop (acceleration level) reward study of the Shuttle to CSV files for MATLAB.

Source: shuttle/outer_loop/double_integrator/logs/<STUDY> written by examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh
(10 seeds per reward, deterministic evaluation on the same 200 goals, LQR evaluated on the same goals).
The per-run metrics come from <STUDY>/outer_loop_by_run.csv (summarize_outer_loop_eval.py); run that first.
The equilibrium offset of each policy is computed from its last checkpoint (needs torch).

Writes data/run_metrics.csv, training_curves.csv, eval_curves.csv, example_episode.csv.
    python3 export_results.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

A_MAX = 7.0                                  # the only line that differs between outer_loop_amax1 and _amax7
STUDY = "outer_loop_amax7_10seeds"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "examples/rl/tasks/shuttle/outer_loop/double_integrator/logs" / STUDY
OUT = HERE / "data"
EXPERIMENTS = ["quad", "alive", "death", "exp", "alive_death", "alive_death200", "lqr"]
BINS = 100
TRAIN_TAGS = {"died": "Episode_Termination_died", "value_loss": "Value loss",
              "final_distance_m": "Metrics_final_distance_to_goal", "std": "Policy _ Standard deviation--"}


def read_scalar(run_dir, tag):
    """(step, value) of one exported TensorBoard scalar."""
    files = [f for f in (run_dir / "download_csvs").glob("*.csv") if tag in f.name]
    if not files:
        return None
    rows = list(csv.reader(files[0].open()))[1:]
    return np.array([[float(r[1]), float(r[2])] for r in rows])


def load_actor(run_dir):
    """Deterministic actor (mean action) of the last checkpoint: 6 -> 64 -> ELU -> 64 -> ELU -> 3."""
    ckpt = sorted((run_dir / "checkpoints").glob("agent_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["policy"]
    net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ELU(), torch.nn.Linear(64, 64), torch.nn.ELU(),
                              torch.nn.Linear(64, 3))
    net.load_state_dict({k[len("net."):]: v for k, v in state.items() if k.startswith("net.")})
    return net.eval()


def equilibrium(net, iterations=50):
    """Position error e_p with pi(e_p, 0) = 0 (Newton), and the output bias pi(0, 0)."""
    policy = lambda ep: net(torch.cat([ep, torch.zeros(3)]).unsqueeze(0))[0]
    ep = torch.zeros(3)
    with torch.no_grad():
        bias = policy(ep).numpy()
    for _ in range(iterations):
        jac = torch.autograd.functional.jacobian(policy, ep)
        ep = ep - torch.linalg.solve(jac, policy(ep).detach())
    return ep.detach().numpy(), bias


def episodes(eval_dir):
    """First episode of every environment, sorted by time: dict env -> arrays."""
    z = np.load(eval_dir / "steps.npz")
    out = {}
    for e in np.unique(z["env"]):
        m = (z["env"] == e) & (z["episode"] == 0)
        o = np.argsort(z["t"][m])
        out[int(e)] = {k: z[k][m][o] for k in ("t", "pos", "vel", "goal", "action")}
    return out


def main():
    OUT.mkdir(exist_ok=True)
    by_run = {r["run"]: r for r in csv.DictReader((ROOT / "outer_loop_by_run.csv").open())}
    metrics, training, curves, example = [], [], [], []

    # goal of the example episodes: median initial distance, from the LQR evaluation (same goals for all)
    lqr_eps = episodes(ROOT / "lqr/eval/lqr")
    d0 = {e: np.linalg.norm(ep["goal"][0] - ep["pos"][0]) for e, ep in lqr_eps.items()}
    example_env = min(d0, key=lambda e: abs(d0[e] - np.median(list(d0.values()))))

    for exp in EXPERIMENTS:
        for eval_dir in sorted((ROOT / exp / "eval").glob("*")):
            if not (eval_dir / "steps.npz").is_file():
                continue
            name = eval_dir.name
            seed = int(name[-2:]) if exp != "lqr" else 0
            r = by_run[name]
            row = [exp, name, seed] + [float(r[k]) for k in
                                       ("survival", "final_err_m", "rmse_pos_m", "t20_s", "J_quad", "u2_mean", "sat_frac")]

            if exp == "lqr":
                row += [0.0, 0.0, np.nan, np.nan, np.nan]
            else:
                run_dir = sorted((ROOT / exp / "train" / name).glob("*_PPO2"))[-1]
                ep_eq, bias = equilibrium(load_actor(run_dir))
                series = {k: read_scalar(run_dir, tag) for k, tag in TRAIN_TAGS.items()}
                vl = series["value_loss"][:, 1]
                row += [float(np.linalg.norm(ep_eq)), float(np.linalg.norm(bias)), float(series["std"][-1, 1]),
                        float(vl.max()), float(vl[-20:].mean())]
                n = min(len(s) for s in series.values())
                for i in range(n):
                    training.append([exp, name, seed, series["died"][i, 0] / 24.0,
                                     *(series[k][i, 1] for k in ("died", "value_loss", "final_distance_m", "std"))])
            metrics.append(row)

            eps = episodes(eval_dir)
            edges = np.linspace(0.0, 10.0, BINS + 1)
            t_all = np.concatenate([ep["t"] for ep in eps.values()])
            err_all = np.concatenate([np.linalg.norm(ep["goal"] - ep["pos"], axis=1) for ep in eps.values()])
            u_all = np.concatenate([np.linalg.norm(A_MAX * np.clip(ep["action"], -1.0, 1.0), axis=1)
                                    for ep in eps.values()])
            b = np.clip(np.digitize(t_all, edges) - 1, 0, BINS - 1)
            for k in range(BINS):
                m = b == k                                  # episodes still running at that time
                if m.any():
                    curves.append([exp, name, seed, 0.5 * (edges[k] + edges[k + 1]), err_all[m].mean(),
                                   u_all[m].mean(), m.sum() / (len(eps) * (edges[1] - edges[0]) / 0.02)])

            ep = eps[example_env]
            u = A_MAX * np.clip(ep["action"], -1.0, 1.0)
            spawn = ep["pos"][0, :2]
            for i in range(len(ep["t"])):
                example.append([exp, name, seed, ep["t"][i], *(ep["pos"][i, :2] - spawn), ep["pos"][i, 2],
                                *(ep["goal"][i, :2] - spawn), ep["goal"][i, 2], *ep["vel"][i], *u[i],
                                np.linalg.norm(ep["goal"][i] - ep["pos"][i])])

    def write(fname, cols, rows):
        with (OUT / fname).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["experiment", "run", "seed"] + cols)
            writer.writerows(rows)

    write("run_metrics.csv", ["survival", "final_err_m", "rmse_m", "t20_s", "J_quad", "u2_mean", "sat_frac",
                              "equilibrium_offset_m", "output_bias", "std_final", "value_loss_peak",
                              "value_loss_final"], metrics)
    write("training_curves.csv", ["iteration", "died", "value_loss", "final_distance_m", "std"], training)
    write("eval_curves.csv", ["t", "err_mean", "u_mean", "fraction_running"], curves)
    write("example_episode.csv", ["t", "x", "y", "z", "goal_x", "goal_y", "goal_z", "vx", "vy", "vz",
                                  "ux", "uy", "uz", "err"], example)
    (OUT / "study.json").write_text(json.dumps({"a_max": A_MAX, "study": STUDY, "example_env": example_env,
                                                "example_initial_distance_m": float(d0[example_env])}, indent=2))
    print(f"{len(metrics)} runs from {STUDY}; example goal = environment {example_env} "
          f"(initial distance {d0[example_env]:.2f} m)")


if __name__ == "__main__":
    main()
