#!/usr/bin/env python3
"""Cross-axis coupling of the trained outer-loop policies (a_max = 7 m/s^2).

The LQR gain matrix is diagonal: the command of one axis depends only on the errors of that axis. A learned MLP
has no such structure. For every surviving policy (last checkpoint) and the states visited in its evaluation
(first episode of every goal, error x = [ep, ev] = [goal - pos, -vel]) the mean action is evaluated twice:
  own   : ep_i += 0.5 m and ev_i += 0.5 m/s  ->  mean |change of u_i|          (axis i perturbed, axis i observed)
  cross : ep_j += 0.5 m and ev_j += 0.5 m/s  ->  mean |change of u_i|, j != i  (another axis perturbed)
averaged over the three axes (and the six ordered pairs for the cross term). The ratio cross/own is the coupling.
The mean Jacobian of the action with respect to [ep, ev] is also reported (signed average, mostly diagonal).

    python3 axis_coupling.py       # writes data/axis_coupling.csv, prints the summary by reward
"""
import csv
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = "outer_loop_amax7_10seeds"
ROOT = HERE.parents[2] / "examples/rl/tasks/shuttle/outer_loop/double_integrator/logs" / STUDY
A_MAX = 7.0
EXPERIMENTS = ["alive", "death", "exp", "alive_death", "alive_death200"]
DELTA = (0.5, 0.5)                    # perturbation of (ep, ev) of one axis: 0.5 m and 0.5 m/s
SAMPLES = 4000


def load_actor(run_dir):
    """Deterministic actor (mean action) of the last checkpoint: 6 -> 64 -> ELU -> 64 -> ELU -> 3."""
    ckpt = sorted((run_dir / "checkpoints").glob("agent_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["policy"]
    net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ELU(), torch.nn.Linear(64, 64), torch.nn.ELU(),
                              torch.nn.Linear(64, 3))
    net.load_state_dict({k[len("net."):]: v for k, v in state.items() if k.startswith("net.")})
    return net.eval()


def visited_states(exp, name, rng):
    z = np.load(ROOT / exp / "eval" / name / "steps.npz")
    m = z["episode"] == 0
    x = np.concatenate([z["goal"][m] - z["pos"][m], -z["vel"][m]], 1).astype(np.float32)
    return torch.tensor(x[rng.choice(len(x), min(SAMPLES, len(x)), replace=False)])


def coupling(net, x):
    with torch.no_grad():
        base = A_MAX * net(x)
        change = np.zeros((3, 3))                       # change[i, j]: mean |du_i| when axis j is perturbed
        for j in range(3):
            xp = x.clone()
            xp[:, j] += DELTA[0]
            xp[:, 3 + j] += DELTA[1]
            change[:, j] = (A_MAX * net(xp) - base).abs().mean(0).numpy()
    own = np.mean(np.diag(change))
    cross = np.mean(change[~np.eye(3, dtype=bool)])
    xr = x.clone().requires_grad_(True)
    out = A_MAX * net(xr)
    jac = torch.stack([torch.autograd.grad(out[:, i].sum(), xr, retain_graph=True)[0] for i in range(3)], 1)
    return own, cross, jac.detach().numpy().mean(0)


def surviving_runs():
    """Runs that survive the evaluation (the policy that never learns to survive is left out, as in the text)."""
    with (HERE / "data" / "run_metrics.csv").open() as f:
        return {r["run"] for r in csv.DictReader(f) if float(r["survival"]) > 0.5}


def main():
    rng = np.random.default_rng(0)
    rows, jacs = [], {}
    alive = surviving_runs()
    for exp in EXPERIMENTS:
        for run in sorted((ROOT / exp / "train").iterdir()):
            if run.name not in alive:
                continue
            net = load_actor(sorted(run.glob("*_PPO2"))[-1])
            own, cross, jac = coupling(net, visited_states(exp, run.name, rng))
            rows.append([exp, run.name, own, cross, 100.0 * cross / own])
            jacs.setdefault(exp, []).append(jac)
    out = HERE / "data" / "axis_coupling.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "run", "own_change_mps2", "cross_change_mps2", "coupling_percent"])
        w.writerows(rows)
    print(f"{'reward':16s}{'own [m/s2]':>14s}{'cross [m/s2]':>16s}{'coupling [%]':>16s}")
    for exp in EXPERIMENTS:
        r = np.array([row[2:] for row in rows if row[0] == exp], dtype=float)
        print(f"{exp:16s}{r[:, 0].mean():8.2f}+-{r[:, 0].std(ddof=1):4.2f}{r[:, 1].mean():10.2f}+-{r[:, 1].std(ddof=1):4.2f}"
              f"{r[:, 2].mean():10.1f}+-{r[:, 2].std(ddof=1):4.1f}")
    r = np.array([row[2:] for row in rows], dtype=float)
    print(f"{'all policies':16s}{r[:, 0].mean():8.2f}{'':6s}{r[:, 1].mean():10.2f}{'':6s}{r[:, 2].mean():10.1f}")
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print("\nmean Jacobian of the action [m/s^2 per m or m/s] (rows u_x u_y u_z; cols ep_x ep_y ep_z ev_x ev_y ev_z)")
    print(np.mean([j for js in jacs.values() for j in js], 0))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
