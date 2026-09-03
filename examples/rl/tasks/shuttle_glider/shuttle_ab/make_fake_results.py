#!/usr/bin/env python3
"""
| File: make_fake_results.py
| Gera um CSV sintetico com o MESMO esquema que o EpisodeRecorder produz, para
| poderes testar o eval_tost.py em 5 segundos, antes de gastares GPU.
|
| Uso:
|   python3 make_fake_results.py --out results_fake.csv --effect 0.00   # sem efeito
|   python3 make_fake_results.py --out fake_big.csv --effect 0.25  # efeito 25%
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_fake.csv")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--base-rmse", type=float, default=0.080)
    p.add_argument("--effect", type=float, default=0.0,
                   help="efeito relativo da ablacao (0.10 = ablacao 10% pior)")
    p.add_argument("--seed-sigma", type=float, default=0.004,
                   help="variabilidade entre seeds de treino")
    p.add_argument("--ep-sigma", type=float, default=0.020,
                   help="variabilidade entre episodios")
    p.add_argument("--rng", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.rng)
    rows = []
    for tag, mult, t5 in (("A_active", 1.0, 0.02), ("C_removed", 1.0 + args.effect, 0.0)):
        for seed in range(args.seeds):
            seed_bias = rng.normal(0.0, args.seed_sigma)
            for ep in range(args.episodes):
                is_lang = ep % 2 == 0
                hard = 1.0 if is_lang else 0.55
                rmse = max(
                    1e-4,
                    args.base_rmse * mult * hard + seed_bias
                    + rng.normal(0.0, args.ep_sigma) * hard,
                )
                rows.append(
                    {
                        "run_tag": tag,
                        "seed": seed,
                        "env_id": ep % 8,
                        "episode_idx": ep,
                        "traj_type": "langevin" if is_lang else "null",
                        "rmse_pos": rmse,
                        "rmse_vel": rmse * 3.0,
                        "terminated": 0,
                        "steps": 500.0,
                        "thrust5_mean_N": max(0.0, rng.normal(t5, 0.005)),
                        "thrust5_impulse_frac": max(0.0, rng.normal(t5 / 30.0, 1e-4)),
                        "power_mean": rng.normal(1.0e10, 2e8),
                        "total_reward": 700.0 - 1000 * rmse,
                    }
                )
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"escrito {args.out} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
