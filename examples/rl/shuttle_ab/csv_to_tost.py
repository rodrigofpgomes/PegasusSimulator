# -*- coding: utf-8 -*-
"""
| File: csv_to_tost.py
| Converte os CSV do EpisodeTrajectoryRecorder do play_multi.py no esquema
| por episodio que o eval_tost.py consome.
|
| Porque existe: o eval_recorder.py esta pendurado no _reset_idx do env, e o
| play_multi.py NUNCA chama _reset_idx -- faz reset_manager.reset_all() a cada
| `episode_steps`. Em avaliacao nao sairia uma unica linha. Mas o recorder que
| ja existe no play_multi.py grava tudo o que precisamos por passo; basta
| agregar.
|
| Entrada  : <record_dir>/trajectories.csv  (+ episodes.csv, opcional)
| Saida    : results_eval.csv com run_tag, seed, env_id, episode_idx,
|            traj_type, rmse_pos, rmse_vel, terminated, steps,
|            thrust5_mean_N, thrust5_impulse_frac, power_mean, total_reward
|
| Exemplo (lemniscate, 3 condicoes no MESMO run, seed 0):
|   python3 csv_to_tost.py \\
|       --records play_records/lemn_seed0 \\
|       --ref-type lemniscate --traj-amplitude 1.5 --traj-period 8 --traj-z 1.5 \\
|       --seed 0 --out results_eval.csv
|
| Depois, para as seeds 1..4, o mesmo comando com --seed N --append.
| E para o goal estatico, --ref-type static (sem os --traj-*).
|
| Analise:
|   python3 eval_tost.py --csv results_eval.csv \\
|       --baseline A_active --ablation C_removed \\
|       --metric rmse_pos --margin-frac 0.10 --by traj_type --plot ab.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes fisicas do shuttle_glider (iguais as do quadcopter_env_v2.py)
# ---------------------------------------------------------------------------
KF = np.array([1.709716e-05] * 4 + [8.54858e-06])   # T_i = KF_i * w_i^2  [N]
MAX_W = np.array([1400.0] * 4 + [3500.0])           # rad/s


def _rotor_speed_cols(df: pd.DataFrame, n_rotors: int) -> list[str]:
    """Prefere actual_w (velocidade real do rotor); cai para cmd_w."""
    for group in ("actual_w", "cmd_w"):
        cols = [f"{group}_{i}" for i in range(n_rotors)]
        if all(c in df.columns for c in cols):
            return cols
    return []


def _infer_n_rotors(df: pd.DataFrame) -> int:
    n = 0
    while f"actual_w_{n}" in df.columns or f"cmd_w_{n}" in df.columns:
        n += 1
    return n


def _rmse_vel(g: pd.DataFrame) -> float:
    """RMSE da velocidade. Se ref_v* nao existir ou for tudo zero, usa |v|.

    ATENCAO: no play_multi.py o make_reference_provider() devolve velocidade
    de referencia identicamente nula, mas o _update_goal() escreve a velocidade
    analitica do lemniscate em reset_manager._goal_vel, e e essa que o recorder
    grava em ref_vx/ref_vy/ref_vz. Verifica na primeira corrida que ref_v* nao
    e tudo zero num lemniscate -- se for, a coluna vem do provider errado e o
    rmse_vel deixa de ser comparavel entre estatico e lemniscate.
    """
    have_ref = all(c in g.columns for c in ("ref_vx", "ref_vy", "ref_vz"))
    v = g[["vx", "vy", "vz"]].to_numpy(dtype=float)
    if have_ref:
        vr = g[["ref_vx", "ref_vy", "ref_vz"]].to_numpy(dtype=float)
        vr = np.nan_to_num(vr)
    else:
        vr = np.zeros_like(v)
    e = np.linalg.norm(v - vr, axis=1)
    return float(np.sqrt(np.mean(e ** 2)))


def aggregate(df: pd.DataFrame, args) -> pd.DataFrame:
    n_rotors = _infer_n_rotors(df)
    w_cols = _rotor_speed_cols(df, n_rotors)
    if not w_cols:
        print("[aviso] sem colunas de rotor -- thrust5 fica NaN", file=sys.stderr)

    kf = KF[:n_rotors] if n_rotors else KF
    max_w = MAX_W[:n_rotors] if n_rotors else MAX_W

    key = ["controller", "env_id", "episode_id"] if "controller" in df.columns \
        else ["env_id", "episode_id"]

    rows = []
    for k, g in df.groupby(key, sort=True):
        if not isinstance(k, tuple):
            k = (k,)
        rec = dict(zip(key, k))

        steps = len(g) * args.record_every
        if steps < args.min_steps:
            continue                        # episodio truncado no fim da corrida

        pe = g["pos_error"].to_numpy(dtype=float)
        rmse_pos = float(np.sqrt(np.mean(pe ** 2)))

        # Nao ha sinal de terminacao em avaliacao: o play_multi.py so faz reset
        # por duracao fixa (step % episode_steps == 0), nunca por violacao de
        # limites. Derivamo-lo do erro de posicao.
        terminated = int(np.nanmax(pe) > args.fail_radius)

        out = {
            "run_tag": str(rec.get("controller", args.run_tag or "run")),
            "seed": int(args.seed),
            "env_id": int(rec["env_id"]),
            "episode_idx": int(rec["episode_id"]),
            "traj_type": args.ref_type,
            "rmse_pos": rmse_pos,
            "rmse_vel": _rmse_vel(g),
            "terminated": terminated,
            "steps": steps,
            "total_reward": np.nan,        # nao existe em avaliacao
        }

        if w_cols:
            w = g[w_cols].to_numpy(dtype=float)
            thrust = kf * (w ** 2)                       # (T, n_rotors) [N]
            tot = thrust.sum(axis=1)
            if n_rotors >= 5:
                out["thrust5_mean_N"] = float(thrust[:, 4].mean())
                out["thrust5_impulse_frac"] = float(
                    thrust[:, 4].sum() / max(tot.sum(), 1e-9)
                )
                out["w5_mean_rads"] = float(w[:, 4].mean())
                out["w5_max_rads"] = float(w[:, 4].max())
            else:
                out["thrust5_mean_N"] = 0.0
                out["thrust5_impulse_frac"] = 0.0
            out["thrust_total_mean_N"] = float(tot.mean())
            out["power_mean"] = float(((w / max_w) ** 3).sum(axis=1).mean())
        else:
            for c in ("thrust5_mean_N", "thrust5_impulse_frac", "power_mean"):
                out[c] = np.nan

        # metricas de heading, ja gravadas pelo recorder quando
        # --vehicle shuttle_glider
        for src, dst in (("heading_error_abs_deg", "heading_err_deg"),
                         ("heading_alignment", "heading_align"),
                         ("att_err_pd", "att_err_pd"),
                         ("z_error", "z_err_mean")):
            if src in g.columns:
                out[dst] = float(np.nanmean(np.abs(g[src].to_numpy(dtype=float))))

        # parametros da referencia, para nao perderes o contexto da corrida
        if args.ref_type == "lemniscate":
            out["traj_amplitude"] = args.traj_amplitude
            out["traj_period"] = args.traj_period
            out["traj_z"] = args.traj_z

        rows.append(out)

    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True,
                   help="pasta(s) --record_dir do play_multi.py")
    p.add_argument("--out", default="results_eval.csv")
    p.add_argument("--append", action="store_true",
                   help="acrescenta ao --out em vez de o substituir")
    p.add_argument("--seed", type=int, default=0,
                   help="seed desta corrida (o play_multi.py nao a grava)")
    p.add_argument("--run-tag", default=None,
                   help="usado so se o CSV nao tiver coluna 'controller'")
    p.add_argument("--ref-type", choices=["static", "lemniscate"], required=True)
    p.add_argument("--traj-amplitude", type=float, default=1.5)
    p.add_argument("--traj-period", type=float, default=8.0)
    p.add_argument("--traj-z", type=float, default=1.5)
    p.add_argument("--record-every", type=int, default=1,
                   help="o mesmo --record_every que passaste ao play_multi.py")
    p.add_argument("--min-steps", type=int, default=50,
                   help="descarta episodios truncados")
    p.add_argument("--fail-radius", type=float, default=1.0,
                   help="m; pos_error acima disto marca terminated=1")
    args = p.parse_args(argv)

    frames = []
    for d in args.records:
        path = d if d.endswith(".csv") else os.path.join(d, "trajectories.csv")
        if not os.path.exists(path):
            print(f"[erro] nao existe: {path}", file=sys.stderr)
            return 1
        df = pd.read_csv(path)
        print(f"[ler] {path}: {len(df)} passos, "
              f"{df['controller'].nunique() if 'controller' in df else 1} controlador(es)")
        frames.append(aggregate(df, args))

    res = pd.concat(frames, ignore_index=True)
    if res.empty:
        print("[erro] nenhum episodio agregado -- ve --min-steps", file=sys.stderr)
        return 1

    if args.append and os.path.exists(args.out):
        old = pd.read_csv(args.out)
        res = pd.concat([old, res], ignore_index=True)

    res.to_csv(args.out, index=False)

    print(f"\n[escrito] {args.out}: {len(res)} episodios")
    print("\nResumo por run_tag x traj_type:")
    cols = [c for c in ("rmse_pos", "rmse_vel", "thrust5_impulse_frac",
                        "terminated", "heading_err_deg") if c in res.columns]
    print(res.groupby(["run_tag", "traj_type"])[cols].mean().to_string())

    n_seeds = res.groupby("run_tag")["seed"].nunique()
    if (n_seeds < 3).any():
        print("\n[aviso] menos de 3 seeds por condicao. O eval_tost.py agrega por")
        print("        seed, logo o intervalo de confianca vai ser inutilizavel.")
        print("        Corre o play_multi.py com --seed 0..4 e junta com --append.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
