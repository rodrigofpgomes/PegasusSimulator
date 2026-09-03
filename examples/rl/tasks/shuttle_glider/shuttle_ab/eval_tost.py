#!/usr/bin/env python3
"""
| File: eval_tost.py
| Analise estatistica A/B/C das ablacoes do 5.o rotor.
|
| Um teste t que NAO rejeita nao prova igualdade. Para um resultado nulo usa-se
| um TESTE DE EQUIVALENCIA (TOST): declara-se equivalencia se o IC a 90% da
| diferenca de medias estiver TODO contido em [-delta, +delta], com delta fixado
| a priori (por omissao 10% do RMSE do baseline).
|
| O bootstrap e' CLUSTER bootstrap sobre SEEDS (a unidade experimental e' o
| treino, nao o episodio -- episodios do mesmo treino nao sao independentes).
|
| Sem scipy: tudo por bootstrap/permutacao.
|
| Uso:
|   python3 eval_tost.py --csv results.csv \
|       --baseline A_active --ablation C_removed \
|       --metric rmse_pos --margin-frac 0.10 --by traj_type --plot ab.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def per_seed_means(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Media do metric por seed -> a unidade experimental."""
    return df.groupby("seed")[metric].mean().to_numpy(dtype=float)


def cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Distribuicao bootstrap de mean(b) - mean(a), reamostrando seeds."""
    out = np.empty(n_boot, dtype=float)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        out[i] = b[ib].mean() - a[ia].mean()
    return out


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, rng) -> float:
    """p-value bilateral por permutacao das etiquetas de condicao."""
    obs = abs(b.mean() - a.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        if abs(perm[na:].mean() - perm[:na].mean()) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# Numero minimo de seeds por condicao para que o TOST tenha significado.
# Ver a guarda em analyse().
MIN_SEEDS = 3


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    d = (b.mean() - a.mean()) / sp
    j = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return d * j


def analyse(a: np.ndarray, b: np.ndarray, margin: float, n_boot: int, rng, label: str):
    diff = b.mean() - a.mean()
    boot = cluster_bootstrap_diff(a, b, n_boot, rng)
    lo90, hi90 = np.percentile(boot, [5.0, 95.0])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    equivalent = (lo90 > -margin) and (hi90 < margin)
    # Guarda de potencia: o bootstrap reamostra SEEDS. Com n<3 o IC colapsa
    # (no limite, n=1 -> IC de largura zero) e o TOST declara equivalencia
    # sempre que |diff| < margem, independentemente da evidencia. Falso positivo.
    n_min = min(len(a), len(b))
    underpowered = n_min < MIN_SEEDS
    if underpowered:
        equivalent = False
    pval = permutation_pvalue(a, b, min(n_boot, 20000), rng)

    print(f"\n=== {label} ===")
    print(f"  n_seeds        : baseline={len(a)}  ablacao={len(b)}")
    print(f"  media baseline : {a.mean():.5f}  (por seed: {np.round(a, 5).tolist()})")
    print(f"  media ablacao  : {b.mean():.5f}  (por seed: {np.round(b, 5).tolist()})")
    print(f"  diferenca      : {diff:+.5f}  ({100.0 * diff / max(a.mean(), 1e-12):+.2f} %)")
    print(f"  IC90% (boot)   : [{lo90:+.5f}, {hi90:+.5f}]")
    print(f"  IC95% (boot)   : [{lo95:+.5f}, {hi95:+.5f}]")
    print(f"  margem delta   : +/- {margin:.5f}")
    print(f"  Hedges g       : {hedges_g(a, b):+.3f}")
    print(f"  p (permutacao) : {pval:.4f}")
    if underpowered:
        print(f"  >>> INSUFICIENTE: apenas {n_min} seed(s) por condicao (minimo {MIN_SEEDS},")
        print("      recomendado 5). O bootstrap reamostra seeds, por isso com n baixo o")
        print("      IC colapsa e o TOST daria um falso positivo. NAO reportar este")
        print("      resultado; correr mais seeds e juntar com 'csv_to_tost.py --append'.")
    elif equivalent:
        print("  >>> EQUIVALENCIA ESTABELECIDA (TOST, alpha=0.05).")
        print("      Reportar: 'a diferenca de RMSE e' de "
              f"{1000 * diff:+.1f} mm, IC90% [{1000 * lo90:+.1f}, {1000 * hi90:+.1f}] mm, "
              f"contido na margem de equivalencia de +/-{1000 * margin:.1f} mm.'")
    elif (lo90 > margin) or (hi90 < -margin):
        print("  >>> DIFERENCA RELEVANTE: o efeito excede a margem. O rotor 5 IMPORTA.")
    else:
        print("  >>> INCONCLUSIVO: IC mais largo que a margem -> mais seeds ou mais")
        print("      episodios por seed (ou margem mal escolhida).")
    return {
        "label": label,
        "n_a": len(a), "n_b": len(b),
        "mean_a": a.mean(), "mean_b": b.mean(),
        "diff": diff, "lo90": lo90, "hi90": hi90,
        "margin": margin, "equivalent": equivalent, "p_perm": pval,
        "underpowered": underpowered,
        "boot": boot,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--baseline", default="A_active")
    p.add_argument("--ablation", default="C_removed")
    p.add_argument("--metric", default="rmse_pos")
    p.add_argument("--margin-frac", type=float, default=0.10,
                   help="margem de equivalencia como fracao da media do baseline")
    p.add_argument("--margin-abs", type=float, default=None,
                   help="margem absoluta (sobrepoe --margin-frac)")
    p.add_argument("--by", default=None, help="coluna de estratificacao, ex. traj_type")
    p.add_argument("--drop-terminated", action="store_true",
                   help="exclui episodios que terminaram por violacao de limites")
    p.add_argument("--min-episode-idx", type=int, default=0,
                   help="descarta os primeiros N episodios de cada env (warm-up)")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    for col in ("run_tag", "seed", args.metric):
        if col not in df.columns:
            print(f"ERRO: coluna '{col}' ausente no CSV. Colunas: {list(df.columns)}")
            return 2

    if args.min_episode_idx > 0 and "episode_idx" in df.columns:
        df = df[df["episode_idx"] >= args.min_episode_idx]
    if args.drop_terminated and "terminated" in df.columns:
        df = df[df["terminated"] == 0]

    rng = np.random.default_rng(args.seed)

    groups = [("global", df)]
    if args.by and args.by in df.columns:
        groups += [(f"{args.by}={v}", d) for v, d in df.groupby(args.by)]

    results = []
    for label, sub in groups:
        da = sub[sub["run_tag"] == args.baseline]
        db = sub[sub["run_tag"] == args.ablation]
        if da.empty or db.empty:
            print(f"\n=== {label} === (ignorado: sem dados para uma das condicoes)")
            continue
        a = per_seed_means(da, args.metric)
        b = per_seed_means(db, args.metric)
        margin = args.margin_abs if args.margin_abs is not None else args.margin_frac * a.mean()
        results.append(analyse(a, b, margin, args.n_boot, rng,
                               f"{label} | {args.metric} | {args.ablation} vs {args.baseline}"))

    # Uso do rotor 5 no baseline: fecha o argumento fisico.
    base = df[df["run_tag"] == args.baseline]
    if not base.empty and "thrust5_mean_N" in base.columns:
        print("\n=== Uso do rotor 5 na condicao baseline ===")
        print(f"  empuxo medio do rotor 5     : {base['thrust5_mean_N'].mean():.4f} N")
        print(f"  fracao do impulso total     : {100 * base['thrust5_impulse_frac'].mean():.3f} %")
        if "power_mean" in base.columns:
            abl = df[df["run_tag"] == args.ablation]
            if not abl.empty:
                print(f"  potencia (sum w^3) baseline : {base['power_mean'].mean():.4g}")
                print(f"  potencia (sum w^3) ablacao  : {abl['power_mean'].mean():.4g}")

    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = len(results)
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
            for ax, r in zip(axes[0], results):
                ax.hist(r["boot"], bins=60, color="#4477aa", alpha=0.85)
                ax.axvline(0.0, color="k", lw=1)
                ax.axvline(r["margin"], color="#cc3311", ls="--", lw=1.2)
                ax.axvline(-r["margin"], color="#cc3311", ls="--", lw=1.2)
                ax.axvline(r["lo90"], color="#009988", ls=":", lw=1.4)
                ax.axvline(r["hi90"], color="#009988", ls=":", lw=1.4)
                ax.set_title(r["label"].split(" | ")[0] +
                             ("\nEQUIVALENTE" if r["equivalent"] else "\nNAO equivalente"),
                             fontsize=9)
                ax.set_xlabel("diferenca de RMSE [m]")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"\n[plot] escrito em {args.plot}")
        except Exception as exc:
            print(f"[plot] falhou: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
