# -*- coding: utf-8 -*-
"""Gera um trajectories.csv sintetico com o esquema exato do
EpisodeTrajectoryRecorder do play_multi.py, para validar o csv_to_tost.py
sem precisar do Isaac Sim."""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(int(os.environ.get("SEED", "7")))
N_ENVS, N_EPS, N_STEPS, N_ROTORS, DT = 4, 3, 500, 5, 0.01
A, T, ZREF = 1.5, 8.0, 1.5
W = 2 * np.pi / T

# thrust5_impulse_frac verdadeiro: ~0 em A/B/C (a hipotese nula do estudo)
COND = {"A_active": dict(err=0.060, w5=40.0), "B_dead": dict(err=0.061, w5=0.0),
        "C_removed": dict(err=0.059, w5=0.0)}

rows = []
for ctrl, p in COND.items():
    for env in range(N_ENVS):
        for ep in range(N_EPS):
            t = np.arange(N_STEPS) * DT
            gx, gy, gz = A * np.sin(W * t), (A / 2) * np.sin(2 * W * t), np.full(N_STEPS, ZREF)
            rvx, rvy = A * W * np.cos(W * t), A * W * np.cos(2 * W * t)
            e = RNG.normal(0, p["err"], (N_STEPS, 3))
            x, y, z = gx + e[:, 0], gy + e[:, 1], gz + e[:, 2]
            vx, vy, vz = rvx + e[:, 0], rvy + e[:, 1], e[:, 2]
            r = {"controller": ctrl, "env_id": env, "episode_id": ep,
                 "episode_step": np.arange(N_STEPS), "global_step": np.arange(N_STEPS),
                 "t": t, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
                 "speed": np.linalg.norm(np.c_[vx, vy, vz], axis=1),
                 "goal_x": gx, "goal_y": gy, "goal_z": gz,
                 "ref_vx": rvx, "ref_vy": rvy, "ref_vz": np.zeros(N_STEPS),
                 "pos_error": np.linalg.norm(e, axis=1),
                 "z_error": e[:, 2], "att_err_ff": np.abs(e[:, 0]),
                 "att_err_pd": np.abs(e[:, 1])}
            for grp in ("cmd_w", "cmd_rpm", "actual_w", "actual_rpm",
                        "l2f_action", "l2f_cmd_rpm", "l2f_current_rpm"):
                for i in range(N_ROTORS):
                    base = 900.0 if i < 4 else p["w5"]
                    r[f"{grp}_{i}"] = np.abs(RNG.normal(base, 20.0, N_STEPS))
            for c in ("fx_body", "fy_body", "fz_body", "body_force_norm",
                      "tx_body", "ty_body", "tz_body", "body_torque_norm"):
                r[c] = RNG.normal(0, 1, N_STEPS)
            for c in ("heading_deg", "heading_ref_deg", "heading_error_deg",
                      "heading_error_abs_deg", "heading_alignment",
                      "delta_v_slipstream_est"):
                r[c] = RNG.normal(30.0 if "abs" in c else 0.0, 5.0, N_STEPS)
            rows.append(pd.DataFrame(r))

os.makedirs("/tmp/pr/lemn_seed0", exist_ok=True)
pd.concat(rows, ignore_index=True).to_csv(
    "/tmp/pr/lemn_seed0/trajectories.csv", index=False)
print("trajectories.csv sintetico escrito")
