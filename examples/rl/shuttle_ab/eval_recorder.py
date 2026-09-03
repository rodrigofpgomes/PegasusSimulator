"""
| File: eval_recorder.py
| Registo de metricas POR EPISODIO num CSV, para alimentar o eval_tost.py.
| Agnostico ao runner (skrl / rsl_rl / rl_games): e' o env que chama .add().
"""

from __future__ import annotations

import csv
import os
import threading

_FIELDS = [
    "run_tag",
    "seed",
    "env_id",
    "episode_idx",
    "traj_type",
    "rmse_pos",
    "rmse_vel",
    "terminated",
    "steps",
    "thrust5_mean_N",
    "thrust5_impulse_frac",
    "power_mean",
    "total_reward",
]


class EpisodeRecorder:
    def __init__(self, path: str, run_tag: str, seed_tag: int, flush_every: int = 256):
        self.path = path
        self.run_tag = run_tag
        self.seed_tag = int(seed_tag)
        self.flush_every = flush_every
        self._rows: list[dict] = []
        self._episode_counter: dict[int, int] = {}
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_FIELDS).writeheader()

    def add(
        self,
        env_ids,
        rmse_pos,
        rmse_vel,
        traj_langevin,
        terminated,
        steps,
        thrust5_mean,
        thrust5_frac,
        power_mean,
        total_reward,
    ):
        ids = [int(i) for i in env_ids.tolist()]
        rp = [float(x) for x in rmse_pos.tolist()]
        rv = [float(x) for x in rmse_vel.tolist()]
        tl = [bool(x) for x in traj_langevin.tolist()]
        td = [bool(x) for x in terminated.tolist()]
        st = [float(x) for x in steps.tolist()]
        t5 = [float(x) for x in thrust5_mean.tolist()]
        f5 = [float(x) for x in thrust5_frac.tolist()]
        pw = [float(x) for x in power_mean.tolist()]
        tr = [float(x) for x in total_reward.tolist()]

        with self._lock:
            for k, env_id in enumerate(ids):
                n = self._episode_counter.get(env_id, 0)
                self._episode_counter[env_id] = n + 1
                self._rows.append(
                    {
                        "run_tag": self.run_tag,
                        "seed": self.seed_tag,
                        "env_id": env_id,
                        "episode_idx": n,
                        "traj_type": "langevin" if tl[k] else "null",
                        "rmse_pos": rp[k],
                        "rmse_vel": rv[k],
                        "terminated": int(td[k]),
                        "steps": st[k],
                        "thrust5_mean_N": t5[k],
                        "thrust5_impulse_frac": f5[k],
                        "power_mean": pw[k],
                        "total_reward": tr[k],
                    }
                )
            if len(self._rows) >= self.flush_every:
                self._flush_locked()

    def _flush_locked(self):
        if not self._rows:
            return
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_FIELDS).writerows(self._rows)
        self._rows.clear()

    def flush(self):
        with self._lock:
            self._flush_locked()
