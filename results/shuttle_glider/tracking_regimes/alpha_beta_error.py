"""RMS error of the angle of attack and sideslip implied by the observed air velocity (estimated velocity rotated
into the estimated body frame, no wind) w.r.t. the true ones, over samples with true speed above 3 m/s (table of
the PX4 mission). Usage: python3 alpha_beta_error.py RUN_DIR [...]  (RUN_DIR with px4_estimate.jsonl)
"""
import json, sys
from pathlib import Path
import numpy as np


def body_velocity(s):
    v, q = s[:, 3:6], s[:, 6:10] / np.linalg.norm(s[:, 6:10], axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    return np.einsum("nji,nj->ni", R, v)   # R^T v


def angles(vb):
    return np.arctan2(-vb[:, 2], vb[:, 0]), np.arcsin(np.clip(vb[:, 1] / np.linalg.norm(vb, axis=1), -1, 1))


for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path(run) / "px4_estimate.jsonl")][1:]
    est = np.array([r["estimated_state"][0] for r in rows]); tru = np.array([r["true_state"][0] for r in rows])
    m = np.linalg.norm(tru[:, 3:6], axis=1) > 3.0
    (ae, be), (at, bt) = angles(body_velocity(est[m])), angles(body_velocity(tru[m]))
    da, db = np.angle(np.exp(1j * (ae - at))), np.angle(np.exp(1j * (be - bt)))
    print(f"{run}: alpha {np.degrees(np.sqrt(np.mean(da ** 2))):.1f} deg, beta {np.degrees(np.sqrt(np.mean(db ** 2))):.1f} deg ({m.sum()} samples)")
