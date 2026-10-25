"""
| File: test_evaluation_references.py
| Description: Evaluation manifest, training-config restore and play_multi reference contracts.
"""

import ast
import json
import math
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

RL_DIR = Path(__file__).resolve().parents[2] / "examples" / "rl"
sys.path.insert(0, str(RL_DIR))

from evaluation_log import validate_manifest
from task_loader import apply_env_overrides, control_timing
sys.path.insert(0, str(RL_DIR / "utils"))
import numpy as np
from lqr_design import closed_loop_poles, discounted_dlqr, double_integrator, pd_gains


def load_references():
    """Load the play_multi reference classes without importing Isaac Sim."""
    names = {"smoothstep_ramp", "TimedTrajectory", "ForwardFlightTrajectory", "LevelTurnTrajectory",
             "HelicalTurnTrajectory", "LongitudinalFlightTrajectory", "LemniscateTrajectory"}
    tree = ast.parse((RL_DIR / "play_multi.py").read_text())
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), "play_multi.py", "exec"), namespace)
    return namespace


@dataclass
class Cfg:
    moment_scale: float = 0.01
    sim_dt: float = 0.01
    decimation: int = 2


class EvaluationManifest(unittest.TestCase):
    def test_skrl_entry_requirements_and_vehicle(self):
        entry = {"name": "a", "type": "ppo2", "task": "t", "checkpoint": "c"}
        self.assertEqual(validate_manifest({"vehicle": "easyglider", "vehicles": [entry]})[1], "easyglider")
        for broken in ({**entry, "task": None}, {**entry, "checkpoint": None}, {**entry, "type": "x"},
                       {**entry, "run_metadata": "/missing.json"}):
            with self.assertRaises(ValueError):
                validate_manifest([broken])
        with self.assertRaises(ValueError):
            validate_manifest([entry, entry])
        with self.assertRaises(ValueError):
            validate_manifest({"vehicle": "easyglider", "vehicles": [{"name": "r", "type": "raptor"}]})

    def test_training_snapshot_then_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_metadata.json"
            path.write_text(json.dumps({"seed": 3, "git_commit": "abc",
                                        "environment": {"moment_scale": 0.12, "sim_dt": 0.01, "decimation": 2,
                                                        "unknown": 1}}))
            cfg = Cfg()
            log = apply_env_overrides(cfg, path, items=["decimation=1"], values={"moment_scale": 0.2})
        self.assertEqual((cfg.moment_scale, cfg.decimation), (0.2, 1))
        self.assertEqual(log["training_timing"], {"sim_dt": 0.01, "decimation": 2})
        self.assertEqual(log["not_in_task_cfg"], ["unknown"])
        self.assertAlmostEqual(control_timing(cfg)["control_hz"], 100.0)
        self.assertAlmostEqual(control_timing(Cfg())["control_hz"], 50.0)


class LqrBaseline(unittest.TestCase):
    def test_discounted_dlqr_matches_scipy_dare(self):
        try:
            from scipy.linalg import solve_discrete_are
        except ImportError:
            self.skipTest("requires scipy")
        dt, gamma = 0.02, 0.99
        A, B = double_integrator(dt)
        Q, R = np.diag([15.0] * 3 + [0.5] * 3), np.eye(3)
        P = solve_discrete_are(np.sqrt(gamma) * A, np.sqrt(gamma) * B, Q, R)
        K = np.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)
        K2, _ = discounted_dlqr(dt, 15.0, 0.5, 1.0, gamma)
        np.testing.assert_allclose(K2, K, atol=1e-8)
        self.assertLess(np.abs(closed_loop_poles(K2, dt)).max(), 1.0)

    def test_pd_gain_layout_and_manifest(self):
        np.testing.assert_allclose(pd_gains(2.0, [1.0, 1.0, 3.0])[2], [0, 0, 2.0, 0, 0, 3.0])
        validate_manifest([{"name": "lqr", "type": "lqr", "gains": {"kp": 1, "kd": 1}}])
        with self.assertRaises(ValueError):
            validate_manifest([{"name": "lqr", "type": "lqr"}])


class ReferenceTrajectories(unittest.TestCase):
    def setUp(self):
        self.ns = load_references()
        self.center = torch.tensor([[0.0, 0.0, 1.5]])

    def sample(self, trajectory, duration=10.0, dt=0.01):
        t = torch.arange(0.0, duration, dt)
        pos, vel, acc = zip(*(trajectory.at_time(torch.tensor([float(x)])) for x in t))
        return torch.cat(pos), torch.cat(vel), torch.cat(acc)

    def test_speed_ramp_is_smooth_bounded_and_consistent(self):
        trajectory = self.ns["ForwardFlightTrajectory"](1, 10.0, 1.5, 2.5, 6.0, 2.5, self.center, "cpu")
        pos, vel, acc = self.sample(trajectory)
        self.assertAlmostEqual(float(acc[0, 0]), 0.0, places=6)
        self.assertLessEqual(float(acc[:, 0].abs().max()), 6.0 + 1e-4)
        torch.testing.assert_close((pos[1:, 0] - pos[:-1, 0]) / 0.01, 0.5 * (vel[1:, 0] + vel[:-1, 0]),
                                   atol=5e-3, rtol=0)
        self.assertAlmostEqual(float(vel[-1, 0]), 10.0, places=4)

    def test_in_flight_start_is_steady(self):
        trajectory = self.ns["ForwardFlightTrajectory"](1, 10.0, 1.5, 2.5, 6.0, 2.5, self.center, "cpu", speed0=10.0)
        _, vel, acc = self.sample(trajectory, duration=1.0)
        torch.testing.assert_close(vel[:, 0], torch.full_like(vel[:, 0], 10.0))
        self.assertEqual(float(acc.abs().max()), 0.0)

    def test_turn_peak_is_centripetal(self):
        trajectory = self.ns["LevelTurnTrajectory"](1, 10.0, 30.0, 1.5, 2.5, 6.0, 2.5, self.center, "cpu",
                                                    speed0=10.0)
        self.assertAlmostEqual(trajectory.peak_acceleration(5.0, 0.01), 100.0 / 30.0, places=3)


if __name__ == "__main__":
    unittest.main()
