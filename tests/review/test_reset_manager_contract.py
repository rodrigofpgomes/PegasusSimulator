"""
| File: test_reset_manager_contract.py
| Description: CPU check of rotor state and command separation during partial resets.
"""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import torch


LOGIC = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic"


def load_reset_manager():
    """Load the production reset manager without Isaac Sim imports."""
    source = LOGIC / "rl/reset_manager.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ResetManager")
    transforms = runpy.run_path(str(LOGIC / "transforms.py"))
    namespace = {"torch": torch, "quaternion_apply": transforms["quaternion_apply"]}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ResetManager"]


class RootView:
    def __init__(self):
        self.positions = torch.zeros((2, 3))
        self.attitudes = torch.zeros((2, 4))
        self.velocities = torch.zeros((2, 6))

    def set_world_poses(self, positions, orientations, indices):
        self.positions[indices] = positions
        self.attitudes[indices] = orientations

    def set_velocities(self, velocities, indices):
        self.velocities[indices] = velocities


class ResetManagerContract(unittest.TestCase):
    def test_partial_reset_sets_rotor_state_without_replacing_motor_command(self):
        manager_class = load_reset_manager()
        thrusters = SimpleNamespace(
            _num_rotors=1,
            min_rotor_velocity=torch.tensor([0.0]),
            max_rotor_velocity=torch.tensor([100.0]),
            _velocity=torch.zeros((2, 1)),
            _input_reference=torch.full((2, 1), 17.0),
        )
        vehicle = SimpleNamespace(
            n_vehicles=2,
            parts_per_vehicle=1,
            _init_pos=torch.zeros((2, 3)),
            _init_orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            _thrusters=thrusters,
            _root_prims=RootView(),
            set_state_batch=lambda *args, **kwargs: None,
        )
        manager = manager_class([vehicle], device="cpu")
        manager.reset_envs(
            torch.tensor([1]),
            init_overrides={
                "orientations": torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
                "rotor_norm": torch.tensor([[0.5]]),
            },
        )

        torch.testing.assert_close(vehicle._root_prims.attitudes[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(thrusters._velocity[:, 0], torch.tensor([0.0, 75.0]))
        torch.testing.assert_close(thrusters._input_reference[:, 0], torch.tensor([17.0, 17.0]))
        # Normalization is computed from the rotor velocity in the environment
        manager.reset_envs(torch.tensor([0]))
        self.assertFalse(hasattr(thrusters, "_reset_rotor_norm"))


if __name__ == "__main__":
    unittest.main()
