"""
| File: test_rl_backend_reset.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: CPU checks for full and partial RL backend resets.
"""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import torch


LOGIC = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic"


class BackendStub:
    def __init__(self, config=None):
        self._vehicle = None

    @property
    def vehicle(self):
        return self._vehicle


class MultirotorBatchStub:
    pass


def load_backend():
    """Load the production backend class without Isaac Sim imports."""
    source = LOGIC / "rl/rl_backend.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RLBackend")
    transforms = runpy.run_path(str(LOGIC / "transforms.py"))
    namespace = {
        "torch": torch,
        "Backend": BackendStub,
        "StateBatch": object,
        "MultirotorBatch": MultirotorBatchStub,
        "quaternion_apply": transforms["quaternion_apply"],
        "quaternion_to_matrix": transforms["quaternion_to_matrix"],
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["RLBackend"]


class RLBackendResetContract(unittest.TestCase):
    def setUp(self):
        backend_class = load_backend()
        self.backend = backend_class(n_vehicles=3, action_mode="rotor_velocity_direct", device="cpu")
        self.selected_vehicle_mode = None
        vehicle = SimpleNamespace(
            n_vehicles=3,
            parts_per_vehicle=2,
            body_index=1,
            device="cpu",
            _thrusters=SimpleNamespace(_num_rotors=1),
            actuator_dim=4,
            set_input_mode=lambda mode: setattr(self, "selected_vehicle_mode", mode),
        )
        self.backend.initialize(vehicle)
        self.backend.start()

        # Distinct values make it clear which environment rows were reset
        self.backend._forces.fill_(1.0)
        self.backend._torques.fill_(2.0)
        self.backend._external_forces.fill_(3.0)
        self.backend._external_torques.fill_(4.0)
        self.backend._input_reference.fill_(5.0)
        self.backend._state_cache.fill_(6.0)

    def test_direct_force_mode_matches_vehicle_interface(self):
        self.assertEqual(self.selected_vehicle_mode, "rotor_velocity_direct")
        self.backend._action_mode = "direct_force"
        self.backend.start()

        self.assertEqual(self.selected_vehicle_mode, "direct_force")

    def test_partial_reset_clears_only_selected_commands(self):
        ids = torch.tensor([1])
        position = torch.tensor([[7.0, 8.0, 9.0]])
        attitude = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        self.backend.set_state(ids, position, attitude)

        for name in ("_forces", "_torques", "_external_forces", "_external_torques", "_input_reference"):
            buffer = getattr(self.backend, name)
            self.assertTrue((buffer[1] == 0.0).all(), name)
            self.assertTrue((buffer[[0, 2]] != 0.0).all(), name)

        torch.testing.assert_close(self.backend._state_cache[1, 0:3], position[0])
        torch.testing.assert_close(self.backend._state_cache[1, 6:10], attitude[0])
        self.assertTrue((self.backend._state_cache[[0, 2]] == 6.0).all())
        self.assertTrue(self.backend._received_first_state)

    def test_full_reset_clears_all_buffers(self):
        self.backend.reset()

        for name in ("_forces", "_torques", "_external_forces", "_external_torques", "_input_reference", "_state_cache"):
            self.assertTrue((getattr(self.backend, name) == 0.0).all(), name)

        self.assertFalse(self.backend._received_first_state)

    def test_rotor_velocity_mode_keeps_wrench_allocation(self):
        self.backend._action_mode = "rotor_velocity"
        self.backend._received_first_state = True
        self.backend.vehicle.force_and_torques_to_velocities = lambda thrust, torque: torch.cat(
            (thrust.unsqueeze(-1), torque), dim=-1
        )
        self.backend._forces[:, 1, 2] = torch.tensor([1.0, 2.0, 3.0])
        self.backend._torques[:, 1, :] = torch.tensor([[4.0, 5.0, 6.0]]).expand(3, -1)

        self.backend.update(0.01)

        torch.testing.assert_close(
            self.backend.input_reference(),
            torch.tensor([[1.0, 4.0, 5.0, 6.0], [2.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0]]),
        )

    def test_inner_loop_requires_multirotor_in_rotor_velocity_mode(self):
        backend_class = load_backend()
        backend = backend_class(n_vehicles=3, action_mode="rotor_velocity", inner_loop=True, device="cpu")
        backend.initialize(self.backend.vehicle)

        with self.assertRaisesRegex(ValueError, "MultirotorBatch"):
            backend.start()

    def test_inner_loop_uses_body_force_without_previous_moment(self):
        backend_class = load_backend()
        backend = backend_class(n_vehicles=2, action_mode="rotor_velocity", inner_loop=True, device="cpu")
        vehicle = MultirotorBatchStub()
        vehicle.n_vehicles = 2
        vehicle.parts_per_vehicle = 2
        vehicle.body_index = 1
        vehicle.device = "cpu"
        vehicle._thrusters = SimpleNamespace(_num_rotors=4)
        vehicle.set_input_mode = lambda mode: setattr(vehicle, "input_mode", mode)
        vehicle.force_and_torques_to_velocities = lambda thrust, torque: torch.cat(
            (thrust.unsqueeze(-1), torque), dim=-1
        )
        backend.initialize(vehicle)
        backend.start()

        # Previous moments must not affect either the tilted or zero-force command
        backend.set_state(
            torch.tensor([0, 1]),
            torch.zeros((2, 3)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
        )
        forces = torch.zeros((2, 2, 3))
        torques = torch.zeros_like(forces)
        forces[0, 1] = torch.tensor([1.0, 0.0, 1.0])
        torques[0, 1, 2] = 0.2
        torques[1, 1, 2] = 0.3
        backend.set_forces_and_torques(forces, torques)
        backend.update(0.01)

        reference = backend.input_reference()
        self.assertEqual(vehicle.input_mode, "rotor_velocity")
        self.assertTrue(torch.isfinite(reference).all())
        torch.testing.assert_close(reference[:, 0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(reference[:, 3], torch.zeros(2))
        self.assertGreater(abs(reference[0, 2].item()), 0.0)

    def test_hide_goal_markers_moves_only_selected_markers(self):
        captured = {}
        self.backend._goal_marker_view = SimpleNamespace(
            set_world_poses=lambda **kwargs: captured.update(kwargs)
        )

        self.backend.hide_goal_markers(torch.tensor([1, 2]))

        torch.testing.assert_close(captured["indices"], torch.tensor([1, 2]))
        torch.testing.assert_close(captured["positions"][:, 2], torch.tensor([-100.0, -100.0]))
        torch.testing.assert_close(captured["orientations"][:, 0], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
