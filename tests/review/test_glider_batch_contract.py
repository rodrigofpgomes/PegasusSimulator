"""
| File: test_glider_batch_contract.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: CPU contract for the four-channel batched Glider vehicle.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


LOGIC = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic"


def load_class(path, name, namespace):
    """Load a production class without importing the Isaac Sim extension."""
    source = LOGIC / path
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class VehicleBatchStub:
    """Provide only the VehicleBatch methods used by GliderBatch in this test."""

    def __init__(self, stage_prefix, vehicle_batch_id, usd_file, n_vehicles, *args):
        self.n_vehicles = n_vehicles
        self.device = "cpu"
        self._backends = tuple(args[5])

    def initialize(self):
        return

    def apply_forces_and_torques_all_parts(self, forces, torques):
        self.applied_forces = forces.clone()
        self.applied_torques = torques.clone()

    @property
    def parts_per_vehicle(self):
        return self._parts_per_vehicle

    @property
    def body_index(self):
        return self._body_index


class AerodynamicsStub:
    n_vehicles = 2

    def update(self, state, delta, dt):
        self.delta = delta.clone()
        return (
            torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
        )


class BackendStub:
    def __init__(self):
        self.command = torch.tensor([[2., .1, .2, .3], [3., -.1, -.2, -.3]])

    def update(self, dt):
        return

    def input_reference(self):
        return self.command


class GliderBatchContract(unittest.TestCase):
    def setUp(self):
        namespace = {
            "torch": torch,
            "State": object,
            "ThrustCurve": object,
            "Sequence": list,
            "VehicleBatch": VehicleBatchStub,
            "GliderBatchConfig": object,
            "InputMode": str,
        }
        transforms = LOGIC / "transforms.py"
        exec(compile(transforms.read_text(), str(transforms), "exec"), namespace)

        ThrustCurve = load_class("thrusters/glider_thrust_curve_batch.py", "GliderThrustCurveBatch", namespace)
        GliderBatch = load_class("vehicles/glider_batch.py", "GliderBatch", namespace)

        self.backend = BackendStub()
        self.aerodynamics = AerodynamicsStub()
        thruster = ThrustCurve(
            config={
                "rotor_constant": [1.0],
                "reaction_moment_coefficient": [0.1],
                "rot_dir": [1],
                "motor_time_constant_up": 0.0,
                "motor_time_constant_down": 0.0,
            },
            n_vehicles=2,
        )
        config = SimpleNamespace(
            sensors=(), graphical_sensors=(), graphs=(), backends=(self.backend,),
            aerodynamics=self.aerodynamics, thrust_curve=thruster,
            propeller_prim_name="rotor", propeller_axis_body=torch.tensor([1., 0., 0.]),
        )
        self.vehicle = GliderBatch(n_vehicles=2, config=config)
        self.vehicle._parts_per_vehicle = 2
        self.vehicle._body_index = 0
        self.vehicle._stage_prefix = "/World/glider"
        body_attitude = torch.tensor([1., 0., 0., 0.])
        rotor_attitude = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
        attitudes = torch.stack((body_attitude, rotor_attitude,
                                 body_attitude, rotor_attitude))
        self.vehicle._vehicle_prims = SimpleNamespace(
            prim_paths=["/World/glider_0/body", "/World/glider_0/rotor",
                        "/World/glider_1/body", "/World/glider_1/rotor"],
            get_world_poses=lambda: (torch.zeros(4, 3), attitudes),
        )
        self.vehicle._state = SimpleNamespace()
        self.vehicle.start()
        self.vehicle._sim_running = True

    def test_backend_command_applies_four_channels_and_reaction_axis(self):
        self.assertEqual(self.vehicle.actuator_dim, 4)
        self.assertEqual(self.vehicle.thrusters._num_rotors, 1)
        self.assertEqual(tuple(self.vehicle.thrusters.max_rotor_velocity.shape), (1,))
        self.assertEqual(self.vehicle.propeller_index, 1)
        self.vehicle.update(0.01)

        torch.testing.assert_close(self.vehicle.surface_deflections, self.backend.command[:, 1:4])
        torch.testing.assert_close(self.vehicle.applied_forces[:, 0],
                                   torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        torch.testing.assert_close(self.vehicle.applied_forces[:, 1],
                                   torch.tensor([[0., 0., 4.], [0., 0., 9.]]),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(self.vehicle.applied_torques[:, 0],
                                   torch.tensor([[.5, .2, .3], [1.3, .5, .6]]))

    def test_direct_command_can_override_one_vehicle_after_partial_reset(self):
        self.vehicle._actuator_reference[0] = torch.tensor([4., .4, .5, .6])
        self.vehicle._direct_reference_mask[0] = True

        command = self.vehicle._get_actuator_reference()

        torch.testing.assert_close(command[0], torch.tensor([4., .4, .5, .6]))
        torch.testing.assert_close(command[1], self.backend.command[1])


if __name__ == "__main__":
    unittest.main()
