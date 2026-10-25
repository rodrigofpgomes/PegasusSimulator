"""
| File: test_scalar_batch_compat.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Scalar (origin/main) behaviour versus the tensor and batched implementations under identical inputs.
"""
import unittest
from offline_runtime import enable
enable()

try:
    import torch
except ImportError:
    torch = None
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


@unittest.skipIf(torch is None, 'PyTorch unavailable in review runtime')
class ScalarBatchCompat(unittest.TestCase):
    def test_scalar_thrust_defaults_match_origin_main(self):
        from pegasus.simulator.logic.thrusters.quadratic_thrust_curve import QuadraticThrustCurve
        curve = QuadraticThrustCurve()
        torch.testing.assert_close(curve._rotor_constant, torch.full((4,), 8.54858e-6))
        torch.testing.assert_close(curve._rolling_moment_coefficient, torch.full((4,), 1e-6))

    def test_batch_thrust_equals_scalar_without_motor_lag(self):
        from pegasus.simulator.logic.thrusters.quadratic_thrust_curve import QuadraticThrustCurve
        from pegasus.simulator.logic.thrusters.quadratic_thrust_curve_batch import QuadraticThrustCurveBatch
        kf, km = [5.84e-6] * 4, [3.504e-7] * 4
        scalar = QuadraticThrustCurve(config=dict(rotor_constant=kf, rolling_moment_coefficient=km))
        batch = QuadraticThrustCurveBatch(config=dict(rotor_constant=kf, reaction_moment_coefficient=km,
                                                      motor_time_constant_up=[1e-6], motor_time_constant_down=[1e-6]), n_vehicles=2)
        ref = torch.tensor([[300., 700., 1200., 0.], [-5., 10., 500., 1500.]])
        batch.set_input_reference(ref)
        forces, _, moment = batch.update(None, 0.01)
        for row in range(2):
            scalar.set_input_reference(ref[row])
            f, _, m = scalar.update(None, 0.01)
            torch.testing.assert_close(forces[row], f)
            torch.testing.assert_close(moment[row], m)

    def test_linear_drag_matches_scalar_equation_and_keeps_state(self):
        import sys, types
        interface = types.ModuleType('pegasus.simulator.logic.interface.pegasus_interface')
        interface.PegasusInterface = object
        sys.modules.setdefault('pegasus.simulator.logic.interface', types.ModuleType('pegasus.simulator.logic.interface'))
        sys.modules.setdefault('pegasus.simulator.logic.interface.pegasus_interface', interface)
        from pegasus.simulator.logic.dynamics import linear_drag
        drag = linear_drag.LinearDrag.__new__(linear_drag.LinearDrag)
        drag._drag_coefficients = torch.diag(torch.tensor([0.5, 0.3, 0.0]))
        drag._drag_force = torch.zeros(3)
        out = drag.update(types.SimpleNamespace(linear_body_velocity=torch.tensor([2., -4., 7.])), 0.01)
        torch.testing.assert_close(out, torch.tensor([-1.0, 1.2, 0.0]))
        torch.testing.assert_close(drag.drag, out)

    @unittest.skipIf(Rotation is None, 'SciPy unavailable')
    def test_ned_frd_conversions_match_scipy_reference(self):
        from pegasus.simulator.logic.state import State
        enu_ned = Rotation.from_quat([0.70711, 0.70711, 0, 0])
        flu_frd = Rotation.from_quat([1.0, 0, 0, 0])
        for seed in range(20):
            q = Rotation.random(random_state=seed).as_quat()
            state = State('cpu')
            state.attitude = torch.tensor([q[3], q[0], q[1], q[2]], dtype=torch.float32)
            state.linear_velocity = torch.tensor([1., -2., 3.])
            reference = enu_ned * Rotation.from_quat(q) * flu_frd
            wxyz = state.get_attitude_ned_frd().numpy()
            xyzw = reference.as_quat()
            expected = torch.tensor([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]).numpy()
            self.assertLess(min(abs(wxyz - expected).max(), abs(wxyz + expected).max()), 1e-5)
            torch.testing.assert_close(state.get_linear_velocity_ned().numpy(),
                                       enu_ned.apply([1., -2., 3.]).astype('float32'), atol=1e-5, rtol=0)

    def test_easyglider_puller_reaction_follows_sdf_turning_direction(self):
        # SDF: puller turningDirection "cw" -> Gazebo motor plugin torque +k_Q * omega^2 about +X (rotor +Z == body +X)
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[2] / 'extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/vehicle_physics.py'
        spec = importlib.util.spec_from_file_location('vehicle_physics_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.EASYGLIDER_PHYSICS["rot_dir"], [1.0])


if __name__ == '__main__':
    unittest.main()
