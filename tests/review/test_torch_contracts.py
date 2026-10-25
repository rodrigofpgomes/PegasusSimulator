"""
| File: test_torch_contracts.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Numerical contracts executable with PyTorch alone; simulator APIs are mocked explicitly.
"""
import ast
import dataclasses
import importlib.util
import math
import types
import unittest
from offline_runtime import enable
enable()
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

ROOT=Path(__file__).resolve().parents[2]/'extensions/pegasus.simulator/pegasus/simulator/logic'

def definitions(path, names, namespace):
    """Execute the production definitions without importing the simulator extension."""
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),namespace)

@unittest.skipIf(torch is None, 'PyTorch unavailable in review runtime')
class TensorContracts(unittest.TestCase):
    def setUp(self):
        self.ns={'torch':torch,'math':math,'dataclass':dataclasses.dataclass,'DEG':math.pi/180}
        source=(ROOT/'transforms.py').read_text()
        exec(compile(source,str(ROOT/'transforms.py'),'exec'),self.ns)
        definitions(Path('rotations.py'),['rot_FLU_to_FRD','rot_ENU_to_NED'],self.ns)

    def test_frd_velocity_is_not_acceleration(self):
        definitions(Path('state.py'),['State'],self.ns)
        state=self.ns['State']('cpu')
        state.linear_velocity=torch.tensor([1.,2.,3.])
        state.linear_acceleration=torch.tensor([11.,22.,33.])
        torch.testing.assert_close(state.get_linear_body_velocity_ned_frd(),torch.tensor([1.,-2.,-3.]))

    def test_reset_rotates_body_rates_and_preserves_other_instances(self):
        definitions(Path('rl/reset_manager.py'),['_reset_vehicle_pose'],self.ns)
        n=3
        class View:
            def __init__(self):
                self.pos=torch.zeros(n,3);self.ori=torch.zeros(n,4);self.vel=torch.zeros(n,6)
            def set_world_poses(self,positions,orientations,indices):self.pos[indices]=positions;self.ori[indices]=orientations
            def set_velocities(self,velocities,indices):self.vel[indices]=velocities
        class ChildView:
            def set_world_poses(self,*args,**kwargs):raise AssertionError('non-root pose write')
            def set_velocities(self,*args,**kwargs):raise AssertionError('non-root velocity write')
        vehicle=types.SimpleNamespace(_root_prims=View(),_vehicle_prims=ChildView())
        q=torch.tensor([[2*math.sqrt(.5),0.,0.,2*math.sqrt(.5)]])
        self.ns['_reset_vehicle_pose'](types.SimpleNamespace(device='cpu'),vehicle,torch.tensor([1]),torch.tensor([[3.,4.,5.]]),q,torch.tensor([[6.,7.,8.]]),torch.tensor([[1.,0.,0.]]))
        torch.testing.assert_close(vehicle._root_prims.vel[1],torch.tensor([6.,7.,8.,0.,1.,0.]),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(vehicle._root_prims.pos[1],torch.tensor([3.,4.,5.]))
        torch.testing.assert_close(vehicle._root_prims.ori[1],q[0]/2,atol=1e-6,rtol=1e-6)
        self.assertTrue((vehicle._root_prims.pos[[0,2]]==0).all())
        self.assertTrue((vehicle._root_prims.vel[[0,2]]==0).all())

    def test_combined_rotors_keep_reset_velocities_in_sync(self):
        self.ns['State'] = object
        self.ns['ThrustCurve'] = object
        self.ns['Sequence'] = list
        definitions(Path('thrusters/quadratic_thrust_curve_batch.py'), ['QuadraticThrustCurveBatch'], self.ns)
        definitions(Path('thrusters/glider_thrust_curve_batch.py'), ['GliderThrustCurveBatch'], self.ns)
        definitions(Path('vehicles/shuttle_glider_batch2.py'), ['_CombinedRotorInterface'], self.ns)

        with self.assertRaisesRegex(ValueError, 'reaction_moment_coefficient'):
            self.ns['QuadraticThrustCurveBatch']({
                'rolling_moment_coefficient': [3.504e-7] * 4,
            }, n_vehicles=2)

        shuttle = self.ns['QuadraticThrustCurveBatch'](config={
            'motor_time_constant_up': 0.0,
            'motor_time_constant_down': 0.0,
        }, n_vehicles=2)
        glider = self.ns['GliderThrustCurveBatch'](config={
            'motor_time_constant_up': 0.0,
            'motor_time_constant_down': 0.0,
        }, n_vehicles=2)
        combined = self.ns['_CombinedRotorInterface'](shuttle, glider, 2, 'cpu')

        reset_velocity = torch.tensor([[100., 200., 300., 400., 500.],
                                       [600., 700., 800., 900., 1000.]])
        combined._velocity[:] = reset_velocity
        combined.sync_to_models()
        torch.testing.assert_close(shuttle._velocity, reset_velocity[:, :4])
        torch.testing.assert_close(glider._velocity, reset_velocity[:, 4:5])

        command = reset_velocity + 10.0
        shuttle.set_input_reference(command[:, :4])
        glider.set_input_reference(command[:, 4:5])
        shuttle.update(None, 0.01)
        glider.update(None, 0.01)
        combined.sync_from_models()

        torch.testing.assert_close(combined._velocity, command)

if __name__=='__main__':unittest.main()
