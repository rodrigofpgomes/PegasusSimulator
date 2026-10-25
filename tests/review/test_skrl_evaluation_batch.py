"""
| File: test_skrl_evaluation_batch.py
| Description: CPU checks for skrl batch decisions and PX4 command ownership.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_glider_batch_contract import LOGIC


ROOT = Path(__file__).resolve().parents[2]


def load_controller():
    """Load the real controller methods without requiring Isaac Sim imports."""
    class RL:
        def update(self, dt):
            return
        def get_state(self):
            return self.state

    source = ROOT / "examples/rl/utils/skrl_agent_controller.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "SkrlAgentBackend")
    namespace = {"RLBackend": RL, "torch": torch, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["SkrlAgentBackend"]


class SkrlEvaluationBatch(unittest.TestCase):
    def make_controller(self, count):
        controller = object.__new__(load_controller())
        controller._px4_config = None
        controller._px4_backend = None
        controller._received_first_state = True
        controller._control_counter = 0
        controller._control_decimation = 2
        controller._algo = "ppo2"
        controller.reset_manager = None
        controller.state = torch.zeros(count, 13)
        controller.state[:, 6] = 1.
        controller.state[:, 0] = torch.arange(count, dtype=torch.float32)

        # Use the task's actual observation function to exercise all state rows.
        task_path = ROOT / "examples/rl/tasks/shuttle/thrust_moment/isaac_lab/quadcopter_env.py"
        task_tree = ast.parse(task_path.read_text())
        method = next(node for node in ast.walk(task_tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_get_observations")
        namespace = {"torch": torch}
        exec(compile((LOGIC / "transforms.py").read_text(), "transforms.py", "exec"), namespace)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(task_path), "exec"), namespace)
        env = SimpleNamespace(backend=controller, _goal_pos=torch.zeros(count, 3), _goal_vel=torch.zeros(count, 3),
                              _pre_physics_step=Mock(), _apply_action=Mock())
        env._get_observations = lambda: namespace["_get_observations"](env)
        controller._env = env
        actions = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 20.
        controller._agent = SimpleNamespace(act=Mock(return_value=(actions, None, {"mean_actions": actions})))
        return controller, actions

    def test_ground_truth_preserves_rows_and_control_decimation(self):
        for count in (1, 3):
            with self.subTest(vehicles=count):
                controller, actions = self.make_controller(count)
                controller.update(0.01)
                controller.update(0.01)
                controller._agent.act.assert_called_once()
                observation = controller._agent.act.call_args.args[0]
                self.assertEqual(observation.shape, (count, 12))
                torch.testing.assert_close(observation[:, 9], -controller.state[:, 0])
                controller._env._pre_physics_step.assert_called_once()
                torch.testing.assert_close(controller._env._pre_physics_step.call_args.args[0], actions)
                self.assertEqual(controller._env._apply_action.call_count, 2)

    def test_px4_rate_mode_uses_motor_outputs_without_task_force_application(self):
        controller, actions = self.make_controller(1)
        motors = torch.tensor([[100., 200., 300., 400.]])
        estimate = controller.state.clone()
        estimate[:, 0] = 5.
        controller._px4_config = {"mode": "body_rate"}
        controller._px4_backend = SimpleNamespace(
            mode="body_rate", ready=True, current=estimate,
            step=Mock(), record=Mock(), send_body_rate=Mock(),
            request_offboard_and_arm=Mock(), _poll=Mock(),
            motor_velocities=Mock(return_value=motors))
        controller.update(0.01)
        controller.update(0.01)
        torch.testing.assert_close(controller._agent.act.call_args.args[0][:, 9], torch.tensor([-5.]))
        self.assertIs(controller._input_reference, motors)
        controller._env._apply_action.assert_not_called()
        self.assertEqual(controller._px4_backend.send_body_rate.call_count, 2)
        torch.testing.assert_close(controller._px4_backend.send_body_rate.call_args.args[0], actions)

    def test_multirotor_uses_first_backend_for_all_batch_rows(self):
        source = LOGIC / "vehicles/multirotor_batch.py"
        tree = ast.parse(source.read_text())
        vehicle_class = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "MultirotorBatch")
        method = next(node for node in vehicle_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "update")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        forces = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        torques = forces * 0.1
        primary = SimpleNamespace(update=Mock(), get_forces_and_torques=Mock(return_value=(forces, torques)))
        secondary = SimpleNamespace(update=Mock(), get_forces_and_torques=Mock())
        vehicle = SimpleNamespace(_sim_running=True, _input_mode="direct_force",
                                  _backends=[primary, secondary], _state=object(),
                                  _forces=torch.zeros_like(forces), _torques=torch.zeros_like(torques),
                                  _body_index=0, _drag=SimpleNamespace(update=Mock(return_value=torch.zeros(2, 3))),
                                  apply_forces_and_torques_all_parts=Mock())
        namespace["update"](vehicle, 0.01)
        secondary.update.assert_called_once_with(0.01)
        secondary.get_forces_and_torques.assert_not_called()
        torch.testing.assert_close(vehicle._forces, forces)
        torch.testing.assert_close(vehicle._torques, torques)

    def test_multirotor_applies_configured_body_drag_on_body_part(self):
        source = LOGIC / "vehicles/multirotor_batch.py"
        tree = ast.parse(source.read_text())
        vehicle_class = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "MultirotorBatch")
        method = next(node for node in vehicle_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "update")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        zeros = torch.zeros(2, 5, 3)
        backend = SimpleNamespace(update=Mock(), get_forces_and_torques=Mock(return_value=(zeros, zeros)))
        drag = torch.tensor([[-1., -2., -3.], [4., 5., 6.]])
        vehicle = SimpleNamespace(_sim_running=True, _input_mode="direct_force", _backends=[backend],
                                  _state=object(), _forces=torch.zeros_like(zeros), _torques=torch.zeros_like(zeros),
                                  _body_index=2, _drag=SimpleNamespace(update=Mock(return_value=drag)),
                                  apply_forces_and_torques_all_parts=Mock())
        namespace["update"](vehicle, 0.01)
        torch.testing.assert_close(vehicle._forces[:, 2, :], drag)
        self.assertEqual(float(vehicle._forces.abs().sum()), float(drag.abs().sum()))


if __name__ == "__main__":
    unittest.main()
