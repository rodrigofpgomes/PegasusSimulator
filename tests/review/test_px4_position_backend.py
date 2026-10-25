"""
| File: test_px4_position_backend.py
| Description: CPU checks for PX4 position targets and motor command ownership.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_glider_batch_contract import load_class
from test_native_px4_pipeline import load_px4_backend


def load_position_backend():
    """Load the production PX4 position controller without Isaac Sim."""
    estimator, namespace = load_px4_backend()
    namespace["PX4MavlinkEstimatorBackend"] = estimator
    namespace["SensorSuiteBatch"] = Mock()
    return load_class("backends/px4_position_backend.py", "PX4PositionBackend", namespace)


def position_backend(feedforward=True):
    backend = object.__new__(load_position_backend())
    backend.config = {"feedforward": feedforward}
    backend.buffer = SimpleNamespace(offset=torch.tensor([1.0, 2.0, 3.0]), system_id=1, component_id=1)
    backend._current_utime = 2500000
    backend.telemetry = SimpleNamespace(mav=SimpleNamespace(set_position_target_local_ned_send=Mock()))
    backend.mavutil = SimpleNamespace(mavlink=SimpleNamespace(MAV_FRAME_LOCAL_NED=1))
    backend.reset_manager = SimpleNamespace(goal_pos=torch.tensor([[4.0, 8.0, 11.0]]),
                                            goal_vel=torch.tensor([[1.0, 2.0, 3.0]]),
                                            goal_acc=torch.tensor([[0.1, 0.2, 0.3]]))
    return backend


class PX4PositionContract(unittest.TestCase):
    def test_position_only_target_uses_aligned_ned_origin(self):
        backend = position_backend(feedforward=False)
        backend._send_goal()
        packet = backend.telemetry.mav.set_position_target_local_ned_send.call_args.args
        self.assertEqual(packet[:5], (2500, 1, 1, 1, 1024 | 2048 | 56 | 448))
        self.assertEqual(packet[5:8], (6.0, 3.0, -8.0))
        self.assertEqual(packet[8:], (0.0,) * 8)

    def test_feedforward_target_sends_reference_velocity_and_acceleration_in_ned(self):
        backend = position_backend()
        backend._send_goal()
        packet = backend.telemetry.mav.set_position_target_local_ned_send.call_args.args
        self.assertEqual(packet[4], 1024 | 2048)
        self.assertEqual(packet[5:8], (6.0, 3.0, -8.0))
        torch.testing.assert_close(torch.tensor(packet[8:14]), torch.tensor([2.0, 1.0, -3.0, 0.2, 0.1, -0.3]))

    def test_position_controller_is_first_backend_command_source(self):
        backend = position_backend()
        backend._prepared = True
        backend._input_reference = torch.zeros((1, 4))
        backend.step = Mock()
        backend._send_goal = Mock()
        backend.request_offboard_and_arm = Mock()
        backend.motor_velocities = Mock(return_value=torch.tensor([[100., 200., 300., 400.]]))
        backend.record = Mock()

        backend.update(0.01)

        backend.step.assert_called_once_with(0.01)
        backend._send_goal.assert_called_once()
        torch.testing.assert_close(backend.input_reference(), torch.tensor([[100., 200., 300., 400.]]))
        backend.record.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
