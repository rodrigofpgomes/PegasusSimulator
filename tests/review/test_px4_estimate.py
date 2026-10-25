"""
| File: test_px4_estimate.py
| Description: CPU checks for the PX4 link: frames, lockstep, simulated-time estimates and encoders.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_native_px4_pipeline import load_px4_backend

PX4Backend, _runtime = load_px4_backend()
EstimateBuffer = _runtime["EstimateBuffer"]
convert_state = _runtime["convert_state"]
EstimateUnavailable = _runtime["EstimateUnavailable"]
ThrusterControl = _runtime["ThrusterControl"]
_runtime["mavutil"].mavlink.MAV_MODE_FLAG_SAFETY_ARMED = 128   # used by the inherited handle_control


class Message(SimpleNamespace):
    def get_type(self):
        return self.kind

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1


def estimate_messages(stamp_ms, flags=EstimateBuffer.REQUIRED_FLAGS, x=0.0):
    """Position, attitude and status reports stamped in PX4 lockstep time."""
    return (Message(kind="LOCAL_POSITION_NED", time_boot_ms=stamp_ms, x=x, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0),
            Message(kind="ATTITUDE_QUATERNION", time_boot_ms=stamp_ms, q1=1.0, q2=0.0, q3=0.0, q4=0.0,
                    rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0),
            Message(kind="ESTIMATOR_STATUS", time_usec=stamp_ms * 1000, flags=flags))


def bridge_with_vehicle():
    """A backend bound to a one-vehicle stub with a four-rotor [0, 1400] rad/s range."""
    bridge = object.__new__(PX4Backend)
    bridge.config = {"motor_order": [0, 1, 2, 3]}
    bridge.buffer = EstimateBuffer()
    bridge._motor_order = [2, 0, 3, 1]
    bridge._rotor_data = ThrusterControl(4, [0.0] * 4, [1400.0] * 4, [0.0] * 4)
    bridge._vehicle = SimpleNamespace(device="cpu")
    bridge.mavutil = SimpleNamespace(mavlink=SimpleNamespace(MAV_MODE_FLAG_SAFETY_ARMED=128))
    bridge._received_first_actuator = True
    bridge._actuator_usec = -1
    bridge._current_utime = 20000
    bridge.config["lockstep_timeout_s"] = 1.0
    return bridge


class PX4EstimateContract(unittest.TestCase):
    def test_ned_frd_state_converts_to_enu_flu(self):
        state = convert_state((2, 3, -4), (5, 6, -7), (1, 0, 0, 0), (8, 9, 10), (1, 2, 3))
        torch.testing.assert_close(state[0, :3], torch.tensor([4.0, 4.0, 7.0]))
        torch.testing.assert_close(state[0, 3:6], torch.tensor([6.0, 5.0, 7.0]))
        torch.testing.assert_close(state[0, 10:13], torch.tensor([8.0, -9.0, -10.0]))
        torch.testing.assert_close(state[0, 6:10].abs(), torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5]))

    def test_estimate_is_selected_at_simulated_time_and_validated(self):
        buf = EstimateBuffer()
        for stamp in (10, 20, 30):
            for message in estimate_messages(stamp, x=float(stamp)):
                buf.ingest(message)
        self.assertTrue(buf.ready_for(0.030))
        self.assertFalse(buf.ready_for(0.040))
        state, stamp = buf.snapshot(0.020)               # one step behind the newest sample
        self.assertEqual(stamp, 0.020)
        self.assertEqual(float(state[0, 1]), 20.0)        # north -> ENU y
        with self.assertRaisesRegex(EstimateUnavailable, "missing"):
            buf.snapshot(0.050)                           # newest earlier sample is 20 ms old
        with self.assertRaisesRegex(EstimateUnavailable, "not arrived"):
            buf.snapshot(0.005)
        for message in estimate_messages(40, flags=EstimateBuffer.REQUIRED_FLAGS | EstimateBuffer.INVALID_FLAGS):
            buf.ingest(message)
        with self.assertRaisesRegex(EstimateUnavailable, "validity"):
            buf.snapshot(0.040)
        with self.assertRaisesRegex(EstimateUnavailable, "backwards"):
            buf.ingest(estimate_messages(5)[0])

    def test_lockstep_waits_for_the_controls_of_the_last_sample(self):
        bridge = bridge_with_vehicle()
        old = SimpleNamespace(time_usec=10000, mode=129, flags=1, controls=[0.0] * 16)
        new = SimpleNamespace(time_usec=20000, mode=129, flags=1, controls=[0.1, 0.2, 0.3, 0.4] + [0.0] * 12)
        bridge._connection = SimpleNamespace(recv_match=Mock(side_effect=[old, new]))
        bridge._receive_actuators()
        self.assertEqual(bridge._connection.recv_match.call_count, 2)
        self.assertEqual(bridge._actuator_usec, 20000)
        self.assertTrue(bridge.armed)
        # rotor i <- PX4 output motor_order[i]; omega = 1400 * u
        torch.testing.assert_close(torch.tensor(bridge._rotor_data.input_reference),
                                   torch.tensor([420.0, 140.0, 560.0, 280.0]))
        bridge._receive_actuators()                      # already synchronized: no new wait
        self.assertEqual(bridge._connection.recv_match.call_count, 2)

        disarmed = SimpleNamespace(time_usec=30000, mode=1, flags=1, controls=[0.5] * 16)
        bridge._current_utime = 30000
        bridge._connection = SimpleNamespace(recv_match=Mock(return_value=disarmed))
        bridge._receive_actuators()
        self.assertFalse(bridge.armed)
        self.assertEqual(bridge._rotor_data.input_reference, [0.0] * 4)

    def test_lockstep_timeout_is_an_error(self):
        bridge = bridge_with_vehicle()
        bridge._connection = SimpleNamespace(recv_match=Mock(return_value=None))
        with self.assertRaisesRegex(EstimateUnavailable, "lockstep"):
            bridge._receive_actuators()

    def test_step_uses_estimate_one_step_behind(self):
        bridge = bridge_with_vehicle()
        bridge.ready = True
        bridge._step_dt = 0.01
        bridge._receive_actuators = Mock()
        bridge.telemetry = None
        for stamp in (10, 20):
            for message in estimate_messages(stamp, x=float(stamp)):
                bridge.buffer.ingest(message)
        bridge.step(0.01)
        self.assertEqual(bridge.current_stamp, 0.010)
        self.assertAlmostEqual(bridge.latency_s, 0.010)
        self.assertEqual(float(bridge.current[0, 1]), 10.0)

    def test_wind_estimate_is_enu_timed_and_optional(self):
        """WIND_COV (north, east) becomes ENU [east, north, 0]; no recent report means no estimate, never zero."""
        bridge = bridge_with_vehicle()
        bridge.ready = True
        bridge._step_dt = 0.01
        bridge._receive_actuators = Mock()
        bridge.telemetry = None
        for stamp in (10, 20):
            for message in estimate_messages(stamp):
                bridge.buffer.ingest(message)
        bridge.step(0.01)
        self.assertIsNone(bridge.current_wind)
        bridge.buffer.ingest(Message(kind="WIND_COV", time_usec=8_000, wind_x=3.0, wind_y=-4.0, var_horiz=0.1))
        bridge.buffer.ingest(Message(kind="WIND_COV", time_usec=15_000, wind_x=9.0, wind_y=9.0, var_horiz=0.1))
        bridge.step(0.01)                       # target 10 ms: the report at 15 ms is still in the future
        torch.testing.assert_close(bridge.current_wind, torch.tensor([[-4.0, 3.0, 0.0]]))
        self.assertAlmostEqual(bridge.wind_stamp, 0.008)
        bridge.buffer.wind_max_age = 0.001
        bridge.step(0.01)
        self.assertIsNone(bridge.current_wind)
        bridge.buffer.ingest(Message(kind="WIND_COV", time_usec=16_000, wind_x=float("nan"), wind_y=0.0, var_horiz=0.1))
        self.assertIsNone(bridge.buffer.wind_enu(0.02))

    def test_offboard_mode_uses_custom_mode_bits(self):
        """pymavlink can label a valid PX4 OFFBOARD heartbeat UNKNOWN."""
        backend = bridge_with_vehicle()
        heartbeat = Message(kind="HEARTBEAT", base_mode=128, custom_mode=6 << 16)
        backend.telemetry = Mock()
        backend.telemetry.recv_match.side_effect = [heartbeat, None]
        backend.mavutil.mode_string_v10 = lambda message: "UNKNOWN"
        backend.received = {}
        backend.command_acks = {}
        backend._poll()
        self.assertTrue(backend.offboard)
        self.assertEqual(backend.heartbeat_mode, ("UNKNOWN", 6 << 16))

    def test_sensor_and_gps_packets_use_px4_units(self):
        packets = []
        bridge = object.__new__(PX4Backend)
        bridge._sensor_data = _runtime["SensorMsg"]()
        bridge._connection = SimpleNamespace(mav=SimpleNamespace(
            hil_sensor_send=lambda *args: packets.append(("sensor", args)),
            hil_gps_send=lambda *args: packets.append(("gps", args))))
        bridge._received_first_hearbeat = True
        bridge._current_utime = 0
        bridge._last_request_usec = 0
        bridge.log_sensors = False
        readings = {
            "IMU": {"linear_acceleration": torch.tensor([0.0, 0.0, -9.81]), "angular_velocity": torch.zeros(3)},
            "Magnetometer": {"magnetic_field": torch.tensor([0.2, 0.0, 0.4])},
            "Barometer": {"absolute_pressure": torch.tensor(1002.0), "pressure_altitude": torch.tensor(94.0),
                          "temperature": torch.tensor(14.4)},
            "GPS": {"fix_type": 3, "latitude": torch.tensor(38.0), "longitude": torch.tensor(-9.0),
                    "altitude": torch.tensor(94.0), "eph": 1.0, "epv": 1.0, "speed": 5.0,
                    "velocity_north": 3.0, "velocity_east": 4.0, "velocity_down": -7.0, "cog": 53.13,
                    "sattelites_visible": 10, "latitude_gt": 38.0, "longitude_gt": -9.0, "altitude_gt": 94.0},
        }
        bridge.update_sensors([{"state": SimpleNamespace(), "readings": readings}], 0.01)
        sensor, gps = packets[0][1], packets[1][1]
        self.assertEqual(sensor[0], 10000)
        self.assertAlmostEqual(float(sensor[13]), 14.4, places=4)    # temperature, not altitude
        self.assertEqual(sensor[14], 7 | 56 | 448 | 6656)
        self.assertEqual(gps[0], 10000)
        self.assertEqual(gps[2:5], (380000000, -90000000, 94000))
        self.assertEqual(gps[5:7], (100, 100))                      # eph/epv in cm
        self.assertEqual(gps[7:11], (500, 300, 400, -700))           # ground speed and NED cm/s
        self.assertEqual(gps[11], 5313)                              # course over ground [cdeg]

    def test_sensors_wait_for_px4_heartbeat(self):
        bridge = object.__new__(PX4Backend)
        bridge._connection = SimpleNamespace(mav=Mock())
        bridge._received_first_hearbeat = False
        bridge._current_utime = 0
        bridge.update_sensors([{"state": None, "readings": {}}], 0.01)
        self.assertEqual(bridge._current_utime, 0)
        bridge._connection.mav.hil_sensor_send.assert_not_called()

    def test_body_rate_setpoint_and_motor_outputs(self):
        calls = []
        bridge = bridge_with_vehicle()
        bridge.config["body_rate_scale_rad_s"] = [2.0, 3.0, 4.0]
        bridge.telemetry = SimpleNamespace(mav=SimpleNamespace(
            set_attitude_target_send=lambda *args: calls.append(args)))
        bridge._current_utime = 50000
        bridge.send_body_rate(torch.tensor([[0.0, 0.5, -0.5, 1.0]]))
        self.assertEqual(calls[0][0:4], (50, 1, 1, 128))
        self.assertEqual(calls[0][5:8], (1.0, 1.5, -4.0))
        self.assertEqual(calls[0][8], 0.5)

        bridge.offboard = bridge.armed = True
        bridge._actuator_usec = 50000
        bridge._rotor_data.update_input_reference([0.1, 0.2, 0.3, 0.4])
        torch.testing.assert_close(bridge.motor_velocities(), torch.tensor([[140.0, 280.0, 420.0, 560.0]]))
        bridge._actuator_usec = 40000
        with self.assertRaisesRegex(EstimateUnavailable, "missing"):
            bridge.motor_velocities()


if __name__ == "__main__":
    unittest.main()
