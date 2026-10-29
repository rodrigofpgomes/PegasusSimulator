"""
| File: px4_mavlink_estimator_backend.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: PX4 SITL link for one vehicle of a VehicleBatch, built on PX4MavlinkBackend.
|
| What is reused from PX4MavlinkBackend (the original Pegasus PX4 backend):
|   - start()/stop(): HIL connection (tcpin, port 4560 + vehicle_id) and PX4LaunchTool
|     (px4_autolaunch launches a fresh PX4 with PX4_SIM_MODEL, killed at stop)
|   - update_sensor() and SensorMsg: IMU, magnetometer, barometer and GPS encoding
|   - send_sensor_msgs()/send_gps_msgs(): HIL_SENSOR and HIL_GPS
|   - handle_control()/ThrusterControl: HIL_ACTUATOR_CONTROLS -> rotor speeds
|   - lockstep: after each HIL_SENSOR, wait for the HIL_ACTUATOR_CONTROLS PX4
|     computed with it (PX4 sends exactly one per sample, also disarmed)
|
| What is added for evaluation with a batch:
|   - sensors arrive through update_sensors() from the VehicleBatch post-physics
|     callback (SensorSuiteBatch), so a sample is the state after physics step k
|   - a second MAVLink link (UDP 14540 + vehicle_id) reads the EKF2 estimate and
|     sends OFFBOARD setpoints; EstimateBuffer converts NED/FRD to ENU/FLU
|   - timing is simulated time: PX4's clock is the HIL_SENSOR time, so the
|     estimate used at a decision is the one at (sample time - estimate_delay_s),
|     one physics step by default, the earliest PX4 streams it. No wall-clock pacing.
|
| Modes: "estimator" (PX4 only estimates; the first backend keeps actuation),
| "body_rate" (PX4 rate loop driven by a body-rate policy) and, in the subclass
| PX4PositionBackend, "position" (PX4 position controller).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from pegasus.simulator.logic.backends.px4_estimate import EstimateBuffer, EstimateUnavailable
from pegasus.simulator.logic.backends.px4_mavlink_backend import (PX4MavlinkBackend, PX4MavlinkBackendConfig,
                                                                  ThrusterControl)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class PX4MavlinkEstimatorBackend(PX4MavlinkBackend):
    """Connect one PX4 SITL instance to a batch of one vehicle."""

    def __init__(self, config: dict):
        """Configure the inherited PX4 link with lockstep and the evaluation options.

        Config keys (all optional): vehicle_id, px4_autolaunch, px4_dir,
        px4_vehicle_model, telemetry_baseport, system_id, component_id,
        motor_order, estimate_delay_s, warmup_timeout_s, warmup_stable_s,
        activation_timeout_s, lockstep_timeout_s, log_sensors, arm_after_warmup (estimator mode: force-arm
        PX4 after the warmup so that EKF2 fuses airspeed and sideslip in flight) and the sensor
        settings read by SensorSuiteBatch (imu, gps, barometer, magnetometer).
        """
        vehicle_id = int(config.get("vehicle_id", 0))
        n_rotors = int(config.get("num_rotors", 4))
        super().__init__(PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "connection_type": "tcpin",
            "connection_ip": config.get("connection_ip", "localhost"),
            "connection_baseport": int(config.get("connection_baseport", 4560)),
            "px4_autolaunch": bool(config.get("px4_autolaunch", True)),
            "px4_dir": os.path.expanduser(config.get("px4_dir", PegasusInterface().px4_path)),
            "px4_vehicle_model": config.get("px4_vehicle_model", "none_shuttle"),
            "enable_lockstep": True,
            "num_rotors": n_rotors,
            # Placeholders sized to num_rotors; initialize() replaces the mapping with the vehicle's rotor limits
            "input_offset": [0.0] * n_rotors,
            "input_scaling": [1.0] * n_rotors,
            "zero_position_armed": [0.0] * n_rotors,
        }))
        self.config = dict(config)
        self.mode = config.get("mode", "estimator")
        if self.mode not in ("estimator", "body_rate"):
            raise ValueError("PX4 mode must be 'estimator' or 'body_rate'")
        self._telemetry_port = (f"udpin:{config.get('telemetry_ip', '127.0.0.1')}:"
                                f"{int(config.get('telemetry_baseport', 14540)) + vehicle_id}")
        self.buffer = EstimateBuffer(
            system_id=int(config.get("system_id", vehicle_id + 1)),
            component_id=int(config.get("component_id", 1)),
        )
        self._motor_order = list(config.get("motor_order", range(self.config_num_rotors)))
        self.telemetry = None
        self.log_path = None
        self.log = None
        self.log_sensors = bool(config.get("log_sensors", False))
        self.sensor_origin = None
        self._reset_link_state()

    @property
    def config_num_rotors(self) -> int:
        """Number of PX4 motor outputs mapped onto vehicle rotors."""
        return int(self.config.get("num_rotors", 4)) if isinstance(self.config, dict) else 4

    def _reset_link_state(self) -> None:
        """Clear estimate, timing and PX4 status for a new SITL process."""
        self.buffer.messages = {kind: [] for kind in EstimateBuffer.KINDS}
        self.current = None
        self.current_stamp = None
        self.current_wind = None
        self.wind_stamp = None
        self.ready = False
        self.state = None
        self.sensor_samples = {}
        self.armed = False
        self.offboard = False
        self.heartbeat_mode = None
        self.command_acks = {}
        self.status_text = None
        self.received = {}
        self._actuator_usec = -1
        self._last_request_usec = -10**12
        self._last_mode_request_usec = -10**12
        self._current_utime = 0

    # =============================================================================
    # Backend lifecycle
    # =============================================================================

    def initialize(self, vehicle) -> None:
        """Bind one vehicle and scale PX4 motor outputs in [0, 1] to its rotor speeds.

        omega_i = min_i + u_i (max_i - min_i), i.e. ThrusterControl with offset 0,
        scaling (max - min) and zero_position_armed = min. With the airframe's
        THR_MDL_FAC = 1, PX4 accounts for thrust being quadratic in omega.
        """
        if vehicle.n_vehicles != 1:
            raise ValueError("A PX4 SITL instance controls one vehicle; use one batch per PX4")
        super().initialize(vehicle)
        self.sensor_origin = vehicle._sensors[0].sensor_origin if vehicle._sensors else None
        n = self.config_num_rotors
        low = vehicle._thrusters.min_rotor_velocity.reshape(-1)[:n].tolist()
        high = vehicle._thrusters.max_rotor_velocity.reshape(-1)[:n].tolist()
        self._rotor_data = ThrusterControl(n, [0.0] * n, [h - l for h, l in zip(high, low)], low)

    def start(self) -> None:
        """Open the HIL link and launch PX4 (inherited), then open the telemetry link."""
        from pymavlink import mavutil

        self.mavutil = mavutil
        if self._is_running:
            return
        self._reset_link_state()
        super().start()
        self.telemetry = mavutil.mavlink_connection(self._telemetry_port, source_system=245)

    def stop(self) -> None:
        """Close both links and the record file; the inherited stop kills an autolaunched PX4."""
        if self.telemetry is not None:
            self.telemetry.close()
            self.telemetry = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.ready = False
        if self._is_running:
            super().stop()

    def close(self) -> None:
        """Alias of stop() used by the evaluation scripts."""
        self.stop()

    def update(self, dt: float) -> None:
        """The command source (first backend) exchanges with PX4 before its decision."""
        return

    def update_state(self, state) -> None:
        """Physical state reaches PX4 only through the simulated sensors."""
        return

    def reset(self) -> None:
        """A teleport invalidates the EKF: restart PX4 and the evaluation instead."""
        if self.ready:
            raise RuntimeError("Restart PX4 SITL and evaluation after a physical teleport")

    def __del__(self):
        """stop() owns cleanup; the PX4LaunchTool kills its process when collected."""
        return

    # =============================================================================
    # Sensors -> PX4 (post-physics)
    # =============================================================================

    def update_sensors(self, samples: list, dt: float) -> None:
        """Encode the sample of the state after this physics step and send it to PX4.

        This is the send half of PX4MavlinkBackend.update(): sensor routing,
        1 Hz heartbeat, HIL_SENSOR and HIL_GPS at time += dt.
        """
        if self._connection is None or not self._received_first_hearbeat:
            return
        sample = samples[0]
        self.state = sample["state"]
        for sensor_type, data in sample["readings"].items():
            self.update_sensor(sensor_type, data)

        self._current_utime += int(round(dt * 1e6))
        if self._current_utime - self._last_request_usec >= 1_000_000:
            self._last_request_usec = self._current_utime
            self.send_heartbeat()
            self.telemetry.mav.heartbeat_send(6, 8, 0, 0, 0)
            # LOCAL_POSITION_NED and ATTITUDE_QUATERNION at 100 Hz, ESTIMATOR_STATUS at 10 Hz and WIND_COV
            # (only streamed while the EKF wind estimate is valid) at 50 Hz
            for message_id, hz in ((31, 100), (32, 100), (230, 10), (231, 50)):
                self.telemetry.mav.command_long_send(self.buffer.system_id, self.buffer.component_id,
                                                     511, 0, message_id, 1e6 / hz, 0, 0, 0, 0, 0)

        if self.log_sensors:
            self._log_sensor_packets(sample["readings"])
        self.send_sensor_msgs(self._current_utime)
        self.send_gps_msgs(self._current_utime)

    def _log_sensor_packets(self, readings: dict) -> None:
        """Keep the transmitted values next to the physical state for diagnosis."""
        d, s = self._sensor_data, self.state
        truth = {"truth_position_enu_m": s.position.tolist(), "truth_velocity_enu_m_s": s.linear_velocity.tolist(),
                 "truth_attitude_enu_flu": s.attitude.tolist(), "truth_rates_flu_rad_s": s.angular_velocity.tolist()}
        if "IMU" in readings:
            self.sensor_samples["hil_sensor"] = {
                "time_usec": self._current_utime,
                "acceleration_frd_m_s2": [float(d.xacc), float(d.yacc), float(d.zacc)],
                "angular_velocity_frd_rad_s": [float(d.xgyro), float(d.ygyro), float(d.zgyro)],
                "magnetic_field_frd_gauss": [float(d.xmag), float(d.ymag), float(d.zmag)],
                "absolute_pressure_hpa": float(d.abs_pressure), "temperature_c": float(d.temperature), **truth}
        if "GPS" in readings:
            self.sensor_samples["hil_gps"] = {
                "time_usec": self._current_utime, "latitude_deg_e7": d.latitude_deg,
                "longitude_deg_e7": d.longitude_deg, "altitude_m_mm": d.altitude,
                "velocity_ned_cm_s": [d.velocity_north, d.velocity_east, d.velocity_down], **truth}

    # =============================================================================
    # PX4 -> simulator (lockstep) and telemetry
    # =============================================================================

    def _wait_for_first_heartbeat(self, timeout: float) -> None:
        """Wait (wall time) until the launched PX4 connects to the HIL link."""
        start = time.monotonic()
        while not self._received_first_hearbeat:
            self.wait_for_first_hearbeat()
            if time.monotonic() - start > timeout:
                raise EstimateUnavailable(f"No PX4 heartbeat on {self._connection_port} after {timeout:.0f} s")
            time.sleep(0.05)

    def _receive_actuators(self) -> None:
        """Lockstep: wait for the HIL_ACTUATOR_CONTROLS of the last HIL_SENSOR.

        As PX4MavlinkBackend.poll_mavlink_messages, the wait only blocks once
        PX4 has produced its first output. motor_order maps PX4 output
        motor_order[i] to vehicle rotor i before handle_control().
        """
        timeout = float(self.config.get("lockstep_timeout_s", 5.0))
        deadline = time.monotonic() + timeout
        while self._actuator_usec < self._current_utime:
            wait = self._received_first_actuator
            msg = self._connection.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=wait,
                                              timeout=max(deadline - time.monotonic(), 0.0) if wait else None)
            if msg is None:
                if wait:
                    raise EstimateUnavailable(f"PX4 lockstep: no HIL_ACTUATOR_CONTROLS for t={self._current_utime} us "
                                              f"within {timeout} s")
                return
            self._received_first_actuator = True
            self._actuator_usec = int(msg.time_usec)
            self.armed = bool(int(msg.mode) & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            controls = list(msg.controls)
            controls[:len(self._motor_order)] = [msg.controls[j] for j in self._motor_order]
            self.handle_control(msg.time_usec, controls, msg.mode, msg.flags)

    def _poll(self) -> None:
        """Read the telemetry link: estimator reports, mode, arming and command replies."""
        if self.telemetry is None:
            return
        for _ in range(512):
            message = self.telemetry.recv_match(blocking=False)
            if message is None:
                break
            kind = message.get_type()
            self.received[kind] = self.received.get(kind, 0) + 1
            if message.get_srcSystem() != self.buffer.system_id:
                continue
            self.buffer.ingest(message)
            if kind == "HEARTBEAT" and message.get_srcComponent() == self.buffer.component_id:
                self.heartbeat_mode = (self.mavutil.mode_string_v10(message), int(message.custom_mode))
                # PX4 main mode is custom_mode bits 16..23 (6 = OFFBOARD); pymavlink may print UNKNOWN
                self.offboard = ((int(message.custom_mode) >> 16) & 0xFF) == 6
            elif kind == "COMMAND_ACK":
                self.command_acks[int(message.command)] = int(message.result)
            elif kind == "STATUSTEXT":
                self.status_text = message.text

    def _estimate_target(self) -> float:
        """Simulated time of the estimate used now: last sample time minus estimate_delay_s."""
        return self._current_utime / 1e6 - float(self.config.get("estimate_delay_s", self._step_dt))

    def _exchange(self, require_estimate: bool) -> None:
        """Receive this sample's actuators, then the estimate at the target time."""
        self._receive_actuators()
        target = self._estimate_target()
        deadline = time.monotonic() + float(self.config.get("lockstep_timeout_s", 5.0))
        while True:
            self._poll()
            if self.buffer.ready_for(target):
                break
            if not require_estimate:
                return
            if time.monotonic() > deadline:
                raise EstimateUnavailable(f"PX4 estimate for t={target:.3f} s did not arrive")
            time.sleep(0.0005)
        self._wait_for_wind(target)
        try:
            state, stamp = self.buffer.snapshot(target)
        except EstimateUnavailable:
            if require_estimate:
                raise
            return
        self.current = state.to(self.vehicle.device)
        self.current_stamp = stamp
        wind = self.buffer.wind_enu(target)
        self.current_wind, self.wind_stamp = (None, None) if wind is None else (
            wind[0].unsqueeze(0).to(self.vehicle.device), wind[1])

    def _wait_for_wind(self, target: float) -> None:
        """Give an active WIND_COV stream the chance to cover ``target`` before the estimate is chosen.

        WIND_COV is streamed at 50 Hz apart from position and attitude, so it can arrive after them. While
        reports are arriving (the newest is at most 0.1 s old) the wait is for one at or after target - 20 ms,
        bounded by 50 ms of wall time because the stream stops whenever the EKF wind estimate is invalid.
        """
        last = self.buffer.latest("WIND_COV")
        if last is None or not target - 0.1 <= last[0] < target - 0.02:
            return
        deadline = time.monotonic() + 0.05
        while last[0] < target - 0.02 and time.monotonic() < deadline:
            self._poll()
            last = self.buffer.latest("WIND_COV")
            time.sleep(0.0005)

    # =============================================================================
    # Evaluation API (called by SkrlAgentBackend / play_multi)
    # =============================================================================

    def connect(self, log_path: Path) -> None:
        """Wait for PX4, restart the sensor histories at the evaluation pose, set the record file."""
        self.log_path = Path(log_path)
        self._wait_for_first_heartbeat(float(self.config.get("connect_timeout_s", 60.0)))
        for suite in self.vehicle._sensors:
            suite.reset()

    def warmup(self, dt: float, render) -> None:
        """Feed stationary sensors with physics stopped until the EKF is valid.

        Runs in lockstep, so it lasts warmup_stable_s of simulated time after the
        first valid estimate (bounded by warmup_timeout_s, simulated). Then the
        PX4 local origin is translated once to the simulated spawn position.
        """
        self._step_dt = float(dt)
        timeout = float(self.config.get("warmup_timeout_s", 60.0))
        stable = float(self.config.get("warmup_stable_s", 2.0))
        good_since = None
        error = "no PX4 telemetry"
        start = self._current_utime / 1e6
        while self._current_utime / 1e6 - start < timeout:
            render()  # World.render does not advance physics
            self.vehicle.update_sensors(dt)
            self._exchange(require_estimate=False)
            now = self._current_utime / 1e6
            try:
                self.buffer.snapshot(self._estimate_target())
                good_since = now if good_since is None else good_since
                if now - good_since >= stable and self.config.get("arm_after_warmup", False) and not self.armed:
                    self.request_force_arm()
                elif now - good_since >= stable:
                    self.buffer.align_position_origin(self.vehicle.state.position[0], self._estimate_target())
                    state, self.current_stamp = self.buffer.snapshot(self._estimate_target())
                    self.current = state.to(self.vehicle.device)
                    self.ready = True
                    return
            except EstimateUnavailable as exc:
                good_since, error = None, str(exc)
        raise EstimateUnavailable(f"PX4 EKF warmup timed out after {timeout} s simulated: {error}; "
                                  f"received={self.received}")

    def step(self, dt: float, sample_sensors: bool = False) -> None:
        """Lockstep exchange before a decision; the estimate is at t - estimate_delay_s.

        Running physics samples sensors only in the post-physics callback, so
        this never creates a second sample for the same physics step.
        """
        if not self.ready:
            raise EstimateUnavailable("PX4 EKF warmup has not completed")
        if sample_sensors:
            self.vehicle.update_sensors(dt)
        self._exchange(require_estimate=True)

    @property
    def latency_s(self) -> float | None:
        """Age of the estimate in use, in simulated time."""
        return None if self.current_stamp is None else self._current_utime / 1e6 - self.current_stamp

    def activate_body_rate(self, dt: float, render) -> None:
        """Stream neutral rate setpoints, then request OFFBOARD and arm (simulated time)."""
        hover = float(self.config.get("initial_thrust", 0.5))
        neutral = torch.tensor([[2.0 * hover - 1.0, 0.0, 0.0, 0.0]])
        self._activate(dt, render, lambda: self.send_body_rate(neutral))

    def _activate(self, dt: float, render, send_setpoint) -> None:
        """Common OFFBOARD activation: setpoints for 1 s, then mode and arm requests."""
        timeout = float(self.config.get("activation_timeout_s", 15.0))
        start = self._current_utime / 1e6
        while self._current_utime / 1e6 - start < timeout:
            render()
            self.step(dt, sample_sensors=True)
            send_setpoint()
            if self._current_utime / 1e6 - start >= 1.0:
                self.request_offboard_and_arm()
            if self.offboard and self.armed:
                return
        raise EstimateUnavailable(
            f"PX4 activation failed: armed={self.armed}, offboard={self.offboard}, acks={self.command_acks}, "
            f"heartbeat={self.heartbeat_mode}, status={self.status_text}")

    def send_body_rate(self, actions: torch.Tensor) -> None:
        """Send [thrust, roll, pitch, yaw] actions in [-1, 1] through PX4's rate loop."""
        action = actions[0].detach().cpu().clamp(-1.0, 1.0)
        scale = torch.as_tensor(self.config["body_rate_scale_rad_s"], dtype=torch.float32)
        thrust = float((action[0] + 1.0) / 2.0)
        rates_frd = action[1:4] * scale * torch.tensor([1.0, -1.0, -1.0])
        self.telemetry.mav.set_attitude_target_send(
            self._current_utime // 1000, self.buffer.system_id, self.buffer.component_id,
            128, [1.0, 0.0, 0.0, 0.0], *[float(value) for value in rates_frd], thrust,
        )

    def request_offboard_and_arm(self) -> None:
        """Request OFFBOARD (176) and arming (400) at most once per simulated second."""
        if self._current_utime - self._last_mode_request_usec < 1_000_000:
            return
        self._last_mode_request_usec = self._current_utime
        if not self.offboard:
            self.telemetry.mav.command_long_send(self.buffer.system_id, self.buffer.component_id,
                                                 176, 0, 1, 6, 0, 0, 0, 0, 0)
        if not self.armed:
            self.telemetry.mav.command_long_send(self.buffer.system_id, self.buffer.component_id,
                                                 400, 0, 1, 0, 0, 0, 0, 0, 0)

    def request_force_arm(self) -> None:
        """Estimator mode with arm_after_warmup: force-arm (400, param2 21196 skips the preflight checks) at most once
        per simulated second. PX4 outputs stay unused; arming only lets the land detector report flight, which EKF2
        needs to fuse airspeed and sideslip."""
        if self._current_utime - self._last_mode_request_usec < 1_000_000:
            return
        self._last_mode_request_usec = self._current_utime
        self.telemetry.mav.command_long_send(self.buffer.system_id, self.buffer.component_id,
                                             400, 0, 1, 21196, 0, 0, 0, 0, 0)

    def motor_velocities(self) -> torch.Tensor:
        """Rotor speeds [rad/s] from the PX4 outputs of the current lockstep sample."""
        if not (self.offboard and self.armed):
            raise EstimateUnavailable("PX4 is not armed in OFFBOARD mode")
        if self._actuator_usec < self._current_utime:
            raise EstimateUnavailable("PX4 motor outputs of the current sample are missing")
        return torch.tensor(self._rotor_data.input_reference, dtype=torch.float32,
                            device=self.vehicle.device).unsqueeze(0)

    def record(self, observation: torch.Tensor | None) -> None:
        """Append estimate, physical state, timing and (optionally) sensors to the record.

        The true state is diagnostic only; it never reaches PX4 or the policy.
        """
        if self.log_path is None:
            return
        physical = self.vehicle.state
        truth = torch.cat((physical.position, physical.linear_velocity,
                           physical.attitude, physical.angular_velocity), dim=-1)
        if self.log is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = self.log_path.open("x", encoding="utf-8")
            metadata = {"mode": self.mode, "position_alignment_enu": self.buffer.offset.tolist(),
                        "estimate_delay_s": float(self.config.get("estimate_delay_s", self._step_dt)),
                        "config": self.config, "sensor_origin_lat_lon_alt": self.sensor_origin}
            self.log.write(json.dumps({"metadata": metadata}, default=str) + "\n")

        sample = {
            "sensor_time_usec": self._current_utime,
            "estimate_time_s": self.current_stamp,
            "estimate_latency_s": self.latency_s,
            "actuator_time_usec": self._actuator_usec,
            "estimated_state": self.current.detach().cpu().tolist(),
            "estimated_wind_enu": None if self.current_wind is None else self.current_wind.detach().cpu().tolist(),
            "policy_observation": None if observation is None else observation.detach().cpu().tolist(),
            "true_state": truth.detach().cpu().tolist(),
            "px4_armed": self.armed,
            "px4_offboard": self.offboard,
            "px4_rotor_reference_rad_s": list(map(float, self._rotor_data.input_reference)),
        }
        if self.log_sensors:
            sample["sensors"] = self.sensor_samples
        self.log.write(json.dumps(sample) + "\n")
        self.log.flush()

    _step_dt = 0.01
