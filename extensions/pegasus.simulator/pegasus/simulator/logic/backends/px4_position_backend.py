"""
| File: px4_position_backend.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: PX4 position control for one Shuttle vehicle batch.
|
| Pegasus supplies the physical vehicle and sensors. PX4 estimates its state,
| tracks the shared evaluation reference in OFFBOARD and returns its motor
| outputs, which this backend applies as rotor speeds (first and only command
| source of its batch). Each physics step runs in lockstep with PX4: the motor
| outputs applied in step k+1 are those PX4 computed from the sensor sample of
| step k, the same one-step delay as the original PX4MavlinkBackend.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pegasus.simulator.logic.backends.px4_estimate import EstimateUnavailable
from pegasus.simulator.logic.backends.px4_mavlink_estimator_backend import PX4MavlinkEstimatorBackend
from pegasus.simulator.logic.sensors.sensor_suite_batch import SensorSuiteBatch

class PX4PositionBackend(PX4MavlinkEstimatorBackend):
    """Connect a single PX4 position controller to a single Shuttle batch."""

    # SET_POSITION_TARGET_LOCAL_NED type_mask bits: 1024 ignore yaw, 2048 ignore yaw rate;
    # 56 ignore velocity, 448 ignore acceleration.
    IGNORE_YAW = 1024 | 2048
    IGNORE_VELOCITY_ACCELERATION = 56 | 448

    def __init__(self, config: dict):
        """Keep the PX4 link and a vehicle-owned Pegasus sensor suite.

        feedforward (default true) sends the reference velocity and acceleration
        with the position, as the policies and the LQR receive them.
        """
        super().__init__({**config, "mode": "estimator"})
        self.config["mode"] = "position"
        self.mode = "position"
        self._px4_config = self.config
        self._px4_sensors = SensorSuiteBatch(1, self.config)
        self.reset_manager = None
        self._prepared = False
        self._input_reference = None

    # =============================================================================
    # Vehicle and evaluation lifecycle
    # =============================================================================

    def start(self) -> None:
        """Set the actuator mode, allocate one rotor command row and start PX4."""
        self.vehicle.set_input_mode("rotor_velocity_direct")
        self._input_reference = torch.zeros((1, self.config_num_rotors), dtype=torch.float32,
                                            device=self.vehicle.device)
        super().start()

    def setup(self, reset_manager) -> None:
        """Use the same goal buffer as the other evaluation controllers."""
        self.reset_manager = reset_manager

    def prepare_px4(self, dt: float, render, log_path: Path) -> None:
        """Connect, initialize EKF2 and activate PX4 position control."""
        try:
            self.connect(log_path)
            self.warmup(dt, render)
            self._activate(dt, render, self._send_goal)
            self._prepared = True
        except Exception:
            self.close()
            raise

    def stop(self) -> None:
        """Release PX4 and prevent subsequent actuator updates."""
        self._prepared = False
        super().stop()

    def update(self, dt: float) -> None:
        """Lockstep exchange, send the current goal, apply PX4's motor outputs."""
        if not self._prepared:
            return
        self.step(dt)
        self._send_goal()
        self.request_offboard_and_arm()
        self._input_reference.copy_(self.motor_velocities())
        self.record(None)

    def input_reference(self) -> torch.Tensor:
        """Return PX4's rotor commands to MultirotorBatch."""
        return self._input_reference

    # =============================================================================
    # PX4 position target
    # =============================================================================

    def _send_goal(self) -> None:
        """Send the shared reference (position, and velocity/acceleration if feedforward)."""
        rm = self.reset_manager
        if self.config.get("feedforward", True):
            self.send_position(rm.goal_pos[0], rm.goal_vel[0], rm.goal_acc[0])
        else:
            self.send_position(rm.goal_pos[0])

    def send_position(self, goal_enu: torch.Tensor, velocity_enu: torch.Tensor | None = None,
                      acceleration_enu: torch.Tensor | None = None) -> None:
        """Send an ENU world reference as a PX4 local NED target.

        The warmup offset maps PX4 local position to the Pegasus world. Yaw and
        yaw rate are omitted, so PX4 keeps its current heading.
        """
        goal = goal_enu.detach().cpu().float() - self.buffer.offset
        enu_to_ned = lambda v: (float(v[1]), float(v[0]), -float(v[2]))
        mask = self.IGNORE_YAW
        vel = acc = (0.0, 0.0, 0.0)
        if velocity_enu is None:
            mask |= self.IGNORE_VELOCITY_ACCELERATION
        else:
            vel = enu_to_ned(velocity_enu.detach().cpu())
            acc = enu_to_ned(acceleration_enu.detach().cpu())
        self.telemetry.mav.set_position_target_local_ned_send(
            self._current_utime // 1000, self.buffer.system_id, self.buffer.component_id,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
            *enu_to_ned(goal), *vel, *acc, 0.0, 0.0,
        )

    def activate_position(self, dt: float, render) -> None:
        """Stream the initial goal before requesting OFFBOARD and arming."""
        self._activate(dt, render, self._send_goal)


__all__ = ["PX4PositionBackend", "EstimateUnavailable"]
