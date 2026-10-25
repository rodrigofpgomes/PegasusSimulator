"""
| File: px4_estimate.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Convert PX4 EKF telemetry into the Pegasus 13-state convention.
|
| PX4 reports local position and velocity in NED, attitude from FRD to NED,
| and body rates in FRD. Pegasus tasks consume ENU position/velocity, attitude
| from FLU to ENU, and body rates in FLU. Position is aligned once at warmup;
| velocity, attitude and rates are never corrected with physical truth.
|
| Timing: in lockstep, the PX4 clock is the time_usec of the last HIL_SENSOR,
| so telemetry timestamps (time_boot_ms) are simulated time. The estimate of
| sensor sample k is only streamed after sample k+1 arrives, so the buffer
| selects the estimate at exactly ``now - delay`` (one physics step by default):
| a fixed, reproducible latency instead of "the latest packet received".
"""

from __future__ import annotations

import math

import torch


class EstimateUnavailable(RuntimeError):
    """PX4 has not supplied a valid estimate for the requested simulated time."""


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two scalar-first quaternions in the order a then b."""
    w, x, y, z = a.unbind(-1)
    W, X, Y, Z = b.unbind(-1)
    return torch.stack((w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
                        w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W), dim=-1)


def convert_state(position, velocity, quaternion, rates, offset) -> torch.Tensor:
    """Return one Pegasus state as a float32 tensor with shape (1, 13).

    Args:
        position: PX4 local NED position [north, east, down] in metres.
        velocity: PX4 NED velocity in metres per second.
        quaternion: PX4 scalar-first attitude from FRD body to NED world.
        rates: PX4 body angular rates [roll, pitch, yaw] in FRD, rad/s.
        offset: ENU translation established once at estimator warmup, metres.
    """
    values = torch.tensor([*position, *velocity, *quaternion, *rates], dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise EstimateUnavailable("PX4 state contains nonfinite values")

    q = values[6:10]
    norm = torch.linalg.vector_norm(q)
    if norm < 0.9 or norm > 1.1:
        raise EstimateUnavailable("PX4 attitude quaternion has an invalid norm")

    # The fixed quaternions rotate NED into ENU and FLU into FRD.
    # q_ENU_FLU = q_ENU_NED * q_NED_FRD * q_FRD_FLU.
    q_enu_ned = q.new_tensor([0.0, 1/math.sqrt(2), 1/math.sqrt(2), 0.0])
    q_frd_flu = q.new_tensor([0.0, 1.0, 0.0, 0.0])
    attitude = _quaternion_multiply(_quaternion_multiply(q_enu_ned, q/norm), q_frd_flu)

    origin = torch.as_tensor(offset, dtype=values.dtype)
    pos = values[:3][[1, 0, 2]] * values.new_tensor([1.0, 1.0, -1.0]) + origin
    vel = values[3:6][[1, 0, 2]] * values.new_tensor([1.0, 1.0, -1.0])
    omega = values[10:13] * values.new_tensor([1.0, -1.0, -1.0])
    return torch.cat((pos, vel, attitude, omega)).float().unsqueeze(0)


class EstimateBuffer:
    """Keep recent PX4 position, attitude, estimator status and wind by simulated time.

    LOCAL_POSITION_NED and ATTITUDE_QUATERNION carry time_boot_ms.
    ESTIMATOR_STATUS and WIND_COV carry time_usec.
    All are PX4 lockstep time, i.e. the simulated time of the HIL_SENSOR stream.
    WIND_COV only exists while the EKF declares its wind estimate valid (it needs
    airspeed or sideslip fusion), so it is optional: wind_enu() returns None without it.
    """

    # ESTIMATOR_ATTITUDE, VELOCITY_HORIZ/VERT, POS_HORIZ_REL, POS_VERT_ABS.
    REQUIRED_FLAGS = 1 | 2 | 4 | 8 | 32
    INVALID_FLAGS = 1024 | 2048  # GPS_GLITCH, ACCEL_ERROR
    KINDS = ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ESTIMATOR_STATUS", "WIND_COV")

    def __init__(self, system_id: int = 1, component_id: int = 1, max_skew: float = 0.0105,
                 status_max_age: float = 0.5, history: int = 64, wind_max_age: float = 0.5):
        """Select one PX4 source; ``max_skew`` bounds the age of the selected sample [s]."""
        self.system_id = system_id
        self.component_id = component_id
        self.max_skew = max_skew
        self.status_max_age = status_max_age
        self.wind_max_age = wind_max_age
        self.history = history
        self.messages = {kind: [] for kind in self.KINDS}
        self.offset = torch.zeros(3)

    @staticmethod
    def stamp(message) -> float:
        """PX4 lockstep time of a message [s]."""
        return message.time_usec / 1e6 if message.get_type() in ("ESTIMATOR_STATUS", "WIND_COV") else message.time_boot_ms / 1000

    def ingest(self, message) -> None:
        """Retain a report from the selected PX4 system/component in time order."""
        kind = message.get_type()
        if kind not in self.messages:
            return
        if message.get_srcSystem() != self.system_id or message.get_srcComponent() != self.component_id:
            return
        stamp = self.stamp(message)
        queue = self.messages[kind]
        if queue and stamp < queue[-1][0]:
            raise EstimateUnavailable("PX4 clock moved backwards; restart SITL and evaluation")
        if queue and stamp == queue[-1][0]:
            queue[-1] = (stamp, message)
        else:
            queue.append((stamp, message))
            del queue[:-self.history]

    def latest(self, kind: str):
        """Return (stamp, message) of the newest report of ``kind`` or None."""
        queue = self.messages[kind]
        return queue[-1] if queue else None

    def _at(self, kind: str, target: float):
        """Newest report not later than ``target`` and no older than max_skew."""
        for stamp, message in reversed(self.messages[kind]):
            if stamp <= target + 1e-6:
                if target - stamp > self.max_skew:
                    raise EstimateUnavailable(f"PX4 {kind} at t={target:.3f} s is missing "
                                              f"(newest earlier sample {stamp:.3f} s)")
                return stamp, message
        raise EstimateUnavailable(f"PX4 {kind} at t={target:.3f} s has not arrived")

    def ready_for(self, target: float) -> bool:
        """True when position and attitude reports up to ``target`` have arrived."""
        return all((last := self.latest(kind)) is not None and last[0] >= target - 1e-6
                   for kind in ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION"))

    def snapshot(self, target: float) -> tuple[torch.Tensor, float]:
        """Convert the estimate at simulated time ``target``; return (state, stamp).

        The EKF validity mask (attitude, horizontal/vertical velocity and
        relative/absolute position) must hold in the newest status report not
        later than ``target``. An unavailable estimate is never replaced with truth.
        """
        p_stamp, p = self._at("LOCAL_POSITION_NED", target)
        q_stamp, q = self._at("ATTITUDE_QUATERNION", target)
        status = [(s, m) for s, m in self.messages["ESTIMATOR_STATUS"] if s <= target + 1e-6]
        if not status or target - status[-1][0] > self.status_max_age:
            raise EstimateUnavailable("PX4 estimator status is missing or stale")
        flags = status[-1][1].flags
        if flags & self.REQUIRED_FLAGS != self.REQUIRED_FLAGS or flags & self.INVALID_FLAGS:
            raise EstimateUnavailable("PX4 EKF validity flags are not satisfied")
        state = convert_state((p.x, p.y, p.z), (p.vx, p.vy, p.vz), (q.q1, q.q2, q.q3, q.q4),
                              (q.rollspeed, q.pitchspeed, q.yawspeed), self.offset)
        return state, min(p_stamp, q_stamp)

    def wind_enu(self, target: float) -> tuple[torch.Tensor, float] | None:
        """Return (wind, stamp) of the EKF wind estimate at ``target``, or None if there is none.

        PX4 reports the horizontal wind velocity in NED (north, east); it is returned in ENU as
        [east, north, 0] [m/s], the vertical wind not being estimated. None means no report not later
        than ``target`` and younger than wind_max_age, e.g. before airspeed fusion starts, so a missing
        estimate is never turned into zero wind here.
        """
        for stamp, message in reversed(self.messages["WIND_COV"]):
            if stamp <= target + 1e-6:
                if target - stamp > self.wind_max_age:
                    return None
                wind = torch.tensor([message.wind_y, message.wind_x, 0.0], dtype=torch.float32)
                return (wind, stamp) if torch.isfinite(wind).all() else None
        return None

    def align_position_origin(self, truth_position, target: float) -> None:
        """Align PX4 local position to the physical spawn point once at warmup."""
        self.offset = torch.zeros(3)
        estimate, _ = self.snapshot(target)
        self.offset = torch.as_tensor(truth_position, dtype=torch.float32).cpu() - estimate[0, :3]
