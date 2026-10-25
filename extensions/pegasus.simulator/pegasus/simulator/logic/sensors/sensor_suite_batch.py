"""
| File: sensor_suite_batch.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Independent Pegasus sensor suites for a batched physical state.
|
| Each vehicle retains its own sensor history and noise stream. The existing
| scalar models are reused through State copies; their computations are not
| vectorized. VehicleBatch invokes this suite after its post-physics state
| update only when a backend consumes update_sensors().
"""

from __future__ import annotations

import torch


class SensorSuiteBatch:
    """Sample IMU, GPS, barometer and magnetometer for each vehicle.

    Config keys "imu", "gps", "barometer" and "magnetometer" are passed to the
    Pegasus sensor models (see each model for its parameters); "sensor_seed"
    seeds the private per-vehicle random streams. An "airspeed" key (possibly
    an empty dict) adds a pitot sensor, which reads the wind of the vehicle's
    aerodynamics model when it has one.
    """

    def __init__(self, n_vehicles: int = 1, config: dict | None = None):
        """Store sensor settings; allocation follows vehicle initialization."""
        self.n_vehicles = n_vehicles
        self.config = dict(config or {})
        self.sensor_origin = None
        self._vehicle = None
        self._coordinates = None
        self._states = []
        self._sensors = []
        self._rng_states = []

    # =============================================================================
    # Lifecycle
    # =============================================================================

    def initialize(self, vehicle, latitude, longitude, altitude) -> None:
        """Use the same geographic origin for all vehicles and sensor models."""
        self._vehicle = vehicle
        self._coordinates = tuple(torch.as_tensor(value).detach().cpu().float() for value in
                                  (latitude, longitude, altitude))
        self.sensor_origin = [float(value) for value in self._coordinates]
        self.reset()

    def reset(self) -> None:
        """Restart sensor histories and private noise streams before warmup."""
        from pegasus.simulator.logic.state import State
        from pegasus.simulator.logic.sensors.imu import IMU
        from pegasus.simulator.logic.sensors.gps import GPS
        from pegasus.simulator.logic.sensors.barometer import Barometer
        from pegasus.simulator.logic.sensors.magnetometer import Magnetometer
        from pegasus.simulator.logic.sensors.airspeed import Airspeed

        self._states = [State("cpu") for _ in range(self.n_vehicles)]
        self._sensors = []
        self._rng_states = []
        seed = int(self.config.get("sensor_seed", 0))
        for index in range(self.n_vehicles):
            # Default rates follow the PX4 gazebo-classic iris (IMU every physics step, magnetometer
            # 100 Hz, barometer 50 Hz, GPS 5 Hz). The GPS bias recursion is applied per sample, so
            # its default random-walk coefficients give gazebo's statistics only at 5 Hz.
            sensors = [
                IMU({"update_rate": 1000.0, **self.config.get("imu", {})}),
                GPS({"update_rate": 5.0, **self.config.get("gps", {})}),
                Barometer({"update_rate": 50.0, **self.config.get("barometer", {})}),
                Magnetometer({"update_rate": 100.0, **self.config.get("magnetometer", {})}),
            ]
            if "airspeed" in self.config:
                # Last in the list so that the barometer stays at index 2 below.
                sensors.append(Airspeed({**self.config["airspeed"]}))
            for sensor in sensors:
                sensor.initialize(self._vehicle, *self._coordinates)
            # Pressure altitude and GPS altitude refer to the same world datum.
            sensors[2]._z_start = torch.tensor(0.0)
            self._sensors.append(sensors)
            self._rng_states.append(torch.Generator().manual_seed(seed + index).get_state())

    # =============================================================================
    # Post-physics sampling
    # =============================================================================

    def update(self, state, dt: float) -> list[dict]:
        """Return each vehicle's scalar state and the readings due this step.

        Sampling uses a private CPU RNG per vehicle. Adding sensors therefore
        leaves the policy and task's torch random stream unchanged.
        """
        samples = []
        sample_dt = torch.tensor(float(dt))
        fields = ("position", "attitude", "linear_velocity", "angular_velocity",
                  "linear_body_velocity", "linear_acceleration")
        # One batch transfer per field avoids a separate GPU synchronization
        # for every sensor and vehicle.
        physical = {name: getattr(state, name).detach().cpu() for name in fields}
        # Wind in the FLU body frame, as the aerodynamics model applies it; vehicles without one have no wind.
        aerodynamics = getattr(self._vehicle, "aerodynamics", None)
        wind_body = getattr(aerodynamics, "_wind_body", None)
        wind_body = None if wind_body is None else wind_body.detach().cpu()
        for index, scalar_state in enumerate(self._states):
            for name in fields:
                setattr(scalar_state, name, physical[name][index].clone())
            scalar_state.wind_body = None if wind_body is None else wind_body[index].clone()
            readings = {}
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._rng_states[index])
                for sensor in self._sensors[index]:
                    data = sensor.update(scalar_state, sample_dt)
                    if data is not None:
                        readings[sensor.sensor_type] = data
                self._rng_states[index] = torch.get_rng_state()
            samples.append({"state": scalar_state, "readings": readings})
        return samples
