"""
| File: airspeed.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Simulates a pitot-static airspeed sensor. Based on gazebo_airspeed_plugin.cpp of PX4 sitl_gazebo-classic.
"""
__all__ = ["Airspeed"]

import torch
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor

DEFAULT_HOME_ALT_AMSL = 488.0


class Airspeed(Sensor):
    """The class that implements a differential pressure (pitot) sensor. This class inherits the base class Sensor.

    The sensor sits at the vehicle CoM, with the pitot tube aligned with the FLU body x axis. As in the gazebo plugin
    there is no lever arm, so the air velocity is the CoM velocity minus the wind, both in the body frame.
    """

    def __init__(self, config={}, device="cpu"):
        """Initialize the Airspeed class

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the Airspeed sensor - it can be empty or only have some of the parameters used by the sensor.

        Examples:
            The dictionary default parameters are

            >>> {"diff_pressure_stddev": 0.01,        # noise standard deviation [hPa]
            >>>  "temperature_msl": 288.15,           # temperature at MSL [K] (15 [C])
            >>>  "lapse_rate": 0.0065,                # reduction in temperature with altitude for troposphere [K/m]
            >>>  "air_density_msl": 1.225,            # air density at MSL [kg/m^3]
            >>>  "update_rate": 10.0}                 # Hz (the gazebo plugin sets the sensor rate to 10 Hz)
        """

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="Airspeed", update_rate=torch.tensor(config.get("update_rate", 10.0), dtype=torch.float32, device=device))

        # Set device
        self.device = device

        # The altitude of the [0, 0, 0] coordinate of the world, overridden by initialize()
        self._origin_alt = DEFAULT_HOME_ALT_AMSL

        self._DIFF_PRESSURE_STDDEV: float = float(config.get("diff_pressure_stddev", 0.01))
        self._TEMPERATURE_MSL: float = float(config.get("temperature_msl", 288.15))
        self._LAPSE_RATE: float = float(config.get("lapse_rate", 0.0065))
        self._AIR_DENSITY_MSL: float = float(config.get("air_density_msl", 1.225))

        # Save the current state measured by the sensor
        self._state = {
            "diff_pressure": torch.tensor(0.0, dtype=torch.float32, device=device),
        }

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: torch.Tensor):
        """Method that implements the logic of a pitot tube. The air velocity in the body frame is the body velocity
        minus the wind velocity expressed in the body frame (attribute wind_body of the state, zero when absent). The
        differential pressure is sign(u) * 0.5 * rho * u^2 along the pitot axis (body x), in hPa, plus gaussian noise.

        Args:
            state (State): The current state of the vehicle.
            dt (torch.Tensor): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """

        wind_body = getattr(state, "wind_body", None)
        u = state.linear_body_velocity[0] - (0.0 if wind_body is None else wind_body[0])

        # Compute the air density at the local altitude and temperature
        alt_amsl = self._origin_alt + state.position[2]
        temperature_local = self._TEMPERATURE_MSL - self._LAPSE_RATE * alt_amsl
        density_ratio = torch.pow(self._TEMPERATURE_MSL / temperature_local, 4.256)
        air_density = self._AIR_DENSITY_MSL / density_ratio

        # Differential pressure in hPa (1 hPa = 100 Pa) plus noise
        noise = self._DIFF_PRESSURE_STDDEV * torch.randn((), device=self.device)
        diff_pressure = torch.sign(u) * 0.005 * air_density * u * u + noise

        self._state = {
            "diff_pressure": diff_pressure,
        }

        return self._state
