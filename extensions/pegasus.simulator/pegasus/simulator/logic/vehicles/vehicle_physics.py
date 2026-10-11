"""
| File: vehicle_physics.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Shared physical parameters for batched vehicle experiments.
|
| These are baseline parameters, not training or reward settings. A task may
| override them to reproduce a particular experiment. Training and evaluation
| must use the resulting task configuration for the same checkpoint.
"""

from copy import deepcopy


# =============================================================================
# Multirotor propulsion
# =============================================================================

IRIS_PHYSICS = {
    "num_rotors": 4,
    "rotor_constant": [5.84e-6] * 4,
    "reaction_moment_coefficient": [3.504e-7] * 4,
    "rot_dir": [-1, -1, 1, 1],
    "min_rotor_velocity": [0.0] * 4,
    "max_rotor_velocity": [1100.0] * 4,
    "motor_time_constant_up": 0.0125,
    "motor_time_constant_down": 0.025,
}

# L2F Crazyflie parameters. The corresponding USD asset is not yet included
# in this branch; the preset remains available for future asset integration.
CRAZYFLIE_PHYSICS = {
    "num_rotors": 4,
    "rotor_constant": [2.8815744627880866e-8] * 4,
    "reaction_moment_coefficient": [1.7187300725171608e-10] * 4,
    "rot_dir": [-1, 1, -1, 1],
    "min_rotor_velocity": [0.0] * 4,
    "max_rotor_velocity": [2272.6281256068564] * 4,
    "motor_time_constant_up": [0.15] * 4,
    "motor_time_constant_down": [0.15] * 4,
}

# Gazebo Shuttle SDF propulsion parameters.
SHUTTLE_PHYSICS = {
    "num_rotors": 4,
    "rotor_constant": [1.709716e-5] * 4,
    "reaction_moment_coefficient": [2.7355456e-7] * 4,
    "rot_dir": [-1, -1, 1, 1],
    "min_rotor_velocity": [0.0] * 4,
    "max_rotor_velocity": [1400.0] * 4,
    "motor_time_constant_up": [0.008] * 4,
    "motor_time_constant_down": [0.020] * 4,
}


# =============================================================================
# Fixed wing and hybrid propulsion
# =============================================================================

EASYGLIDER_PHYSICS = {
    "rotor_constant": [8.54858e-6],
    "reaction_moment_coefficient": [1.0258e-7],
    "rot_dir": [-1.0],
    "min_rotor_velocity": 0.0,
    "max_rotor_velocity": 3500.0,
    "motor_time_constant_up": 0.0125,
    "motor_time_constant_down": 0.025,
}

SHUTTLE_GLIDER_PHYSICS = {
    "shuttle_thrust_cfg": SHUTTLE_PHYSICS,
    "glider_thrust_cfg": EASYGLIDER_PHYSICS,
    "aerodynamics_cfg": {},
}

VEHICLE_PHYSICS = {
    "Iris": IRIS_PHYSICS,
    "Crazyflie": CRAZYFLIE_PHYSICS,
    "Shuttle": SHUTTLE_PHYSICS,
    "Easyglider": EASYGLIDER_PHYSICS,
    "Shuttle_glider2": SHUTTLE_GLIDER_PHYSICS,
}


def get_vehicle_physics(vehicle: str) -> dict:
    """Return an independent copy of a vehicle's baseline physical parameters."""
    return deepcopy(VEHICLE_PHYSICS[vehicle])
