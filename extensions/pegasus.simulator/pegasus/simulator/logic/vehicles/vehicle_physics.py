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

import math
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

# L2F Crazyflie parameters used with the cf2x USDA asset.
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

# Puller of the generic PX4 fixed-wing model (PX4-gazebo-models advanced_plane): motorConstant 8.54858e-6,
# momentConstant 0.01, maxRotVelocity 3500, time constants 0.0125/0.025 s. It is turningDirection "cw", which
# the Gazebo motor plugin turns into a reaction torque of +k_Q * omega^2 about the rotor axis (+X). Quadrotor
# "ccw" motors map to rot_dir -1.
EASYGLIDER_PHYSICS = {
    "rotor_constant": [8.54858e-6],
    "reaction_moment_coefficient": [8.54858e-8],      # 8.54858e-6 * 0.01
    "rot_dir": [1.0],
    "min_rotor_velocity": 0.0,
    "max_rotor_velocity": 3500.0,
    "motor_time_constant_up": 0.0125,
    "motor_time_constant_down": 0.025,
}

# EasyGlider control surface limits from the manufacturer's recommended throws (datasheet), independent of
# the SDF aerodynamic coefficients. Each throw d is the trailing-edge displacement and c the measured
# hinge-to-trailing-edge chord, so the deflection is asin(d / c). Positive deflection is trailing edge down
# (elevator, each aileron) and trailing edge to the left (rudder).
EASYGLIDER_SURFACE_CHORDS_MM = {"aileron": 39.0, "elevator": 34.0, "rudder": 68.0}
EASYGLIDER_SURFACE_THROWS_MM = {            # (trailing edge up / right, trailing edge down / left)
    "aileron": (15.0, 6.0),                  # differential: each aileron goes 15 mm up and 6 mm down
    "elevator": (8.0, 8.0),
    "rudder": (15.0, 15.0),
}


def _throw_deg(surface: str, index: int) -> float:
    return math.degrees(math.asin(EASYGLIDER_SURFACE_THROWS_MM[surface][index] / EASYGLIDER_SURFACE_CHORDS_MM[surface]))


# Per-aileron range (trailing edge down positive): -22.62 deg (up) to +8.85 deg (down)
EASYGLIDER_AILERON_SURFACE_RANGE_DEG = (-_throw_deg("aileron", 0), _throw_deg("aileron", 1))

# The model uses one equivalent antisymmetric command delta_a = (delta_a_l - delta_a_r) / 2 (roll derivatives
# of the SDF are those of the pair). With the differential throws the pair always spans 22.62 + 8.85 deg, so the
# equivalent command is symmetric: +-15.74 deg. The small symmetric (camber) component of the differential is neglected.
_AILERON_EQUIVALENT_DEG = 0.5 * (_throw_deg("aileron", 0) + _throw_deg("aileron", 1))

EASYGLIDER_SURFACE_LIMITS_DEG = {
    "elevator": (-_throw_deg("elevator", 1), _throw_deg("elevator", 0)),            # +-13.61
    "aileron": (-_AILERON_EQUIVALENT_DEG, _AILERON_EQUIVALENT_DEG),                 # +-15.74 (equivalent command)
    "rudder": (-_throw_deg("rudder", 1), _throw_deg("rudder", 0)),                  # +-12.74
}

# Same limits in radians, in the command order [delta_e, delta_a, delta_r]
EASYGLIDER_SURFACE_MIN_RAD = tuple(math.radians(EASYGLIDER_SURFACE_LIMITS_DEG[k][0]) for k in ("elevator", "aileron", "rudder"))
EASYGLIDER_SURFACE_MAX_RAD = tuple(math.radians(EASYGLIDER_SURFACE_LIMITS_DEG[k][1]) for k in ("elevator", "aileron", "rudder"))

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
