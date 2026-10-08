#!/usr/bin/env python3
"""
| File: 4_python_single_batch_glider_orbit.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description:
|   Closed-loop circular-orbit test for the EasyGlider batch model.
|
| The backend implements the fixed-wing successive-loop-closure autopilot
| (course/roll, altitude/pitch, airspeed/throttle and yaw damping) and the
| circular-orbit path-following law.
"""

# Isaac Sim startup
from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': False})

# Imports after SimulationApp
import csv
import math
from pathlib import Path

import numpy as np
import omni.timeline
import torch

from isaacsim.core.utils.prims import create_prim
from isaacsim.core.api.objects import GroundPlane
from isaacsim.util.debug_draw import _debug_draw

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.glider_batch import GliderBatch, GliderBatchConfig

# Import the custom python control backend
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/utils')
from glider_controller_batch import GliderAutopilotBackend

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEG = math.pi / 180.0

DT = 1.0 / 250.0
RENDER_DT = 1.0 / 60.0

# Run long enough for more than two complete turns:
#   T_orbit ~= 2*pi*R / V
#           ~= 27.3 s for R=50 m, V=11.5 m/s.
DURATION = 70.0
LOG_PERIOD = 0.05
N_VEHICLES = 1

# -----------------------------------------------------------------------------
# Visualization settings
# -----------------------------------------------------------------------------

# The ground is automatically made larger than the orbit diameter:
#     GROUND_SIZE = 2 * (ORBIT_RADIUS + GROUND_MARGIN)
GROUND_MARGIN = 15.0

# Desired orbit debug drawing.
ORBIT_DASH_COUNT = 72
ORBIT_DASH_FRACTION = 0.58
ORBIT_LINE_WIDTH = 3.0
ORBIT_COLOR = (31/255, 119/255, 180/255, 1.0)

# Actual flown trajectory.
DRAW_ACTUAL_TRAJECTORY = True
ACTUAL_LINE_WIDTH = 3.0
ACTUAL_COLOR = (1.0, 0.0, 0.0, 1.0)
ACTUAL_DRAW_PERIOD = 0.10

# Top-view camera. This scale is chosen so the complete ground/orbit remains
# visible for the usual Isaac Sim camera field of view.
TOP_CAMERA_SCALE = 2.0


# =============================================================================
# Orbit definition
# =============================================================================

ORBIT_CENTER_ENU = torch.tensor([0.0, 0.0, 60.0], dtype=torch.float32)

ORBIT_RADIUS = 50.0

# +1 = clockwise
# -1 = counter-clockwise
ORBIT_DIRECTION = +1

AIRSPEED_COMMAND = 11.5
ALTITUDE_COMMAND = float(ORBIT_CENTER_ENU[2])

# Derived visualization dimensions.
GROUND_SIZE = 2.0 * (ORBIT_RADIUS + GROUND_MARGIN)
TOP_CAMERA_Z = ALTITUDE_COMMAND + TOP_CAMERA_SCALE * (ORBIT_RADIUS + GROUND_MARGIN)


# =============================================================================
# Initial condition
# =============================================================================

# For a clockwise orbit the tangent direction here is South.
INITIAL_POSITION_ENU = torch.tensor([ORBIT_CENTER_ENU[0] + ORBIT_RADIUS, ORBIT_CENTER_ENU[1], ORBIT_CENTER_ENU[2]], dtype=torch.float32)

if ORBIT_DIRECTION == +1:
    # South, measured clockwise from North.
    INITIAL_COURSE = math.pi
else:
    # North.
    INITIAL_COURSE = 0.0


# Powered-flight trim of the current EasyGlider model.
INITIAL_PITCH_UP = 0.98152 * DEG

# Coordinated-turn bank angle used only as an initial condition.
#
#   phi ~= lambda atan(V^2 / (R g))
#
# This reduces the initial transient. The controller is still responsible for
# maintaining the orbit after the simulation starts.
G = 9.80665

INITIAL_ROLL = float(ORBIT_DIRECTION) * math.atan(AIRSPEED_COMMAND ** 2 / (ORBIT_RADIUS * G))


# =============================================================================
# Aerodynamic / propulsion configuration
# =============================================================================

AERODYNAMICS_CFG = {
    "geometry": {
        "rho": 1.2041,
        "S": 0.416,
        "b": 1.800,
        "c": 0.416 / 1.800,
        "AR": 1.800**2 / 0.416,
        "e": 0.8523,
        "r_ref": (0.37, 0.0, 0.0552),
    },

    # Empty means: use the validated defaults stored in GliderCoefficients.
    "coefficients": {},

    "eval_at_ref": True,
}


THRUST_CFG = {
    # Current preliminary propulsion model:
    #
    #   T = k_T Omega^2
    #   Q = k_M T
    #
    # These will later be replaced by CT(J), CQ(J).
    "thrust_constant": 8.5e-6,
    "moment_constant": 0.018,

    "reaction_moment_sign": -1.0,

    "min_rotor_velocity": 0.0,
    "max_rotor_velocity": 1100.0,

    "motor_time_constant": 0.0,

    "air_density": 1.2041,
    "propeller_diameter": 0.2286,
}


# =============================================================================
# Autopilot configuration
# =============================================================================

AUTOPILOT_CFG = {
    "trim": {
        "airspeed": 11.5,

        # Textbook sign used inside the backend:
        # positive pitch_up_deg = nose-up.
        "pitch_up_deg": 0.98152,

        # EasyGlider actuator convention.
        "delta_e_deg": 2.39950,
        "delta_a_deg": 0.10682,
        "delta_r_deg": 0.0,

        "omega_p": 364.509,
    },

    "limits": {
        "delta_e_deg": 20.0,
        "delta_a_deg": 20.0,
        "delta_r_deg": 20.0,

        "roll_command_deg": 35.0,
        "pitch_command_min_deg": -15.0,
        "pitch_command_max_deg": 20.0,

        "omega_min": 0.0,
        "omega_max": 1100.0,
    },

    # Conservative initial gains.
    #
    # They follow the control architecture from the book, but they have not
    # yet been identified from the exact EasyGlider linearized model.
    "gains": {
        "kp_phi": 0.65,
        "kd_phi": 0.03,

        "kp_chi": 2.00,
        "ki_chi": 0.45,

        "kp_theta": -0.55,
        "kd_theta": -0.10,

        "kp_h": 0.045,
        "ki_h": 0.0075,

        "kp_V": 0.080,
        "ki_V": 0.020,

        "k_r": 0.15,
        "p_wo": 0.50,
    },

    "orbit": {"k_orbit": 4.0},

    "integrator_limits": {"course": math.radians(60.0), "altitude": 30.0, "airspeed": 20.0},

    "gravity": G,
}


# =============================================================================
# Rotation helpers
# =============================================================================

def euler_enu_flu_to_quaternion(roll: float, pitch_flu: float, course: float) -> torch.Tensor:
    """
    Build a quaternion [w,x,y,z] for the body-to-world orientation.

    Parameters
    roll: FLU roll [rad]. Positive roll means right wing down.

    pitch_flu: FLU pitch [rad]. Positive means nose-down.

    course: Desired heading/course [rad], measured clockwise from North.

    Notes
    Standard ENU yaw is measured counter-clockwise from +East. Therefore:
        yaw_ENU = pi/2 - course.
    """

    yaw = math.pi / 2.0 - course

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)

    cp = math.cos(pitch_flu / 2.0)
    sp = math.sin(pitch_flu / 2.0)

    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    # Standard Rz(yaw) Ry(pitch) Rx(roll) quaternion.
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = torch.tensor([[w, x, y, z]], dtype=torch.float32)

    return q / torch.linalg.vector_norm(q, dim=1, keepdim=True)


def course_velocity_enu(speed: float, course: float) -> torch.Tensor:
    """
    Horizontal ENU velocity for course measured clockwise from North.

        v_E = V sin(chi)
        v_N = V cos(chi)
    """

    return torch.tensor([speed * math.sin(course), speed * math.cos(course), 0.0], dtype=torch.float32)


# =============================================================================
# Application
# =============================================================================

class GliderOrbitTest:

    def __init__(self):

        # ---------------------------------------------------------------------
        # Pegasus world
        # ---------------------------------------------------------------------

        self.pg = PegasusInterface()

        self.pg.set_world_settings(physics_dt=DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0, device='cpu')

        self.pg.initialize_world()

        self.world = self.pg.world
        self.timeline = omni.timeline.get_timeline_interface()

        # ---------------------------------------------------------------------
        # Ground / lighting
        # ---------------------------------------------------------------------

        # Explicit square ground plane whose side is always larger than the
        # orbit diameter.  GroundPlane.size is the length of each edge.
        self.world.scene.add(
            GroundPlane(
                prim_path="/World/OrbitGround",
                name="orbit_ground",
                size=GROUND_SIZE,
                z_position=0.0,
                color=np.array([0.30, 0.30, 0.30]),
            )
        )

        create_prim('/World/Light', 'DomeLight', attributes={'inputs:intensity': 3500.0})

        # ---------------------------------------------------------------------
        # Top-down camera
        # ---------------------------------------------------------------------

        # A tiny y offset avoids an exactly singular vertical look-at while the
        # view remains visually indistinguishable from a true top view.
        self.pg.set_viewport_camera(
            camera_position=[float(ORBIT_CENTER_ENU[0]), float(ORBIT_CENTER_ENU[1]) - 0.01, float(TOP_CAMERA_Z)],
            camera_target=[float(ORBIT_CENTER_ENU[0]), float(ORBIT_CENTER_ENU[1]), float(ALTITUDE_COMMAND)],
        )

        # Debug-draw interface used for desired and actual trajectories.
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self._last_actual_draw_point = None

        self.draw_reference_orbit()

        # ---------------------------------------------------------------------
        # Backend
        # ---------------------------------------------------------------------

        self.backend = GliderAutopilotBackend(n_vehicles=N_VEHICLES, config=AUTOPILOT_CFG, device='cpu')

        self.backend.set_orbit(
            center_enu=ORBIT_CENTER_ENU,
            radius=ORBIT_RADIUS,
            direction=ORBIT_DIRECTION,
            altitude=ALTITUDE_COMMAND,
            airspeed=AIRSPEED_COMMAND,
        )

        # ---------------------------------------------------------------------
        # Glider
        # ---------------------------------------------------------------------

        config = GliderBatchConfig(
            n_vehicles=N_VEHICLES,

            thrust_curve_cfg=THRUST_CFG,
            aerodynamics_cfg=AERODYNAMICS_CFG,

            backends=[self.backend],

            propeller_prim_name="rotor",
            propeller_axis_body=(1.0, 0.0, 0.0),
        )

        init_pos = INITIAL_POSITION_ENU.reshape(1, 3)

        # Backend pitch sign is nose-up positive.
        # FLU Euler pitch has the opposite sign.
        initial_pitch_flu = -INITIAL_PITCH_UP

        init_orientation = euler_enu_flu_to_quaternion(roll=INITIAL_ROLL, pitch_flu=initial_pitch_flu, course=INITIAL_COURSE)

        self.glider = GliderBatch(
            stage_prefix="/World/easyglider",
            usd_file=ROBOTS["Easyglider"],
            vehicle_batch_id=0,
            n_vehicles=N_VEHICLES,
            init_pos=init_pos,
            init_orientation=init_orientation,
            spacing=3.0,
            config=config,
        )

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        self.rows = []
        self.output_path = Path(__file__).with_name('easyglider_orbit_test.csv')

    # =========================================================================
    # Initial state
    # =========================================================================

    def set_initial_state(self):

        velocity_world = course_velocity_enu(AIRSPEED_COMMAND, INITIAL_COURSE).to(dtype=torch.float32, device=self.glider.device)

        # Initial velocity of the body prim is set directly in the PhysX scene.
        root_velocity = torch.zeros((N_VEHICLES, 6), dtype=torch.float32, device=self.glider.device)
        root_velocity[:, 0:3] = velocity_world.unsqueeze(0)

        self.glider._root_prims.set_velocities(root_velocity)

        # Refresh the cached StateBatch immediately from PhysX so the backend
        # sees the intended airspeed/course before the next control update.
        # This uses the normal VehicleBatch state path and avoids calling
        # set_state_batch() solely for initialization.
        self.glider.update_state(0.0)

    # =========================================================================
    # Trajectory visualization
    # =========================================================================

    def draw_reference_orbit(self):
        """Draw the commanded circular orbit as dashed arc segments."""

        center_e = float(ORBIT_CENTER_ENU[0])
        center_n = float(ORBIT_CENTER_ENU[1])
        z = float(ALTITUDE_COMMAND)

        start_points = []
        end_points = []

        dtheta = 2.0 * math.pi / ORBIT_DASH_COUNT
        dash_angle = ORBIT_DASH_FRACTION * dtheta

        for i in range(ORBIT_DASH_COUNT):
            theta0 = i * dtheta
            theta1 = theta0 + dash_angle

            p0 = (center_e + ORBIT_RADIUS * math.cos(theta0), center_n + ORBIT_RADIUS * math.sin(theta0), z)

            p1 = (center_e + ORBIT_RADIUS * math.cos(theta1), center_n + ORBIT_RADIUS * math.sin(theta1), z)

            start_points.append(p0)
            end_points.append(p1)

        self.draw.draw_lines(start_points, end_points, [ORBIT_COLOR] * len(start_points), [ORBIT_LINE_WIDTH] * len(start_points))

        # Mark the orbit centre as a small point.
        self.draw.draw_points([(center_e, center_n, z)], [(1.0, 1.0, 1.0, 1.0)], [8.0])

    def draw_actual_trajectory_segment(self):
        """Append one solid segment of the trajectory actually flown."""

        if not DRAW_ACTUAL_TRAJECTORY:
            return

        position = self.glider.state.position[0].detach().to(dtype=torch.float32).cpu()

        current = (float(position[0]), float(position[1]), float(position[2]))

        if self._last_actual_draw_point is not None:
            self.draw.draw_lines([self._last_actual_draw_point], [current], [ACTUAL_COLOR], [ACTUAL_LINE_WIDTH])

        self._last_actual_draw_point = current

    # =========================================================================
    # Logging
    # =========================================================================

    def record(self, t: float):

        state = self.glider.state

        pos = state.position[0].detach().cpu()
        vel = state.linear_velocity[0].detach().cpu()
        rates = state.angular_velocity[0].detach().cpu()

        actuator = self.backend.actuator_reference[0].detach().cpu()

        east = float(pos[0])
        north = float(pos[1])
        altitude = float(pos[2])

        de = east - float(ORBIT_CENTER_ENU[0])
        dn = north - float(ORBIT_CENTER_ENU[1])

        distance = math.sqrt(de * de + dn * dn)

        radial_error = distance - ORBIT_RADIUS

        chi = float(self.backend.course[0]) if self.backend.course is not None else float('nan')

        chi_c = float(self.backend.course_command[0]) if self.backend.course_command is not None else float('nan')

        phi = float(self.backend.roll[0]) if self.backend.roll is not None else float('nan')

        phi_c = float(self.backend.roll_command[0]) if self.backend.roll_command is not None else float('nan')

        phi_ff = float(self.backend.roll_feedforward[0]) if self.backend.roll_feedforward is not None else float('nan')

        theta = float(self.backend.pitch[0]) if self.backend.pitch is not None else float('nan')

        theta_c = float(self.backend.pitch_command[0]) if self.backend.pitch_command is not None else float('nan')

        Va = float(self.glider.aerodynamics.airspeed[0])

        alpha = float(self.glider.aerodynamics.alpha[0])

        beta = float(self.glider.aerodynamics.beta[0])

        self.rows.append(
            [
                t,

                east,
                north,
                altitude,

                float(vel[0]),
                float(vel[1]),
                float(vel[2]),

                distance,
                radial_error,

                Va,
                alpha / DEG,
                beta / DEG,

                chi / DEG,
                chi_c / DEG,

                phi / DEG,
                phi_c / DEG,
                phi_ff / DEG,

                theta / DEG,
                theta_c / DEG,

                float(rates[0]),
                float(rates[1]),
                float(rates[2]),

                float(actuator[0]) / DEG,
                float(actuator[1]) / DEG,
                float(actuator[2]) / DEG,
                float(actuator[3]),

                float(self.glider.thrusters.force[0, 0]),
            ]
        )

    def save_log(self):

        with self.output_path.open("w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "t",

                    "east",
                    "north",
                    "altitude",

                    "v_east",
                    "v_north",
                    "v_up",

                    "orbit_distance",
                    "radial_error",

                    "Va",
                    "alpha_deg",
                    "beta_deg",

                    "course_deg",
                    "course_command_deg",

                    "roll_deg",
                    "roll_command_deg",
                    "roll_feedforward_deg",

                    "pitch_up_deg",
                    "pitch_command_deg",

                    "p_flu_rad_s",
                    "q_flu_rad_s",
                    "r_flu_rad_s",

                    "delta_e_deg",
                    "delta_a_deg",
                    "delta_r_deg",
                    "omega_p_rad_s",

                    "thrust_N",
                ]
            )

            writer.writerows(self.rows)

        print(f'\nSaved orbit log: {self.output_path}')

    # =========================================================================
    # Final summary
    # =========================================================================

    def print_summary(self):

        state = self.glider.state

        pos = state.position[0]

        east = float(pos[0])
        north = float(pos[1])
        altitude = float(pos[2])

        radial_distance = math.sqrt((east - float(ORBIT_CENTER_ENU[0])) ** 2 + (north - float(ORBIT_CENTER_ENU[1])) ** 2)

        radial_error = radial_distance - ORBIT_RADIUS

        Va = float(self.glider.aerodynamics.airspeed[0])

        alpha = float(self.glider.aerodynamics.alpha[0]) / DEG

        beta = float(self.glider.aerodynamics.beta[0]) / DEG

        chi = float(self.backend.course[0]) / DEG

        chi_c = float(self.backend.course_command[0]) / DEG

        phi = float(self.backend.roll[0]) / DEG

        phi_c = float(self.backend.roll_command[0]) / DEG

        theta = float(self.backend.pitch[0]) / DEG

        theta_c = float(self.backend.pitch_command[0]) / DEG

        actuator = self.backend.actuator_reference[0].detach().cpu()

        print('\n' + '=' * 80)

        print('EASYGLIDER CLOSED-LOOP ORBIT TEST')

        print('=' * 80)

        print(f'Orbit radius command : {ORBIT_RADIUS:10.3f} m')

        print(f'Final orbit distance : {radial_distance:10.3f} m')

        print(f'Final radial error   : {radial_error:+10.3f} m')

        print(f'Altitude command     : {ALTITUDE_COMMAND:10.3f} m')

        print(f'Final altitude       : {altitude:10.3f} m')

        print(f'Altitude error       : {altitude - ALTITUDE_COMMAND:+10.3f} m')

        print(f'Airspeed command     : {AIRSPEED_COMMAND:10.3f} m/s')

        print(f'Final airspeed       : {Va:10.3f} m/s')

        print(f'Alpha                : {alpha:10.3f} deg')

        print(f'Beta                 : {beta:10.3f} deg')

        print(f'Course               : {chi:10.3f} deg')

        print(f'Course command       : {chi_c:10.3f} deg')

        print(f'Roll                 : {phi:10.3f} deg')

        print(f'Roll command         : {phi_c:10.3f} deg')

        print(f'Pitch (nose-up sign) : {theta:10.3f} deg')

        print(f'Pitch command        : {theta_c:10.3f} deg')

        print('-' * 80)

        print(f'Elevator             : {float(actuator[0]) / DEG:+10.3f} deg')

        print(f'Aileron              : {float(actuator[1]) / DEG:+10.3f} deg')

        print(f'Rudder               : {float(actuator[2]) / DEG:+10.3f} deg')

        print(f'Propeller speed      : {float(actuator[3]):10.3f} rad/s')

        print(f'Thrust               : {float(self.glider.thrusters.force[0, 0]):10.3f} N')

        print('=' * 80)

        # Broad sanity checks only. They are deliberately not tight enough to
        # classify gain tuning as a model failure.
        passed = (
            abs(radial_error) < 15.0
            and abs(
                altitude
                - ALTITUDE_COMMAND
            ) < 10.0
            and abs(
                Va
                - AIRSPEED_COMMAND
            ) < 4.0
            and abs(beta) < 10.0
        )

        print('RESULT: ' + ('PASS (bounded closed-loop orbit)' if passed else 'FAIL / controller tuning required'))

        print('=' * 80)

    # =========================================================================
    # Run
    # =========================================================================

    def run(self):

        # Initialize PhysX views.
        self.world.reset()

        # Start timeline. VehicleBatch.start() and backend.start() are called by
        # the corresponding Pegasus callbacks.
        self.timeline.play()

        # One first step ensures prim views and callbacks are initialized.
        self.world.step(render=True)

        self.set_initial_state()

        start_time = float(self.world.current_time)

        next_log = 0.0
        next_actual_draw = 0.0

        while simulation_app.is_running():

            self.world.step(render=True)

            # Use actual simulated time rather than manually adding DT.
            t = float(self.world.current_time) - start_time

            if t >= next_log:

                self.record(t)

                next_log += LOG_PERIOD

            if t >= next_actual_draw:

                self.draw_actual_trajectory_segment()

                next_actual_draw += ACTUAL_DRAW_PERIOD

            # --------------------------------------------------------------
            # Basic divergence checks
            # --------------------------------------------------------------

            state = self.glider.state

            altitude = float(state.position[0, 2])

            Va = float(self.glider.aerodynamics.airspeed[0])

            rates_norm = float(torch.linalg.vector_norm(state.angular_velocity[0]))

            if (not math.isfinite(altitude) or not math.isfinite(Va) or altitude < 2.0 or Va > 50.0 or rates_norm > 10.0):

                print('\nOrbit test aborted: trajectory diverged.')

                break

            if t >= DURATION:
                break

        self.save_log()
        self.print_summary()

        self.timeline.stop()

        simulation_app.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    GliderOrbitTest().run()
