#!/usr/bin/env python3
"""
Minimal EasyGlider flight test for Pegasus Simulator.

Modes:
    TEST_MODE = "glide"
        Tests VehicleBatch + GliderAerodynamicsBatch with Omega_p = 0.
        This is the recommended first integration test.

    TEST_MODE = "powered"
        Tests VehicleBatch + aerodynamics + thrust model around a powered
        straight-and-level trim point.

The actuator vector is:
    [delta_e, delta_a, delta_r, Omega_p]
    [rad,     rad,     rad,     rad/s]
"""

# -----------------------------------------------------------------------------
# Isaac Sim must be started before importing most Isaac/Pegasus modules.
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import csv
import math
from pathlib import Path

import torch
import omni.timeline
from isaacsim.core.utils.prims import create_prim

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from pegasus.simulator.logic.vehicles.glider_batch import GliderBatch, GliderBatchConfig



DEG = math.pi / 180.0

# =============================================================================
# Test configuration
# =============================================================================

TEST_MODE = "powered"       # "glide" or "powered"

DT = 1.0 / 250.0
DURATION = 20.0
LOG_PERIOD = 0.05

N_VEHICLES = 1
INITIAL_ALTITUDE = 60.0


# =============================================================================
# Model configuration
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
    # The coefficients dictionary may be omitted when the dataclass defaults
    # already contain the validated EasyGlider coefficients.
    "coefficients": {},
    "eval_at_ref": True,
}

THRUST_CFG = {
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
# Trim points
# =============================================================================

if TEST_MODE == "glide":
    # Approximate steady-glide trim for the current aerodynamic model.
    #
    # Va       ~= 11.1 m/s
    # alpha    ~= 1.54 deg
    # gamma    ~= -4.33 deg  (descending flight path)
    # sink     ~= 0.84 m/s
    # delta_e  ~= +0.18 deg
    #
    VA0 = 11.1
    ALPHA0 = 1.5380 * DEG
    GAMMA0 = -4.3335 * DEG

    DELTA_E = 0.1797 * DEG
    DELTA_A = 0.0
    DELTA_R = 0.0
    OMEGA_P = 0.0

elif TEST_MODE == "powered":
    # Approximate straight-and-level trim for the current model:
    #
    # Va       = 11.5 m/s
    # alpha    ~= 0.982 deg
    # delta_e  ~= +2.399 deg
    # delta_a  ~= +0.107 deg
    # Omega_p  ~= 364.5 rad/s
    #
    # IMPORTANT:
    # The powered test assumes that GliderBatch applies the thrust force along
    # the physical +x_B direction even though the rotor rigid prim has its own
    # local orientation.
    VA0 = 11.5
    ALPHA0 = 0.98152 * DEG
    GAMMA0 = 0.0

    DELTA_E = 2.39950 * DEG
    DELTA_A = 0.10682 * DEG
    DELTA_R = 0.0
    OMEGA_P = 364.509

else:
    raise ValueError("TEST_MODE must be 'glide' or 'powered'")


# In the user's FLU convention, positive Euler pitch about +y_B is nose-down.
#
# physical nose-up angle = gamma + alpha
# therefore:
THETA0 = -(GAMMA0 + ALPHA0)


# =============================================================================
# Helpers
# =============================================================================

def pitch_quaternion(theta: float) -> torch.Tensor:
    """Return [w, x, y, z] for a rotation theta about +y."""
    return torch.tensor(
        [[
            math.cos(theta / 2.0),
            0.0,
            math.sin(theta / 2.0),
            0.0,
        ]],
        dtype=torch.float32,
    )


def quaternion_to_euler(q):
    """Quaternion [w,x,y,z] -> roll, pitch, yaw [rad]."""
    w, x, y, z = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# =============================================================================
# Application
# =============================================================================

class GliderFlightTest:

    def __init__(self):
        self.pg = PegasusInterface()

        self.pg.set_world_settings(
            physics_dt=DT,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
            device="cpu",
        )
        self.pg.initialize_world()

        self.world = self.pg.world
        self.timeline = omni.timeline.get_timeline_interface()

        # Ground plane only provides visual/geometric reference.
        self.world.scene.add_default_ground_plane()

        create_prim(
            "/World/Light",
            "DomeLight",
            attributes={"inputs:intensity": 2500.0},
        )

        self.pg.set_viewport_camera(
            camera_position=[-12.0, -18.0, INITIAL_ALTITUDE + 7.0],
            camera_target=[10.0, 0.0, INITIAL_ALTITUDE],
        )

        config = GliderBatchConfig(
            n_vehicles=N_VEHICLES,
            thrust_curve_cfg=THRUST_CFG,
            aerodynamics_cfg=AERODYNAMICS_CFG,
            backends=[],
            propeller_prim_name="rotor",
            propeller_axis_body=(1.0, 0.0, 0.0),
        )

        init_pos = torch.tensor(
            [[0.0, 0.0, INITIAL_ALTITUDE]],
            dtype=torch.float32,
        )

        init_orientation = pitch_quaternion(THETA0)

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

        # Always use an explicit (N,4) command shape.
        self.glider.set_actuator_reference(
            [[DELTA_E, DELTA_A, DELTA_R, OMEGA_P]]
        )

        self.log = []
        self.output = Path(__file__).with_name(
            f"easyglider_{TEST_MODE}_test.csv"
        )

    def set_initial_velocity(self):
        """
        Apply the trim inertial velocity after the prim views have been
        initialized by GliderBatch.start().
        """

        # ENU inertial velocity.
        #
        # gamma > 0: climb
        vx = VA0 * math.cos(GAMMA0)
        vz = VA0 * math.sin(GAMMA0)

        velocity_world = torch.tensor(
            [vx, 0.0, vz],
            dtype=torch.float32,
            device=self.glider.device,
        )

        # Set the same translational velocity on all rigid parts initially.
        n_parts_total = self.glider._vehicle_prims.count

        rigid_velocities = torch.zeros(
            (n_parts_total, 6),
            dtype=torch.float32,
            device=self.glider.device,
        )
        rigid_velocities[:, 0] = vx
        rigid_velocities[:, 2] = vz

        self.glider._vehicle_prims.set_velocities(rigid_velocities)

        # Synchronize Pegasus cached state immediately.
        pos, attitude = self.glider._root_prims.get_world_poses()

        self.glider.set_state_batch(
            env_ids=torch.tensor(
                [0], dtype=torch.long, device=self.glider.device
            ),
            positions=torch.as_tensor(
                pos, dtype=torch.float32, device=self.glider.device
            ),
            attitudes=torch.as_tensor(
                attitude, dtype=torch.float32, device=self.glider.device
            ),
            linear_velocity=velocity_world.unsqueeze(0),
            angular_velocity=torch.zeros(
                (1, 3),
                dtype=torch.float32,
                device=self.glider.device,
            ),
        )

    def record(self, t):
        state = self.glider.state

        pos = state.position[0].detach().cpu()
        vel = state.linear_velocity[0].detach().cpu()
        rates = state.angular_velocity[0].detach().cpu()
        quat = state.attitude[0].detach().cpu()

        roll, pitch, yaw = quaternion_to_euler(quat)

        aero = self.glider.aerodynamics

        Va = float(aero.airspeed[0])
        alpha = float(aero.alpha[0])
        beta = float(aero.beta[0])

        speed = float(torch.linalg.vector_norm(vel))

        thrust = float(self.glider.thrusters.force[0, 0])
        omega_p = float(self.glider.thrusters.velocity[0, 0])

        self.log.append([
            t,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(vel[0]),
            float(vel[1]),
            float(vel[2]),
            speed,
            Va,
            alpha / DEG,
            beta / DEG,
            roll / DEG,
            pitch / DEG,
            yaw / DEG,
            float(rates[0]),
            float(rates[1]),
            float(rates[2]),
            thrust,
            omega_p,
        ])

    def save(self):
        with self.output.open("w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "t",
                "x", "y", "z",
                "vx", "vy", "vz",
                "speed",
                "Va",
                "alpha_deg",
                "beta_deg",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "p_rad_s",
                "q_rad_s",
                "r_rad_s",
                "thrust_N",
                "omega_p_rad_s",
            ])

            writer.writerows(self.log)

        print(f"\nSaved log: {self.output}")

    def summary(self):
        state = self.glider.state
        aero = self.glider.aerodynamics

        pos = state.position[0]
        rates = state.angular_velocity[0]

        Va = float(aero.airspeed[0])
        alpha = float(aero.alpha[0]) / DEG
        beta = float(aero.beta[0]) / DEG

        roll, pitch, yaw = quaternion_to_euler(
            state.attitude[0].detach().cpu()
        )

        print("\n" + "=" * 72)
        print(f"EASYGLIDER {TEST_MODE.upper()} TEST")
        print("=" * 72)
        print(f"Initial altitude : {INITIAL_ALTITUDE:8.3f} m")
        print(f"Final altitude   : {float(pos[2]):8.3f} m")
        print(f"Delta altitude   : {float(pos[2]) - INITIAL_ALTITUDE:+8.3f} m")
        print(f"Airspeed         : {Va:8.3f} m/s")
        print(f"Alpha            : {alpha:8.3f} deg")
        print(f"Beta             : {beta:8.3f} deg")
        print(f"Roll             : {roll / DEG:8.3f} deg")
        print(f"Pitch            : {pitch / DEG:8.3f} deg")
        print(f"Yaw              : {yaw / DEG:8.3f} deg")
        print(f"p, q, r          : {float(rates[0]):+.4f}, "
              f"{float(rates[1]):+.4f}, {float(rates[2]):+.4f} rad/s")

        if TEST_MODE == "powered":
            print(f"Thrust           : "
                  f"{float(self.glider.thrusters.force[0,0]):8.3f} N")
            print(f"Propeller speed  : "
                  f"{float(self.glider.thrusters.velocity[0,0]):8.3f} rad/s")

        print("=" * 72)

    def run(self):
        # Build/initialize PhysX views.
        self.world.reset()

        # Starting the timeline triggers GliderBatch.start().
        self.timeline.play()

        # One rendered step lets the timeline callback initialize the vehicle.
        self.world.step(render=True)

        self.set_initial_velocity()

        start_time = self.world.current_time
        next_log = 0.0

        while simulation_app.is_running():

            self.world.step(render=True)

            # Actual simulated time
            t = self.world.current_time - start_time

            if t >= next_log:
                self.record(t)
                next_log += LOG_PERIOD

            z = float(self.glider.state.position[0, 2])
            Va = float(self.glider.aerodynamics.airspeed[0])

            rates = float(
                torch.linalg.vector_norm(
                    self.glider.state.angular_velocity[0]
                )
            )

            if (
                not math.isfinite(z)
                or not math.isfinite(Va)
                or z < 1.0
                or Va > 50.0
                or rates > 10.0
            ):
                print("\nTest aborted: trajectory diverged.")
                break

            if t >= DURATION:
                break

        self.save()
        self.summary()

        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    GliderFlightTest().run()
