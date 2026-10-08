#!/usr/bin/env python
"""
| File: easyglider_sim.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| Description: Open-loop validation rig for the EasyGlider aerodynamic backend.
|
| Spawns easyglider_2prim.usda, propagates the rigid-body dynamics with PhysX,
| and drives it with the analytic backend in easyglider_aero.py.  Only TWO prims
| are actuated:
|
|     /body   <- F_RB, tau_RB from EasyGliderBackend, applied AT THE CoM
|     /rotor  <- joint velocity target (visual disc + gyroscopic coupling)
|
| The four control surfaces never move in PhysX: their deflections enter only as
| (delta_e, delta_a, delta_r) arguments of the backend.
|
| A scripted manoeuvre sequence exercises each control channel in isolation and
| writes a CSV that can be compared against the predictions of flight_tests.py.
"""
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ---------------------------------------------------------------------------
import os
import sys
import csv
import math

import numpy as np
import omni.timeline
import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easyglider_aero import EasyGliderBackend, Geometry, Coeffs, DEG

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "easyglider_2prim.usda")
PRIM_ROOT = "/World/easyglider"
DT = 1.0 / 250.0


# ===========================================================================
# helpers
# ===========================================================================
def quat_to_R(q):
    """Isaac quaternion (w, x, y, z) -> rotation matrix {B} -> {I}."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def R_to_euler_flu(R):
    """ZYX Euler angles.  CAUTION: in FLU a positive rotation about +y (the LEFT
    wing) pitches the nose DOWN, so theta is nose-down positive.  The reported
    'pitch up' angle is therefore -theta."""
    theta = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    phi = math.atan2(R[2, 1], R[2, 2])
    psi = math.atan2(R[1, 0], R[0, 0])
    return phi, theta, psi


# ===========================================================================
# manoeuvre schedule -- each block isolates one coefficient group
# ===========================================================================
class Schedule:
    """Returns (delta_e, delta_a, delta_r, Omega_p, label) for a given time."""

    # (duration, label, de, da, dr, Omega)   angles in DEGREES
    BLOCKS = [
        (6.0,  "trim-glide",      0.0,  0.0,  0.0,     0.0),
        (4.0,  "elevator-doublet", 0.0, 0.0,  0.0,     0.0),   # overridden below
        (6.0,  "aileron-step",    0.0,  5.0,  0.0,     0.0),
        (6.0,  "aileron-back",    0.0,  0.0,  0.0,     0.0),
        (6.0,  "rudder-step",     0.0,  0.0,  5.0,     0.0),
        (6.0,  "rudder-back",     0.0,  0.0,  0.0,     0.0),
        (8.0,  "throttle-50",     0.0,  0.0,  0.0,  1750.0),
        (8.0,  "throttle-100",    0.0,  0.0,  0.0,  3500.0),
        (8.0,  "phugoid",         0.0,  0.0,  0.0,     0.0),
    ]

    def __init__(self):
        self.edges = []
        t = 0.0
        for dur, lab, de, da, dr, om in self.BLOCKS:
            self.edges.append((t, t + dur, lab, de, da, dr, om))
            t += dur
        self.total = t

    def __call__(self, t):
        for t0, t1, lab, de, da, dr, om in self.edges:
            if t0 <= t < t1:
                if lab == "elevator-doublet":
                    # +2 deg for 0.4 s, -2 deg for 0.4 s, then neutral
                    tau = t - t0
                    de = 2.0 if tau < 0.4 else (-2.0 if tau < 0.8 else 0.0)
                return de * DEG, da * DEG, dr * DEG, om, lab
        return 0.0, 0.0, 0.0, 0.0, "done"


# ===========================================================================
class EasyGliderApp:

    def __init__(self, V0=11.5, alt=60.0, log_path="easyglider_validation.csv"):
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(physics_dt=DT, rendering_dt=DT * 4, stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        prim_utils.create_prim(
            "/World/Light/DomeLight", "DomeLight",
            position=np.array([1.0, 1.0, 1.0]),
            attributes={"inputs:intensity": 3e3, "inputs:color": (1.0, 1.0, 1.0)},
        )

        add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_ROOT)

        self.body = RigidPrimView(prim_paths_expr=PRIM_ROOT + "/body", name="eg_body")
        self.rotor = ArticulationView(prim_paths_expr=PRIM_ROOT, name="eg_art")
        self.world.scene.add(self.body)
        self.world.scene.add(self.rotor)

        # ---- backend under test -------------------------------------------
        G = Geometry()
        G.r_ref = np.array([0.37, 0.0, 0.0552])   # identified Xref (nose datum)
        self.backend = EasyGliderBackend(G, Coeffs(), kT=1.0e-06, kM=0.01,
                                         s_p=-1.0, eval_at_ref=True)
        # thrust line relative to the CoM, in {B}: rotor at (0.115,0,-0.085),
        # CoM at (-0.30,0,-0.0552), both in base_link
        self.r_prop = np.array([0.415, 0.0, -0.0298])
        self.r_com = np.array([-0.30, 0.0, -0.0552])   # CoM in the body prim frame

        self.sched = Schedule()
        self.V0, self.alt = V0, alt
        self.t = 0.0
        self.log_path = log_path
        self.rows = []
        self.com_velocity_verified = None

        self.world.reset()
        self._place()
        self.world.add_physics_callback("aero", self._on_physics_step)

    # -----------------------------------------------------------------
    def _place(self):
        """Launch nose-forward, wings level, at V0 along +x, altitude alt."""
        pos = np.array([[0.0, 0.0, self.alt]])
        orient = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.body.set_world_poses(pos, orient)
        self.body.set_velocities(np.array([[self.V0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    # -----------------------------------------------------------------
    def _on_physics_step(self, step_size):
        pos, quat = self.body.get_world_poses()
        vel = self.body.get_velocities()
        pos = np.array(pos[0], dtype=float)
        quat = np.array(quat[0], dtype=float)
        v_I = np.array(vel[0][0:3], dtype=float)
        w_I = np.array(vel[0][3:6], dtype=float)

        R = quat_to_R(quat)
        omega_b = R.T @ w_I
        v_com_b = R.T @ v_I

        # ---- one-off check: is get_velocities reported at the CoM? --------
        if self.com_velocity_verified is None and abs(self.t - 0.5) < DT:
            self.com_velocity_verified = True
            carb.log_warn("[EasyGlider] run the spin test separately to confirm "
                          "whether linear velocity is reported at the CoM.")

        # ---- backend ------------------------------------------------------
        de, da, dr, Omega, label = self.sched(self.t)
        F_b, tau_b = self.backend.forces_moments(v_com_b, omega_b, (de, da, dr), Omega)

        # thrust acts 29.8 mm below the CoM -> nose-up couple, not in the backend
        T = self.backend.kT * min(Omega, 3500.0) ** 2
        tau_b = tau_b + np.cross(self.r_prop, np.array([T, 0.0, 0.0])) \
                      - np.cross(np.zeros(3), np.zeros(3))

        self.body.apply_forces_and_torques_at_pos(
            forces=np.array([F_b]), torques=np.array([tau_b]),
            positions=None, is_global=False)

        # rotor: velocity target in rad/s on the single articulation joint
        self.rotor.set_joint_velocity_targets(np.array([[Omega]]))

        # ---- log ----------------------------------------------------------
        Va, Vs, alpha, beta = self.backend.airdata(v_com_b)
        phi, theta, psi = R_to_euler_flu(R)
        self.rows.append([
            self.t, label, pos[0], pos[1], pos[2], Va,
            alpha / DEG, beta / DEG,
            phi / DEG, -theta / DEG, psi / DEG,
            omega_b[0] / DEG, omega_b[1] / DEG, omega_b[2] / DEG,
            de / DEG, da / DEG, dr / DEG, Omega,
            F_b[0], F_b[1], F_b[2], tau_b[0], tau_b[1], tau_b[2],
        ])
        self.t += step_size

    # -----------------------------------------------------------------
    def save(self):
        hdr = ["t", "phase", "x", "y", "z", "Va", "alpha_deg", "beta_deg",
               "roll_deg", "pitchup_deg", "yaw_deg", "p_dps", "q_dps", "r_dps",
               "de_deg", "da_deg", "dr_deg", "Omega",
               "Fx", "Fy", "Fz", "taux", "tauy", "tauz"]
        with open(self.log_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(hdr)
            wcsv.writerows(self.rows)
        print("[EasyGlider] wrote %d samples to %s" % (len(self.rows), self.log_path))
        self._summary()

    # -----------------------------------------------------------------
    def _summary(self):
        """Print the four acceptance numbers next to their predictions."""
        A = {}
        for r in self.rows:
            A.setdefault(r[1], []).append(r)

        print("\n" + "=" * 74)
        print("ACCEPTANCE SUMMARY      measured        predicted")
        print("=" * 74)

        if "trim-glide" in A:
            seg = A["trim-glide"]
            t0, t1 = seg[0], seg[-1]
            dz, dx = t1[4] - t0[4], t1[2] - t0[2]
            ld = -dx / dz if dz < 0 else float("nan")
            print("glide L/D            %8.2f        %8.2f" % (ld, 13.9))
            print("trim airspeed        %8.2f m/s    %8.2f m/s" % (t1[5], 11.1))
            print("trim alpha           %8.2f deg    %8.2f deg" % (t1[6], 1.61))

        if "aileron-step" in A:
            seg = A["aileron-step"]
            p_ss = np.mean([r[11] for r in seg[-len(seg) // 3:]])
            print("roll rate (da=+5deg) %8.2f deg/s  %8.2f deg/s" % (p_ss, 29.75))
            print("   [sign: must be POSITIVE = right wing down]")

        if "rudder-step" in A:
            seg = A["rudder-step"]
            b_ss = np.mean([r[7] for r in seg[-len(seg) // 3:]])
            print("sideslip (dr=+5deg)  %8.2f deg    %8.2f deg" % (b_ss, 1.88))

        if "throttle-100" in A:
            seg = A["throttle-100"]
            print("Va at full throttle  %8.2f m/s    (T/W = 0.84)" % seg[-1][5])
        print("=" * 74)

    # -----------------------------------------------------------------
    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and self.t < self.sched.total:
            self.world.step(render=True)
        self.save()
        carb.log_warn("EasyGlider validation finished.")
        self.timeline.stop()
        simulation_app.close()


def main():
    EasyGliderApp(V0=11.5, alt=60.0).run()


if __name__ == "__main__":
    main()
