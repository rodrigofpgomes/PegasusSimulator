#!/usr/bin/env python
"""
EasyGlider validation harness  --  Isaac Sim / Pegasus
======================================================

Two-tier protocol.

  TIER 1  BENCH TESTS -- validate the *integration*, not the aerodynamics.
          Gravity and/or aerodynamics are switched off so that each test has a
          closed-form answer.  If any of these fails, no flight test is
          meaningful.
            B1  free fall            -> mass, USD, no spurious rotation
            B2  pure torque impulse  -> Ixx, Iyy, Izz and axis directions
            B3  pure force at CoM    -> positions=None, is_global, m
            B4  static thrust        -> kT

  TIER 2  FLIGHT TESTS -- validate the aerodynamic coefficients.
          State is RE-INITIALISED to trim before every stage, so no stage can
          inherit the divergence of the previous one.  All results are reported
          NON-DIMENSIONALLY (p_hat, r_hat, CL, Cm) because those are invariant
          to airspeed; deg/s readings are not.
            F1  glide          40 s  -> CD0, CL0, L/D
            F2  elevator doublet 8 s -> Cmq, Cmde, Iyy
            F3  aileron step     2 s -> Clda, Clp, roll sign
            F4  rudder step      4 s -> Cndr, Cnb   (with wings-level hold)
            F5  phugoid         40 s -> Cma, CD0

Every stage carries an abort guard.  A stage that trips it is reported as
INVALID rather than silently producing a number.
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ---------------------------------------------------------------------------
import os
import sys
import math
import numpy as np

import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from easyglider_aero import EasyGliderBackend, Geometry, Coeffs, DEG   # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USD_PATH = os.path.abspath(os.path.dirname(__file__)) + "/easyglider_2prim.usda"
PRIM_ROOT = "/World/easyglider"

DT = 1.0 / 250.0
LOG_EVERY = 5                    # -> 50 Hz in the CSV

ALT = 1000.0                     # start high; the Gridroom floor is at z = 0
V0 = 11.5                        # trim airspeed [m/s]
ALPHA0 = 1.61 * DEG              # trim angle of attack [rad]

GRAV = 9.80665
MASS = 1.475                     # body prim mass in the USD
I_USD = np.array([0.197563, 0.1458929, 0.1477])

# thrust line offset: CoM -> propeller hub, in {B}
R_PROP = np.array([0.415, 0.0, -0.0298])
M_ROTOR = 0.005                  # rotor prim mass in the USD (second rigid body)
R_REF = np.array([0.37, 0.0, 0.0552])      # CoM -> AVL reference point


# ---------------------------------------------------------------------------
# Small maths helpers  (FLU body, ENU world, ZYX Euler)
# ---------------------------------------------------------------------------
def quat_to_R(q):
    """Isaac quaternion (w, x, y, z) -> rotation matrix {B} -> {I}."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_from_euler(roll, pitch, yaw):
    """ZYX Euler -> R {B} -> {I}.  pitch is the FLU Euler angle: POSITIVE = NOSE DOWN."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def R_to_quat(R):
    """Rotation matrix -> Isaac quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def euler_from_R(R):
    """R -> (roll, pitch, yaw).  pitch POSITIVE = NOSE DOWN in FLU."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Stage description
# ---------------------------------------------------------------------------
class Stage:
    """One test.  Subclasses override setup / command / finish."""

    name = "stage"
    duration = 1.0
    gravity = True
    aero = True
    prop = True
    reinit = True          # teleport back to trim before starting
    guard = True           # abort on departure
    thrust_line = True     # apply r_prop x F_prop and the propeller reaction torque

    def __init__(self):
        self.rec = []      # list of dicts recorded during the stage
        self.invalid = ""

    def setup(self, app):
        """Called once, after the state has been re-initialised."""

    def command(self, app, tau):
        """tau = time since the stage started.  Returns (de, da, dr, Omega)."""
        return 0.0, 0.0, 0.0, 0.0

    def external(self, app, tau):
        """Extra body-frame (force, torque) added on top of the aero model."""
        return np.zeros(3), np.zeros(3)

    def finish(self, app):
        """Return a list of (label, measured, predicted, unit, tol_frac)."""
        return []

    # -- helpers available to subclasses ------------------------------------
    def window(self, t0, t1):
        return [s for s in self.rec if t0 <= s["tau"] <= t1]

    def mean(self, key, t0, t1):
        w = self.window(t0, t1)
        return float(np.mean([s[key] for s in w])) if w else float("nan")


# ===========================================================================
# TIER 1 -- bench tests
# ===========================================================================
class B1FreeFall(Stage):
    name = "B1 free fall"
    duration = 2.0
    gravity = True
    aero = False
    prop = False
    guard = False

    def finish(self, app):
        w = self.window(0.5, 2.0)
        t = np.array([s["tau"] for s in w])
        vz = np.array([s["vz"] for s in w])
        az = float(np.polyfit(t, vz, 1)[0])
        wmax = max(s["wnorm"] for s in w)
        return [
            ("vertical acceleration", az, -GRAV, "m/s^2", 0.01),
            ("spurious body rate", math.degrees(wmax), 0.0, "deg/s", None),
        ]


class B2Inertia(Stage):
    """Pure torque about one body axis, no gravity, no aero -> I = tau / wdot."""

    duration = 2.0
    gravity = False
    aero = False
    prop = False
    guard = False
    TAU = 0.05        # N.m -- small enough to stay in the small-angle regime

    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        self.name = "B2 inertia %s" % "xyz"[axis]

    def external(self, app, tau):
        t = np.zeros(3)
        t[self.axis] = self.TAU
        return np.zeros(3), t

    def finish(self, app):
        w = self.window(0.2, 2.0)
        t = np.array([s["tau"] for s in w])
        om = np.array([s["omega"][self.axis] for s in w])
        wdot = float(np.polyfit(t, om, 1)[0])
        I = self.TAU / wdot if abs(wdot) > 1e-9 else float("nan")
        # cross-axis leakage: the other two components must stay near zero
        others = [j for j in range(3) if j != self.axis]
        leak = max(max(abs(s["omega"][j]) for j in others) for s in w)
        # The articulation is a TWO-body system.  A pure torque on the body
        # rotates body+rotor about the composite CoM, so the effective inertia
        # is the body value plus the rotor's parallel-axis term.  Ignoring this
        # makes Iyy and Izz look 0.6 % too large.
        r = R_PROP
        add = [M_ROTOR * (r[1] ** 2 + r[2] ** 2),
               M_ROTOR * (r[0] ** 2 + r[2] ** 2),
               M_ROTOR * (r[0] ** 2 + r[1] ** 2)]
        I_pred = I_USD[self.axis] + add[self.axis]
        return [
            ("I%s%s" % ("xyz"[self.axis], "xyz"[self.axis]),
             I, I_pred, "kg.m^2", 0.02),
            ("cross-axis leakage", math.degrees(leak), 0.0, "deg/s", None),
        ]


class B3ForceAtCoM(Stage):
    """Pure body-frame force, no gravity, no aero.

    Validates three things at once:
      - the magnitude of the acceleration  -> mass
      - its direction in world axes        -> is_global / frame handling
      - the ABSENCE of angular acceleration -> positions=None really is the CoM
    """

    name = "B3 force at CoM"
    duration = 2.0
    gravity = False
    aero = False
    prop = False
    guard = False
    F = np.array([2.0, 0.0, 3.0])        # body frame, deliberately off-axis

    def external(self, app, tau):
        return self.F.copy(), np.zeros(3)

    def finish(self, app):
        w = self.window(0.2, 2.0)
        t = np.array([s["tau"] for s in w])
        out = []
        # Two subtleties, both confirmed numerically against run 3:
        #  - the whole ARTICULATION accelerates, so the relevant mass is the
        #    system mass (body 1.475 + rotor 0.005), not the body prim mass;
        #  - the induced rotation (3.44 deg/s) turns the body axes by ~1.26 deg
        #    on average over the fit window, so the world-frame prediction has
        #    to use the MEAN attitude over the window, not R(t=0).
        R = np.mean(np.array([s["R"] for s in w]), axis=0)
        a_pred = R @ self.F / (MASS + M_ROTOR)
        for i, ax in enumerate("xyz"):
            v = np.array([s["v_world"][i] for s in w])
            a = float(np.polyfit(t, v, 1)[0])
            out.append(("world accel %s" % ax, a, float(a_pred[i]), "m/s^2", 0.02))
        # A force at the BODY CoM is not at the SYSTEM CoM: the rotor shifts it
        # by m_r*|r|/(m+m_r) ~ 1.4 mm, which produces a small but REAL rotation.
        # Predicting exactly zero here is wrong; predict the composite value.
        d_sys = M_ROTOR * R_PROP / (MASS + M_ROTOR)
        I_sys = np.array(I_USD) + M_ROTOR * float(np.dot(R_PROP, R_PROP))
        wdot = np.cross(-d_sys, self.F) / I_sys
        w_pred = float(np.linalg.norm(wdot)) * self.duration
        wmax = max(s["wnorm"] for s in w)
        out.append(("induced body rate", math.degrees(wmax),
                    math.degrees(w_pred), "deg/s", 0.25))
        return out


class B4Thrust(Stage):
    """Propeller only, no gravity, no aero -> T = m * a."""

    name = "B4 static thrust"
    duration = 2.0
    gravity = False
    aero = False
    prop = True
    guard = False
    thrust_line = False   # a pure axial force: otherwise the -0.365 N.m couple
                          # pitches the aircraft over and du/dt measures nothing
    OMEGA = 3500.0

    def command(self, app, tau):
        return 0.0, 0.0, 0.0, self.OMEGA

    def finish(self, app):
        w = self.window(0.2, 2.0)
        t = np.array([s["tau"] for s in w])
        vb = np.array([s["u"] for s in w])
        a = float(np.polyfit(t, vb, 1)[0])
        T = MASS * a
        T_pred = app.be.kT * self.OMEGA ** 2
        return [
            ("static thrust", T, T_pred, "N", 0.02),
            ("T/W", T / (MASS * GRAV), T_pred / (MASS * GRAV), "-", 0.05),
        ]


# ===========================================================================
# TIER 2 -- flight tests
# ===========================================================================
class F1Glide(Stage):
    """40 s is ~6 phugoid periods, enough to average the oscillation out."""

    name = "F1 glide"
    duration = 40.0

    def finish(self, app):
        w = self.window(5.0, 40.0)
        t = np.array([s["tau"] for s in w])
        z = np.array([s["z"] for s in w])
        sink = -float(np.polyfit(t, z, 1)[0])
        Va = self.mean("Va", 5.0, 40.0)
        ld = math.sqrt(max(Va * Va - sink * sink, 0.0)) / sink if sink > 1e-6 else float("nan")
        return [
            ("mean airspeed", Va, 11.10, "m/s", 0.06),
            ("sink rate", sink, 0.80, "m/s", 0.15),
            ("L/D", ld, 13.90, "-", 0.12),
            ("mean alpha", self.mean("alpha", 5.0, 40.0) / DEG, 1.61, "deg", None),
        ]


class F2ElevatorDoublet(Stage):
    name = "F2 elevator doublet"
    duration = 8.0
    AMP = 4.0 * DEG   # the short period has a 0.125 s half-life: hit it hard
    T1 = 0.15         # and briefly, or the ring-down is over before we look

    def command(self, app, tau):
        if tau < self.T1:
            de = self.AMP
        elif tau < 2 * self.T1:
            de = -self.AMP
        else:
            de = 0.0
        return de, 0.0, 0.0, 0.0

    def finish(self, app):
        # The short period is NOT identifiable from this signal.  At zeta =
        # 0.694 the overshoot is exp(-pi*z/sqrt(1-z^2)) = 4.8 % of the first
        # peak, while the phugoid rides on q with a comparable residual
        # amplitude, so q never crosses back through zero: it slides straight
        # into the slow drift.  All three estimators were checked on synthetic
        # signals with the true modal content -- extremum pair, zero crossings
        # and a doubly integrated ODE fit -- and they return nan or errors of
        # 25 to 90 %.  So (wn, zeta) is reported for INFORMATION ONLY, and the
        # graded quantity is the one that is actually well posed: the pitch
        # acceleration in the instant the elevator moves.  That validates
        # Cm_delta_e and Iyy end to end, before alpha has had time to react.
        w = self.window(0.0, 0.02)
        t = np.array([s["tau"] for s in w])
        q = np.array([s["omega"][1] for s in w])
        qd = float(np.polyfit(t, q, 1)[0]) if len(t) >= 3 else float("nan")
        s0 = self.rec[0]
        Va, a0 = s0["Va"], s0["alpha"]
        v_b = np.array([Va * math.cos(a0), 0.0, -Va * math.sin(a0)])
        _, T_cmd = app.be.forces_moments(v_b, np.zeros(3),
                                         (self.AMP, 0.0, 0.0), 0.0)
        I_yy = float(I_USD[1]) + M_ROTOR * float(np.dot(R_PROP, R_PROP))
        qd_pred = float(T_cmd[1]) / I_yy
        wl = self.window(2 * self.T1, 2.0)
        wn, zeta = _fit_ringdown(np.array([s["tau"] for s in wl]),
                                 np.array([s["omega"][1] for s in wl]))
        return [
            ("initial pitch accel", qd, qd_pred, "rad/s^2", 0.10),
            ("short-period wn", wn, None, "rad/s", None),
            ("short-period zeta", zeta, None, "-", None),
        ]


class F3AileronStep(Stage):
    """Only 2 s: the roll mode settles in 0.105 s, and 2 s is far too short for
    the spiral (t_double = 5.3 s) to contaminate the reading."""

    name = "F3 aileron step"
    duration = 2.0
    AMP = 5.0 * DEG

    def command(self, app, tau):
        return 0.0, self.AMP, 0.0, 0.0

    def finish(self, app):
        C, G = app.be.C, app.be.G
        p_hat_pred = -C.Clda * self.AMP / C.Clp
        p_hat = self.mean("p_hat", 0.5, 1.5)
        return [
            ("p_hat (roll)", p_hat, p_hat_pred, "-", 0.08),
            ("roll sign (must be > 0)", math.copysign(1.0, p_hat), 1.0, "-", None),
            ("bank angle at 2 s", self.rec[-1]["roll"] / DEG, None, "deg", None),
        ]


class F4RudderStep(Stage):
    """Steady sideslip needs the wings held level, otherwise the aircraft just
    rolls off into a turn and beta is meaningless.  A proportional-derivative
    hold on the ailerons does the job; it does not touch the yaw axis, so
    Cndr / Cnb are still measured open-loop."""

    name = "F4 rudder step"
    duration = 6.0
    AMP = 5.0 * DEG
    KP, KD = 3.0, 0.6

    def command(self, app, tau):
        s = app.last
        da = -self.KP * s["roll"] - self.KD * s["omega"][0]
        da = float(np.clip(da, -0.35, 0.35))
        return 0.0, da, self.AMP, 0.0

    def finish(self, app):
        beta = self.mean("beta", 3.0, 6.0) / DEG
        roll = self.mean("roll", 3.0, 6.0) / DEG
        return [
            ("steady sideslip", beta, 1.88, "deg", 0.15),
            ("residual bank (must be small)", roll, 0.0, "deg", None),
        ]


class F5Phugoid(Stage):
    name = "F5 phugoid"
    duration = 40.0

    def setup(self, app):
        # kick the phugoid by launching 15 % fast; it is a speed/altitude mode
        app.reinit(V=V0 * 1.15)

    def finish(self, app):
        w = self.window(2.0, 40.0)
        t = np.array([s["tau"] for s in w])
        Va = np.array([s["Va"] for s in w])
        wn, zeta = _fit_second_order(t, Va - float(np.mean(Va)))
        return [
            ("phugoid period", 2 * math.pi / wn if wn > 1e-6 else float("nan"),
             6.71, "s", 0.20),
            ("phugoid zeta", zeta, 0.028, "-", None),
        ]


# ---------------------------------------------------------------------------
def _fit_ringdown(t, y):
    """Estimate (wn, zeta) from the FIRST pair of opposite-sign extrema of a
    ring-down.  Valid up to zeta ~ 0.8, where there are too few cycles for a
    zero-crossing fit.  Successive opposite-sign extrema are half a damped
    period apart, and ln|a0/a1| = pi*zeta/sqrt(1-zeta^2)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) < 10:
        return float("nan"), float("nan")
    ext = [(t[i], y[i]) for i in range(1, len(y) - 1)
           if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0.0]
    if len(ext) < 2:
        return float("nan"), float("nan")
    t0, a0 = ext[0]
    nxt = next(((tt, aa) for tt, aa in ext[1:] if aa * a0 < 0.0), None)
    if nxt is None or abs(a0) < 1e-9 or abs(nxt[1]) < 1e-9:
        return float("nan"), float("nan")
    t1, a1 = nxt
    delta = math.log(abs(a0 / a1))
    zeta = float(np.clip(delta / math.sqrt(math.pi ** 2 + delta ** 2), 0.0, 0.99))
    wd = math.pi / max(t1 - t0, 1e-6)
    return wd / math.sqrt(max(1.0 - zeta ** 2, 1e-6)), zeta


def _fit_second_order(t, y):
    """Estimate (wn, zeta) of a damped sinusoid from zero crossings and the
    decay of successive peaks.  Deliberately dependency-free."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) < 10:
        return float("nan"), float("nan")
    # zero crossings -> damped period
    sgn = np.sign(y)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross) < 3:
        return float("nan"), float("nan")
    tc = t[cross]
    Td = 2.0 * float(np.mean(np.diff(tc)))
    wd = 2.0 * math.pi / Td
    # peak decay -> logarithmic decrement
    peaks = []
    for i in range(1, len(y) - 1):
        if abs(y[i]) > abs(y[i - 1]) and abs(y[i]) > abs(y[i + 1]):
            peaks.append((t[i], abs(y[i])))
    if len(peaks) < 2:
        return wd, 0.0
    t0, a0 = peaks[0]
    t1, a1 = peaks[-1]
    n = max(1.0, (t1 - t0) / (Td / 2.0))
    if a1 <= 0.0 or a0 <= 0.0:
        return wd, 0.0
    delta = math.log(a0 / a1) / n
    zeta = delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)
    zeta = float(np.clip(zeta, -0.99, 0.99))
    wn = wd / math.sqrt(max(1.0 - zeta ** 2, 1e-6))
    return wn, zeta


# ===========================================================================
# Application
# ===========================================================================
class EasyGliderValidator:

    def __init__(self, log_path="easyglider_validation2.csv"):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        ws = dict(self.pg._world_settings)
        ws["physics_dt"] = DT
        ws["rendering_dt"] = DT * 4
        ws["device"] = "cpu"
        self.pg._world = World(**ws)
        self.world = self.pg.world

        # bare stage: no floor to crash into, no clutter
        prim_utils.create_prim(
            "/World/Light/DomeLight", "DomeLight",
            attributes={"inputs:intensity": 3e3, "inputs:color": (1.0, 1.0, 1.0)},
        )
        add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_ROOT)

        self.body = RigidPrimView(PRIM_ROOT + "/body", name="eg_body")
        self.art = ArticulationView(PRIM_ROOT, name="eg_art")
        self.world.scene.add(self.body)
        self.world.scene.add(self.art)

        # -- aerodynamic backend ------------------------------------------
        geom = Geometry()
        geom.r_ref = R_REF.copy()
        self.be = EasyGliderBackend(geom=geom, coef=Coeffs())

        self.log_path = log_path
        self.rows = []
        self.last = {"roll": 0.0, "omega": np.zeros(3)}

        self.stages = [
            B1FreeFall(),
            B2Inertia(0), B2Inertia(1), B2Inertia(2),
            B3ForceAtCoM(),
            B4Thrust(),
            F1Glide(),
            F2ElevatorDoublet(),
            F3AileronStep(),
            F4RudderStep(),
            F5Phugoid(),
        ]
        self.idx = -1
        self.stage = None
        self.t = 0.0
        self.tau = 0.0
        self.nstep = 0
        self.done = False

        self.world.reset()
        self.world.add_physics_callback("eg", self._on_physics_step)

    # -- state handling ----------------------------------------------------
    def set_gravity(self, on):
        self.world.get_physics_context().set_gravity(-GRAV if on else 0.0)

    def reinit(self, V=V0, alpha=ALPHA0, gamma=0.0, alt=ALT, yaw=0.0):
        """Teleport to a clean initial condition.

        pitch_euler = -(alpha + gamma) because in FLU a positive rotation about
        +y (the LEFT wing) pitches the nose DOWN.
        """
        pitch = -(alpha + gamma)
        R = R_from_euler(0.0, pitch, yaw)
        pos = np.array([[0.0, 0.0, alt]])
        quat = R_to_quat(R).reshape(1, 4)
        v_world = np.array([[V * math.cos(gamma) * math.cos(yaw),
                             V * math.cos(gamma) * math.sin(yaw),
                             V * math.sin(gamma)]])
        self.body.set_world_poses(positions=pos, orientations=quat)
        self.body.set_velocities(np.concatenate([v_world, np.zeros((1, 3))], axis=1))
        try:
            n = self.art.num_dof
            self.art.set_joint_positions(np.zeros((1, n)))
            self.art.set_joint_velocities(np.zeros((1, n)))
        except Exception:
            pass

    def next_stage(self):
        if self.stage is not None:
            self.stage.results = self.stage.finish(self)
        self.idx += 1
        if self.idx >= len(self.stages):
            self.done = True
            return
        self.stage = self.stages[self.idx]
        self.tau = 0.0
        self.set_gravity(self.stage.gravity)
        if self.stage.reinit:
            self.reinit()
        self.stage.setup(self)
        carb.log_warn("[eg] stage %s" % self.stage.name)

    # -- physics step ------------------------------------------------------
    def _on_physics_step(self, step_size):
        if self.done:
            return
        if self.stage is None or self.tau >= self.stage.duration:
            self.next_stage()
            if self.done:
                return

        st = self.stage

        pos, quat = self.body.get_world_poses()
        vel = self.body.get_velocities()
        p_w = np.asarray(pos[0], dtype=float)
        q_w = np.asarray(quat[0], dtype=float)
        v_w = np.asarray(vel[0][:3], dtype=float)
        w_w = np.asarray(vel[0][3:], dtype=float)

        R = quat_to_R(q_w)
        v_b = R.T @ v_w                       # velocity of the CoM in {B}
        w_b = R.T @ w_w
        roll, pitch, yaw = euler_from_R(R)

        self.last = {"roll": roll, "omega": w_b}

        de, da, dr, Om = st.command(self, self.tau)

        if st.aero:
            F, T = self.be.forces_moments(v_b, w_b, (de, da, dr), Om if st.prop else 0.0)
        else:
            F, T = np.zeros(3), np.zeros(3)
            if st.prop and Om > 0.0:
                Fp, Qp = self.be.propeller(Om, v_b[0])
                F = np.array([Fp, 0.0, 0.0])
                T = (np.array([self.be.s_p * Qp, 0.0, 0.0])
                     if st.thrust_line else np.zeros(3))

        # the thrust line passes 29.8 mm BELOW the CoM -> nose-up couple
        if st.prop and Om > 0.0 and st.thrust_line:
            Fp_only, _ = self.be.propeller(Om, v_b[0])
            T = T + np.cross(R_PROP, np.array([Fp_only, 0.0, 0.0]))

        Fe, Te = st.external(self, self.tau)
        F = F + Fe
        T = T + Te

        self.body.apply_forces_and_torques_at_pos(
            forces=F.reshape(1, 3), torques=T.reshape(1, 3),
            positions=None, is_global=False)

        # -- record --------------------------------------------------------
        Va, Vs, alpha, beta = self.be.airdata(v_b)
        G = self.be.G
        sample = {
            "tau": self.tau, "t": self.t,
            "x": p_w[0], "y": p_w[1], "z": p_w[2],
            "vz": v_w[2], "v_world": v_w, "u": v_b[0],
            "Va": Va, "alpha": alpha, "beta": beta,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "omega": w_b, "wnorm": float(np.linalg.norm(w_b)),
            "p_hat": w_b[0] * G.b / (2.0 * Vs),
            "r_hat": w_b[2] * G.b / (2.0 * Vs),
            "R": R,
        }
        st.rec.append(sample)

        if self.nstep % LOG_EVERY == 0:
            self.rows.append([
                "%.4f" % self.t, st.name, "%.3f" % p_w[0], "%.3f" % p_w[1],
                "%.3f" % p_w[2], "%.4f" % Va, "%.4f" % (alpha / DEG),
                "%.4f" % (beta / DEG), "%.3f" % (roll / DEG),
                "%.3f" % (-pitch / DEG), "%.3f" % (yaw / DEG),
                "%.4f" % w_b[0], "%.4f" % w_b[1], "%.4f" % w_b[2],
                "%.5f" % sample["p_hat"], "%.5f" % sample["r_hat"],
                "%.3f" % (de / DEG), "%.3f" % (da / DEG), "%.3f" % (dr / DEG),
                "%.1f" % Om,
                "%.4f" % F[0], "%.4f" % F[1], "%.4f" % F[2],
                "%.5f" % T[0], "%.5f" % T[1], "%.5f" % T[2],
            ])

        # -- abort guard ---------------------------------------------------
        if st.guard and not st.invalid:
            if abs(alpha) > 30 * DEG:
                st.invalid = "alpha exceeded 30 deg at tau = %.2f s" % self.tau
            elif sample["wnorm"] > 5.0:
                st.invalid = "body rate exceeded 5 rad/s at tau = %.2f s" % self.tau
            elif p_w[2] < 100.0:
                st.invalid = "altitude below 100 m at tau = %.2f s" % self.tau
            elif Va > 40.0:
                st.invalid = "airspeed exceeded 40 m/s at tau = %.2f s" % self.tau
            if st.invalid:
                carb.log_warn("[eg] %s INVALID: %s" % (st.name, st.invalid))
                self.tau = st.duration      # cut the stage short

        self.tau += step_size
        self.t += step_size
        self.nstep += 1

    # -- reporting ---------------------------------------------------------
    def save(self):
        import csv
        head = ["t", "stage", "x", "y", "z", "Va", "alpha_deg", "beta_deg",
                "roll_deg", "pitchup_deg", "yaw_deg", "p", "q", "r",
                "p_hat", "r_hat", "de_deg", "da_deg", "dr_deg", "Omega",
                "Fx", "Fy", "Fz", "taux", "tauy", "tauz"]
        with open(self.log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(self.rows)
        print("\nlog written to %s  (%d rows)" % (self.log_path, len(self.rows)))

    def summary(self):
        line = "=" * 78
        print("\n" + line)
        print("EASYGLIDER VALIDATION SUMMARY")
        print(line)
        print("%-30s %12s %12s %8s  %s" % ("quantity", "measured", "predicted",
                                           "unit", "verdict"))
        print("-" * 78)
        npass = nfail = 0
        for st in self.stages:
            res = getattr(st, "results", None)
            print("" if st is self.stages[0] else "")
            if st.invalid:
                print("%-30s %s" % (st.name, "** INVALID: " + st.invalid))
                nfail += 1
                continue
            print("%s" % st.name)
            for label, meas, pred, unit, tol in (res or []):
                if pred is None:
                    print("  %-28s %12.4f %12s %8s" % (label, meas, "--", unit))
                    continue
                if tol is None:
                    ok = abs(meas - pred) < 1.0
                else:
                    denom = abs(pred) if abs(pred) > 1e-9 else 1.0
                    ok = abs(meas - pred) / denom <= tol
                npass += int(ok)
                nfail += int(not ok)
                print("  %-28s %12.4f %12.4f %8s  %s"
                      % (label, meas, pred, unit, "PASS" if ok else "** FAIL"))
        print("-" * 78)
        print("%d passed, %d failed" % (npass, nfail))
        print(line)

    # -- main loop ---------------------------------------------------------
    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.done:
            self.world.step(render=True)
        self.save()
        self.summary()
        self.timeline.stop()
        simulation_app.close()


def main():
    app = EasyGliderValidator()
    app.run()


if __name__ == "__main__":
    main()
