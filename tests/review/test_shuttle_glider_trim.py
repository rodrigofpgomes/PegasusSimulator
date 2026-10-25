"""
| File: test_shuttle_glider_trim.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: The Shuttle-Glider reset trim is an equilibrium of the six-degree-of-freedom model of the trim analysis.
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from offline_runtime import enable

enable()
ROOT = Path(__file__).resolve().parents[2]
from pegasus.simulator.logic.dynamics.glider_aerodynamics import GliderAerodynamicsBatch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "shuttle_glider_trim_analysis", ROOT / "examples/rl/diagnostics/shuttle_glider_trim.py")
analysis = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_spec.loader.exec_module(analysis)
sys.argv = _argv
_spec = importlib.util.spec_from_file_location(
    "shuttle_glider_trim", ROOT / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/shuttle_glider_trim.py")
trim_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trim_mod)


def make_trim(veh, table="limits"):
    aero = GliderAerodynamicsBatch({"geometry": {"r_com": tuple(veh["r_com_aero"])}}, 1, "cpu")
    return aero, trim_mod.ShuttleGliderTrim(
        mass=veh["m"], gravity=analysis.G0, rotor_positions=veh["r_rotor"], puller_position=veh["r_puller"],
        reference_position=veh["r_bodycom"] + veh["r_com_aero"], rotor_constant=veh["kf"], moment_constant=veh["km"],
        rot_dir=veh["rot_dir"], puller_constant=veh["kf_p"], puller_moment_constant=veh["km_p"],
        puller_rot_dir=veh["rot_dir_p"], min_omega=[0.0] * 5, max_omega=list(veh["w_max"]) + [veh["w_max_p"]],
        aerodynamics=aero, table=table)


class ShuttleGliderTrimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.veh = analysis.load_vehicle()
        cls.aero, cls.trim = make_trim(cls.veh)

    def residual(self, out, i, speed, bank=0.0, turn_rate=0.0):
        T = out["omega"][i, :4].double().numpy() ** 2 * self.veh["kf"]
        Tp = float(out["omega"][i, 4]) ** 2 * self.veh["kf_p"]
        x = np.concatenate([[float(out["alpha"][i]), 0.0, bank], T, [Tp], out["surfaces"][i].double().numpy()])
        return analysis.residual(x, self.veh, speed, 0.0, turn_rate, surfaces=True)

    def test_aero_wrench_matches_batched_model(self):
        speed = torch.tensor([3.0, 9.0, 14.0])
        alpha = torch.tensor([0.02, 0.2, -0.05])
        surfaces = torch.tensor([[0.1, -0.05, 0.02], [-0.2, 0.1, -0.1], [0.0, 0.0, 0.0]])
        F, M = self.trim.aero_wrench(speed, alpha, surfaces)
        r_bc = torch.tensor(self.veh["r_bodycom"], dtype=torch.float32)
        for i in range(3):
            v = speed[i] * torch.stack((torch.cos(alpha[i]), torch.tensor(0.0), -torch.sin(alpha[i])))
            Fb, Mb = self.aero.forces_moments(v.unsqueeze(0), torch.zeros(1, 3), surfaces[i:i + 1])
            torch.testing.assert_close(F[i], Fb[0], atol=1e-4, rtol=0)
            torch.testing.assert_close(M[i], Mb[0] + torch.cross(r_bc, Fb[0], dim=0), atol=1e-4, rtol=0)

    def test_straight_flight_resets_are_equilibria(self):
        speed = torch.arange(0.0, 17.0, 0.5)
        alloc = make_trim(self.veh, "alloc")[1]
        for trim, use_surfaces in ((self.trim, False), (self.trim, True), (alloc, True)):
            out = trim.solve(speed, use_surfaces=use_surfaces)
            for i, V in enumerate(speed.tolist()):
                r = self.residual(out, i, V)
                # The aileron side force (CY_da, below 0.02 N) is the only term without a balancing actuator.
                self.assertLess(abs(r[1]), 0.02)
                # Rotors stopped at their lower limit (rear pair from 13 m/s): the closed-form thrust (a few mN below
                # zero) is clamped, which leaves that residual.
                clamped = float(out["thrust"][i].clamp(max=0.0).abs().sum())
                self.assertLess(np.abs(np.delete(r, 1)).max(), 2e-3 + 2.0 * clamped, (use_surfaces, V, r))
            T = out["thrust"]
            self.assertTrue(torch.all(T >= trim_mod.TRIM_ROTOR_RESERVE - trim_mod.TRIM_THRUST_TOLERANCE))
            if not use_surfaces:
                self.assertTrue(torch.all(out["surfaces"] == 0.0))

    def test_turn_resets_balance_forces(self):
        speed = torch.tensor([8.0, 10.0, 12.0, 13.0])
        accel = torch.tensor([3.2, 5.0, 6.0, 4.83])
        bank = -torch.atan(accel / analysis.G0)          # left turn
        out = self.trim.solve(speed, bank=bank, use_surfaces=True)
        for i in range(4):
            V = float(speed[i])
            # Turn rate a_c / V. The reset neglects the rate terms (CLq, Cmq, Clr, Cnr, ... and w x I w), which
            # leave up to 2 % of the weight (lift lost to CLq * q) and a few tenths of a N m unbalanced.
            r = self.residual(out, i, V, bank=float(bank[i]), turn_rate=float(accel[i]) / V)
            self.assertLess(np.abs(r[:3]).max(), 0.03 * analysis.G0 * self.veh["m"], r)
            self.assertLess(np.abs(r[3:]).max(), 0.5, r)

    def test_table_respects_surface_travel(self):
        limits = {"elevator": 13.6089, "aileron": 15.7349, "rudder": 12.7436}
        for name, table in trim_mod.TRIM_TABLES.items():
            for surface, lim in limits.items():
                self.assertLessEqual(max(abs(v) for v in table[surface]), lim + 1e-2, (name, surface))
            # Minimum-power trims stay below the angle of maximum lift (14.6 deg)
            self.assertLessEqual(max(table["alpha"]), 14.0, name)
            self.assertEqual(len(table["alpha"]), len(trim_mod.TRIM_SPEED_KNOTS))


if __name__ == "__main__":
    unittest.main()
