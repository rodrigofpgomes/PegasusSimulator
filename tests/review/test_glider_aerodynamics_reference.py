"""
| File: test_glider_aerodynamics_reference.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: EasyGlider aerodynamic coefficients versus the reference SDF, reference point geometry and surface limits.
"""
import ast
import math
import unittest
from pathlib import Path

from offline_runtime import enable
enable()

try:
    import torch
except ImportError:
    torch = None

LOGIC = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic"
DEG = math.pi / 180.0

# Coefficients of easy_glider4.sdf (AdvancedLiftDrag plugin, FRD, angles per rad, controls per deg):
# https://github.com/Filipe-Cavalheiro/PX4-gazebo-models/blob/main/tools/avl_automation/easy_glider4/easy_glider4.sdf
SDF = dict(
    CL0=0.36986, CLa=3.679875, CD0=0.00759, Cem0=-0.57912, Cema=-6.591829, AR=7.788461538461539, eff=0.8523,
    CYb=-0.187754, Cellb=-0.062425, Cenb=0.108424,
    CYp=0.032506, Cellp=-0.392788, Cenp=-0.033202,
    CDq=0.549541, CLq=-16.413084, Cemq=-35.807747,
    CYr=0.222424, Cellr=0.137367, Cenr=-0.129973,
    alpha_stall=0.3391428111, area=0.416, mac=0.15407407407407406, air_density=1.2041,
    Cref=0.416 / 1.8,  # AVL reference chord (easy_glider4.avl); the SDF 'mac' is 2/3 of it
    elevator=dict(CD=-0.000044, CL=-0.001961, Cem=-0.008387),
    aileron_left=dict(CY=0.000107, Cell=-0.001596),
    rudder=dict(CY=-0.001237, Cell=-0.000142, Cen=0.000711),
)


@unittest.skipIf(torch is None, "PyTorch unavailable in review runtime")
class GliderAerodynamicsReference(unittest.TestCase):
    def setUp(self):
        from pegasus.simulator.logic.dynamics import glider_aerodynamics as module
        self.module = module
        self.C = module.GliderCoefficients()
        self.G = module.GliderGeometry()

    def test_coefficients_match_sdf_after_frame_conversion(self):
        C, S = self.C, SDF
        # Same-frame values
        for ours, theirs in ((C.CL0, S["CL0"]), (C.CLa, S["CLa"]), (C.CD0, S["CD0"]), (C.alpha0, S["alpha_stall"])):
            self.assertAlmostEqual(ours, theirs, places=9)
        # FRD -> FLU: y and z axes flip. Moments about y, z and the rates q, r change sign.
        self.assertAlmostEqual(C.Cm0, -S["Cem0"], places=9)
        self.assertAlmostEqual(C.Cma, -S["Cema"], places=9)
        self.assertAlmostEqual(C.Cmq, S["Cemq"], places=9)       # moment and q both flip
        self.assertAlmostEqual(C.CYb, -S["CYb"], places=9)
        self.assertAlmostEqual(C.CYp, -S["CYp"], places=9)
        self.assertAlmostEqual(C.CYr, S["CYr"], places=9)        # force and r both flip
        self.assertAlmostEqual(C.Clb, S["Cellb"], places=9)
        self.assertAlmostEqual(C.Clp, S["Cellp"], places=9)
        self.assertAlmostEqual(C.Clr, -S["Cellr"], places=9)
        self.assertAlmostEqual(C.Cnb, -S["Cenb"], places=9)
        self.assertAlmostEqual(C.Cnp, -S["Cenp"], places=9)
        self.assertAlmostEqual(C.Cnr, S["Cenr"], places=9)       # moment and r both flip
        # The SDF CL_*/CD_* rate and control slots hold CZ/CX (down/forward positive): lift = -CZ, drag = -CX.
        # Lift and drag are frame independent, only the q sign flips.
        self.assertAlmostEqual(C.CLq, S["CLq"], places=9)        # -(-CZq) * (-1 for q) = CZq
        self.assertAlmostEqual(C.CDq, S["CDq"], places=9)        # -(CXq) * (-1 for q) = CXq
        self.assertAlmostEqual(C.CLde, -S["elevator"]["CL"] / DEG, places=9)
        self.assertAlmostEqual(C.CDde, -S["elevator"]["CD"] / DEG, places=9)
        self.assertAlmostEqual(C.Cmde, -S["elevator"]["Cem"] / DEG, places=9)
        # Elevons are antisymmetric for the aileron command; the rudder keeps its own sign convention
        self.assertAlmostEqual(C.CYda, 2.0 * S["aileron_left"]["CY"] / DEG, places=9)
        self.assertAlmostEqual(C.Clda, -2.0 * S["aileron_left"]["Cell"] / DEG, places=9)
        self.assertAlmostEqual(C.CYdr, S["rudder"]["CY"] / DEG, places=9)
        self.assertAlmostEqual(C.Cldr, -S["rudder"]["Cell"] / DEG, places=9)
        self.assertAlmostEqual(C.Cndr, S["rudder"]["Cen"] / DEG, places=9)
        # Geometry
        self.assertAlmostEqual(self.G.S, S["area"])
        self.assertAlmostEqual(self.G.c, S["Cref"])                # AVL Cref, not the SDF mac
        self.assertAlmostEqual(self.G.AR, S["AR"], places=9)
        self.assertAlmostEqual(self.G.rho, S["air_density"])
        self.assertAlmostEqual(self.G.e, S["eff"])

    def test_pitch_rate_and_elevator_signs_are_physical(self):
        # Pitch-up (negative q in FLU) must increase lift and be damped by a nose-down moment; trailing-edge-down
        # elevator increases lift together with a nose-down moment.
        aero = self.module.GliderAerodynamicsBatch({"geometry": {"r_com": (0., 0., 0.)}}, 1, "cpu")
        v, zero = torch.tensor([[15., 0., 0.]]), torch.zeros(1, 3)
        q_up = torch.tensor([[0., -0.5, 0.]])
        F0, M0 = aero.forces_moments(v, zero, zero)
        F1, M1 = aero.forces_moments(v, q_up, zero)
        self.assertGreater(float(F1[0, 2] - F0[0, 2]), 0.0)
        self.assertGreater(float(M1[0, 1] - M0[0, 1]), 0.0)
        F2, M2 = aero.forces_moments(v, zero, torch.tensor([[0.1, 0., 0.]]))
        self.assertGreater(float(F2[0, 2] - F0[0, 2]), 0.0)
        self.assertGreater(float(M2[0, 1] - M0[0, 1]), 0.0)

    def test_pitch_rate_force_response_matches_avl_body_axis_derivatives(self):
        # Independent reference from the raw AVL body-axis derivatives written in the SDF (FRD, X forward, Z down):
        # dFx = qS*CXq*q_hat, dFz = qS*CZq*q_hat with q_hat from the FRD pitch rate. The SDF CDq/CLq slots hold CXq/CZq.
        CXq, CZq = SDF["CDq"], SDF["CLq"]
        aero = self.module.GliderAerodynamicsBatch({"geometry": {"r_com": (0., 0., 0.)}}, 1, "cpu")
        V, q_flu = 15.0, 0.4
        v, zero = torch.tensor([[V, 0., 0.]]), torch.zeros(1, 3)
        F0, _ = aero.forces_moments(v, zero, zero)
        F1, _ = aero.forces_moments(v, torch.tensor([[0., q_flu, 0.]]), zero)
        qS = 0.5 * SDF["air_density"] * V ** 2 * SDF["area"]
        q_hat_frd = -q_flu * SDF["Cref"] / (2.0 * V)                     # FRD pitch rate is -q_FLU
        expected_dFx = qS * CXq * q_hat_frd                              # X is forward in both frames
        expected_dFz = -(qS * CZq * q_hat_frd)                           # FRD z is down, FLU z is up
        self.assertAlmostEqual(float(F1[0, 0] - F0[0, 0]), expected_dFx, places=3)
        self.assertAlmostEqual(float(F1[0, 2] - F0[0, 2]), expected_dFz, places=3)
        # q_FLU > 0 is a nose-down rotation: the tail moves up, tail lift drops and so does the total lift
        self.assertLess(expected_dFz, 0.0)

    def test_rate_terms_act_along_lift_and_drag_axes_at_nonzero_alpha(self):
        aero = self.module.GliderAerodynamicsBatch({"geometry": {"r_com": (0., 0., 0.)}}, 1, "cpu")
        V, al = 15.0, math.radians(8.0)
        v = torch.tensor([[V * math.cos(al), 0., -V * math.sin(al)]]); zero = torch.zeros(1, 3)
        F0, _ = aero.forces_moments(v, zero, zero)
        F1, _ = aero.forces_moments(v, torch.tensor([[0., -0.5, 0.]]), zero)
        dF = (F1 - F0)[0]
        drag_dir = torch.tensor([-math.cos(al), 0., math.sin(al)])      # body-frame direction of drag force
        lift_dir = torch.tensor([math.sin(al), 0., math.cos(al)])
        # Rate terms are lift (CLq) and drag (CDq) increments: their components follow the same axes
        dL = -16.413084 * (0.5 * SDF["air_density"] * V ** 2 * SDF["area"]) * (-0.5 * SDF["Cref"] / (2.0 * V))
        dD = 0.549541 * (0.5 * SDF["air_density"] * V ** 2 * SDF["area"]) * (-0.5 * SDF["Cref"] / (2.0 * V))
        self.assertAlmostEqual(float(torch.dot(dF, lift_dir)), dL, places=3)
        self.assertAlmostEqual(float(torch.dot(dF, drag_dir)), dD, places=3)
        self.assertLess(abs(float(dF[1])), 1e-6)                        # no sideforce in the longitudinal plane

    def test_moment_is_transferred_once_to_the_force_application_point(self):
        r = (0.3427, 0.0, 0.016)
        at_ref = self.module.GliderAerodynamicsBatch({"geometry": {"r_com": (0., 0., 0.)}, "eval_at_ref": False}, 2, "cpu")
        at_com = self.module.GliderAerodynamicsBatch({"geometry": {"r_com": r}, "eval_at_ref": False}, 2, "cpu")
        v = torch.tensor([[14., 1., -2.], [9., -.5, 1.]])
        w = torch.tensor([[.1, -.2, .3], [0., .4, -.1]])
        d = torch.tensor([[.05, .1, -.05], [-.1, 0., .1]])
        F_ref, M_ref = at_ref.forces_moments(v, w, d)
        F_com, M_com = at_com.forces_moments(v, w, d)
        torch.testing.assert_close(F_com, F_ref)
        torch.testing.assert_close(M_com, M_ref + torch.cross(torch.tensor(r).expand_as(F_ref), F_ref, dim=1))

    def test_level_trim_at_10_m_s_needs_more_elevator_than_the_datasheet_range(self):
        # Operating limit: with the reference point at the nose and the AVL reference chord,
        # level flight at 10 m/s needs more nose-up elevator than the datasheet throw of 13.61 deg.
        self._assert_level_trim({10.0: True}, {}, within=False)

    def test_default_level_trim_is_inside_the_datasheet_elevator_range(self):
        # With the reference point at the nose and the AVL reference chord, level flight from 12 to 24 m/s needs |delta_e| <= 13.61 deg
        self._assert_level_trim({12.0: True, 15.0: True, 18.0: True, 24.0: True}, {}, within=True)

    def _assert_level_trim(self, speeds, aero_cfg, within):
        aero = self.module.GliderAerodynamicsBatch(aero_cfg, 1, "cpu")
        weight, r_puller_z, zero = 1.495 * 9.81, -0.020454, torch.zeros(1, 3)

        def wrench(V, alpha, de):
            v = torch.tensor([[V * math.cos(alpha), 0., -V * math.sin(alpha)]])
            F, tau = aero.forces_moments(v, zero, torch.tensor([[de, 0., 0.]]))
            return float(F[0, 0]), float(F[0, 2]), float(tau[0, 1])

        def pitch_residual(V, de):
            lo, hi = -0.3, 0.6                                           # alpha from vertical force balance
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if wrench(V, mid, de)[1] < weight * math.cos(mid) else (lo, mid)
            Fx, _, My = wrench(V, mid, de)
            thrust = max(weight * math.sin(mid) - Fx, 0.0)               # thrust along +X applied below the CoM
            return My + r_puller_z * thrust

        for V in speeds:
            lo, hi = -0.9, 0.9                                           # bisection: pitch moment grows with de
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if pitch_residual(V, mid) < 0.0 else (lo, mid)
            if within:
                self.assertLessEqual(abs(math.degrees(mid)), 13.61, msg=f"V={V}")
            else:
                self.assertGreater(abs(math.degrees(mid)), 13.61, msg=f"V={V}")

    def test_default_geometry_is_forward_reference_point(self):
        self.assertEqual(self.G.r_com, self.module.EASYGLIDER_REFERENCE_FROM_COM)
        # AVL origin (nose) at the propeller station of the asset: base_link + 0.045 m
        self.assertAlmostEqual(self.G.r_com[0], 0.34269228536660874, places=9)
        self.assertAlmostEqual(self.G.r_com[2], 0.016, places=9)
        self.assertEqual(self.module.EASYGLIDER_REFERENCE_FROM_CAD_COM[0], 0.29769228536660874)

    def test_merged_vehicle_reference_point_follows_glider_origin(self):
        tree = ast.parse((LOGIC / "vehicles/shuttle_glider_batch2.py").read_text())
        names = {"_GLIDER_COM", "_MERGED_COM", "_MERGED_R_COM"}
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names]
        ns = {"EASYGLIDER_REFERENCE_FROM_COM": self.module.EASYGLIDER_REFERENCE_FROM_COM}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "shuttle_glider_batch2.py", "exec"), ns)
        # USDA glider xform (0, 0, -0.28028) is its CoM; the reference point is 0.3427 m ahead and 0.016 m above,
        # and the merged body CoM (the force application point) is at z = -0.0031
        self.assertAlmostEqual(ns["_MERGED_R_COM"][0], 0.34269, places=4)
        self.assertAlmostEqual(ns["_MERGED_R_COM"][2], -0.28028 + 0.016 + 0.0031, places=4)

    def test_surface_limits_follow_the_datasheet_throws(self):
        spec = Path(LOGIC / "vehicles/vehicle_physics.py")
        ns = {}
        exec(compile(spec.read_text(), str(spec), "exec"), ns)
        lim = ns["EASYGLIDER_SURFACE_LIMITS_DEG"]
        self.assertAlmostEqual(lim["elevator"][1], 13.61, places=2)
        self.assertAlmostEqual(lim["rudder"][1], 12.74, places=2)
        self.assertAlmostEqual(lim["aileron"][1], 15.735, places=3)
        for key in lim:
            self.assertAlmostEqual(lim[key][0], -lim[key][1])
        self.assertAlmostEqual(ns["EASYGLIDER_AILERON_SURFACE_RANGE_DEG"][0], -22.62, places=2)
        self.assertAlmostEqual(ns["EASYGLIDER_AILERON_SURFACE_RANGE_DEG"][1], 8.85, places=2)
        self.assertEqual(ns["EASYGLIDER_SURFACE_MAX_RAD"],
                         tuple(math.radians(lim[k][1]) for k in ("elevator", "aileron", "rudder")))


if __name__ == "__main__":
    unittest.main()
