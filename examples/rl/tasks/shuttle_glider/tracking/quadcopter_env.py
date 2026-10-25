"""
| File: quadcopter_env.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Shuttle-Glider trajectory tracking with direct actuation: five rotor speeds (four lift rotors and the
| puller) and three control surfaces (elevator, aileron, rudder), trained without a curriculum.
|
| Episodes (10 s) draw one segment reference (trajectory.py) from a fixed mixture of flight regimes: hover,
| acceleration from hover, deceleration (to hover when it fits in the episode), cruise with a speed change, turns with
| roll-in/roll-out, and climb/descent. Every episode starts in the 6-DOF trim of its first reference sample
| (pegasus.simulator.logic.rl.shuttle_glider_trim: actuator limits only, minimum shaft power), with a small perturbation
| or, in a fraction of the episodes, a large one scaled with the speed.
|
| The reward is the same in every regime: tracking (position, velocity, heading along the path, body rate about the
| reference turn rate), rotor shaft power normalised by the hover power, a surface deflection cost with the
| power/surface ratio of the trim table (so the trim is the steady-state optimum of the reward), and the command change.
| There is no angle-of-attack, sideslip or regime-specific term. The observation is expressed in the body frame and
| does not contain the absolute yaw (see _get_observations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.rl.shuttle_glider_trim import ShuttleGliderTrim, interpolate_table, trim_attitude
from pegasus.simulator.logic.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from pegasus.simulator.logic.vehicles.vehicle_physics import EASYGLIDER_SURFACE_MAX_RAD, EASYGLIDER_SURFACE_MIN_RAD

from .trajectory import SegmentReference, smoothstep

EPISODE_TYPES = ("hover", "accel", "decel", "cruise", "turn", "climb")


@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):
    # [pos error 3, vel error 3, air velocity 3, ref acceleration 3, ref heading 3, gravity 3, body rates 3,
    #  last action 8, rotor speeds 5, surface deflections 3], all in the body frame
    observation_space: int = 37
    action_space: int = 8
    state_space: int = 0

    sim_dt: float = 0.01
    decimation: int = 1
    episode_length_s: float = 10.0

    action_mode: str = "rotor_velocity_direct"
    vehicle: str = "Shuttle_glider2"
    gravity: float = 9.81

    # Surface actuators: datasheet travel (vehicle_physics) and rate limit
    surface_rate_max_deg_s: float = 150.0

    # --- reference mixture (probabilities in EPISODE_TYPES order) ---
    trajectory_generator: bool = True      # False: external references (play_multi sets the goals)
    episode_type_probs: tuple = (0.15, 0.15, 0.15, 0.15, 0.25, 0.15)
    speed_max: float = 14.0                # [m/s]
    accel_max: float = 3.0                 # peak along-track acceleration of a speed increase [m/s^2]
    # Peak deceleration by speed [m/s^2]: about 60 % of the steady-descent equivalent of the 6-DOF trim (the puller
    # cannot brake; at high speed the wing cannot add drag without lift). From 14 m/s an episode only slows down.
    decel_speed_knots: tuple = (0.0, 8.0, 10.0, 12.0, 14.0)
    decel_max: tuple = (2.4, 2.3, 1.4, 0.7, 0.45)
    ramp_min_s: float = 2.5                # minimum duration of a speed change [s]
    radial_acc_max: float = 6.0            # turn radius >= V^2 / radial_acc_max [m/s^2]
    turn_radius_min: float = 15.0
    turn_radius_max: float = 80.0
    climb_angle_max_deg: float = 5.0       # vertical speed <= max(climb_vz_min, V sin(angle))
    climb_vz_min: float = 0.5

    # --- initial states: trim of the first reference sample plus a perturbation ---
    large_perturbation_prob: float = 0.3
    small_perturbation: tuple = (0.1, 0.2, 3.0, 0.1, 0.02)   # position [m], velocity [m/s], angle [deg], rate [rad/s], rotor
    large_perturbation_hover: tuple = (0.3, 1.0, 30.0, 1.0, 0.1)
    large_perturbation_fast: tuple = (0.3, 1.0, 10.0, 0.3, 0.1)   # at speed_max; linear in between

    # --- observation scaling (clipped identically in training and evaluation) ---
    obs_pos_scale: float = 0.5
    obs_pos_clip: float = 2.0
    obs_vel_scale: float = 1.0
    obs_vel_clip: float = 3.0
    obs_air_scale: float = 15.0
    obs_acc_scale: float = 6.0
    obs_rate_scale: float = 5.0

    # --- reward ---
    constant: float = 1.5
    w_pos: float = 1.0
    w_vel: float = 0.1
    w_heading: float = 0.5        # 1 - cos(heading error) of the body x axis projected on the horizontal plane
    w_rate: float = 0.05          # ||omega_b - R^T (0, 0, psi_dot_ref)||
    # Rotor shaft power / hover power. 0.1 and 0.2 drive the policy to alpha 11-14 deg (CLmax at 14.6 deg): decelerations
    # stall, the critic degrades after ~2000 iterations and so does the tracking; 0.05 keeps alpha ~8 deg at the same power
    w_power: float = 0.05
    # Surface cost = w_power * surface_power_equivalent_w / P_hover * sum (delta / delta_max)^2: the ratio of the trim
    # table (examples/rl/diagnostics/shuttle_glider_trim.py, LIMITS_SURFACE_COST_W)
    surface_power_equivalent_w: float = 25.0
    # (A) Low dynamic pressure: the surface cost is multiplied by 1 + gain * exp(-q / q_ref). Without airflow a deflection
    # produces no moment, so it is pure waste; at cruise (q ~ 90-120 Pa) the factor returns to ~1 and the trim stays optimal.
    # 0 disables it (the policy then leaves the surfaces at arbitrary deflections in hover).
    surface_low_q_gain: float = 0.0
    surface_low_q_ref: float = 15.0    # [Pa], ~5 m/s
    # (B) Opposing moments [per N m]: sum over roll/pitch/yaw of min(|M_surf|, |M_rot|) where the surface control moment
    # and the lift-rotor moment have opposite signs (the part that cancels: internal load and wasted power). 0 disables it.
    w_conflict: float = 0.0
    # (C) Moment cancelled inside the lift rotors [per N m]: front pair vs rear pair in roll, left pair vs right pair in
    # pitch. A diagonal (yaw) differential cancels in both, so this prices rotor yaw (lever kq/kf ~0.016 m vs 0.248 m)
    # against the rudder. 0 disables it.
    w_rotor_cancel: float = 0.0
    # Smoothing [N m] of the cancelled moment of (B) and (C). The overlap o = -sign(M_1) sign(M_2) min(|M_1|, |M_2|) is
    # continuous (+min when opposed, -min when aligned) and the cost is a Huber hinge of it: 0 for o <= 0, o^2 / (2 eps)
    # up to eps, o - eps / 2 beyond. 0 gives the exact min(|M_1|, |M_2|) for opposite signs, whose kink at a sign
    # change makes the policy chatter (aileron moment flips sign in 9 % of the steps of a steady turn of conflict).
    conflict_eps: float = 0.0
    # (D) Lift-rotor thrust differentials: w_rotor_diff sum ((T_i - mean T) / T_max)^2. The rotors give a moment at a cost
    # independent of speed and the surfaces at w_s (M / M_max(q))^2, so the moments move to the surfaces as q grows
    # (pitch from ~8 m/s, roll from ~4.5 m/s, yaw sooner). 0.1 keeps the trim power within +2 W; 0.3-1 lowers alpha and
    # costs 6-20 W from 6 m/s (shuttle_glider_trim.py --limits-table --alloc). 0 disables it.
    w_rotor_diff: float = 0.0
    # (D') Where the surfaces lack authority the rotors are the only moment source (pitching over to accelerate from
    # hover): a uniform D degraded the training (wls). rotor_diff_q_ref > 0 [Pa] scales D by q / (q + q_ref) (wls_q).
    rotor_diff_q_ref: float = 0.0
    # (D'') rotor_diff_tau > 0 [s] applies D to an exponential moving average of the differentials instead: steady
    # antagonism pays, manoeuvres and the exploration noise are filtered (wls_ema). The average is appended to the
    # observation (x obs_rotor_diff_scale) so the reward stays Markov: observation_space must then be 41.
    rotor_diff_tau: float = 0.0
    obs_rotor_diff_scale: float = 10.0
    # (F) Filtered lateral moments: w_moment_ema sum over roll and yaw of [EMA(M_rotors)^2 + EMA(M_surfaces)^2] [per
    # N^2 m^2], with M_rotors the moment of the lift rotors about the CoM and M_surfaces the control moment of the
    # surfaces (qS b (Cl_da da + Cl_dr dr), qS b (Cn_da da + Cn_dr dr)). Convex in the commands, so the exploration noise
    # does not bias it, and averaged over moment_ema_tau, so manoeuvres are free: crossed controls (surfaces against
    # rotors in steady flight, wls_ema: ~0.7 N m on each side) pay on both sides. Pitch is left out, since the rotor
    # pitch moment is needed at every speed. The averages are observed (x obs_moment_scale): +4 observations.
    w_moment_ema: float = 0.0
    moment_ema_tau: float = 2.0       # [s]
    obs_moment_scale: float = 1.0
    # (E) Low dynamic pressure, L1: w_surface_l1 exp(-q / surface_l1_q_ref) sum (sqrt((d / d_max)^2 + eps^2) - eps).
    # Unlike the quadratic cost of (A), its gradient does not vanish near zero, so hover deflections go to zero
    # (surfq still left 1-2 deg). 0 disables it.
    w_surface_l1: float = 0.0
    surface_l1_q_ref: float = 5.0      # [Pa], ~2.9 m/s: factor 0.14 at 4 m/s, ~0 from 6 m/s
    surface_l1_eps: float = 0.01
    # Trim table of the initial states: "limits" (power + quadratic surface cost), "alloc" (also B-E with the weights of
    # the alloc variant) or "wls" (also D and E only); each is the steady-state optimum of its reward
    trim_table: str = "limits"
    w_d_action: float = 0.5       # ||a_t - a_t-1|| of the sampled actions before the [-1, 1] clip, eight channels
    termination_penalty: float = 100.0

    # --- termination (per axis) ---
    max_pos_error_per_axis: float = 3.0
    max_vel_error_per_axis: float = 6.0
    max_ang_vel_per_axis: float = 35.0

    # --- external references (trajectory_generator=False) ---
    heading_hold_speed: float = 0.5       # below this reference speed the heading is held
    heading_rate_speed_floor: float = 1.0  # psi_dot = (v x a)_z / max(|v_xy|^2, floor^2)

    # --- evaluation: one episode type with fixed parameters ("" = training mixture) ---
    test_mode: bool = False
    eval_episode: str = ""
    eval_speed: float = 0.0
    eval_speed_end: float = 0.0
    eval_turn_radius: float = 40.0
    eval_vz: float = 0.0
    # Surface ablation (evaluation only): "trim" replaces the policy surface targets by the trim table at the measured
    # airspeed, "zero" by zero deflection; "" keeps the policy
    surface_override: str = ""

    randomize_init_state: bool = False    # the task supplies its own initial states
    goal_pos_xy_range: list | None = None
    goal_pos_z_range: list | None = None


class QuadcopterEnv(PegasusEnv):
    cfg: QuadcopterEnvCfg

    NUM_ROTORS = 5
    REWARD_TERMS = ("pos", "vel", "heading", "rate", "power", "surface", "surface_l1", "conflict", "rotor_cancel",
                    "rotor_diff", "moment", "d_action", "total")

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)
        if cfg.eval_episode and cfg.eval_episode not in EPISODE_TYPES:
            raise ValueError(f"eval_episode must be one of {EPISODE_TYPES}, got {cfg.eval_episode!r}")
        if len(cfg.episode_type_probs) != len(EPISODE_TYPES):
            raise ValueError(f"episode_type_probs needs {len(EPISODE_TYPES)} values ({EPISODE_TYPES})")
        self._trajectory: SegmentReference | None = None
        self._trim: ShuttleGliderTrim | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        super().setup()
        n, dev = self.num_envs, self.device
        self._last_action = torch.zeros((n, self.cfg.action_space), device=dev)
        self._prev_action = torch.zeros_like(self._last_action)
        self._action_history_obs = torch.zeros_like(self._last_action)
        # Unclipped policy samples for the command-change cost
        self._raw_action = torch.zeros_like(self._last_action)
        self._prev_raw_action = torch.zeros_like(self._last_action)
        self._surface_deflections = torch.zeros((n, 3), device=dev)
        self._rotor_diff_ema = torch.zeros((n, 4), device=dev)
        self._moment_ema = torch.zeros((n, 4), device=dev)
        expected_obs = 37 + (4 if self.cfg.rotor_diff_tau > 0.0 else 0) + (4 if self.cfg.w_moment_ema > 0.0 else 0)
        if self.cfg.observation_space != expected_obs:
            raise ValueError(f"observation_space must be {expected_obs} with rotor_diff_tau={self.cfg.rotor_diff_tau} "
                             f"and w_moment_ema={self.cfg.w_moment_ema}")
        self._ref_heading = torch.zeros(n, device=dev)
        self._ref_heading_rate = torch.zeros(n, device=dev)
        self._psi_track = torch.zeros(n, device=dev)       # held heading of external references
        self._types = torch.zeros(n, dtype=torch.long, device=dev)
        self._episode_sums = {k: torch.zeros(n, device=dev) for k in self.REWARD_TERMS}
        self._stats = {k: torch.zeros(n, device=dev)
                       for k in ("steps", "pos_error", "power", "alpha_deg", "alpha_steps", "rotor_sat", "surface_sat")}

        lo = torch.tensor(EASYGLIDER_SURFACE_MIN_RAD, dtype=torch.float32, device=dev)
        hi = torch.tensor(EASYGLIDER_SURFACE_MAX_RAD, dtype=torch.float32, device=dev)
        self._surface_lo, self._surface_hi = lo, hi

        vehicle = self.backend._vehicle
        if self.cfg.goal_pos_xy_range is None:
            self.cfg.goal_pos_xy_range = [-0.5, 0.5]
        if self.cfg.goal_pos_z_range is None:
            z = vehicle._init_pos[0, 2].item()
            self.cfg.goal_pos_z_range = [z - 0.5, z + 0.5]
        self.backend.create_goal_markers(root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0))
        self.reset_manager.set_goal_cfg(self.cfg)

        # Trim, shaft-power constants and hover power
        self._trim = ShuttleGliderTrim.from_vehicle(vehicle, self.cfg.gravity, dev, table=self.cfg.trim_table)
        self._k_m = torch.cat((self._trim.kq.abs(), torch.tensor([abs(self._trim.kq_p)], device=dev)))
        hover = self._trim.solve(torch.zeros(1, device=dev), use_surfaces=True)
        self._p_hover = float((self._k_m * hover["omega"][0] ** 3).sum())
        self._w_surface = self.cfg.w_power * self.cfg.surface_power_equivalent_w / self._p_hover
        self._decel_knots = torch.tensor(self.cfg.decel_speed_knots, dtype=torch.float32, device=dev)
        self._decel_max = torch.tensor(self.cfg.decel_max, dtype=torch.float32, device=dev)

        if self.cfg.trajectory_generator:
            # One extra sample covers the terminal observation at the time limit
            self._trajectory = SegmentReference(n, self.max_episode_length + 1, self.step_dt, dev)
            self._reset_reference_and_state(torch.arange(n, device=dev))

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    def _sync_reference(self):
        """Current reference into the shared goal buffers, with its heading and heading rate."""
        rm = self.reset_manager
        if self._trajectory is not None:
            ids = torch.arange(self.num_envs, device=self.device)
            pos, vel, acc, psi, psi_dot = self._trajectory.sample(ids, self.episode_length_buf)
            rm._goal_pos[:], rm._goal_vel[:], rm._goal_acc[:] = pos, vel, acc
            self._ref_heading, self._ref_heading_rate = psi, psi_dot
            return
        # External reference: heading of the reference velocity, held at low speed
        v, a = rm.goal_vel, rm.goal_acc
        speed_sq = v[:, 0].square() + v[:, 1].square()
        moving = speed_sq > self.cfg.heading_hold_speed ** 2
        self._psi_track = torch.where(moving, torch.atan2(v[:, 1], v[:, 0]), self._psi_track)
        self._ref_heading = self._psi_track
        self._ref_heading_rate = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / speed_sq.clamp_min(self.cfg.heading_rate_speed_floor ** 2)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        # The command-change cost uses the samples before the clip: measured after it, a mean held at a bound clips
        # half of the exploration noise, which rewards saturating channels without effect (the surfaces in hover)
        self._prev_raw_action = self._raw_action
        self._raw_action = actions.clone()
        action = actions.clamp(-1.0, 1.0)
        self._prev_action = self._last_action.clone()
        self._last_action = action
        self._action_history_obs = action.clone()

    def _normalize_surfaces(self, surfaces: torch.Tensor) -> torch.Tensor:
        return surfaces / torch.where(surfaces >= 0.0, self._surface_hi, -self._surface_lo)

    def _apply_action(self):
        """Rotor speeds from actions 0..4; surface targets from actions 5..7 through the rate limit."""
        a = self._last_action
        target = torch.where(a[:, 5:8] >= 0.0, a[:, 5:8] * self._surface_hi, -a[:, 5:8] * self._surface_lo)
        if self.cfg.surface_override == "zero":
            target = torch.zeros_like(target)
        elif self.cfg.surface_override == "trim":
            _, vel_w, ang_b, R = self._state()
            airspeed = torch.linalg.norm(self._air_velocity_body(vel_w, ang_b, R), dim=1)
            target = torch.deg2rad(torch.stack([interpolate_table(airspeed, self._trim.table[k]) for k in
                                                ("elevator", "aileron", "rudder")], dim=1))
        step_max = math.radians(self.cfg.surface_rate_max_deg_s) * self.cfg.sim_dt
        self._surface_deflections = self._surface_deflections + (target - self._surface_deflections).clamp(-step_max, step_max)

        thrusters = self.backend._vehicle._thrusters
        min_w, max_w = thrusters.min_rotor_velocity, thrusters.max_rotor_velocity
        command = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        command[:, 0:5] = (a[:, 0:5] + 1.0) * 0.5 * (max_w - min_w) + min_w
        command[:, 5:8] = self._surface_deflections
        self.backend._input_reference = command

    def _rotor_speeds_normalized(self) -> torch.Tensor:
        thrusters = self.backend._vehicle._thrusters
        min_w, max_w = thrusters.min_rotor_velocity, thrusters.max_rotor_velocity
        return 2.0 * (thrusters._velocity - min_w) / (max_w - min_w) - 1.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _state(self):
        state = self.backend.get_state()
        pos, vel_w, quat, ang_b = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        R = quaternion_to_matrix(quat)
        return pos, vel_w, ang_b, R

    def _air_velocity_body(self, vel_w, ang_b, R) -> torch.Tensor:
        """Air-relative velocity at the aerodynamic reference point in the body frame (as airdata_from_state)."""
        aero = self.backend._vehicle.aerodynamics
        v = (R.transpose(1, 2) @ vel_w.unsqueeze(-1)).squeeze(-1) - aero._wind_body
        if aero.eval_at_ref:
            v = v + torch.cross(ang_b, aero._r_com.expand_as(v), dim=1)
        return v

    def _get_observations(self) -> dict:
        """Body-frame observation without the absolute yaw.

        The rotation matrix is replaced by the gravity direction in the body frame (roll and pitch) and the reference
        heading in the body frame (heading error); the angle of attack and sideslip by the air velocity, which is
        continuous at every airspeed.
        """
        self._sync_reference()
        c = self.cfg
        pos, vel_w, ang_b, R = self._state()
        Rt = R.transpose(1, 2)
        to_body = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)
        rm = self.reset_manager
        e_p = (to_body(pos - rm.goal_pos) / c.obs_pos_scale).clamp(-c.obs_pos_clip, c.obs_pos_clip)
        e_v = (to_body(vel_w - rm.goal_vel) / c.obs_vel_scale).clamp(-c.obs_vel_clip, c.obs_vel_clip)
        heading = torch.stack((torch.cos(self._ref_heading), torch.sin(self._ref_heading),
                               torch.zeros_like(self._ref_heading)), dim=1)
        obs = torch.cat([e_p, e_v,
                         self._air_velocity_body(vel_w, ang_b, R) / c.obs_air_scale,
                         to_body(rm.goal_acc) / c.obs_acc_scale,
                         to_body(heading),
                         -R[:, 2, :],                       # gravity direction in the body frame
                         ang_b / c.obs_rate_scale,
                         self._action_history_obs,
                         self._rotor_speeds_normalized(),
                         self._normalize_surfaces(self._surface_deflections)], dim=1)
        if c.rotor_diff_tau > 0.0:
            obs = torch.cat([obs, self._rotor_diff_ema * c.obs_rotor_diff_scale], dim=1)
        if c.w_moment_ema > 0.0:
            obs = torch.cat([obs, self._moment_ema * c.obs_moment_scale], dim=1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def _heading_cost(self, R: torch.Tensor) -> torch.Tensor:
        """1 - cos of the angle between the horizontal projection of the body x axis and the reference heading."""
        x_h = R[:, 0:2, 0]
        d = torch.stack((torch.cos(self._ref_heading), torch.sin(self._ref_heading)), dim=1)
        cos_e = (x_h * d).sum(dim=1) / torch.linalg.norm(x_h, dim=1).clamp_min(1e-3)
        return 1.0 - cos_e.clamp(-1.0, 1.0)

    def _power(self) -> torch.Tensor:
        """Rotor shaft power sum k_m omega^3 over the five rotors, normalised by the hover power."""
        omega = self.backend._vehicle._thrusters._velocity
        return (self._k_m * omega.clamp_min(0.0) ** 3).sum(dim=1) / self._p_hover

    def _cancelled(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Moment [N m] by which a and b cancel: min(|a|, |b|) for opposite signs, Huber-smoothed by cfg.conflict_eps."""
        overlap = -torch.sign(a) * torch.sign(b) * torch.minimum(a.abs(), b.abs())
        eps = self.cfg.conflict_eps
        if eps <= 0.0:
            return overlap.clamp_min(0.0)
        return torch.where(overlap < eps, overlap.clamp_min(0.0).square() / (2.0 * eps), overlap - 0.5 * eps)

    def _lateral_moments(self, qbar, thrust, surfaces) -> torch.Tensor:
        """Roll and yaw moments [N m] of the lift rotors (about the CoM) and of the surface controls: (n, 4) =
        (rotor roll, rotor yaw, surface roll, surface yaw)."""
        G, C, trim = self._trim.G, self._trim.C, self._trim
        qSb = qbar * G.S * G.b
        _, da, dr = surfaces.unbind(dim=1)
        return torch.stack((thrust @ trim.r_rotor[:, 1], thrust @ (trim.kq / trim.kf),
                            qSb * (C.Clda * da + C.Cldr * dr), qSb * (C.Cnda * da + C.Cndr * dr)), dim=1)

    def _moment_conflict(self, qbar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(surfaces vs lift rotors, inside the lift rotors): moment [N m] cancelled, summed over the axes.

        Surface moment: control derivatives only (qS b Clda da, qS c Cmde de, qS b Cndr dr), in the body frame of the
        aerodynamic model; rotor moment about the assembly CoM: (sum y T, -sum x T, sum (k_q / k_f) T) as in the trim.
        Inside the rotors: roll of the front pair vs the rear pair, pitch of the left pair vs the right pair.
        """
        G, C, trim = self._trim.G, self._trim.C, self._trim
        qS = qbar * G.S
        de, da, dr = self._surface_deflections.unbind(dim=1)
        m_surf = torch.stack((qS * G.b * C.Clda * da, qS * G.c * C.Cmde * de, qS * G.b * C.Cndr * dr), dim=1)
        thrust = trim.kf * self.backend._vehicle._thrusters._velocity[:, :4].clamp_min(0.0).square()
        roll_i, pitch_i = thrust * trim.r_rotor[:, 1], -thrust * trim.r_rotor[:, 0]
        m_rot = torch.stack((roll_i.sum(dim=1), pitch_i.sum(dim=1), thrust @ (trim.kq / trim.kf)), dim=1)
        surf_rot = self._cancelled(m_surf, m_rot).sum(dim=1)
        front, left = (trim.r_rotor[:, 0] > 0).float(), (trim.r_rotor[:, 1] > 0).float()
        return surf_rot, (self._cancelled(roll_i @ front, roll_i @ (1.0 - front))
                          + self._cancelled(pitch_i @ left, pitch_i @ (1.0 - left)))

    def eval_extras(self) -> dict[str, torch.Tensor]:
        """Actuator and aerodynamic state for offline figures (play.py --metrics-dir): rotor speeds [rad/s], surface
        deflections [rad], air velocity at the aerodynamic reference [m/s, body], aerodynamic force and moment of the
        last physics step [N, N m, body], control moments of the surfaces and of the lift rotors (+ puller reaction)
        [N m, body; as _moment_conflict]."""
        G, C, trim = self._trim.G, self._trim.C, self._trim
        _, vel_w, ang_b, R = self._state()
        v_air = self._air_velocity_body(vel_w, ang_b, R)
        qS = 0.5 * G.rho * v_air.square().sum(dim=1) * G.S
        de, da, dr = self._surface_deflections.unbind(dim=1)
        m_surf = torch.stack((qS * G.b * (C.Clda * da + C.Cldr * dr), qS * G.c * C.Cmde * de,
                              qS * G.b * (C.Cnda * da + C.Cndr * dr)), dim=1)
        omega = self.backend._vehicle._thrusters._velocity
        thrust = trim.kf * omega[:, :4].clamp_min(0.0).square()
        m_rot = torch.stack((thrust @ trim.r_rotor[:, 1], -(thrust @ trim.r_rotor[:, 0]), thrust @ (trim.kq / trim.kf)),
                            dim=1)
        wrench = self.backend._vehicle.aero_wrench_local
        zeros = torch.zeros_like(v_air)
        f_aero, m_aero = ((zeros, zeros) if wrench is None else
                          (torch.as_tensor(w, dtype=torch.float32, device=self.device).reshape(-1, 3) for w in wrench))
        return {"omega": omega, "surfaces": self._surface_deflections, "v_air": v_air, "f_aero": f_aero,
                "m_aero": m_aero, "m_surf": m_surf, "m_rot": m_rot}

    def _failure_mask(self, e_p, e_v, ang_b) -> torch.Tensor:
        c = self.cfg
        return ((e_p.abs() > c.max_pos_error_per_axis).any(dim=1) | (e_v.abs() > c.max_vel_error_per_axis).any(dim=1)
                | (ang_b.abs() > c.max_ang_vel_per_axis).any(dim=1))

    def update_filters(self, qbar: torch.Tensor | None = None, thrust: torch.Tensor | None = None) -> None:
        """Advance the moving averages of the allocation terms by one control step: lift-rotor thrust differentials
        and lateral moments, both observed. _get_rewards calls it in training; evaluation loops that do not compute
        rewards (play_multi) call it once per control step before building the observation."""
        c = self.cfg
        if c.rotor_diff_tau <= 0.0 and c.w_moment_ema <= 0.0:
            return
        if thrust is None:
            thrust = self._trim.kf * self.backend._vehicle._thrusters._velocity[:, :4].clamp_min(0.0).square()
        if qbar is None:
            _, vel_w, ang_b, R = self._state()
            qbar = 0.5 * self._trim.G.rho * self._air_velocity_body(vel_w, ang_b, R).square().sum(dim=1)
        if c.rotor_diff_tau > 0.0:
            thrust_diff = (thrust - thrust.mean(dim=1, keepdim=True)) / self._trim.T_max
            self._rotor_diff_ema += min(self.step_dt / c.rotor_diff_tau, 1.0) * (thrust_diff - self._rotor_diff_ema)
        if c.w_moment_ema > 0.0:
            moments = self._lateral_moments(qbar, thrust, self._surface_deflections)
            self._moment_ema += min(self.step_dt / c.moment_ema_tau, 1.0) * (moments - self._moment_ema)

    def _get_rewards(self) -> torch.Tensor:
        self._sync_reference()
        c = self.cfg
        pos, vel_w, ang_b, R = self._state()
        rm = self.reset_manager
        e_p, e_v = pos - rm.goal_pos, vel_w - rm.goal_vel
        rate_ref = R[:, 2, :] * self._ref_heading_rate.unsqueeze(1)      # R^T (0, 0, psi_dot)
        power = self._power()
        surfaces = self._normalize_surfaces(self._surface_deflections)
        v_air = self._air_velocity_body(vel_w, ang_b, R)
        airspeed = torch.linalg.norm(v_air, dim=1)
        qbar = 0.5 * self._trim.G.rho * airspeed.square()
        low_q = 1.0 + c.surface_low_q_gain * torch.exp(-qbar / c.surface_low_q_ref)
        conflict, rotor_cancel = self._moment_conflict(qbar)
        l1_q = c.w_surface_l1 * torch.exp(-qbar / c.surface_l1_q_ref)
        eps = c.surface_l1_eps
        thrust = self._trim.kf * self.backend._vehicle._thrusters._velocity[:, :4].clamp_min(0.0).square()
        thrust_diff = (thrust - thrust.mean(dim=1, keepdim=True)) / self._trim.T_max
        self.update_filters(qbar, thrust)
        if c.rotor_diff_tau > 0.0:
            thrust_diff = self._rotor_diff_ema
        diff_q = qbar / (qbar + c.rotor_diff_q_ref) if c.rotor_diff_q_ref > 0.0 else 1.0

        terms = {
            "pos": c.w_pos * torch.linalg.norm(e_p, dim=1),
            "vel": c.w_vel * torch.linalg.norm(e_v, dim=1),
            "heading": c.w_heading * self._heading_cost(R),
            "rate": c.w_rate * torch.linalg.norm(ang_b - rate_ref, dim=1),
            "power": c.w_power * power,
            "surface": self._w_surface * low_q * surfaces.square().sum(dim=1),
            "surface_l1": l1_q * (torch.sqrt(surfaces.square() + eps**2) - eps).sum(dim=1),
            "conflict": c.w_conflict * conflict,
            "rotor_cancel": c.w_rotor_cancel * rotor_cancel,
            "rotor_diff": c.w_rotor_diff * diff_q * thrust_diff.square().sum(dim=1),
            "moment": c.w_moment_ema * self._moment_ema.square().sum(dim=1),
            "d_action": c.w_d_action * torch.linalg.norm(self._raw_action - self._prev_raw_action, dim=1),
        }
        reward = c.constant - sum(terms.values())
        died = self._failure_mask(e_p, e_v, ang_b)
        reward[died] = -c.termination_penalty

        for key, value in terms.items():
            self._episode_sums[key] -= value
        self._episode_sums["total"] += reward

        # Diagnostics (not rewarded): angle of attack only where it is defined
        fast = (airspeed > 3.0).float()
        s = self._stats
        s["steps"] += 1.0
        s["pos_error"] += torch.linalg.norm(e_p, dim=1)
        s["power"] += power
        s["alpha_deg"] += fast * torch.rad2deg(torch.atan2(-v_air[:, 2], v_air[:, 0]))
        s["alpha_steps"] += fast
        s["rotor_sat"] += (self._last_action[:, :5].abs() >= 0.99).float().mean(dim=1)
        s["surface_sat"] += (surfaces.abs() >= 0.99).float().mean(dim=1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, vel_w, ang_b, _ = self._state()
        rm = self.reset_manager
        terminated = self._failure_mask(pos - rm.goal_pos, vel_w - rm.goal_vel, ang_b)
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        log = self.extras.setdefault("log", {})
        log["Episode_Termination/died"] = self.reset_terminated[env_ids].float().mean()
        log["Episode_Termination/timeout"] = self.reset_time_outs[env_ids].float().mean()
        for key, value in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = value[env_ids].mean()
            value[env_ids] = 0.0
        s = self._stats
        steps = s["steps"][env_ids].clamp_min(1.0)
        survived = (self.reset_time_outs[env_ids].bool() & ~self.reset_terminated[env_ids].bool()).float()
        for k, name in enumerate(EPISODE_TYPES):
            mask = self._types[env_ids] == k
            if torch.any(mask):
                log[f"Tracking/{name}/pos_error"] = (s["pos_error"][env_ids] / steps)[mask].mean()
                log[f"Tracking/{name}/survival"] = survived[mask].mean()
                log[f"Tracking/{name}/power"] = (s["power"][env_ids] / steps)[mask].mean()
        alpha_steps = s["alpha_steps"][env_ids]
        if torch.any(alpha_steps > 0):
            log["Aerodynamics/alpha_deg"] = s["alpha_deg"][env_ids].sum() / alpha_steps.sum()
        log["Actuators/rotor_saturation"] = (s["rotor_sat"][env_ids] / steps).mean()
        log["Actuators/surface_saturation"] = (s["surface_sat"][env_ids] / steps).mean()
        for value in s.values():
            value[env_ids] = 0.0
        if self._trajectory is not None:
            self._reset_reference_and_state(env_ids)
        self._call_reset_callbacks(env_ids)

    def _decel_limit(self, speed: torch.Tensor) -> torch.Tensor:
        """Peak deceleration admitted at a speed (piecewise linear in decel_speed_knots)."""
        k = self._decel_knots
        v = speed.clamp(float(k[0]), float(k[-1]))
        upper = torch.bucketize(v, k).clamp(1, k.numel() - 1)
        lower = upper - 1
        blend = (v - k[lower]) / (k[upper] - k[lower])
        return self._decel_max[lower] + blend * (self._decel_max[upper] - self._decel_max[lower])

    def _sample_episodes(self, n: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Episode types and segment parameters (trajectory.SegmentReference.PARAMS) of n new episodes."""
        c, dev = self.cfg, self.device
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)
        sign = lambda: torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        never = torch.full((n,), 1.0e3, device=dev)
        zeros = torch.zeros(n, device=dev)
        probs = torch.tensor(c.episode_type_probs, dtype=torch.float32, device=dev)
        types = torch.multinomial(probs / probs.sum(), n, replacement=True)
        is_type = lambda name: types == EPISODE_TYPES.index(name)
        p = {"V0": zeros.clone(), "V1": zeros.clone(), "t_v": never.clone(), "T_v": torch.ones(n, device=dev),
             "kappa": zeros.clone(), "t_k": never.clone(), "T_k": torch.ones(n, device=dev), "t_ko": never.clone(),
             "vz": zeros.clone(), "t_z": never.clone(), "T_z": torch.ones(n, device=dev), "t_zo": never.clone(),
             "psi0": u(-math.pi, math.pi)}
        D, vmax = c.episode_length_s, c.speed_max

        def speed_change(mask, V0, V1, t_v):
            dV = V1 - V0
            a = torch.where(dV >= 0.0, torch.full_like(V0, c.accel_max), self._decel_limit(V0))
            T = torch.maximum(torch.full_like(V0, c.ramp_min_s), 1.5 * dV.abs() / a)
            for key, value in (("V0", V0), ("V1", V1), ("t_v", t_v), ("T_v", T)):
                p[key] = torch.where(mask, value, p[key])

        # Acceleration from hover (ends before D for speed_max / accel_max <= 4.7 s)
        t_v = u(0.5, 2.0)
        speed_change(is_type("accel"), zeros, u(2.0, vmax), t_v)
        # Deceleration, to hover when the admissible deceleration allows it within the episode
        V0 = u(2.0, vmax)
        avail = D - t_v - 0.5
        dv_max = self._decel_limit(V0) * avail / 1.5
        V1 = torch.where(dv_max >= V0, zeros, V0 - u(0.5, 1.0) * dv_max)
        speed_change(is_type("decel"), V0, V1, t_v)
        # Cruise, half of them with a speed change
        V0 = u(1.0, vmax)
        target = u(1.0, vmax)
        dV = (target - V0).clamp(-self._decel_limit(V0) * avail / 1.5, c.accel_max * avail / 1.5)
        V1 = torch.where(torch.rand(n, device=dev) < 0.5, V0, V0 + dV)
        speed_change(is_type("cruise"), V0, V1, t_v)
        # Turns at constant speed: half start on the arc, the others roll in; roll-out when it fits
        turn = is_type("turn")
        V0 = u(3.0, vmax)
        r_min = torch.maximum(torch.full_like(V0, c.turn_radius_min), V0.square() / c.radial_acc_max)
        radius = r_min + torch.rand(n, device=dev) * (c.turn_radius_max - r_min).clamp_min(5.0)
        T_k = u(1.5, 3.0)
        t_k = torch.where(torch.rand(n, device=dev) < 0.5, -T_k, u(0.0, 3.0))
        for key, value in (("V0", V0), ("V1", V0), ("kappa", sign() / radius), ("T_k", T_k), ("t_k", t_k),
                           ("t_ko", t_k + T_k + u(2.0, 6.0))):
            p[key] = torch.where(turn, value, p[key])
        # Climb or descent pulse at constant speed (from hover upwards)
        climb = is_type("climb")
        V0 = u(0.0, vmax)
        vz_max = torch.clamp_min(V0 * math.sin(math.radians(c.climb_angle_max_deg)), c.climb_vz_min)
        T_z = u(1.5, 3.0)
        t_z = u(0.0, 3.0)
        for key, value in (("V0", V0), ("V1", V0), ("vz", sign() * u(0.3, 1.0) * vz_max), ("T_z", T_z),
                           ("t_z", t_z), ("t_zo", t_z + T_z + u(1.0, 4.0))):
            p[key] = torch.where(climb, value, p[key])
        return types, p

    def _evaluation_episodes(self, n: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fixed parameters of eval_episode (accel/decel start after 1 s; turns start on the arc; no roll-out)."""
        c, dev = self.cfg, self.device
        full = lambda x: torch.full((n,), float(x), device=dev)
        V, V_end = full(c.eval_speed), full(c.eval_speed_end)
        p = {"V0": V, "V1": V, "t_v": full(1.0e3), "T_v": full(1.0), "kappa": full(0.0), "t_k": full(-1.0),
             "T_k": full(1.0), "t_ko": full(1.0e3), "vz": full(0.0), "t_z": full(1.0), "T_z": full(2.0),
             "t_zo": full(1.0e3), "psi0": (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi}
        kind = c.eval_episode
        if kind == "hover":
            p["V0"] = p["V1"] = full(0.0)
        elif kind in ("accel", "decel"):
            V0, V1 = (full(0.0), V) if kind == "accel" else (V, V_end)
            a = c.accel_max if kind == "accel" else self._decel_limit(V0)
            p.update(V0=V0, V1=V1, t_v=full(1.0), T_v=torch.clamp_min(1.5 * (V1 - V0).abs() / a, c.ramp_min_s))
        elif kind == "turn":
            p["kappa"] = full(1.0 / max(c.eval_turn_radius, 1e-3))
            p["vz"] = full(c.eval_vz)                       # helical turn when eval_vz != 0 (not in the training mixture)
        elif kind == "climb":
            p["vz"] = full(c.eval_vz)
        return torch.full((n,), EPISODE_TYPES.index(kind), dtype=torch.long, device=dev), p

    def _reset_reference_and_state(self, env_ids: torch.Tensor):
        """New reference, trimmed state of its first sample plus a perturbation, actuator history at the trim."""
        n, dev, c = env_ids.numel(), self.device, self.cfg
        types, p = self._evaluation_episodes(n) if c.eval_episode else self._sample_episodes(n)
        self._types[env_ids] = types

        # First reference sample: speed V0, heading psi0, curvature kappa * S_in(0)
        V0, psi0 = p["V0"], p["psi0"]
        kappa0 = p["kappa"] * smoothstep(-p["t_k"] / p["T_k"])[0]
        a_c = V0.square() * kappa0
        bank = -torch.atan(a_c / c.gravity)                 # left turn (kappa > 0): left wing down
        trim = self._trim.solve(V0, bank=bank, use_surfaces=True)
        R = trim_attitude(psi0, trim["alpha"], bank)
        yaw_rate = torch.stack((torch.zeros_like(V0), torch.zeros_like(V0), V0 * kappa0), dim=1)
        ang_vel = (R.transpose(1, 2) @ yaw_rate.unsqueeze(-1)).squeeze(-1)
        lin_vel = torch.stack((V0 * torch.cos(psi0), V0 * torch.sin(psi0), torch.zeros_like(V0)), dim=1)
        rotor_norm = trim["rotor_norm"].clone()

        # Perturbation: small, or large with the angle and rate bounds interpolated between hover and speed_max
        small = torch.tensor(c.small_perturbation, dtype=torch.float32, device=dev)
        frac = (V0 / c.speed_max).clamp(0.0, 1.0).unsqueeze(1)
        large = (torch.tensor(c.large_perturbation_hover, dtype=torch.float32, device=dev) * (1.0 - frac)
                 + torch.tensor(c.large_perturbation_fast, dtype=torch.float32, device=dev) * frac)
        is_large = (torch.rand(n, 1, device=dev) < c.large_perturbation_prob).float()
        mag = is_large * large + (1.0 - is_large) * small            # (n, 5)
        rnd = lambda k: torch.rand(n, k, device=dev) * 2.0 - 1.0
        pos_offset = rnd(3) * mag[:, 0:1]
        lin_vel = lin_vel + rnd(3) * mag[:, 1:2]
        R = R @ euler_angles_to_matrix(rnd(3) * torch.deg2rad(mag[:, 2:3]), convention="ZYX")
        ang_vel = ang_vel + rnd(3) * mag[:, 3:4]
        rotor_norm[:, :4] = (rotor_norm[:, :4] + rnd(4) * mag[:, 4:5]).clamp(-1.0, 1.0)

        overrides = dict(mask=torch.ones(n, dtype=torch.bool, device=dev), position_offset=pos_offset,
                         orientations=matrix_to_quaternion(R), linear_velocity=lin_vel,
                         angular_velocity=ang_vel, rotor_norm=rotor_norm)
        self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=False, randomize_state=False,
                                      init_overrides=overrides)
        self.episode_length_buf[env_ids] = 0
        centers = self.backend._vehicle._init_pos.to(device=dev, dtype=torch.float32)
        self._trajectory.reset(env_ids, centers, p)
        self._sync_reference()
        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)

        # Surfaces at the trim; the first command change is measured from the trimmed command
        self._surface_deflections[env_ids] = trim["surfaces"]
        thrust0 = trim["thrust"].clamp_min(0.0)
        self._rotor_diff_ema[env_ids] = (thrust0 - thrust0.mean(dim=1, keepdim=True)) / self._trim.T_max
        qbar0 = 0.5 * self._trim.G.rho * V0.square()
        self._moment_ema[env_ids] = self._lateral_moments(qbar0, thrust0, trim["surfaces"])
        reset_action = torch.cat((self._normalized_rotor_velocities(env_ids),
                                  self._normalize_surfaces(trim["surfaces"])), dim=1)
        self._action_history_obs[env_ids] = reset_action
        self._last_action[env_ids] = reset_action
        self._prev_action[env_ids] = reset_action
        self._raw_action[env_ids] = reset_action
        self._prev_raw_action[env_ids] = reset_action

    def _trim_from_speed(self, V0: torch.Tensor, requested_alpha: torch.Tensor | None = None):
        """Straight-flight trim (alpha, surfaces, rotor_norm) for the in-flight starts of play_multi."""
        out = self._trim.solve(V0, use_surfaces=True)
        return out["alpha"], out["surfaces"], out["rotor_norm"]
