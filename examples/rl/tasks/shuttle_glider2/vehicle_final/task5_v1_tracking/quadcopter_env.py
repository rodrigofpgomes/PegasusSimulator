"""
Shuttle + EasyGlider: three controlled reward comparisons.
Common physical trim, 46-D observation (absolute R), explicit desired velocity,
fixed reward weights, exact hover mixture and physical surface rate limiter.
PPO presets retained; no second difference and no CAPS in this comparison.
See README.md and MEMORIA_IMPLEMENTACOES.md for assumptions and integration.
Original author: Rodrigo Gomes. Original license: BSD-3-Clause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.rl.reset_manager import InitStateCfg
from pegasus.simulator.logic.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix

from .curriculum_manager import CurriculumManager, bounded_alpha_trim_rad
from .trajectory import TrackReferenceGenerator
from .reward_terms import compute_reward_terms


vehicle_physics_cfg = {
    "shuttle_thrust_cfg": {
        "num_rotors": 4,
        "rotor_constant": [1.709716e-05, 1.709716e-05, 1.709716e-05, 1.709716e-05],
        "rolling_moment_coefficient": [1e-06, 1e-06, 1e-06, 1e-06],
        "rot_dir": [-1, -1, 1, 1],
        "min_rotor_velocity": [0, 0, 0, 0],
        "max_rotor_velocity": [1400, 1400, 1400, 1400],
        "motor_time_constant": [0.008, 0.008, 0.008, 0.008],
    },
    "glider_thrust_cfg": {
        "thrust_constant": 8.5e-6,
        "moment_constant": 0.018,
        "reaction_moment_sign": -1.0,
        "min_rotor_velocity": 0.0,
        # Full puller authority (was capped at 1100 in curriculum2/3). At 3500 rad/s
        # the puller gives ~104 N (>2x weight), enough to reach the >=12 m/s
        # wing-offload regime within one episode.
        "max_rotor_velocity": 3500.0,
        "motor_time_constant": 0.0,
    },
    "aerodynamics_cfg": {'eval_at_ref': True,
 'geometry': {'rho': 1.2041,
              'S': 0.416,
              'b': 1.8,
              'c': 0.15407407407407406,
              'AR': 7.788461538461539,
              'e': 0.8523,
              'r_origin': (0.0, 0.0, -0.2803),
              'r_com': (0.0, 0.0, -0.2772)},
 'coefficients': {'CL0': 0.36986,
                  'CLa': 3.679875,
                  'CLq': 16.413084,
                  'CLde': 0.11235702362515444,
                  'CD0': 0.00759,
                  'CDq': -0.549541,
                  'CDde': 0.002521014298575622,
                  'Cm0': 0.57912,
                  'Cma': 6.591829,
                  'Cmq': -35.807747,
                  'Cmde': 0.48053970277622143,
                  'CY0': 0.0,
                  'CYb': 0.187754,
                  'CYp': -0.032506,
                  'CYr': 0.222424,
                  'CYda': 0.012261296815799617,
                  'CYdr': -0.07087487925768284,
                  'Cl0': 0.0,
                  'Clb': -0.062425,
                  'Clp': -0.392788,
                  'Clr': -0.137367,
                  'Clda': 0.18288812820575878,
                  'Cldr': 0.00813600069085769,
                  'Cn0': 0.0,
                  'Cnb': -0.108424,
                  'Cnp': 0.033202,
                  'Cnr': -0.129973,
                  'Cnda': 0.0,
                  'Cndr': 0.04073729923380153,
                  'alpha0': 0.3391428111,
                  'M_sig': 15.0}},
}


@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):
    observation_space: int = 40
    action_space: int = 8
    state_space: int = 0

    sim_dt: float = 0.01
    decimation: int = 1
    episode_length_s: float = 8.0

    action_mode: str = "rotor_velocity_direct"
    vehicle: str = "Shuttle_glider2"
    vehicle_physics_cfg: Any = field(default_factory=lambda: vehicle_physics_cfg)
    
    # Aileron
    delta_a_min = math.radians(-8.85)
    delta_a_max = math.radians(22.62)

    # Elevator
    delta_e_min = math.radians(-13.61)
    delta_e_max = math.radians(13.61)

    # Rudder
    delta_r_min = math.radians(-12.74)
    delta_r_max = math.radians(12.74)

    # Common stationary objective. Only reward_variant differs between packages.
    reward_variant: int = 1
    w_pos: float = 1.0
    w_vel: float = 0.3
    reward_velocity_scale: float = 1.0  # fixed [m/s], independent of ref speed
    w_quad_effort: float = 0.040
    w_puller_effort: float = 0.005
    w_d_rotor: float = 0.20            # mean of five command differences squared
    w_surf: float = 0.003              # small neutral-deflection cost
    w_surf_rate: float = 0.02          # physical rate, no second command cost
    w_hover_puller_effort: float = 0.040  # soft preference, never an action mask
    w_hover_ang_rate: float = 0.03
    hover_rate_scale: float = 1.0       # rad/s; no desired yaw/heading
    # Variant 2 adds physically motivated angle and actuator-margin terms.
    w_beta: float = 0.02
    beta_scale_deg: float = 5.0
    w_alpha: float = 0.03
    alpha_scale_deg: float = 3.0
    alpha_soft_max_deg: float = 15.0
    alpha_soft_min_deg: float = -4.0
    w_rotor_margin: float = 0.01
    rotor_margin_newtons: float = 2.0  # per lift rotor, lower AND upper limits
    w_elevator_margin: float = 0.01
    elevator_soft_ratio: float = 0.8
    aero_on_speed: float = 4.0
    aero_full_speed: float = 8.0
    # Variant 3 additionally tests modest offload; never pitch offload.
    w_ctrl_offload: float = 0.01
    w_yaw_offload: float = 0.005
    offload_turn_on_acc: float = 0.2    # reference radial acceleration [m/s^2]
    offload_turn_full_acc: float = 1.0
    surface_rate_max_deg_s: float = 150.0
    constant: float = 1.5
    termination_penalty: float = 200.0
    cost_clip: float = 6.0

    # Static observation normalisation (per-channel physical scales). Keeps every
    # channel O(1) WITHOUT the non-stationarity of a running normaliser, which
    # matters because the curriculum shifts the observation distribution across
    # levels. Set use_static_obs_norm=False to feed raw observations (A/B test).
    use_static_obs_norm: bool = True
    obs_scale_pos: float = 10.0     # position error [m]
    obs_scale_vel: float = 13.0     # velocity error / airspeed [m/s]
    obs_scale_acc: float = 15.0     # reference acceleration [m/s^2]
    obs_scale_angle: float = 0.5    # alpha, beta [rad]
    obs_scale_rate: float = 6.0     # body angular rates [rad/s]

    # Termination: position/velocity error norms; other limits per body axis.
    max_pos_error_per_axis: float = 3.0
    max_lin_vel_error_per_axis: float = 6.0
    max_absolute_lin_vel: float = 25.0
    max_ang_vel_per_axis: float = 35.0

    trajectory_generator: bool = True
    trajectory_ramp_duration: float = 2.5
    trajectory_a_max: float = 6.0          # feasibility bound [m/s^2]
    randomize_heading: bool = True
    trajectory_radial_acc_max: float = 6.0
    cruise_perturbation_scale: float = 0.1
    reset_rotor_jitter: float = 0.02

    # Physical constants for trim seeding (merged shuttle+glider).
    vehicle_mass: float = 4.9950
    gravity: float = 9.81

    curriculum_enabled: bool = True
    curriculum_start_level: int = 0
    curriculum_evaluation_episodes: int = 2048
    curriculum_required_consecutive_windows: int = 3 #5
    curriculum_previous_level_prob: float = 0.20
    curriculum_current_level_prob: float = 0.80
    curriculum_next_level_prob: float = 0.00

    randomize_init_state: bool = True
    init_state_cfg: InitStateCfg = field(default_factory=lambda: InitStateCfg(
        max_linear_velocity=0.05, max_angular_velocity=0.05, max_angle_deg=2.0,
        max_position=(0.02, 0.02, 0.01), guidance_prob=0.1))

    goal_pos_xy_range: list | None = None
    goal_pos_z_range: list | None = None

    test_mode: bool = False
    
    # Hybrid cruise-start trim
    trim_elevator_deg: float = -9.0
    trim_elevator_on_speed: float = 6.0
    trim_elevator_full_speed: float = 10.0

    # Rotor positions relative to the complete-vehicle CoM/origin [m]
    trim_front_x: float = 0.248
    trim_rear_x: float = -0.248
    trim_puller_z: float = -0.30073463808811335

    # Required thrust reserve in each front/rear rotor pair
    trim_pair_reserve: float = 4.0
    trim_elevator_soft_ratio: float = 0.8
    trim_search_iterations: int = 24


class QuadcopterEnv(PegasusEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)
        self._last_action = None
        self._prev_action = None
        
        
        self._action_history_obs = None
        self._trajectory: TrackReferenceGenerator | None = None
        self._curriculum: CurriculumManager | None = None
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._tracking_sums: dict[str, torch.Tensor] = {}
        self._rotor_metrics: dict[str, torch.Tensor] = {}

        self._death_cause: dict[str, torch.Tensor] = {}

        self._lvl: dict[str, torch.Tensor] = {}

        self._control_surfaces = None
        self._prev_surfaces = None

        self._surface_min = None
        self._surface_max = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self):
        if self.cfg.reward_variant not in (1, 2, 3):
            raise ValueError("reward_variant must be 1, 2 or 3")
        if not self.cfg.reward_velocity_scale > 0.0:
            raise ValueError("reward_velocity_scale must be positive")
        super().setup()


        self._last_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_action = torch.zeros_like(self._last_action)
        

        self._action_history_obs = torch.zeros_like(self._last_action)

        self._control_surfaces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_surfaces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._surface_min = torch.tensor([self.cfg.delta_e_min, self.cfg.delta_a_min, self.cfg.delta_r_min], dtype=torch.float32, device=self.device)
        self._surface_max = torch.tensor([self.cfg.delta_e_max, self.cfg.delta_a_max, self.cfg.delta_r_max], dtype=torch.float32, device=self.device)

        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device)
            for k in ("pos", "vel", "d_rotor", "hover_ang_rate", "quad_effort",
                      "puller_effort", "hover_puller_effort", "surf", "surf_rate", "beta", "alpha",
                      "rotor_margin", "elevator_margin", "ctrl_offload", "yaw_offload", "total")}
        self._tracking_sums = {k: torch.zeros(self.num_envs, device=self.device)
            for k in ("pos_error", "vel_error", "ang_rate", "steps")}
        self._rotor_metrics = {k: torch.zeros(self.num_envs, device=self.device)
            for k in ("steps", "puller_speed", "puller_active", "quad_effort", "puller_effort",
                      "airspeed", "elevator", "aileron", "rudder", "elevator_sat",
                      "aileron_sat", "rudder_sat", "roll_diff", "pitch_diff", "yaw_diff",
                      "command_change_rms", "rotor_lower_margin", "rotor_upper_margin")}

        self._death_cause = {k: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                             for k in ("pos_error", "vel_error", "abs_velocity", "ang_velocity")}

        if self.cfg.goal_pos_xy_range is None:
            self.cfg.goal_pos_xy_range = [-0.5, 0.5]
        if self.cfg.goal_pos_z_range is None:
            spawn_z = self.backend._vehicle._init_pos[0, 2].item()
            self.cfg.goal_pos_z_range = [spawn_z - 0.5, spawn_z + 0.5]

        self.backend.create_goal_markers(root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0))
        self.reset_manager.set_goal_cfg(self.cfg)

        self._curriculum = CurriculumManager(
            device=self.device, start_level=self.cfg.curriculum_start_level,
            evaluation_episodes=self.cfg.curriculum_evaluation_episodes,
            required_consecutive_windows=self.cfg.curriculum_required_consecutive_windows,
            previous_level_prob=self.cfg.curriculum_previous_level_prob,
            current_level_prob=self.cfg.curriculum_current_level_prob,
            next_level_prob=self.cfg.curriculum_next_level_prob)

        self._build_level_table()
        self._read_aero_and_thrust_constants()
        self._apply_initial_state_curriculum()

        all_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.trajectory_generator:
            self._trajectory = TrackReferenceGenerator(
                num_envs=self.num_envs, episode_steps=self.max_episode_length,
                dt=self.cfg.sim_dt * self.cfg.decimation, device=self.device,
                levels=self._curriculum.levels, ramp_duration=self.cfg.trajectory_ramp_duration,
                a_max=self.cfg.trajectory_a_max)
            self._reset_trajectory_and_state(all_ids, first_setup=True)
        else:
            self.reset_manager._randomize_goals(all_ids)

    def _build_level_table(self):
        levels = self._curriculum.levels
        def col(attr, transform=lambda x: x):
            return torch.tensor([transform(getattr(level, attr)) for level in levels],
                                dtype=torch.float32, device=self.device)
        self._lvl = {name: col(name) for name in (
            "speed_min", "speed_max", "rehearsal_prob", "rehearsal_min",
            "cruise_prob", "turn_probability", "turn_radius_min", "turn_radius_max",
            "vertical_amp", "vertical_period", "hover_probability",
            "max_position", "max_linear_velocity", "max_angular_velocity")}
        self._lvl["alpha_trim"] = col("alpha_trim_deg", math.radians)
        self._lvl["max_angle"] = col("max_angle_deg", math.radians)


    def _read_aero_and_thrust_constants(self):
        vehicle = self.backend._vehicle
        aero = vehicle.aerodynamics

        self._rho = aero.G.rho
        self._S = aero.G.S
        self._AR = aero.G.AR
        self._e = aero.G.e
        self._c = aero.G.c

        self._CL0 = aero.C.CL0
        self._CLa = aero.C.CLa
        self._CLde = aero.C.CLde
        self._CD0 = aero.C.CD0
        self._Cm0 = aero.C.Cm0
        self._Cma = aero.C.Cma
        self._Cmde = aero.C.Cmde

        r_origin = torch.as_tensor(aero.G.r_origin, dtype=torch.float32, device=self.device)
        self._r_aero_x = float(r_origin[0])
        self._r_aero_z = float(r_origin[2])

        self._W = self.cfg.vehicle_mass * self.cfg.gravity

        # thrust constants: combined interface stores [4 shuttle, 1 puller]
        rc = vehicle._thrusters._rotor_constant.to(self.device).reshape(-1)

        self._k_shuttle = float(rc[0])
        self._k_puller = float(rc[-1])

        self._min_w = vehicle._thrusters.min_rotor_velocity.to(self.device).reshape(-1)
        self._max_w = vehicle._thrusters.max_rotor_velocity.to(self.device).reshape(-1)

    # ------------------------------------------------------------------
    # Trim seeding for cruise starts
    # ------------------------------------------------------------------

    def _trim_aero_wrench(self, V0: torch.Tensor, alpha: torch.Tensor, delta_e: torch.Tensor):
        """Aerodynamic longitudinal wrench about the vehicle origin."""

        aero = self.backend._vehicle.aerodynamics
        qS = 0.5 * self._rho * V0.square() * self._S

        # Same nonlinear lift/drag model used by the simulation.
        CL, CD = aero.CL_CD(alpha, delta_e)

        lift = qS * (CL + self._CLde * delta_e)
        drag = (qS * CD).clamp_min(0.0)

        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        Fx = -drag * ca + lift * sa
        Fz = +drag * sa + lift * ca

        tau_ref = qS * self._c * (self._Cm0 + self._Cma * alpha + self._Cmde * delta_e)

        # tau_origin = tau_ref + r_origin x F_aero
        tau_origin = tau_ref + self._r_aero_z * Fx - self._r_aero_x * Fz

        return Fx, Fz, tau_origin

    def _trim_from_speed(self, V0: torch.Tensor, requested_alpha: torch.Tensor):
        """Longitudinal reset only; preserve force AND pitch equilibrium.

        Start below the old alpha cap and reduce alpha to satisfy BOTH lower
        and upper thrust reserve bounds. This does not impose alpha on policy.
        Lateral moments/reaction torque are not claimed to be trimmed here.
        """
        requested = bounded_alpha_trim_rad(V0, requested_alpha)
        elevator_gate = ((V0 - self.cfg.trim_elevator_on_speed)
                         / (self.cfg.trim_elevator_full_speed - self.cfg.trim_elevator_on_speed)).clamp(0.0, 1.0)
        de_min = self.cfg.delta_e_min * self.cfg.trim_elevator_soft_ratio
        de_max = self.cfg.delta_e_max * self.cfg.trim_elevator_soft_ratio
        delta_e = (math.radians(self.cfg.trim_elevator_deg) * elevator_gate).clamp(de_min, de_max)
        reserve = self.cfg.trim_pair_reserve
        pair_front_max = self._k_shuttle * (self._max_w[0].square() + self._max_w[2].square())
        pair_rear_max = self._k_shuttle * (self._max_w[1].square() + self._max_w[3].square())
        puller_max = self._k_puller * self._max_w[4].square()
        if reserve <= 0.0:
            raise ValueError("trim_pair_reserve must be positive")

        def equilibrium(alpha):
            Fx, Fz, tau_aero = self._trim_aero_wrench(V0, alpha, delta_e)
            Tp = self._W * torch.sin(alpha) - Fx
            Tq = self._W * torch.cos(alpha) - Fz
            My = tau_aero + self.cfg.trim_puller_z * Tp
            Tf = (My - self.cfg.trim_rear_x * Tq) / (self.cfg.trim_front_x - self.cfg.trim_rear_x)
            Tr = Tq - Tf
            valid = ((Tp >= 0.0) & (Tp <= puller_max)
                     & (Tf >= reserve) & (Tf <= pair_front_max - reserve)
                     & (Tr >= reserve) & (Tr <= pair_rear_max - reserve))
            return Tp, Tf, Tr, valid

        low = torch.zeros_like(requested)
        high = requested.clone()
        if not torch.all(equilibrium(low)[3]):
            raise RuntimeError("No feasible alpha=0 longitudinal reset for the configured physical model/reserve")
        # The current 0..alpha_cap branch is monotone in limiting rear-pair
        # feasibility. Keep already feasible requests unchanged.
        requested_ok = equilibrium(requested)[3]
        for _ in range(self.cfg.trim_search_iterations):
            mid = 0.5 * (low + high)
            ok = equilibrium(mid)[3]
            low = torch.where(ok, mid, low)
            high = torch.where(ok, high, mid)
        # Keep a small interior alpha margin for float32 rounding at the boundary.
        alpha_trim = torch.where(requested_ok, requested, (low - 1e-5).clamp_min(0.0))
        Tp, Tf, Tr, valid = equilibrium(alpha_trim)
        if not torch.all(valid):
            raise RuntimeError("Infeasible hybrid reset; verify coefficients, arm positions and reserve")
        omega = torch.zeros((V0.shape[0], 5), dtype=V0.dtype, device=self.device)
        wf = torch.sqrt(0.5 * Tf / self._k_shuttle)
        wr = torch.sqrt(0.5 * Tr / self._k_shuttle)
        omega[:, 0] = wf
        omega[:, 2] = wf
        omega[:, 1] = wr
        omega[:, 3] = wr
        omega[:, 4] = torch.sqrt(Tp / self._k_puller)
        if torch.any(omega > self._max_w[:5].unsqueeze(0) + 1e-4):
            raise RuntimeError("Reset thrust exceeds a rotor's maximum: clipping would break equilibrium")
        span = (self._max_w[:5] - self._min_w[:5]).clamp_min(1e-6).unsqueeze(0)
        rotor_norm = 2.0 * (omega - self._min_w[:5].unsqueeze(0)) / span - 1.0
        surface_trim = torch.zeros((V0.shape[0], 3), dtype=V0.dtype, device=self.device)
        surface_trim[:, 0] = delta_e
        return alpha_trim, surface_trim, rotor_norm


    def _normalize_surfaces(self, surfaces: torch.Tensor) -> torch.Tensor:
        limits = torch.where(surfaces >= 0.0, self._surface_max.unsqueeze(0), (-self._surface_min).unsqueeze(0))

        return surfaces / limits.clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Curriculum helpers
    # ------------------------------------------------------------------
    def _sample_trajectory_levels(self, n: int) -> torch.Tensor:
        if self._curriculum is None:
            return torch.zeros(n, dtype=torch.long, device=self.device)
        if not self.cfg.curriculum_enabled:
            return torch.full((n,), self._curriculum.current_level, dtype=torch.long, device=self.device)
        return self._curriculum.sample_levels(n)

    def _apply_initial_state_curriculum(self):
        if self._curriculum is None or not self.cfg.curriculum_enabled:
            return
        spec = self._curriculum.level
        values = {"max_position": spec.max_position, "max_linear_velocity": spec.max_linear_velocity,
                  "max_angular_velocity": spec.max_angular_velocity, "max_angle_deg": spec.max_angle_deg}
        for reset_cfg in (self.cfg.init_state_cfg, getattr(self.reset_manager, "init_state_cfg", None)):
            if reset_cfg is None:
                continue
            for name, value in values.items():
                if hasattr(reset_cfg, name):
                    setattr(reset_cfg, name, value)

    def _log_curriculum_state(self):
        if self._curriculum is None:
            return
        self.extras.setdefault("log", {})
        log = self.extras["log"]
        def scalar(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(())
        log["Curriculum/level"] = scalar(self._curriculum.current_level)
        log["Curriculum/good_windows"] = scalar(self._curriculum.good_windows)
        if self._curriculum.last_window_pos_error == self._curriculum.last_window_pos_error:
            log["Curriculum/window_pos_error"] = scalar(self._curriculum.last_window_pos_error)
            log["Curriculum/window_vel_error"] = scalar(self._curriculum.last_window_vel_error)
            log["Curriculum/window_success_rate"] = scalar(self._curriculum.last_window_success_rate)

    def _update_curriculum_from_completed_episodes(self, env_ids: torch.Tensor):
        if self._curriculum is None or not self.cfg.curriculum_enabled or self.cfg.test_mode:
            return
        valid = self._tracking_sums["steps"][env_ids] > 0
        if self._trajectory is not None:
            valid &= self._trajectory.env_levels[env_ids] == self._curriculum.current_level
        if not torch.any(valid):
            return
        ids = env_ids[valid]
        steps = self._tracking_sums["steps"][ids].clamp_min(1.0)
        mean_pos_error = self._tracking_sums["pos_error"][ids] / steps
        mean_vel_error = self._tracking_sums["vel_error"][ids] / steps
        #spec = self._curriculum.level
        #success = (self.reset_time_outs[ids].bool() & ~self.reset_terminated[ids].bool()
                   #& (mean_pos_error <= spec.pos_error_threshold)
                   #& (mean_vel_error <= spec.vel_error_threshold))
        
        success = self.reset_time_outs[ids].bool() & ~self.reset_terminated[ids].bool()
        
        changed = self._curriculum.observe_episodes(mean_pos_error, mean_vel_error, success)
        self._log_curriculum_state()
        if changed:
            self._apply_initial_state_curriculum()
            self._log_curriculum_state()
            spec = self._curriculum.level
            print(f"[Curriculum] -> level {self._curriculum.current_level} ({spec.name}) | "
                  f"V=[{spec.speed_min:.1f},{spec.speed_max:.1f}] cruise_p={spec.cruise_prob:.2f} "
                  f"turn_p={spec.turn_probability:.2f} a_trim={spec.alpha_trim_deg:.1f}deg")

    # ------------------------------------------------------------------
    # Reference + coordinated reset
    # ------------------------------------------------------------------
    def _sync_trajectory_reference(self, env_ids: torch.Tensor | None = None):
        if self._trajectory is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        step_ids = self.episode_length_buf[env_ids].to(dtype=torch.long)
        pos_ref, vel_ref, acc_ref = self._trajectory.current(env_ids, step_ids)
        self.reset_manager._goal_pos[env_ids] = pos_ref
        self.reset_manager._goal_vel[env_ids] = vel_ref
        self.reset_manager._goal_acc[env_ids] = acc_ref

    def set_tracking_reference(self, env_ids, position, velocity, acceleration):
        """Evaluation API: explicit position/velocity/acceleration, no yaw.

        Construct cfg with trajectory_generator=False and curriculum_enabled=False.
        Call after setup/reset and before the next policy step. This helper
        refuses to fight the training generator or silently reshape references.
        """
        if self._trajectory is not None:
            raise RuntimeError("External references require trajectory_generator=False")
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        for key, value in (("_goal_pos", position), ("_goal_vel", velocity), ("_goal_acc", acceleration)):
            data = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if data.shape == (3,):
                data = data.unsqueeze(0).expand(ids.numel(), -1)
            if data.shape != (ids.numel(), 3):
                raise ValueError("Each reference must have shape (3,) or (len(env_ids),3)")
            getattr(self.reset_manager, key)[ids] = data

    def _log_regime_metrics(self, env_ids):
        if self._trajectory is None:
            return
        log = self.extras.setdefault("log", {})
        modes = self._trajectory.env_modes[env_ids]
        for mode, name in ((TrackReferenceGenerator.HOVER, "hover"),
                           (TrackReferenceGenerator.STRAIGHT, "straight"),
                           (TrackReferenceGenerator.TURN, "turn")):
            mask = modes == mode
            if not torch.any(mask):
                continue
            ids = env_ids[mask]
            steps = self._tracking_sums["steps"][ids].clamp_min(1.0)
            for key in ("pos_error", "vel_error", "ang_rate"):
                log[f"Tracking/{name}/{key}"] = (self._tracking_sums[key][ids] / steps).mean()
            for key in ("quad_effort", "puller_effort", "command_change_rms",
                        "rotor_lower_margin", "rotor_upper_margin", "roll_diff", "pitch_diff", "yaw_diff"):
                log[f"Actuation/{name}/{key}"] = (self._rotor_metrics[key][ids] / steps).mean()
            log[f"Tracking/{name}/survival"] = (self.reset_time_outs[ids].bool() & ~self.reset_terminated[ids].bool()).float().mean()
            log[f"Trajectory/{name}/completed_fraction"] = mask.float().mean()


    def _reset_trajectory_and_state(self, env_ids: torch.Tensor, first_setup: bool = False):
        n = env_ids.numel()
        centers = self.backend._vehicle._init_pos.to(device=self.device, dtype=torch.float32)
        levels = self._sample_trajectory_levels(n)
        g = lambda key: self._lvl[key][levels]
        rnd = lambda: torch.rand(n, device=self.device)
        rehearse = rnd() < g("rehearsal_prob")
        speed_lo = torch.where(rehearse, g("rehearsal_min"), g("speed_min"))
        speed_target = speed_lo + rnd() * (g("speed_max") - speed_lo)
        # True point mass at hover, even when the final level has speed_max=13.
        hover = (g("speed_max") <= 0.0) | (rnd() < g("hover_probability"))
        speed_target = torch.where(hover, torch.zeros_like(speed_target), speed_target)
        cruise = (rnd() < g("cruise_prob")) & ~hover
        speed0 = torch.where(cruise, speed_target, torch.zeros_like(speed_target))
        turn = (rnd() < g("turn_probability")) & ~hover
        modes = torch.where(hover, torch.full_like(levels, TrackReferenceGenerator.HOVER),
            torch.where(turn, torch.full_like(levels, TrackReferenceGenerator.TURN),
                        torch.full_like(levels, TrackReferenceGenerator.STRAIGHT)))
        turn_radius = g("turn_radius_min") + rnd() * (g("turn_radius_max") - g("turn_radius_min"))
        if self.cfg.trajectory_radial_acc_max <= 0.0:
            raise ValueError("trajectory_radial_acc_max must be positive")
        # Bound the centripetal acceleration as well as the tangential ramp.
        min_radius = speed_target.square() / self.cfg.trajectory_radial_acc_max
        turn_radius = torch.where(turn, torch.maximum(turn_radius, min_radius), turn_radius)
        turn_dir = torch.where(rnd() < 0.5, -torch.ones(n, device=self.device), torch.ones(n, device=self.device))
        vertical_amp = g("vertical_amp") * torch.where(rnd() < 0.5, -torch.ones(n, device=self.device), torch.ones(n, device=self.device))
        vertical_amp = torch.where(hover, torch.zeros_like(vertical_amp), vertical_amp)
        vertical_period = g("vertical_period").clamp_min(1.0)
        alpha_trim, surface_trim, rotor_norm = self._trim_from_speed(speed0, g("alpha_trim"))
        heading = (rnd() * 2.0 - 1.0) * math.pi if self.cfg.randomize_heading else torch.zeros(n, device=self.device)
        angles = torch.stack((heading, -alpha_trim, torch.zeros_like(heading)), dim=1)
        lin_vel = torch.stack((speed0 * torch.cos(heading), speed0 * torch.sin(heading), torch.zeros(n, device=self.device)), dim=1)
        pos_offset = torch.zeros(n, 3, device=self.device)
        ang_vel = torch.zeros_like(pos_offset)
        if self.cfg.randomize_init_state:
            scale = torch.where(cruise, torch.full_like(speed0, self.cfg.cruise_perturbation_scale), torch.ones_like(speed0))
            pos_offset = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * g("max_position") * scale.unsqueeze(1)
            lin_vel += (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * (g("max_linear_velocity") * scale).unsqueeze(1)
            ang_vel = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * (g("max_angular_velocity") * scale).unsqueeze(1)
            angles += (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * (g("max_angle") * scale).unsqueeze(1)
            # Hover/accelerating starts now begin near the hover rotor equilibrium.
            jitter = (torch.rand(n, 4, device=self.device) * 2.0 - 1.0) * self.cfg.reset_rotor_jitter
            rotor_norm[:, :4] = (rotor_norm[:, :4] + (~cruise).float().unsqueeze(1) * jitter).clamp(-1.0, 1.0)
        ori = matrix_to_quaternion(euler_angles_to_matrix(angles, convention="ZYX"))
        overrides = dict(mask=torch.ones(n, dtype=torch.bool, device=self.device),
                         position_offset=pos_offset, orientations=ori, linear_velocity=lin_vel,
                         angular_velocity=ang_vel, rotor_norm=rotor_norm)
        # The supplied ResetManager's overrides are used for every episode;
        # perturbations have already been sampled with each episode's own level.
        self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=False,
                                      randomize_state=False, init_overrides=overrides)
        self.episode_length_buf[env_ids] = 0
        self._trajectory.reset(env_ids, centers, levels, heading, speed0, speed_target,
                               modes, turn_radius, turn_dir, vertical_amp, vertical_period)
        self._sync_trajectory_reference(env_ids)
        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)
        reset_rotor = self.backend._vehicle._thrusters._reset_rotor_norm[env_ids]
        reset_action = torch.zeros((n, self.cfg.action_space), device=self.device)
        reset_action[:, :5] = reset_rotor
        surface_state = torch.where(cruise.unsqueeze(1), surface_trim, torch.zeros_like(surface_trim))
        reset_action[:, 5:8] = self._normalize_surfaces(surface_state)
        self._action_history_obs[env_ids] = reset_action
        self._last_action[env_ids] = reset_action
        self._prev_action[env_ids] = reset_action
        self._control_surfaces[env_ids] = surface_state
        self._prev_surfaces[env_ids] = surface_state



    # ------------------------------------------------------------------
    # PegasusEnv interface
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._sync_trajectory_reference()
        # All eight channels remain policy-controlled in every regime,
        # including hover. Only the actuator's physical range is enforced.
        action = actions.clamp(-1.0, 1.0).clone()
        self._prev_action = self._last_action.clone()
        self._last_action = action
        self._action_history_obs = action.clone()
        self._prev_surfaces = self._control_surfaces.clone()


    def _apply_action(self):
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        half = 0.5 * (max_w - min_w)
        center = min_w + half

        # rotor commands: first 5 actions -> rotor angular velocity
        omega = self._last_action[:, :5] * half + center

        # Decode actions [-1,1] -> radians for [elevator, aileron, rudder].
        elevator_action = self._last_action[:, 5]
        aileron_action = self._last_action[:, 6]
        rudder_action = self._last_action[:, 7]
        
        elevator_target = torch.where(elevator_action >= 0.0, elevator_action * self.cfg.delta_e_max, elevator_action * (-self.cfg.delta_e_min))
        aileron_target = torch.where(aileron_action >= 0.0, aileron_action * self.cfg.delta_a_max, aileron_action * (-self.cfg.delta_a_min))
        rudder_target = torch.where(rudder_action >= 0.0, rudder_action * self.cfg.delta_r_max, rudder_action * (-self.cfg.delta_r_min))
        
        surface_step_max = math.radians(self.cfg.surface_rate_max_deg_s) * self.cfg.sim_dt
        target = torch.stack([elevator_target, aileron_target, rudder_target], dim=-1)
        
        step = (target - self._control_surfaces).clamp(-surface_step_max, surface_step_max)
        
        self._control_surfaces = self._control_surfaces + step
        
        command = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        command[:, 0:5] = omega
        command[:, 5:8] = self._control_surfaces

        self.backend._input_reference = command

    def _body_frame(self):
        state = self.backend.get_state()
        pos = state[:, 0:3]; vel_w = state[:, 3:6]; quat = state[:, 6:10]; ang_b = state[:, 10:13]
        R = quaternion_to_matrix(quat)
        Rt = R.transpose(1, 2)
        return state, pos, vel_w, quat, ang_b, R, Rt

    def _get_observations(self) -> dict:
        self._sync_trajectory_reference()
        
        _, pos, vel_w, quat, ang_b, R, Rt = self._body_frame()
        
        goal_pos = self.reset_manager.goal_pos
        goal_vel = self.reset_manager.goal_vel
        goal_acc = self.reset_manager.goal_acc
        
        pos_err_b = (Rt @ (pos - goal_pos).unsqueeze(-1)).squeeze(-1)
        vel_err_b = (Rt @ (vel_w - goal_vel).unsqueeze(-1)).squeeze(-1)
        
        vel_body = (Rt @ vel_w.unsqueeze(-1)).squeeze(-1)
        
        goal_vel_b = (Rt @ goal_vel.unsqueeze(-1)).squeeze(-1)
        goal_acc_b = (Rt @ goal_acc.unsqueeze(-1)).squeeze(-1)
        aero = self.backend._vehicle.aerodynamics
        Va, _, alpha, beta = aero.airdata_from_state(vel_body, ang_b)
        alpha = alpha.unsqueeze(1)
        beta = beta.unsqueeze(1)
        airspeed = Va.unsqueeze(1)
        R_flat = R.reshape(self.num_envs, 9)  # measured absolute attitude
        thrusters = self.backend._vehicle._thrusters
        min_w = thrusters.min_rotor_velocity
        max_w = thrusters.max_rotor_velocity
        rotor_speeds_norm = 2.0 * (thrusters._velocity - min_w) / (max_w - min_w).clamp_min(1e-6) - 1.0
        surfaces_norm = self._normalize_surfaces(self._control_surfaces)
        if self.cfg.use_static_obs_norm:
            pos_err_b = pos_err_b / self.cfg.obs_scale_pos
            vel_err_b = vel_err_b / self.cfg.obs_scale_vel
            vel_body = vel_body / self.cfg.obs_scale_vel
            goal_vel_b = goal_vel_b / self.cfg.obs_scale_vel
            goal_acc_b = goal_acc_b / self.cfg.obs_scale_acc
            alpha = alpha / self.cfg.obs_scale_angle
            beta = beta / self.cfg.obs_scale_angle
            airspeed = airspeed / self.cfg.obs_scale_vel
            ang = ang_b / self.cfg.obs_scale_rate
        else:
            airspeed = airspeed / 15.0
            ang = ang_b
        # The first 40 channels retain the original order. No yaw, desired
        # heading, heading error or Rrel is added. Last six channels are speeds.
        obs = torch.cat((pos_err_b, vel_err_b, alpha, beta, airspeed,
                         goal_acc_b, R_flat, ang, self._action_history_obs,
                         rotor_speeds_norm, surfaces_norm), dim=1) # vel_body, goal_vel_b
        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        self._sync_trajectory_reference()
        _, pos, vel_w, quat, ang_b, R, Rt = self._body_frame()
        pos_err_b = (Rt @ (pos - self.reset_manager.goal_pos).unsqueeze(-1)).squeeze(-1)
        vel_err_b = (Rt @ (vel_w - self.reset_manager.goal_vel).unsqueeze(-1)).squeeze(-1)
        vel_body = (Rt @ vel_w.unsqueeze(-1)).squeeze(-1)
        aero = self.backend._vehicle.aerodynamics
        Va, _, alpha, beta = aero.airdata_from_state(vel_body, ang_b)
        thrusters = self.backend._vehicle._thrusters
        surfaces_norm = self._normalize_surfaces(self._control_surfaces)
        terms, metrics = compute_reward_terms(
            self.cfg, pos_err_b, vel_err_b, ang_b, alpha, beta, Va,
            self.reset_manager.goal_vel, self.reset_manager.goal_acc,
            self._last_action, self._prev_action,
            thrusters._velocity, thrusters.min_rotor_velocity, thrusters.max_rotor_velocity,
            thrusters._rotor_constant.reshape(-1), surfaces_norm,
            self._control_surfaces - self._prev_surfaces,
            self.cfg.sim_dt * self.cfg.decimation)
        cost = sum(terms.values()).clamp(max=self.cfg.cost_clip)
        reward = self.cfg.constant - cost
        died = self._death_mask(pos_err_b, vel_err_b, vel_w, ang_b)
        reward[died] = -self.cfg.termination_penalty
        for key, term in terms.items():
            self._episode_sums[key] += -term
        self._episode_sums["total"] += reward
        self._tracking_sums["pos_error"] += metrics["pos_error"]
        self._tracking_sums["vel_error"] += metrics["vel_error"]
        self._tracking_sums["ang_rate"] += torch.linalg.norm(ang_b, dim=1)
        self._tracking_sums["steps"] += 1.0
        self._rotor_metrics["steps"] += 1.0
        self._rotor_metrics["puller_speed"] += thrusters._velocity[:, 4]
        self._rotor_metrics["puller_active"] += (thrusters._velocity[:, 4] > 0.1 * thrusters.max_rotor_velocity[4]).float()
        self._rotor_metrics["airspeed"] += Va
        for key in ("quad_effort", "puller_effort", "roll_diff", "pitch_diff", "yaw_diff",
                    "command_change_rms", "rotor_lower_margin", "rotor_upper_margin"):
            value = metrics[key]
            if key.endswith("_diff"):
                value = value.abs()
            self._rotor_metrics[key] += value
        for index, key in enumerate(("elevator", "aileron", "rudder")):
            self._rotor_metrics[key] += self._control_surfaces[:, index].abs()
            self._rotor_metrics[key + "_sat"] += (surfaces_norm[:, index].abs() >= 0.99).float()
        self._log_curriculum_state()
        return reward


    def _death_mask(self, pos_err_b, vel_err_b, vel_w, ang_b):
        # Norm limits are invariant to vehicle attitude. sqrt(3)*3m formerly
        # passed for some orientations despite having the same tracking norm.
        c_pos_error = torch.linalg.norm(pos_err_b, dim=1) > self.cfg.max_pos_error_per_axis
        c_vel_error = torch.linalg.norm(vel_err_b, dim=1) > self.cfg.max_lin_vel_error_per_axis
        c_abs_velocity = torch.linalg.norm(vel_w, dim=1) > self.cfg.max_absolute_lin_vel
        c_ang_velocity = (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)
        if self._death_cause:
            self._death_cause["pos_error"] = c_pos_error
            self._death_cause["vel_error"] = c_vel_error
            self._death_cause["abs_velocity"] = c_abs_velocity
            self._death_cause["ang_velocity"] = c_ang_velocity
        return c_pos_error | c_vel_error | c_abs_velocity | c_ang_velocity


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, pos, vel_w, quat, ang_b, R, Rt = self._body_frame()
        goal_pos = self.reset_manager.goal_pos
        goal_vel = self.reset_manager.goal_vel

        pos_err_b = (Rt @ (pos - goal_pos).unsqueeze(-1)).squeeze(-1)
        vel_err_b = (Rt @ (vel_w - goal_vel).unsqueeze(-1)).squeeze(-1)

        terminated = self._death_mask(pos_err_b, vel_err_b, vel_w, ang_b)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        state = self.backend.get_state()
        final_dist = torch.linalg.norm(self.reset_manager.goal_pos[env_ids] - state[env_ids, 0:3], dim=1).mean()

        self.extras.setdefault("log", {})
        log = self.extras["log"]
        log["Metrics/final_distance_to_goal"] = final_dist
        log["Episode_Termination/died"] = self.reset_terminated[env_ids].float().mean()
        log["Episode_Termination/timeout"] = self.reset_time_outs[env_ids].float().mean()

        term = self.reset_terminated[env_ids]
        n_term = term.float().sum().clamp_min(1.0)
        for name, mask in self._death_cause.items():
            log[f"Episode_Termination/cause_{name}"] = (mask[env_ids] & term).float().sum() / n_term

        for key, value in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = value[env_ids].mean()

        current_mask = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        if self._trajectory is not None:
            current_mask = self._trajectory.env_levels[env_ids] == self._curriculum.current_level
        if torch.any(current_mask):
            cids = env_ids[current_mask]
            steps = self._rotor_metrics["steps"][cids].clamp_min(1.0)
            log["Rotors/puller_speed"] = (self._rotor_metrics["puller_speed"][cids] / steps).mean()
            log["Rotors/puller_active"] = (self._rotor_metrics["puller_active"][cids] / steps).mean()
            log["Rotors/quad_effort"] = (self._rotor_metrics["quad_effort"][cids] / steps).mean()
            log["Rotors/puller_effort"] = (self._rotor_metrics["puller_effort"][cids] / steps).mean()
            log["Aerodynamics/Va"] = (self._rotor_metrics["airspeed"][cids] / steps).mean()
            log["Surfaces/elevator"] = (self._rotor_metrics["elevator"][cids] / steps).mean()
            log["Surfaces/aileron"] = (self._rotor_metrics["aileron"][cids] / steps).mean()
            log["Surfaces/rudder"] = (self._rotor_metrics["rudder"][cids] / steps).mean()
            log["Surfaces/elevator_saturation"] = (
                self._rotor_metrics["elevator_sat"][cids] / steps).mean()
            log["Surfaces/aileron_saturation"] = (
                self._rotor_metrics["aileron_sat"][cids] / steps).mean()
            log["Surfaces/rudder_saturation"] = (
                self._rotor_metrics["rudder_sat"][cids] / steps).mean()
            modes = self._trajectory.env_modes[cids]
            log["Trajectory/turn_fraction"] = (modes == TrackReferenceGenerator.TURN).float().mean()

        self._log_regime_metrics(env_ids)
        self._update_curriculum_from_completed_episodes(env_ids)
        self._apply_initial_state_curriculum()

        if self._trajectory is not None:
            self._reset_trajectory_and_state(env_ids)
        else:
            self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=True,
                                          randomize_state=self.cfg.randomize_init_state)
            self.episode_length_buf[env_ids] = 0

        for value in self._episode_sums.values():
            value[env_ids] = 0.0
        for value in self._tracking_sums.values():
            value[env_ids] = 0.0
        for value in self._rotor_metrics.values():
            value[env_ids] = 0.0

        self._call_reset_callbacks(env_ids)
