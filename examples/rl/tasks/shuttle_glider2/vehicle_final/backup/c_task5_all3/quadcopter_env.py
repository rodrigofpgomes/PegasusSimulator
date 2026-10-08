"""
| File: quadcopter_env.py  (curriculum4 - stage 2)
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Shuttle + EasyGlider PPO/SAC task, stage 2 (rotors + control surfaces).
|
|   1. Observation is expressed in the body frame and carries the aerodynamic
|      state explicitly (alpha, beta, Va), instead of world-frame errors.
|   2. Reference generation is airspeed/heading-parameterised, forward-biased,
|      feasibility-bounded and C2 (TrackReferenceGenerator).
|   3. Initial conditions include approximate force-balanced "cruise starts"
|      (matched attitude, body-forward velocity and seeded rotor speeds), so RL
|      sees the high-q_bar regime directly. Pitch-moment trim is not imposed.
|   4. Reward adds an angle-of-attack (stall-margin) term and normalises the
|      velocity cost by the reference speed.
|   5. Stage 2 adds three aerodynamic control surfaces (elevator, aileron,
|      rudder) as extra actions. In turns, a gated "control-offload" term asks
|      the aileron to progressively replace lift-rotor ROLL differential as
|      actual airspeed grows (surface authority is proportional to ~V_a^2).
|
| Observation (40 dims):
|   pos_error_body   (3)   vel_error_body (3)   [alpha, beta, Va] (3)
|   goal_acc_body    (3)   R_flat (9)           ang_b (3)
|   action_history   (8)   rotor_speeds_norm (5)  surfaces_norm (3)
|
| Action (8 dims): four shuttle lift rotors + EasyGlider puller (normalised
| [-1,1]) + [delta_e, delta_a, delta_r] (normalised [-1,1], mapped to correspondent limits).
| The puller cap is set in vehicle_physics_cfg (default 3500 rad/s ~ full authority).
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

DE_MAX = math.radians(25.0)   # elevator
DA_MAX = math.radians(25.0)   # aileron
DR_MAX = math.radians(25.0)   # rudder

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
    "aerodynamics_cfg": {},
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

    # Base reward weights (aero-shaping weights come from the curriculum level).
    w_pos: float = 1.0
    w_vel: float = 0.3
    
    w_d_rotor: float = 0.2
    w_d_surface: float = 0.05 #0.05
    
    w_dd_rotor: float = 0.0 #0.02
    w_dd_surface: float = 0.0 #0.005 #0.05
    
    w_surf: float = 0.03 #0.01
    w_surf_rate: float = 0.02
    
    surface_rate_max_deg_s: float = 150.0

    w_ctrl_offload: float = 0.0 #0.03
    surf_on_speed: float = 6.0        # surfaces start being effective ~6 m/s
    surf_full_speed: float = 10.0     # full ctrl-offload weight from here up
    
    w_puller_effort: float = 0.005

    w_ang_rate: float = 0.03
    
    quad_offload_on_speed: float = 6.0
    quad_offload_full_speed: float = 10.0
    
    constant: float = 1.5
    termination_penalty: float = 200.0
    cost_clip: float = 6.0

    # Angle-of-attack safe band (stall blend alpha0 ~ 19.4 deg).
    alpha_soft_max_deg: float = 15.0
    alpha_soft_min_deg: float = -4.0

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

    # Termination bounds (per body axis unless noted).
    max_pos_error_per_axis: float = 3.0
    max_lin_vel_error_per_axis: float = 6.0
    max_absolute_lin_vel: float = 25.0
    max_ang_vel_per_axis: float = 35.0

    trajectory_generator: bool = True
    trajectory_ramp_duration: float = 2.5
    trajectory_a_max: float = 6.0          # feasibility bound [m/s^2]
    randomize_heading: bool = True

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
    trim_pair_reserve: float = 2.0


class QuadcopterEnv(PegasusEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        super().__init__(cfg, backend, reset_manager)
        self._last_action = None
        self._prev_action = None
        
        self._prev_prev_action = None
        
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
        super().setup()

        self._last_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_action = torch.zeros_like(self._last_action)
        
        self._prev_prev_action = torch.zeros_like(self._last_action)

        self._action_history_obs = torch.zeros_like(self._last_action)

        self._control_surfaces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_surfaces = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._surface_min = torch.tensor([self.cfg.delta_e_min, self.cfg.delta_a_min, self.cfg.delta_r_min], dtype=torch.float32, device=self.device)
        self._surface_max = torch.tensor([self.cfg.delta_e_max, self.cfg.delta_a_max, self.cfg.delta_r_max], dtype=torch.float32, device=self.device)

        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device)
                              for k in ("pos", "vel", "d_rotor", "d_surface", "dd_rotor", "dd_surface", "heading", "beta", "alpha",
                                        "quad_effort", "puller_effort", "surf", "surf_rate", "ang_rate", "total")}
        self._tracking_sums = {k: torch.zeros(self.num_envs, device=self.device)
                              for k in ("pos_error", "vel_error", "steps")}
        self._rotor_metrics = {k: torch.zeros(self.num_envs, device=self.device)
                              for k in ("steps", "puller_speed", "puller_active", "quad_effort", "puller_effort",
                                        "airspeed", "elevator", "aileron", "rudder",
                                        "elevator_sat", "aileron_sat", "rudder_sat")}

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
            return torch.tensor([transform(getattr(l, attr)) for l in levels],
                                dtype=torch.float32, device=self.device)
        self._lvl = {
            "speed_min": col("speed_min"), "speed_max": col("speed_max"),
            "rehearsal_prob": col("rehearsal_prob"), "rehearsal_min": col("rehearsal_min"),
            "cruise_prob": col("cruise_prob"), "turn_probability": col("turn_probability"),
            "turn_radius_min": col("turn_radius_min"), "turn_radius_max": col("turn_radius_max"),
            "vertical_amp": col("vertical_amp"), "vertical_period": col("vertical_period"),
            "alpha_trim": col("alpha_trim_deg", math.radians),
            "w_heading": col("w_heading"), "w_beta": col("w_beta"),
            "w_alpha": col("w_alpha"), "w_quad_effort": col("w_quad_effort"),
        }

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

    def _trim_from_speed(self, V0: torch.Tensor, requested_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hybrid aerodynamic/surface/rotor trim for level cruise."""

        # Aerodynamically feasible AoA envelope.
        alpha_trim = bounded_alpha_trim_rad(V0, requested_alpha)

        # Elevator progresses smoothly from 0 deg at 6 m/s to -9 deg at 10 m/s.
        elevator_gate = torch.clamp((V0 - self.cfg.trim_elevator_on_speed) / (self.cfg.trim_elevator_full_speed - self.cfg.trim_elevator_on_speed), 0.0, 1.0)

        delta_e = math.radians(self.cfg.trim_elevator_deg) * elevator_gate

        Fx, Fz, tau_aero = self._trim_aero_wrench(V0, alpha_trim, delta_e)

        # Exact force equilibrium in the longitudinal plane.
        T_puller = self._W * torch.sin(alpha_trim) - Fx
        T_quad = self._W * torch.cos(alpha_trim) - Fz

        # Puller pitch moment about the complete-vehicle origin.
        pitch_moment = tau_aero + self.cfg.trim_puller_z * T_puller

        x_front = self.cfg.trim_front_x
        x_rear = self.cfg.trim_rear_x

        # My + z_p Tp - x_f Tf - x_r Tr = 0
        # Tf + Tr = T_quad
        T_front = (pitch_moment - x_rear * T_quad) / (x_front - x_rear)

        T_rear = T_quad - T_front

        # Detect an invalid trim instead of silently hiding it with clamp().
        invalid = (
            (T_puller < -1e-4)
            | (T_quad < -1e-4)
            | (T_front < self.cfg.trim_pair_reserve - 1e-4)
            | (T_rear < self.cfg.trim_pair_reserve - 1e-4)
        )

        if torch.any(invalid):
            bad = invalid.nonzero(as_tuple=False).squeeze(1)
            raise RuntimeError(
                "Infeasible hybrid trim for "
                f"{bad.numel()} environments. "
                "Reduce alpha_trim or elevator demand."
            )

        T_puller = T_puller.clamp_min(0.0)
        T_front = T_front.clamp_min(0.0)
        T_rear = T_rear.clamp_min(0.0)

        omega = torch.zeros((V0.shape[0], 5), dtype=V0.dtype, device=self.device)

        # Rotors 0/2 front; rotors 1/3 rear.
        omega_front = torch.sqrt((0.5 * T_front / self._k_shuttle).clamp_min(0.0))
        omega_rear = torch.sqrt((0.5 * T_rear / self._k_shuttle).clamp_min(0.0))
        omega_puller = torch.sqrt((T_puller / self._k_puller).clamp_min(0.0))

        omega[:, 0] = omega_front
        omega[:, 2] = omega_front
        omega[:, 1] = omega_rear
        omega[:, 3] = omega_rear
        omega[:, 4] = omega_puller

        omega = torch.maximum(omega, self._min_w[:5].unsqueeze(0))
        omega = torch.minimum(omega, self._max_w[:5].unsqueeze(0))

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

    def _reset_trajectory_and_state(self, env_ids: torch.Tensor, first_setup: bool = False):
        """Sample per-env task, build matched cruise-start overrides, reset & sync."""
        n = env_ids.numel()
        centers = self.backend._vehicle._init_pos.to(device=self.device, dtype=torch.float32)
        levels = self._sample_trajectory_levels(n)

        g = lambda key: self._lvl[key][levels]
        rnd = lambda: torch.rand(n, device=self.device)

        # Rehearsal mixture: most envs draw the reference speed from the current
        # focus band [speed_min, speed_max]; a fraction (rehearsal_prob) draws from
        # the full mastered envelope [rehearsal_min, speed_max] to avoid forgetting
        # slower regimes and the accelerate/decelerate transitions.
        rehearse = rnd() < g("rehearsal_prob")
        speed_lo = torch.where(rehearse, g("rehearsal_min"), g("speed_min"))
        speed_target = speed_lo + rnd() * (g("speed_max") - speed_lo)
        is_moving = g("speed_max") > 0.0
        cruise = (rnd() < g("cruise_prob")) & is_moving
        speed0 = torch.where(cruise, speed_target, torch.zeros_like(speed_target))

        turn = (rnd() < g("turn_probability")) & is_moving
        modes = torch.where(is_moving,
                            torch.where(turn, torch.full_like(levels, TrackReferenceGenerator.TURN),
                                        torch.full_like(levels, TrackReferenceGenerator.STRAIGHT)),
                            torch.full_like(levels, TrackReferenceGenerator.HOVER))
        turn_radius = g("turn_radius_min") + rnd() * (g("turn_radius_max") - g("turn_radius_min"))
        turn_dir = torch.where(rnd() < 0.5, -torch.ones(n, device=self.device), torch.ones(n, device=self.device))
        vertical_amp = g("vertical_amp") * torch.where(rnd() < 0.5, -torch.ones(n, device=self.device),
                                                       torch.ones(n, device=self.device))
        vertical_period = g("vertical_period").clamp_min(1.0)

        requested_alpha_trim = g("alpha_trim")
        alpha_trim, surface_trim, rotor_norm = self._trim_from_speed(speed0, requested_alpha_trim)

        if self.cfg.randomize_heading:
            heading = (rnd() * 2.0 - 1.0) * math.pi
        else:
            heading = torch.zeros(n, device=self.device)

        # cruise-start overrides (trim attitude, body-forward velocity, trim rotors)
        angles = torch.stack((heading, -alpha_trim, torch.zeros_like(heading)), dim=1)
        ori = matrix_to_quaternion(euler_angles_to_matrix(angles, convention="ZYX"))

        lin_vel = torch.stack([speed0 * torch.cos(heading), speed0 * torch.sin(heading), torch.zeros(n, device=self.device)], dim=1)

        overrides = {
            "mask": cruise,
            "position_offset": torch.zeros(n, 3, device=self.device),
            "orientations": ori,
            "linear_velocity": lin_vel,
            "angular_velocity": torch.zeros(n, 3, device=self.device),
            "rotor_norm": rotor_norm,
        }

        self.reset_manager.reset_envs(env_ids=env_ids, randomize_goals=False,
                                      randomize_state=self.cfg.randomize_init_state,
                                      init_overrides=overrides)
        self.episode_length_buf[env_ids] = 0
        self._trajectory.reset(env_ids, centers, levels, heading, speed0, speed_target,
                               modes, turn_radius, turn_dir, vertical_amp, vertical_period)
        self._sync_trajectory_reference(env_ids)
        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)

        # Seed all three action histories from the same reset command.
        # Surfaces start at cruise trim or neutral for a non-cruise start.
        reset_rotor = self.backend._vehicle._thrusters._reset_rotor_norm[env_ids]   # (n, 5)
        reset_action = torch.zeros((n, self.cfg.action_space), device=self.device)
        reset_action[:, :reset_rotor.shape[1]] = reset_rotor
        
        surface_action = self._normalize_surfaces(surface_trim)
        surface_action = torch.where(cruise.unsqueeze(1), surface_action, torch.zeros_like(surface_action))
        surface_state = torch.where(cruise.unsqueeze(1), surface_trim, torch.zeros_like(surface_trim))

        reset_action[:, 5:8] = surface_action

        self._action_history_obs[env_ids] = reset_action
        self._last_action[env_ids] = reset_action
        self._prev_action[env_ids] = reset_action
        self._prev_prev_action[env_ids] = reset_action

        self._control_surfaces[env_ids] = surface_state
        self._prev_surfaces[env_ids] = surface_state


    # ------------------------------------------------------------------
    # PegasusEnv interface
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        action = actions.clamp(-1.0, 1.0)
        
        self._prev_prev_action = self._prev_action.clone()
        
        self._prev_action = self._last_action.clone()
        self._last_action = action
        self._action_history_obs = action.clone()
        
        # Physical surface state at the start of the policy step.
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
        goal_acc_b = (Rt @ (goal_acc).unsqueeze(-1)).squeeze(-1)

        aero = self.backend._vehicle.aerodynamics

        vel_body = (Rt @ vel_w.unsqueeze(-1)).squeeze(-1)
        Va, _, alpha, beta = aero.airdata_from_state(linear_body_velocity=vel_body, angular_body_velocity=ang_b)

        alpha = alpha.unsqueeze(1)
        beta = beta.unsqueeze(1)
        airspeed = Va.unsqueeze(1)

        R_flat = R.reshape(self.num_envs, 9)   # already O(1); rotation entries in [-1, 1]
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        rotor_w = self.backend._vehicle._thrusters._velocity
        rotor_speeds_norm = (rotor_w - min_w) / (max_w - min_w) * 2.0 - 1.0  # already [-1, 1]

        surfaces_norm = self._normalize_surfaces(self._control_surfaces)   # ~[-1, 1]

        # Static per-channel normalisation by physical scale (stationary across the
        # curriculum). R_flat, action history, rotor speeds and surfaces are O(1).
        # NOTE: absolute world position is NOT in the observation (only pos_err_b),
        # and neither v_body nor goal_vel_b are included: the actual velocity is
        # captured by (alpha, beta, Va) and the reference velocity is recoverable
        # from vel_err_b, so both would be redundant.
        if self.cfg.use_static_obs_norm:
            pos_err_b = pos_err_b / self.cfg.obs_scale_pos
            vel_err_b = vel_err_b / self.cfg.obs_scale_vel
            goal_acc_b = goal_acc_b / self.cfg.obs_scale_acc
            alpha = alpha / self.cfg.obs_scale_angle
            beta = beta / self.cfg.obs_scale_angle
            Va = airspeed / self.cfg.obs_scale_vel
            ang = ang_b / self.cfg.obs_scale_rate
        else:
            Va = airspeed / 15.0   # keep airspeed roughly O(1) even when raw
            ang = ang_b

        obs = torch.cat([pos_err_b, vel_err_b, alpha, beta, Va,
                         goal_acc_b, R_flat, ang,
                         self._action_history_obs, rotor_speeds_norm, surfaces_norm], dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._sync_trajectory_reference()

        _, pos, vel_w, quat, ang_b, R, Rt = self._body_frame()

        goal_pos = self.reset_manager.goal_pos
        goal_vel = self.reset_manager.goal_vel

        aero = self.backend._vehicle.aerodynamics

        # Calculate aerodynamic state at the aerodynamic reference point.
        vel_body = (Rt @ vel_w.unsqueeze(-1)).squeeze(-1)
        Va, _, alpha, beta = aero.airdata_from_state(
            linear_body_velocity=vel_body,
            angular_body_velocity=ang_b,
        )

        pos_err_b = (Rt @ (pos - goal_pos).unsqueeze(-1)).squeeze(-1)
        vel_err_b = (Rt @ (vel_w - goal_vel).unsqueeze(-1)).squeeze(-1)

        # Position tracking cost
        pos_cost = torch.linalg.norm(pos_err_b, dim=1)
        pos_term = self.cfg.w_pos * pos_cost

        # Velocity tracking cost, normalised by the reference speed (xy-plane)
        ref_speed_xy = torch.linalg.norm(goal_vel[:, 0:2], dim=1)
        vel_cost = torch.linalg.norm(vel_err_b, dim=1) / (1.0 + ref_speed_xy)
        vel_term = self.cfg.w_vel * vel_cost
        
        # Penalise large changes in the rotor commands and control surfaces.
        d_action = self._last_action - self._prev_action
        d_rotor_cost = d_action[:, :5].square().sum(dim=1)# torch.linalg.norm(d_action[:, :5], dim=1) # d_action[:, :5].square().mean(dim=1)
        d_rotor_term = self.cfg.w_d_rotor * d_rotor_cost
        
        d_surface_cost = d_action[:, 5:].square().sum(dim=1) # torch.linalg.norm(d_action[:, 5:], dim=1) # d_action[:, 5:].square().mean(dim=1)
        d_surface_term = self.cfg.w_d_surface * d_surface_cost

        control_dt = self.cfg.sim_dt * self.cfg.decimation
        surf_step_max = math.radians(self.cfg.surface_rate_max_deg_s) * control_dt
        surf_rate_cost = ((self._control_surfaces - self._prev_surfaces) / surf_step_max).square().mean(dim=1)
        surf_rate_term = self.cfg.w_surf_rate * surf_rate_cost

        # Penalise large second-order changes in the rotor commands and control surfaces.
        dd_action = self._last_action - 2.0 * self._prev_action + self._prev_prev_action
        dd_rotor_cost = dd_action[:, :5].square().mean(dim=1)
        dd_rotor_term = self.cfg.w_dd_rotor * dd_rotor_cost
        
        dd_surface_cost = dd_action[:, 5:].square().mean(dim=1)
        dd_surface_term = self.cfg.w_dd_surface * dd_surface_cost
        
        # Penalise large control surface deflections (aero shaping).
        surfaces_norm = self._normalize_surfaces(self._control_surfaces)   # ~[-1, 1]
        surf_cost = surfaces_norm.square().mean(dim=1)
        surf_term = self.cfg.w_surf * surf_cost

        # Heading: body-x should point along the reference velocity (xy).
        forward_xy = R[:, 0:2, 0]
        ref_dir = goal_vel[:, 0:2] / ref_speed_xy.clamp_min(1e-6).unsqueeze(1)
        forward_dir = forward_xy / torch.linalg.norm(forward_xy, dim=1).clamp_min(1e-6).unsqueeze(1)
        heading_cost = (1.0 - torch.sum(ref_dir * forward_dir, dim=1).clamp(-1.0, 1.0))

        beta_cost = ((beta.abs() / math.radians(15.0)).square()).clamp(max=4.0)

        a_hi = math.radians(self.cfg.alpha_soft_max_deg)
        a_lo = math.radians(self.cfg.alpha_soft_min_deg)
        alpha_cost = ((alpha - a_hi).clamp_min(0.0).square()
                      + (a_lo - alpha).clamp_min(0.0).square()) / (math.radians(10.0) ** 2)
        alpha_cost = alpha_cost.clamp(max=4.0)

        # Penalise high rotor speeds (energy efficiency).
        actual_w = self.backend._vehicle._thrusters._velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        quad_effort_cost = torch.mean((actual_w[:, :4] / max_w[:4].unsqueeze(0)).square(), dim=1)
        puller_effort_cost = (actual_w[:, 4] / max_w[4]).square()

        # Per-env aero-shaping weights from the level actually being run.
        lv = self._trajectory.env_levels
        w_heading = self._lvl["w_heading"][lv]
        w_beta = self._lvl["w_beta"][lv]
        w_alpha = self._lvl["w_alpha"][lv]
        w_quad = self._lvl["w_quad_effort"][lv]

        forward_gate = ((ref_speed_xy - 2.0) / 3.0).clamp(0.0, 1.0)

        u_heading = torch.clamp((ref_speed_xy - 0.3) / (2.0 - 0.3), 0.0, 1.0)
        heading_gate = u_heading.square() * (3.0 - 2.0 * u_heading)
        heading_term = heading_gate * w_heading * heading_cost

        beta_term = forward_gate * w_beta * beta_cost
        alpha_term = forward_gate * w_alpha * alpha_cost

        quad_gate = torch.clamp((Va - self.cfg.quad_offload_on_speed)/ (self.cfg.quad_offload_full_speed - self.cfg.quad_offload_on_speed), 0.0, 1.0)
        quad_term = quad_gate * w_quad * quad_effort_cost

        puller_term = forward_gate * self.cfg.w_puller_effort * puller_effort_cost

        ang_rate = torch.linalg.norm(ang_b, dim=1)
        ang_rate_cost = (torch.relu(ang_rate - 1.0) / 3.0).square().clamp(max=4.0)
        ang_rate_term = self.cfg.w_ang_rate * ang_rate_cost

        cost = (pos_term + vel_term + d_rotor_term + d_surface_term + dd_rotor_term + dd_surface_term
                + surf_rate_term + heading_term + beta_term + alpha_term + surf_term + quad_term 
                + puller_term + ang_rate_term)

        cost = cost.clamp(max=self.cfg.cost_clip)
        reward = self.cfg.constant - cost

        died = self._death_mask(pos_err_b, vel_err_b, vel_w, ang_b)
        reward[died] = -self.cfg.termination_penalty

        self._episode_sums["pos"] += -pos_term
        self._episode_sums["vel"] += -vel_term
        self._episode_sums["d_rotor"] += -d_rotor_term
        self._episode_sums["d_surface"] += -d_surface_term
        self._episode_sums["dd_rotor"] += -dd_rotor_term
        self._episode_sums["dd_surface"] += -dd_surface_term
        self._episode_sums["heading"] += -heading_term
        self._episode_sums["beta"] += -beta_term
        self._episode_sums["alpha"] += -alpha_term
        self._episode_sums["quad_effort"] += -quad_term
        self._episode_sums["puller_effort"] += -puller_term
        self._episode_sums["surf"] += -surf_term
        self._episode_sums["surf_rate"] += -surf_rate_term
        self._episode_sums["ang_rate"] += -ang_rate_term
        self._episode_sums["total"] += reward

        self._tracking_sums["pos_error"] += pos_cost
        self._tracking_sums["vel_error"] += torch.linalg.norm(vel_err_b, dim=1)
        self._tracking_sums["steps"] += 1.0

        puller_speed = actual_w[:, 4]
        self._rotor_metrics["steps"] += 1.0
        self._rotor_metrics["puller_speed"] += puller_speed
        self._rotor_metrics["puller_active"] += (puller_speed > 0.10 * max_w[4]).float()
        self._rotor_metrics["quad_effort"] += quad_effort_cost
        self._rotor_metrics["puller_effort"] += puller_effort_cost
        self._rotor_metrics["airspeed"] += Va

        surface_abs = self._control_surfaces.abs()
        surface_saturated = surfaces_norm.abs() >= 0.99
        self._rotor_metrics["elevator"] += surface_abs[:, 0]
        self._rotor_metrics["aileron"] += surface_abs[:, 1]
        self._rotor_metrics["rudder"] += surface_abs[:, 2]
        self._rotor_metrics["elevator_sat"] += surface_saturated[:, 0].float()
        self._rotor_metrics["aileron_sat"] += surface_saturated[:, 1].float()
        self._rotor_metrics["rudder_sat"] += surface_saturated[:, 2].float()

        self._log_curriculum_state()
        return reward

    def _death_mask(self, pos_err_b, vel_err_b, vel_w, ang_b):
        c_pos_error = (pos_err_b.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        c_vel_error = (vel_err_b.abs() > self.cfg.max_lin_vel_error_per_axis).any(dim=1)
        c_abs_velocity = torch.linalg.norm(vel_w, dim=1) > self.cfg.max_absolute_lin_vel
        c_ang_velocity = (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

        if self._death_cause:
            self._death_cause["pos_error"]    = c_pos_error
            self._death_cause["vel_error"]    = c_vel_error
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
