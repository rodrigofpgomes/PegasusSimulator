"""
| File: quadcopter_env.py (raptor_pretrain)
| Description: Shuttle_glider 34D observation, H8 Frenet reference, signed rotor-5 bonus.

Observation (34 dims):
    pos_error                (3) - position error in world frame (pos - goal / reference)
    vel_error                (3) - velocity error in world frame
    R_flat                   (9) - rotation matrix, row-major
    ang_b                    (3) - angular velocity in body frame
    self._action_history_obs (5) - previous normalised motor command (ActionHistory length=1)
    rotor_speeds_norm        (5) - actual rotor speeds normalised to [-1, 1]
    acc_ref_b                (3) - reference acceleration in the BODY frame
    vel_ref_b                (3) - reference VELOCITY in the BODY frame

Action (5 dims):
    Normalised rotor velocity in [-1, 1], mapped to [min_w, max_w] = [0, 0, 0, 0, 0]..[1400, 1400, 1400, 1400, 3500].

Reward (static weights, no curriculum):
    r = constant - (w_pos*|pos_error| + w_vel*|vel_error| + w_d_action*|Δaction|). Termination penalty replaces the reward when the episode ends early. Cost is clipped at cost_clip=3.0.

Trajectory: null/Langevin-like reference mixture with ping-pong replay (pos, vel, acc).

Timing: decimation=1, sim_dt=0.01 -> 100 Hz, 500 steps = 5 s per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.rl.reset_manager import InitStateCfg
from pegasus.simulator.logic.transforms import quaternion_to_matrix

from .trajectory import RaptorLikeTrajectory

_SHUTTLE_GLIDER_PHYSICS_CFG = {
    "num_rotors": 5,
    "rotor_constant":             [1.709716e-05, 1.709716e-05, 1.709716e-05, 1.709716e-05, 8.54858e-06],
    "rolling_moment_coefficient": [1e-06, 1e-06, 1e-06, 1e-06, 0.0],
    "rot_dir":                    [-1, -1, 1, 1, 1],
    "min_rotor_velocity":         [0, 0, 0, 0, 0],
    "max_rotor_velocity":         [1400, 1400, 1400, 1400, 1000],
    "motor_time_constant":        [0.008, 0.008, 0.008, 0.008, 0.0125],
}

@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):

    # --- spaces ---
    """Configuration for the quadcopter hover task: observation/action/state spaces, reward scales and termination bounds."""
    observation_space: int = 34
    action_space: int = 5
    state_space: int = 0

    # --- timing (100 Hz, 5 s episodes) ---
    sim_dt: float = 0.01
    decimation: int = 1
    episode_length_s: float = 5.0

    # --- vehicle ---
    action_mode: str = "rotor_velocity_direct"
    vehicle: str = "Shuttle_glider"

    # Physics parameters forwarded to MultirotorBatchConfig (None = simulator defaults)
    vehicle_physics_cfg: Any = field(default_factory=lambda: _SHUTTLE_GLIDER_PHYSICS_CFG)

    # --- reward weights (RAPTOR sample_dynamics_parameters.cpp) ---
    w_pos:      float = 1.0    # position squared error
    w_vel:      float = 0.3    # velocity cost
    w_d_action: float = 1.0    # delta-action squared, rotors 1-4 only
    # Rotor 5 gets its own smoothness weight. With a single ||d_action|| over
    # all five dimensions, modulating the puller is punished exactly as much as
    # modulating lift -- the behaviour we want is actively discouraged.
    w_d_action_5: float = 0.1
    # Heading term: 0.5*(1 - cos(heading error)) gated on a moving reference.
    # 0.0 disables it (control variant).
    w_align:    float = 0.0
    # Sideslip term: |v_lateral_body|, same gate.
    w_lateral:  float = 0.0

    # --- rotor-5 terms (family F) -----------------------------------------
    # Bonus on the SIGNED useful thrust: (f5 / f5_max) * cos(heading error).
    # This replaces the geometric heading penalty of family E. It pays for
    # partial alignment AND for modulation, and it does not demand that the
    # nose lock onto the velocity vector -- which is physically impossible
    # below T ~ 7 s anyway. It is exactly the signed quantity the evaluation
    # reports as thrust_5_along_ref.
    w_use:      float = 0.0
    # Cost on the rotor-5 LEVEL, (f5 / f5_max). Without it, holding the puller
    # high is free: family E learned a near-constant 4.33 N bias at 71 % of the
    # command range, pointing 86 deg away from travel, trimmed out with 7.9 deg
    # of permanent tilt. w_d_action_5 only penalises CHANGES, never the level.
    # Net effect of the pair: reward = f5_frac * (w_use * align - w_level), so
    # the puller pays for itself only when align > w_level / w_use.
    w_level:    float = 0.0
    # Feasibility gate: skip the bonus when the reference demands a heading
    # rate the airframe cannot track. Yaw authority is 5.92 rad/s^2; a T = 6 s
    # lemniscate exceeds it 22 % of the time and a T = 4 s one 60 %. None
    # disables the gate.
    hr_ref_max: float | None = None
    constant:   float = 1.5
    termination_penalty: float = 200.0
    cost_clip:  float = 3.0

    # --- termination + goal range: scaled from vehicle geometry at setup() ---
    # RAPTOR: max_pos_error = max_rotor_distance * 20 (per axis)
    #         goal_range     = max_rotor_distance * 10
    # Set to None to trigger auto-scaling in setup(); override with a float to fix manually.
    max_pos_error_per_axis: float = 1.0
    max_lin_vel_per_axis: float = 3.0
    max_ang_vel_per_axis: float = 35.0
    min_upright: float = -0.17

    use_raptor_trajectory: bool = True
    # With the H8 generator this is the probability of a MOVING reference
    # (frenet or lemniscate); 1 - p is the static condition.
    trajectory_mixture_langevin_prob: float = 0.65
    langevin_gamma: float = 1.0
    langevin_omega: float = 2.0
    langevin_sigma: float = 0.5
    #langevin_alpha: float = 0.01

    goal_pos_xy_range: list | None = None
    goal_pos_z_range:  list | None = None

    # --- initial state randomisation (RAPTOR init_90_deg) ---
    randomize_init_state: bool = True
    init_state_cfg: InitStateCfg = field(default_factory=lambda: InitStateCfg(max_angle_deg=90.0, guidance_prob=0.1))

    # --- observation clamping during test/evaluation ---
    test_mode: bool = False
    clamp_observations_in_test: bool = True

    obs_pos_error_limit: float = 0.3
    obs_vel_error_limit: float = 0.5


class QuadcopterEnv(PegasusEnv):
    """Quadcopter hover task environment. See this phase's README for the exact reward formulation."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        """Initializes the task: caches config/backend/reset-manager and allocates the per-episode reward logging buffers."""
        super().__init__(cfg, backend, reset_manager)
        self._last_action: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self._action_history_obs: torch.Tensor | None = None

        self._trajectory = None

        self._episode_sums: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Allocates device buffers once the simulation timeline is active."""
        super().setup()
        self._last_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_action = torch.zeros_like(self._last_action)
        self._action_history_obs = torch.zeros_like(self._last_action)
        
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ("pos", "vel", "d_action", "d_action_5", "align",
                      "lateral", "use", "total",
                      # aux_* are diagnostic accumulators, not reward terms.
                      "aux_gate", "aux_align", "aux_f5", "aux_f5frac",
                      "aux_cs_x", "aux_cs_y", "aux_cs_xx", "aux_cs_yy",
                      "aux_cs_xy", "aux_t5signed", "aux_feasible")
        }

        # Auto-scale termination and goal range from vehicle geometry (RAPTOR behaviour)
        #rotor_pos = self.backend._vehicle._rotor_positions_body[0]  # (num_rotors, 3)
        #max_rotor_dist = rotor_pos.norm(dim=1).max().item()
        if self.cfg.max_pos_error_per_axis is None:
            self.cfg.max_pos_error_per_axis = 1.0
            # self.cfg.max_pos_error_per_axis = max_rotor_dist * 20.0
        if self.cfg.goal_pos_xy_range is None:
            # r = max_rotor_dist * 10.0
            self.cfg.goal_pos_xy_range = [-0.5, 0.5]
        if self.cfg.goal_pos_z_range is None:
            # r = max_rotor_dist * 10.0
            spawn_z = self.backend._vehicle._init_pos[0, 2].item()
            self.cfg.goal_pos_z_range = [spawn_z - 0.5, spawn_z + 0.5]

        #print(f"[QuadcopterEnv] max_rotor_dist={max_rotor_dist:.4f}m  "
        #      f"termination={self.cfg.max_pos_error_per_axis:.3f}m  "
        #      f"goal_xy={self.cfg.goal_pos_xy_range}  goal_z={self.cfg.goal_pos_z_range}")

        self.backend.create_goal_markers(
            root_path="/World/GoalMarkers", size=0.15, color=(1.0, 0.0, 0.0)
        )

        self.reset_manager.set_goal_cfg(self.cfg)

        all_ids = torch.arange(self.num_envs, device=self.device)

        if self.cfg.use_raptor_trajectory:
            self._trajectory = RaptorLikeTrajectory(
                num_envs=self.num_envs,
                episode_steps=self.max_episode_length,
                dt=self.cfg.sim_dt * self.cfg.decimation,
                device=self.device,
                gamma=self.cfg.langevin_gamma,
                omega=self.cfg.langevin_omega,
                sigma=self.cfg.langevin_sigma,
                #alpha=self.cfg.langevin_alpha,
                mixture_langevin_prob=self.cfg.trajectory_mixture_langevin_prob,
            )

            centers = self.backend._vehicle._init_pos.to(
                device=self.device, dtype=torch.float32
            )

            self._trajectory.reset(all_ids, centers)
            self._sync_trajectory_reference(all_ids)
        else:
            self.reset_manager._randomize_goals(all_ids)


    def _sync_trajectory_reference(self, env_ids: torch.Tensor | None = None):
        """Updates the reference-trajectory target used by observations/rewards for the current step."""
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


    # ------------------------------------------------------------------
    # PegasusEnv interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Stores and rescales the raw policy actions before the physics substeps."""
        action = actions.clamp(-1.0, 1.0)

        self._prev_action = self._last_action.clone()
        self._last_action = action

        self._action_history_obs = action.clone()

    def _apply_action(self):
        """Map normalised action [-1,1] -> rotor velocity [min_w, max_w] and send."""
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        half = 0.5 * (max_w - min_w)
        center = min_w + half
        omega = self._last_action * half + center   # (N, 4)  rad/s
        self.backend._input_reference = omega
        self.backend._vehicle._thrusters.set_input_reference(omega)

    def _get_observations(self) -> dict:
        """Builds the observation dict for the policy (and critic, when a state space is defined)."""
        self._sync_trajectory_reference()

        state = self.backend.get_state()
        pos   = state[:, 0:3]
        vel_w = state[:, 3:6]
        quat  = state[:, 6:10]   # w,x,y,z
        ang_b = state[:, 10:13]

        pos_error = pos - self.reset_manager.goal_pos
        vel_error = vel_w - self.reset_manager.goal_vel

        # Clamp observations only during test/evaluation
        if self.cfg.test_mode and self.cfg.clamp_observations_in_test:
            pos_error = pos_error.clamp(-self.cfg.obs_pos_error_limit, self.cfg.obs_pos_error_limit)
            vel_error = vel_error.clamp(-self.cfg.obs_vel_error_limit, self.cfg.obs_vel_error_limit)

        # goal_acc = self.reset_manager.goal_acc

        R = quaternion_to_matrix(quat)
        R_flat = R.reshape(self.num_envs, 9)

        # Rotor speeds normalised to [-1, 1]: matches rl-tools RotorSpeeds observation
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        rpm   = self.backend._vehicle._thrusters._velocity  # (N, 4) actual rotor speeds
        rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0

        # Reference acceleration in the body frame. This is the only signal the
        # puller can act on, and it is exactly what the 28D ablation removed.
        acc_ref_b = torch.bmm(
            R.transpose(1, 2), self.reset_manager.goal_acc.unsqueeze(-1)
        ).squeeze(-1)

        # Reference VELOCITY in the body frame.
        #
        # This is the fix for the defect that made family E fail. The
        # observation carried only vel_error = v - goal_vel, so goal_vel was
        # NOT recoverable without knowing v, which is also absent. The heading
        # reward was therefore a function of an unobservable quantity: the
        # policy was being graded on something it could not see. Measured
        # consequence: w_align raised total yaw travel from 158 to 331 deg per
        # 20 s episode while corr(yaw rate, demanded yaw rate) stayed at -0.06.
        # The gradient existed; the signal did not.
        #
        # It also makes the heading error directly readable: cos and sin of it
        # are vel_ref_b[0] and vel_ref_b[1] divided by the xy norm.
        vel_ref_b = torch.bmm(
            R.transpose(1, 2), self.reset_manager.goal_vel.unsqueeze(-1)
        ).squeeze(-1)

        obs = torch.cat([pos_error, vel_error, R_flat, ang_b, self._action_history_obs, rotor_speeds_norm, acc_ref_b, vel_ref_b], dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Computes and logs the reward signal for the current timestep."""
        self._sync_trajectory_reference()

        state = self.backend.get_state()
        pos = state[:, 0:3]
        vel_w = state[:, 3:6]
        quat  = state[:, 6:10]
        ang_b = state[:, 10:13]

        pos_error = pos - self.reset_manager.goal_pos
        vel_error = vel_w - self.reset_manager.goal_vel

        d_action = self._last_action - self._prev_action

        pos_cost = torch.linalg.norm(pos_error, dim=1)
        vel_cost = torch.linalg.norm(vel_error, dim=1)
        d_action_cost = torch.linalg.norm(d_action[:, :4], dim=1)
        d_action_5_cost = d_action[:, 4].abs()

        # --- heading / sideslip shaping ---------------------------------------
        # Body +X is the puller axis. align = cos(angle between the nose and the
        # direction of travel demanded by the reference.
        R_rew = quaternion_to_matrix(quat)
        fwd_xy = R_rew[:, :2, 0]
        fwd_hat = fwd_xy / torch.linalg.norm(fwd_xy, dim=1, keepdim=True).clamp(min=1e-6)
        ref_xy = self.reset_manager.goal_vel[:, :2]
        ref_n = torch.linalg.norm(ref_xy, dim=1)
        ref_hat = ref_xy / ref_n.unsqueeze(1).clamp(min=1e-6)
        # Same threshold the evaluation recorder uses (ref_speed_xy > 0.15), so
        # the reward and the reported metric agree. Static episodes are excluded.
        gate = (ref_n > 0.15).float()
        align = (fwd_hat * ref_hat).sum(dim=1).clamp(-1.0, 1.0)
        align_cost = 0.5 * (1.0 - align) * gate      # 0 aligned .. 1 backwards
        vel_b = torch.bmm(R_rew.transpose(1, 2), vel_w.unsqueeze(-1)).squeeze(-1)
        lateral_cost = vel_b[:, 1].abs() * gate

        # --- demanded heading rate, for the feasibility gate ------------------
        # For a planar curve, hr = (v x a)_z / |v|^2. Computed from goal_vel and
        # goal_acc directly, so no history buffer is needed.
        gv = self.reset_manager.goal_vel
        ga = self.reset_manager.goal_acc
        hr_ref = (gv[:, 0] * ga[:, 1] - gv[:, 1] * ga[:, 0]) / (
            gv[:, 0] ** 2 + gv[:, 1] ** 2
        ).clamp(min=1e-4)
        if self.cfg.hr_ref_max is None:
            feasible = torch.ones_like(gate)
        else:
            feasible = (hr_ref.abs() < self.cfg.hr_ref_max).float()

        # --- rotor-5 diagnostics (logging only, no gradient) ------------------
        thr = self.backend._vehicle._thrusters
        w5 = thr._velocity[..., 4]
        kf5 = torch.as_tensor(thr._rotor_constant, device=self.device, dtype=torch.float32)[..., 4]
        mw5 = torch.as_tensor(thr.max_rotor_velocity, device=self.device, dtype=torch.float32)[..., 4]
        f5 = kf5 * w5 * w5
        a_dem_x = torch.bmm(
            R_rew.transpose(1, 2), self.reset_manager.goal_acc.unsqueeze(-1)
        ).squeeze(-1)[:, 0]
        f5_frac = (w5 / mw5).clamp(0.0, 1.0) ** 2       # thrust fraction, f5/f5_max
        # SIGNED useful thrust, in [-1, 1]. The evaluation harness reports the
        # rectified max(T5 * align, 0), which is positive by construction when
        # align ~ 0 and T5 is large -- family E scored 1.45 N that way while the
        # signed value was 0.08 N. Never optimise the rectified version.
        use_signed = f5_frac * align * gate * feasible
        level_cost = f5_frac * gate

        cost = (self.cfg.w_pos * pos_cost + self.cfg.w_vel * vel_cost
                + self.cfg.w_d_action * d_action_cost
                + self.cfg.w_d_action_5 * d_action_5_cost)
        cost = cost.clamp(max=self.cfg.cost_clip)        

        # CRITICAL: the shaping terms are subtracted OUTSIDE the clip. Inside it
        # they would saturate cost_clip and zero the position gradient -- that is
        # exactly the mechanism that broke the v4 heading environments.
        shaping = self.cfg.w_align * align_cost + self.cfg.w_lateral * lateral_cost

        # The bonus is ADDED outside the clip, for the same reason the shaping is
        # subtracted outside it: inside, it would saturate cost_clip and zero the
        # position gradient.
        bonus = self.cfg.w_use * use_signed - self.cfg.w_level * level_cost

        reward = self.cfg.constant - cost - shaping + bonus

        # Termination penalty replaces the normal reward when the episode ends early
        died = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        died |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        died |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

        reward[died] = -self.cfg.termination_penalty

        self._episode_sums["pos"] += -self.cfg.w_pos * pos_cost
        self._episode_sums["vel"] += -self.cfg.w_vel * vel_cost
        self._episode_sums["d_action"] += -self.cfg.w_d_action * d_action_cost
        self._episode_sums["d_action_5"] += -self.cfg.w_d_action_5 * d_action_5_cost
        self._episode_sums["align"] += -self.cfg.w_align * align_cost
        self._episode_sums["lateral"] += -self.cfg.w_lateral * lateral_cost
        self._episode_sums["aux_gate"] += gate
        self._episode_sums["aux_align"] += align * gate
        self._episode_sums["aux_f5"] += f5
        self._episode_sums["aux_f5frac"] += w5 / mw5
        self._episode_sums["aux_cs_x"] += f5 * gate
        self._episode_sums["aux_cs_y"] += a_dem_x * gate
        self._episode_sums["aux_cs_xx"] += f5 * f5 * gate
        self._episode_sums["aux_cs_yy"] += a_dem_x * a_dem_x * gate
        self._episode_sums["aux_cs_xy"] += f5 * a_dem_x * gate
        self._episode_sums["use"] += bonus
        self._episode_sums["aux_t5signed"] += f5 * align * gate
        self._episode_sums["aux_feasible"] += feasible * gate
        self._episode_sums["total"] += reward

        return reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        """Returns the (terminated, truncated) flags for the current timestep."""
        state = self.backend.get_state()
        pos   = state[:, 0:3]
        vel_w = state[:, 3:6]
        quat  = state[:, 6:10]
        ang_b = state[:, 10:13]

        R = quaternion_to_matrix(quat)

        pos_error = pos - self.reset_manager.goal_pos
        vel_error = vel_w - self.reset_manager.goal_vel
 
        terminated = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        terminated |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        terminated |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Resets the selected environments and updates the logging statistics."""
        if env_ids.numel() == 0:
            return

        state = self.backend.get_state()

        final_dist = torch.linalg.norm(
            self.reset_manager.goal_pos[env_ids] - state[env_ids, 0:3], dim=1
        ).mean()

        self.extras.setdefault("log", {})
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist
        self.extras["log"]["Episode_Termination/died"]    = self.reset_terminated[env_ids].float().mean()
        self.extras["log"]["Episode_Termination/timeout"] = self.reset_time_outs[env_ids].float().mean()
        # Rotor-5 / heading metrics, computed BEFORE the accumulators are zeroed.
        # The correlation and the alignment are averaged over MOVING episodes only:
        # a static episode has zero demand variance and would otherwise contribute
        # a forced zero, halving the reported number.
        gate_sum = self._episode_sums["aux_gate"][env_ids]
        moving = gate_sum > 0
        ep_len = self.episode_length_buf[env_ids].float().clamp(min=1.0)

        self.extras["log"]["Metrics/moving_frac"] = moving.float().mean()
        self.extras["log"]["Metrics/thrust5_N"] = (
            self._episode_sums["aux_f5"][env_ids] / ep_len
        ).mean()
        self.extras["log"]["Metrics/thrust5_frac"] = (
            self._episode_sums["aux_f5frac"][env_ids] / ep_len
        ).mean()

        if moving.any():
            g = gate_sum[moving]
            # SIGNED useful rotor-5 thrust in newtons. This is the headline
            # number: it is what the test reports as thrust_5_along_ref and it
            # must reach ~1.0 N to matter (the test itself demands 1.03 N).
            self.extras["log"]["Metrics/thrust5_signed_N"] = (
                self._episode_sums["aux_t5signed"][env_ids][moving] / g
            ).mean()
            self.extras["log"]["Metrics/feasible_frac"] = (
                self._episode_sums["aux_feasible"][env_ids][moving] / g
            ).mean()
            self.extras["log"]["Metrics/heading_align"] = (
                self._episode_sums["aux_align"][env_ids][moving] / g
            ).mean()
            mx = self._episode_sums["aux_cs_x"][env_ids][moving] / g
            my = self._episode_sums["aux_cs_y"][env_ids][moving] / g
            vxx = (self._episode_sums["aux_cs_xx"][env_ids][moving] / g - mx * mx).clamp(min=0.0)
            vyy = (self._episode_sums["aux_cs_yy"][env_ids][moving] / g - my * my).clamp(min=0.0)
            cov = self._episode_sums["aux_cs_xy"][env_ids][moving] / g - mx * my
            corr = cov / torch.sqrt(vxx * vyy + 1e-12)
            self.extras["log"]["Metrics/corr_thrust5_accdem"] = corr.mean()

        for k, v in self._episode_sums.items():
            if not k.startswith("aux_"):
                self.extras["log"][f"Episode_Reward/{k}"] = v[env_ids].mean()
            self._episode_sums[k][env_ids] = 0.0

        if self._trajectory is not None:
            self.reset_manager.reset_envs(
                env_ids=env_ids,
                randomize_goals=False,
                randomize_state=self.cfg.randomize_init_state,
            )
        else:
            self.reset_manager.reset_envs(
                env_ids=env_ids,
                randomize_goals=True,
                randomize_state=self.cfg.randomize_init_state,
            )

        self.episode_length_buf[env_ids] = 0

        if self._trajectory is not None:
            centers = self.backend._vehicle._init_pos.to(device=self.device, dtype=torch.float32)
            self._trajectory.reset(env_ids, centers)
            self._sync_trajectory_reference(env_ids)

        self.backend.update_goal_markers(self.reset_manager.goal_pos[env_ids], env_ids=env_ids)

        self._action_history_obs[env_ids] = self.backend._vehicle._thrusters._reset_rotor_norm[env_ids]

        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

        self._call_reset_callbacks(env_ids)
