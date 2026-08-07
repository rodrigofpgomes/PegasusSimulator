"""
| File: quadcopter_env.py (raptor_pretrain)
| Description: Shuttle_glider task with 31D observation, velocity cost and cost clipping.

Observation (31 dims):
    pos_error                (3) - position error in world frame (pos - goal / reference)
    vel_error                (3) - velocity error in world frame
    R_flat                   (9) - rotation matrix, row-major
    goal_acc                 (3) - reference acceleration in world frame
    ang_b                    (3) - angular velocity in body frame
    self._action_history_obs (5) - previous normalised motor command (ActionHistory length=1)
    rotor_speeds_norm        (5) - actual rotor speeds normalised to [-1, 1]

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
    "max_rotor_velocity":         [1400, 1400, 1400, 1400, 3500],
    "motor_time_constant":        [0.008, 0.008, 0.008, 0.008, 0.0125],
}

@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):

    # --- spaces ---
    """Configuration for the quadcopter hover task: observation/action/state spaces, reward scales and termination bounds."""
    observation_space: int = 31
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
    w_d_action: float = 1.0    # delta-action squared (action smoothness)
    constant:   float = 1.5
    termination_penalty: float = 200.0
    cost_clip:  float = 3.0

    # --- termination + goal range: scaled from vehicle geometry at setup() ---
    # RAPTOR: max_pos_error = max_rotor_distance * 20 (per axis)
    #         goal_range     = max_rotor_distance * 10
    # Set to None to trigger auto-scaling in setup(); override with a float to fix manually.
    max_pos_error_per_axis: float = 1.0
    max_lin_vel_per_axis: float = 2.0
    max_ang_vel_per_axis: float = 35.0
    min_upright: float = -0.17

    use_raptor_trajectory: bool = True
    trajectory_mixture_langevin_prob: float = 0.5
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
            for k in ("pos", "vel", "d_action", "total")
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

        goal_acc = self.reset_manager.goal_acc

        R_flat = quaternion_to_matrix(quat).reshape(self.num_envs, 9)

        # Rotor speeds normalised to [-1, 1]: matches rl-tools RotorSpeeds observation
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        rpm   = self.backend._vehicle._thrusters._velocity  # (N, 4) actual rotor speeds
        rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0

        obs = torch.cat([pos_error, vel_error, R_flat, goal_acc, ang_b, self._action_history_obs, rotor_speeds_norm], dim=1)
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
        d_action_cost = torch.linalg.norm(d_action, dim=1)

        cost = self.cfg.w_pos * pos_cost + self.cfg.w_vel * vel_cost + self.cfg.w_d_action * d_action_cost
        cost = cost.clamp(max=self.cfg.cost_clip)          

        reward = self.cfg.constant - cost

        upright   = quaternion_to_matrix(quat)[:, 2, 2]

        # Termination penalty replaces the normal reward when the episode ends early
        died = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        died |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        died |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)
        died |= (upright < self.cfg.min_upright)           # remove esta linha na variante B

        reward[died] = -self.cfg.termination_penalty

        self._episode_sums["pos"] += -self.cfg.w_pos * pos_cost
        self._episode_sums["vel"] += -self.cfg.w_vel * vel_cost
        self._episode_sums["d_action"] += -self.cfg.w_d_action * d_action_cost
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
 
        upright = R[:, 2, 2]                       

        terminated = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        terminated |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        terminated |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)
        terminated |= (upright < self.cfg.min_upright)

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
        for k, v in self._episode_sums.items():
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
