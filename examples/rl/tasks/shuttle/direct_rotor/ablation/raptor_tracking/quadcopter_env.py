"""
| File: quadcopter_env.py
| Adapted by: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Ablation of the RAPTOR-style tracking baseline (shuttle/direct_rotor/baseline/raptor_tracking):
|              selectable orientation cost, optional velocity cost and optional reference feedforward in the
|              observation. The defaults reproduce the baseline.
"""


from __future__ import annotations

from dataclasses import dataclass, field
import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.rl.reset_manager import InitStateCfg
from pegasus.simulator.logic.transforms import quaternion_to_matrix

from .trajectory import LemniscateTrajectory, RaptorLikeTrajectory


@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):

    # --- spaces ---
    """Configure direct rotor control, tracking reward and episode bounds."""
    observation_space: int = 26
    action_space: int = 4
    state_space: int = 0

    # --- timing (100 Hz, 5 s episodes) ---
    sim_dt: float = 0.01
    decimation: int = 1
    episode_length_s: float = 5.0

    # --- vehicle ---
    action_mode: str = "rotor_velocity_direct"
    vehicle: str = "Shuttle"


    # Reward weights for this ablation
    w_pos:      float = 1.0    # Euclidean position error
    w_ori:      float = 0.1    # weight of the orientation cost selected by ori_cost
    ori_cost:   str = "qz"     # "qz": 2 acos(1 - |q_z|) (baseline); "trace": 0.5 tr(I - Rd^T R); "none"
    w_vel:      float = 0.0    # Euclidean velocity error relative to the reference
    ang_cost:   str = "none"   # angular-rate cost: "none"; "yaw": |omega_z|; "norm": ||omega_B||
    w_ang:      float = 0.0    # weight of the angular-rate cost selected by ang_cost
    w_d_action: float = 1.0    # Euclidean change in normalized rotor command
    constant:   float = 1.5
    termination_penalty: float = 100.0

    # --- termination bounds and goal ranges ---
    max_pos_error_per_axis: float = 1.0
    max_lin_vel_per_axis: float = 2.0
    max_ang_vel_per_axis: float = 35.0

    use_raptor_trajectory: bool = True
    trajectory_mixture_langevin_prob: float = 0.5
    langevin_gamma: float = 1.0
    langevin_omega: float = 2.0
    langevin_sigma: float = 0.5
    langevin_alpha: float = 0.01

    goal_pos_xy_range: list | None = None
    goal_pos_z_range:  list | None = None

    # --- initial state randomisation (RAPTOR init_90_deg) ---
    randomize_init_state: bool = True
    init_state_cfg: InitStateCfg = field(default_factory=lambda: InitStateCfg(max_angle_deg=90.0, guidance_prob=0.1))

    # --- observation clamping during test/evaluation ---
    test_mode: bool = False
    clamp_observations_in_test: bool = False

    # --- observation options (observation_space is set from them by the task) ---
    obs_ref_acc:  bool = False   # append the reference acceleration (world frame, 3)
    obs_attitude: str = "R"      # "R": rotation matrix; "R_err": Rd^T R, Rd from reference acceleration and yaw_ref
    yaw_ref:      float = 0.0    # desired yaw of Rd [rad]

    # --- evaluation reference (play.py --env-cfg eval_reference=lemniscate episode_length_s=20 lem_peak_acc=2) ---
    eval_reference: str = "training"   # "training": fixed point / Langevin mixture; "lemniscate": figure eight
    lem_amplitude:  float = 1.0        # [m] half width of the figure eight
    lem_peak_acc:   float = 2.0        # [m/s^2] peak acceleration of the reference (sets its frequency)
    lem_ramp_s:     float = 2.0        # [s] time to ramp the reference speed up from rest

    obs_pos_error_limit: float = 0.3
    obs_vel_error_limit: float = 0.5


class QuadcopterEnv(PegasusEnv):
    """Position, selectable orientation, velocity and action-change costs; terminal penalty 100."""
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, backend, reset_manager):
        """Initializes the task: caches config/backend/reset-manager and allocates the per-episode reward logging buffers."""
        # The observation size follows the observation options (also after a run-metadata restore)
        if (cfg.ori_cost not in ("qz", "trace", "none") or cfg.obs_attitude not in ("R", "R_err")
                or cfg.ang_cost not in ("none", "yaw", "norm") or cfg.eval_reference not in ("training", "lemniscate")):
            raise ValueError(f"ori_cost={cfg.ori_cost!r}, obs_attitude={cfg.obs_attitude!r}, ang_cost={cfg.ang_cost!r}")
        cfg.observation_space = 26 + (3 if cfg.obs_ref_acc else 0)
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
            for k in ("pos", "ori", "vel", "ang", "d_action", "total")
        }

        # Fill optional task bounds and goal ranges with fixed values
        if self.cfg.max_pos_error_per_axis is None:
            self.cfg.max_pos_error_per_axis = 1.0
        if self.cfg.goal_pos_xy_range is None:
            self.cfg.goal_pos_xy_range = [-0.5, 0.5]
        if self.cfg.goal_pos_z_range is None:
            spawn_z = self.backend._vehicle._init_pos[0, 2].item()
            self.cfg.goal_pos_z_range = [spawn_z - 0.5, spawn_z + 0.5]


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
                alpha=self.cfg.langevin_alpha,
                mixture_langevin_prob=self.cfg.trajectory_mixture_langevin_prob,
            )

            if self.cfg.eval_reference == "lemniscate":
                self._trajectory = LemniscateTrajectory(
                    num_envs=self.num_envs,
                    episode_steps=self.max_episode_length,
                    dt=self.cfg.sim_dt * self.cfg.decimation,
                    device=self.device,
                    amplitude=self.cfg.lem_amplitude,
                    peak_acc=self.cfg.lem_peak_acc,
                    ramp_s=self.cfg.lem_ramp_s,
                )

            centers = self.backend._vehicle._init_pos.to(
                device=self.device, dtype=torch.float32
            )

            self._trajectory.reset(all_ids, centers)
            self._sync_trajectory_reference(all_ids)
        else:
            self.reset_manager._randomize_goals(all_ids)

    def _sync_trajectory_reference(self, env_ids: torch.Tensor | None = None):
        """Copy the current trajectory sample into the shared goal buffers."""
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

    def _desired_rotation(self) -> torch.Tensor:
        """Rd from the reference acceleration and the fixed desired yaw (thrust axis along a_r + g e_z)."""
        acc = self.reset_manager.goal_acc
        yaw = torch.full((acc.shape[0],), self.cfg.yaw_ref, device=self.device, dtype=acc.dtype)
        gravity = torch.zeros_like(acc)
        gravity[:, 2] = 9.81
        z_b = acc + gravity
        z_b = z_b / z_b.norm(dim=1, keepdim=True).clamp_min(1e-8)
        x_c = torch.stack([yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], dim=1)
        y_b = torch.cross(z_b, x_c, dim=1)
        y_b = y_b / y_b.norm(dim=1, keepdim=True).clamp_min(1e-8)
        x_b = torch.cross(y_b, z_b, dim=1)
        return torch.stack([x_b, y_b, z_b], dim=2)

    # ------------------------------------------------------------------
    # PegasusEnv interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Clamp policy commands and retain the previous command for smoothing cost."""
        action = actions.clamp(-1.0, 1.0)

        self._prev_action = self._last_action.clone()
        self._last_action = action

        self._action_history_obs = action.clone()

    def _apply_action(self):
        """Map four normalized actions to rotor speeds in rad/s."""
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        half = 0.5 * (max_w - min_w)
        center = min_w + half
        omega = self._last_action * half + center   # (N, 4)  rad/s
        self.backend._input_reference = omega
        self.backend._vehicle._thrusters.set_input_reference(omega)

    def _get_observations(self) -> dict:
        """Build the policy observation from tracking errors and actuator state."""
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

        R = quaternion_to_matrix(quat)
        if self.cfg.obs_attitude == "R_err":
            R = torch.bmm(self._desired_rotation().transpose(1, 2), R)
        R_flat = R.reshape(self.num_envs, 9)

        # Rotor speeds normalised to [-1, 1]: matches rl-tools RotorSpeeds observation
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        rpm   = self.backend._vehicle._thrusters._velocity  # (N, 4) actual rotor speeds
        rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0

        obs = [pos_error, R_flat, vel_error, ang_b, self._action_history_obs, rotor_speeds_norm]
        if self.cfg.obs_ref_acc:
            obs.append(self.reset_manager.goal_acc)
        obs = torch.cat(obs, dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Combine active tracking costs and replace failures with the death penalty."""
        self._sync_trajectory_reference()

        state = self.backend.get_state()
        pos = state[:, 0:3]
        vel_w = state[:, 3:6]
        quat  = state[:, 6:10]
        ang_b = state[:, 10:13]

        pos_error = pos - self.reset_manager.goal_pos

        if self.cfg.ori_cost == "qz":
            q = quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp_min(1e-8)
            acos_arg = torch.clamp(1.0 - q[:, 3].abs(), -1.0, 1.0)
            ori_cost = 2.0 * torch.acos(acos_arg)
        elif self.cfg.ori_cost == "trace":
            # Psi = 0.5 tr(I - Rd^T R) = 1 - cos(rotation angle between R and Rd), in [0, 2]
            E = torch.bmm(self._desired_rotation().transpose(1, 2), quaternion_to_matrix(quat))
            ori_cost = (0.5 * (3.0 - E.diagonal(dim1=1, dim2=2).sum(dim=1))).clamp(0.0, 2.0)
        else:
            ori_cost = torch.zeros(self.num_envs, device=self.device)

        vel_cost = torch.linalg.norm(vel_w - self.reset_manager.goal_vel, dim=1)
        if self.cfg.ang_cost == "yaw":
            ang_cost = ang_b[:, 2].abs()
        elif self.cfg.ang_cost == "norm":
            ang_cost = torch.linalg.norm(ang_b, dim=1)
        else:
            ang_cost = torch.zeros(self.num_envs, device=self.device)

        d_action = self._last_action - self._prev_action

        pos_cost = torch.linalg.norm(pos_error, dim=1)
        d_action_cost = torch.linalg.norm(d_action, dim=1)

        cost = (self.cfg.w_pos * pos_cost + self.cfg.w_ori * ori_cost + self.cfg.w_vel * vel_cost
                + self.cfg.w_ang * ang_cost + self.cfg.w_d_action * d_action_cost)

        reward = self.cfg.constant - cost

        # Termination penalty replaces the normal reward when the episode ends early
        died = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        died |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        died |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

        reward[died] = -self.cfg.termination_penalty

        self._episode_sums["ori"] += -self.cfg.w_ori * ori_cost
        self._episode_sums["pos"] += -self.cfg.w_pos * pos_cost
        self._episode_sums["vel"] += -self.cfg.w_vel * vel_cost
        self._episode_sums["ang"] += -self.cfg.w_ang * ang_cost
        self._episode_sums["d_action"] += -self.cfg.w_d_action * d_action_cost
        self._episode_sums["total"] += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        """End failed episodes at the active bounds and truncate at the time limit."""
        state = self.backend.get_state()
        pos   = state[:, 0:3]
        vel_w = state[:, 3:6]
        ang_b = state[:, 10:13]

        pos_error = pos - self.reset_manager.goal_pos

        terminated = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        terminated |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        terminated |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Log completed episodes, reset their states and renew trajectory references."""
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

        self._action_history_obs[env_ids] = self._normalized_rotor_velocities(env_ids)

        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

        self._call_reset_callbacks(env_ids)
