"""
| File: quadcopter_env.py (raptor_pretrain)
| Description: Shuttle_glider 28D observation ablation without goal_acc.

Observation (28 dims):
    pos_error                (3) - position error in world frame (pos - goal / reference)
    vel_error                (3) - velocity error in world frame
    R_flat                   (9) - rotation matrix, row-major
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
import torch.nn.functional as F

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
    "rotor_axes_body":            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]]
}

_VERTICAL_TAIL_PHYSICS_CFG = {
    "S_vtail": 0.022078,  # Vertical-tail reference area in square meters.
    "CL_beta": 3.0,       # linear side-force coefficient slope (rad^-1)
    "CL_max": 0.8,        # maximum magnitude of the side-force coefficient.
    "CD0": 0.03,          # zero-sideslip profile drag coefficient.
    "induced_k": 0.10,    # induced-drag coefficient.
    "CD_crossflow": 1.0,   # crossflow drag coefficient.
    "prop_radius": 0.13,  # Radius of the fifth rotor propeller in meters.
    "wake_factor": 1.5,   # factor to account for the wake expansion and decay of the propeller slipstream.
    "coverage_factor": 0.3 # factor to account for the fraction of the vertical tail covered by the propeller slipstream.
}

@dataclass
class QuadcopterEnvCfg(PegasusEnvCfg):

    # --- spaces ---
    """Configuration for the quadcopter hover task: observation/action/state spaces, reward scales and termination bounds."""
    observation_space: int = 28
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
    vertical_tail_physics_cfg: Any = field(default_factory=lambda: _VERTICAL_TAIL_PHYSICS_CFG)
    
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
        """Map normalized actions to rotor velocities and apply aerodynamic forces."""
        
        # Get the minimum/maximum rotor velocities for all five rotors.
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        
        # Compute half of each rotor's allowed velocity range.
        half = 0.5 * (max_w - min_w)
        center = min_w + half
        
        # Map normalized actions from [-1, 1] to physical rotor velocities.
        omega = self._last_action * half + center   # (N, 4)  rad/s
        self.backend._input_reference = omega
        
        # Send the rotor velocity command to the motor/thruster model.
        self.backend._vehicle._thrusters.set_input_reference(omega)

        state = self.backend.get_state()
        vel_w = state[:, 3:6]
        quat  = state[:, 6:10]   # w,x,y,z

        R = quaternion_to_matrix(quat)

        vel_b = (R.transpose(1, 2) @ vel_w.unsqueeze(-1)).squeeze(-1)
        ang_b = state[:, 10:13]
        
        # Create a zero world-frame wind vector for every environment.
        wind_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=vel_w.dtype)
        
        # Set a constant 5 m/s wind in the positive world Y direction.
        wind_w[:, 1] = 5.0

        # Transform the wind velocity from the world frame to the body frame.
        wind_b = R.transpose(1, 2) @ wind_w.unsqueeze(-1)
        
        # Compute the aerodynamic force and torque generated by the vertical tail.
        force_tail_b, torque_tail_b = self.vertical_tail_wrench(v_com_b=vel_b, omega_b=ang_b, wind_b=wind_b.squeeze(-1))

        external_forces = torch.zeros((self.num_envs, self.backend.parts_per_vehicle, 3), device=self.device, dtype=vel_b.dtype)
        external_torques = torch.zeros_like(external_forces)

        body_idx = self.backend._vehicle.body_index

        external_forces[:, body_idx, :] = force_tail_b
        external_torques[:, body_idx, :] = torque_tail_b

        self.backend.set_external_forces_and_torques(external_forces, external_torques)

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

        R_flat = quaternion_to_matrix(quat).reshape(self.num_envs, 9)

        # Rotor speeds normalised to [-1, 1]: matches rl-tools RotorSpeeds observation
        min_w = self.backend._vehicle._thrusters.min_rotor_velocity
        max_w = self.backend._vehicle._thrusters.max_rotor_velocity
        rpm   = self.backend._vehicle._thrusters._velocity  # (N, 4) actual rotor speeds
        rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0

        obs = torch.cat([pos_error, vel_error, R_flat, ang_b, self._action_history_obs, rotor_speeds_norm], dim=1)
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

        # Termination penalty replaces the normal reward when the episode ends early
        died = (pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)
        died |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)
        died |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)

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


    def vertical_tail_wrench(self, v_com_b, omega_b, wind_b=None, rho=1.225):
        """Compute the body-frame force and torque generated by the vertical tail."""
        
        # Get the PyTorch device and numerical data type used by the vehicle velocity tensor.
        device = v_com_b.device
        dtype = v_com_b.dtype

        # Define the center-of-mass position in the vehicle body frame.
        p_com = torch.tensor([-0.06945947, 0.0, 0.10253066], device=device, dtype=dtype)
    
        # Define the vertical-tail center-of-pressure position in the body frame.
        p_vtail = torch.tensor([-0.848256, 0.0, -0.093542], device=device, dtype=dtype)

        # Compute the vector from the center of mass to the vertical tail.
        r_vtail = p_vtail - p_com
        r_vtail = r_vtail.expand_as(v_com_b)

        if wind_b is None:
            wind_b = torch.zeros_like(v_com_b)

        # Compute the local air-relative velocity at the vertical-tail position.
        v_cp_base_b = v_com_b + torch.cross(omega_b,r_vtail, dim=-1) - wind_b
    
        # Read the actual angular velocity of the fifth rotor.
        omega_5 = self.backend._vehicle._thrusters._velocity[:, 4]

        # Read the thrust coefficient of the fifth rotor.
        kf_5 = self.backend._vehicle._thrusters._rotor_constant[4]
        
        # Add the fifth rotor's slipstream velocity to the tail airflow.
        v_cp_b = self.puller_slipstream(omega_5=omega_5, v_cp_b=v_cp_base_b, kf_5=kf_5, rho=rho, prop_radius=0.13, wake_factor=1.5, coverage_factor=0.3)
        
        # Copy the tail airflow before removing the vertical component.
        v_xy = v_cp_b.clone()
        
        # Remove the Z component because the vertical tail mainly acts in the XY plane.
        v_xy[..., 2] = 0.0

        # Compute the magnitude of the horizontal air-relative velocity.
        airspeed_xy = torch.linalg.vector_norm(v_xy, dim=-1, keepdim=True)

        # Compute the normalized horizontal airflow direction.
        v_hat = v_xy / airspeed_xy.clamp_min(1e-6)

        # Extract the longitudinal/lateral component of the tail airflow.
        vx = v_xy[..., 0]
        vy = v_xy[..., 1]

        # Compute the sideslip angle using the horizontal velocity components.
        beta = torch.atan2(vy, vx)

        S_vtail = 0.022078  # Vertical-tail reference area in square meters.
        CL_beta = 3.0       # linear side-force coefficient slope (rad^-1)
        CL_max = 0.8        # maximum magnitude of the side-force coefficient.
        CD0 = 0.03          # zero-sideslip profile drag coefficient.
        induced_k = 0.10    # induced-drag coefficient.
        CD_crossflow = 1.0  # crossflow drag coefficient.

        # Compute the unsaturated linear side-force coefficient.
        cl_linear = CL_beta * beta
        cl = CL_max * torch.tanh(cl_linear / CL_max) # smoothly saturate the side-force coefficient.

        # Activate the lift model mainly when forward airflow is present.
        forward_gate = torch.sigmoid((vx - 0.5) / 0.15)
        
        # Apply the forward-flight activation factor to the side-force coefficient.
        cl = cl * forward_gate

        # Compute profile, induced, and crossflow drag contributions.
        cd = CD0 + induced_k * cl.square() + CD_crossflow * torch.sin(beta).square()

        # Compute the dynamic pressure at the vertical tail.
        dynamic_pressure = 0.5 * rho * airspeed_xy.squeeze(-1).square()

        # Allocate the vertical-tail span-axis vector.
        span_axis = torch.zeros_like(v_xy)
        span_axis[..., 2] = 1.0

        # Compute the lateral-force direction perpendicular to the airflow and span axis.
        side_force_hat = F.normalize(torch.cross(v_hat, span_axis, dim=-1), dim=-1, eps=1e-6)

        # Compute the common aerodynamic force scaling factor.
        force_scale = (dynamic_pressure * S_vtail).unsqueeze(-1)

        # Compute the restoring lateral force generated by the vertical tail.
        force_side_b = force_scale * cl.unsqueeze(-1) * side_force_hat
        
        # Compute the aerodynamic drag force opposite to the airflow direction.
        force_drag_b = -force_scale * cd.unsqueeze(-1) * v_hat

        # Add the side-force and drag-force contributions.
        force_b = force_side_b + force_drag_b

        # Remove aerodynamic forces when the airflow magnitude is too small.
        valid = airspeed_xy >= 0.2
        force_b = torch.where(valid, force_b, torch.zeros_like(force_b))

        # Compute the moment about the center of mass using r cross F.
        torque_b = torch.cross(r_vtail, force_b, dim=-1)

        return force_b, torque_b
    
    
    @staticmethod
    def puller_slipstream(omega_5: torch.Tensor, v_cp_b: torch.Tensor, kf_5: float, rho: float = 1.225, prop_radius: float = 0.13, wake_factor: float = 1.5, coverage_factor: float = 0.3):
        """Add the puller propeller slipstream velocity to the vertical-tail airflow."""
        
        # Get the PyTorch device and numerical data type used by the vehicle velocity tensor.
        device = v_cp_b.device
        dtype = v_cp_b.dtype

        # Define the fifth rotor axis as the positive body X direction.
        rotor_axis_b = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

        # Compute the fifth rotor thrust using the quadratic rotor model.
        thrust_5 = kf_5 * omega_5.square()

        # Compute the propeller disk area
        disk_area = torch.pi * prop_radius**2

        # Project the existing tail airflow onto the fifth rotor axis.
        v_axial = torch.sum(v_cp_b * rotor_axis_b, dim=-1)

        # Prevent the simplified momentum model from using reverse axial flow.
        v_axial_model = v_axial.clamp_min(0.0)

        # Estimate the induced velocity at the propeller disk.
        v_induced = 0.5 * (torch.sqrt(v_axial_model.square() + 2.0 * thrust_5 / (rho * disk_area + 1e-6)) - v_axial_model)

        # Estimate the slipstream velocity increment reaching the vertical tail.
        delta_v = wake_factor * coverage_factor * v_induced
        
        v_tail_b = v_cp_b + delta_v.unsqueeze(-1) * rotor_axis_b
        
        return v_tail_b