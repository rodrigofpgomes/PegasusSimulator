"""
sac_controller.py
Pegasus batched backend that runs a trained skrl SAC policy (actor only).

The checkpoint format matches what skrl's SequentialTrainer saves:
    {
        "policy": <Policy.state_dict()>,
        ...
    }

Observation layout (26 dims):
  [pos_error(3), R_flat(9), vel_error(3), ang_vel_b(3), last_action(4), rotor_speeds_norm(4)]
  where pos_error = pos_w - goal_pos_w, vel_error = vel_w - goal_vel_w,
  rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2 - 1.

Action: normalised motor command in [-1, 1], mapped linearly to [min_w, max_w].
"""

__all__ = ["SACBackend"]

import torch
import torch.nn as nn
import numpy as np

from pegasus.simulator.logic.backends.backend import Backend as _BaseBackend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf


class SACBackend(_BaseBackend):
    """Batched Pegasus backend running a trained skrl SAC actor."""

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int,
        act_dim: int,
        n_vehicles: int,
        reset_manager=None,
        action_mode: str = "rotor_velocity_direct",
        device: str = "cuda:0",
    ):
        super().__init__(config=None)
        self._checkpoint_path = checkpoint_path
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._n_vehicles = n_vehicles
        self.reset_manager = reset_manager
        self._action_mode = action_mode
        self._ext_device = device

        self._vehicle = None
        self._device = None

        self._input_ref = None
        self._state_cache = None
        self._received_first_state = False
        self._last_action = None  # (N, 4) normalised, fed back as part of obs

        self._policy_net = None
        self._log_std = None

        self._goal_marker_view = None
        self._goal_marker_paths = []

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------

    def initialize(self, vehicle):
        self._vehicle = vehicle

    def setup(self, reset_manager):
        self.reset_manager = reset_manager

    def start(self):
        self._n_vehicles = self._vehicle.n_vehicles
        self._device = self._vehicle.device

        self._input_ref = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)
        self._last_action = torch.zeros((self._n_vehicles, self._act_dim), dtype=torch.float32, device=self._device)

        self._received_first_state = False
        self._vehicle.set_input_mode(self._action_mode)

        self._load_checkpoint()
        self._prime_motors()

    def stop(self):
        pass

    def reset(self):
        if self._input_ref is not None:
            self._input_ref.zero_()
            self._state_cache.zero_()
            self._last_action.zero_()
            self._prime_motors()
        self._received_first_state = False

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if not self._received_first_state or self._policy_net is None:
            return

        pos_w = self._state_cache[:, 0:3]
        vel_w = self._state_cache[:, 3:6]
        quat  = self._state_cache[:, 6:10]
        ang_b = self._state_cache[:, 10:13]

        goal_pos = self.reset_manager.goal_pos.to(self._device, dtype=torch.float32)
        goal_vel = self.reset_manager.goal_vel.to(self._device, dtype=torch.float32)
        pos_error = pos_w - goal_pos
        vel_error = vel_w - goal_vel
        R_flat = quaternion_to_matrix(quat).reshape(self._n_vehicles, 9)

        min_w = self._vehicle._thrusters.min_rotor_velocity.to(self._device, dtype=torch.float32)
        max_w = self._vehicle._thrusters.max_rotor_velocity.to(self._device, dtype=torch.float32)
        rpm   = self._vehicle._thrusters._velocity
        rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0

        obs = torch.cat([pos_error, R_flat, vel_error, ang_b, self._last_action, rotor_speeds_norm], dim=1)

        # Forward pass - deterministic mean: net outputs [mu, log_std], tanh(mu) = action
        out = self._policy_net(obs)
        mean_clipped = torch.tanh(out[:, :self._act_dim])
        self._last_action = mean_clipped

        omega = self._action_to_omega(mean_clipped)
        self._input_ref = omega

    def update_state(self, state: StateBatch):
        self._state_cache = torch.cat(
            [state.position, state.linear_velocity, state.attitude, state.angular_velocity], dim=-1
        )
        self._received_first_state = True

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self) -> torch.Tensor:
        return self._input_ref

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def set_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ):
        if self._state_cache is None or env_ids.numel() == 0:
            return
        lin_vel = linear_velocity if linear_velocity is not None else torch.zeros((env_ids.numel(), 3), device=self._device)
        ang_vel = angular_velocity if angular_velocity is not None else torch.zeros((env_ids.numel(), 3), device=self._device)
        self._state_cache[env_ids, 0:3] = positions.to(self._device)
        self._state_cache[env_ids, 3:6] = lin_vel.to(self._device)
        self._state_cache[env_ids, 6:10] = attitudes.to(self._device)
        self._state_cache[env_ids, 10:13] = ang_vel.to(self._device)
        self._last_action[env_ids] = 0.0
        self._prime_motors(env_ids=env_ids)
        self._received_first_state = True

    def get_state(self) -> torch.Tensor:
        return self._state_cache

    # ------------------------------------------------------------------
    # Goal marker
    # ------------------------------------------------------------------

    def create_goal_marker(self, root_path: str = "/World/GoalMarkers", size: float = 0.15, color: tuple = (1.0, 0.0, 0.0)):
        stage = self._vehicle._world.stage
        if not stage.GetPrimAtPath(root_path).IsValid():
            prim_utils.create_prim(root_path, "Xform")
        self._goal_marker_paths = []
        for i in range(self._n_vehicles):
            prim_path = f"{root_path}/goal_{i}"
            if not stage.GetPrimAtPath(prim_path).IsValid():
                prim_utils.create_prim(prim_path, "Cube", translation=[0.0, 0.0, -100.0], scale=[size, size, size])
            cube = UsdGeom.Cube(stage.GetPrimAtPath(prim_path))
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self._goal_marker_paths.append(prim_path)
        self._goal_marker_view = XFormPrimView(
            prim_paths_expr=f"{root_path}/goal_*",
            name=f"goal_marker_view_{root_path.replace('/', '_')}",
        )

    def update_goal_marker(self, positions: torch.Tensor):
        if self._goal_marker_view is None:
            return
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        positions = positions.to(device=self._device, dtype=torch.float32)
        orientations = torch.zeros((positions.shape[0], 4), device=self._device, dtype=torch.float32)
        orientations[:, 0] = 1.0
        self._goal_marker_view.set_world_poses(positions=positions, orientations=orientations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self):
        ck = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)

        # Reconstruct the actor network (same architecture as sac_cfg.Policy)
        # Output is 2*act_dim: [mu(4), log_std(4)] — state-dependent log_std
        self._policy_net = nn.Sequential(
            nn.Linear(self._obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),            nn.ReLU(),
            nn.Linear(64, self._act_dim * 2),
        ).to(self._device)

        policy_sd = ck["policy"]
        net_sd = {k[len("net."):]: v for k, v in policy_sd.items() if k.startswith("net.")}
        self._policy_net.load_state_dict(net_sd)
        self._policy_net.eval()

        print(f"[SACBackend] Loaded checkpoint: {self._checkpoint_path}", flush=True)

    def _action_to_omega(self, action_norm: torch.Tensor) -> torch.Tensor:
        min_w = self._vehicle._thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32)
        max_w = self._vehicle._thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32)
        half   = 0.5 * (max_w - min_w)
        center = min_w + half
        return action_norm * half + center

    def _prime_motors(self, env_ids: torch.Tensor | None = None):
        if self._input_ref is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._n_vehicles, device=self._device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self._device, dtype=torch.long)
        min_w = self._vehicle._thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32)
        max_w = self._vehicle._thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32)
        omega_mid = (0.5 * (min_w + max_w)).unsqueeze(0).expand(env_ids.numel(), -1)
        self._input_ref[env_ids] = omega_mid
        self._vehicle._thrusters._input_reference[env_ids] = omega_mid
        self._vehicle._thrusters._velocity[env_ids] = omega_mid
