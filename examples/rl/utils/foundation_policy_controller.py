"""
foundation_policy_controller.py
Pegasus batched backend running the RAPTOR (foundation_policy) quadrotor policy.

Observation layout (22 dims, numpy):
  [pos_error(3), R_flat(9), vel_w(3), ang_vel_b(3), last_action(4)]
  where pos_error = pos_w - goal_pos_w.

Action: normalized motor commands in [-1, 1], mapped linearly to [min_w, max_w] of the
vehicle's own rotor velocity range. This makes the backend vehicle-agnostic (Iris, Crazyflie, etc.).
"""

__all__ = ["RaptorBackend"]

import itertools
import math
import numpy as np
import torch

from pegasus.simulator.logic.backends.backend import Backend as _BaseBackend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

from foundation_policy import Raptor

# RAPTOR action order matches L2F Crazyflie rotor layout (crazy_flie.h):
#   action[0]: x+, y-   action[1]: x-, y-   action[2]: x-, y+   action[3]: x+, y+
_RAPTOR_ROTOR_XY = (
    ( 0.028, -0.028),
    (-0.028, -0.028),
    (-0.028,  0.028),
    ( 0.028,  0.028),
)


class RaptorBackend(_BaseBackend):
    """Batched Pegasus backend running the RAPTOR foundation policy."""

    def __init__(self, n_vehicles: int, reset_manager=None, action_mode: str = "rotor_velocity"):
        super().__init__(config=None)
        self._n_vehicles = int(n_vehicles)
        self._action_mode = action_mode
        self.reset_manager = reset_manager

        self._vehicle = None
        self._device = None

        self._forces = None
        self._torques = None
        self._input_ref = None
        self._state_cache = None
        self._received_first_state = False

        self.policy = None
        self._last_action_norm = None  # numpy (n_vehicles, 4), fed back into policy

        self._raptor_to_pegasus = None  # long tensor (4,): action[i] -> pegasus rotor index
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

        self._forces = torch.zeros(
            (self._n_vehicles, self._vehicle.parts_per_vehicle, 3),
            dtype=torch.float32, device=self._device,
        )
        self._torques = torch.zeros_like(self._forces)
        self._input_ref = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)

        self._last_action_norm = np.zeros((self._n_vehicles, 4), dtype=np.float32)

        self.policy = Raptor()
        self.policy.reset()

        self._received_first_state = False
        self._vehicle.set_input_mode(self._action_mode)
        self._raptor_to_pegasus = self._infer_rotor_order()
        self._prime_motors()

    def stop(self):
        pass

    def reset(self):
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._state_cache.zero_()
            self._last_action_norm[:] = 0.0
            self._prime_motors()

        if self.policy is not None:
            self.policy.reset()

        self._received_first_state = False

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if not self._received_first_state or self.policy is None:
            return

        if self._action_mode != "rotor_velocity":
            raise RuntimeError("RaptorBackend requires action_mode='rotor_velocity'.")

        pos_w = self._state_cache[:, 0:3]       # (N, 3) torch
        vel_w = self._state_cache[:, 3:6]       # (N, 3) torch
        quat_wxyz = self._state_cache[:, 6:10]  # (N, 4) torch  w,x,y,z
        ang_vel_b = self._state_cache[:, 10:13] # (N, 3) torch

        # Goal position
        goal_pos = self.reset_manager.goal_pos
        if isinstance(goal_pos, torch.Tensor):
            goal_pos_np = goal_pos.cpu().numpy()
        else:
            goal_pos_np = np.asarray(goal_pos, dtype=np.float32)
        
        # Goal velocity
        goal_vel = self.reset_manager.goal_vel
        if isinstance(goal_vel, torch.Tensor):
            goal_vel_np = goal_vel.cpu().numpy()
        else:            
            goal_vel_np = np.asarray(goal_vel, dtype=np.float32)

        pos_error = pos_w.cpu().numpy() - goal_pos_np  # (N, 3)
        vel_error = vel_w.cpu().numpy() - goal_vel_np  # (N, 3)

        # Rotation matrix (N, 3, 3) -> (N, 9) numpy
        rot_mat = quaternion_to_matrix(quat_wxyz)  # (N, 3, 3) torch
        rot_flat = rot_mat.reshape(self._n_vehicles, 9).cpu().numpy()

        ang_vel_np = ang_vel_b.cpu().numpy()

        # observation: [pos_error(3), R_flat(9), vel(3), ang_vel(3), last_action(4)]
        obs = np.concatenate([pos_error, rot_flat, vel_error, ang_vel_np, self._last_action_norm], axis=1)

        action_norm = self.policy.evaluate_step(obs)  # (N, 4), raw (may exceed [-1,1])
        action_norm_clipped = np.clip(action_norm, -1.0, 1.0)
        self._last_action_norm[:] = action_norm_clipped  # feed back clamped, matching L2F C++ behaviour

        omega = self._action_norm_to_omega(action_norm_clipped)

        # Reorder from RAPTOR action order to Pegasus USD rotor order.
        omega_pegasus = torch.empty_like(omega)
        omega_pegasus[:, self._raptor_to_pegasus] = omega
        self._input_ref = omega_pegasus

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

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        self._state_cache[env_ids, 0:3] = positions.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 3:6] = linear_velocity.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 6:10] = attitudes.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 10:13] = angular_velocity.to(self._device, dtype=torch.float32)

        self._last_action_norm[env_ids.cpu().numpy()] = 0.0
        self._prime_motors(env_ids=env_ids)

        # GRU state cannot be reset per-environment; reset the full history.
        if self.policy is not None:
            self.policy.reset()

        self._received_first_state = True

    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forces, self._torques

    def get_state(self) -> torch.Tensor:
        return self._state_cache

    # ------------------------------------------------------------------
    # Debug accessors (used by EpisodeTrajectoryRecorder)
    # ------------------------------------------------------------------

    @property
    def last_action_norm(self) -> torch.Tensor | None:
        if self._last_action_norm is None:
            return None
        return torch.from_numpy(self._last_action_norm)

    # ------------------------------------------------------------------
    # Goal marker (visual reference cube)
    # ------------------------------------------------------------------

    def create_goal_marker(self, root_path: str = "/World/GoalMarkers", size: float = 0.15, color: tuple = (1.0, 0.0, 0.0)):
        """Spawns one cube per environment to visualise the goal position."""
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
            prim_paths_expr=f"{root_path}/goal_*", name=f"goal_marker_view_{root_path.replace('/', '_')}"
        )

    def update_goal_marker(self, positions: torch.Tensor):
        """Moves goal cubes to the given world positions (N, 3)."""
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

    def _infer_rotor_order(self) -> torch.Tensor:
        """Returns p where p[i] is the Pegasus USD rotor index for RAPTOR action i."""
        identity = torch.arange(4, dtype=torch.long, device=self._device)
        rotor_positions = self._vehicle._rotor_positions_body
        if rotor_positions is None:
            return identity

        pegasus_xy = rotor_positions[0, :, :2].to(dtype=torch.float32, device=self._device)
        if pegasus_xy.shape != (4, 2) or not torch.isfinite(pegasus_xy).all():
            return identity

        raptor_xy = torch.tensor(_RAPTOR_ROTOR_XY, dtype=torch.float32, device=self._device)

        peg_norm = pegasus_xy / pegasus_xy.abs().max().clamp_min(1e-6)
        rap_norm = raptor_xy / raptor_xy.abs().max().clamp_min(1e-6)

        best_perm, best_cost = list(range(4)), float("inf")
        for perm in itertools.permutations(range(4)):
            cost = torch.sum((peg_norm[list(perm)] - rap_norm) ** 2).item()
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)

        return torch.tensor(best_perm, dtype=torch.long, device=self._device)

    def _action_norm_to_omega(self, action_norm: np.ndarray) -> torch.Tensor:
        """Map RAPTOR action [-1, 1] directly to the vehicle's [min_w, max_w] in rad/s.

        Using the vehicle's own rotor limits avoids the L2F RPM constants being
        applied to vehicles with different operating ranges (e.g. Iris vs Crazyflie).
        The action is treated as a linear interpolation over the rotor speed range:
          -1  →  min_rotor_velocity
          +1  →  max_rotor_velocity
        """
        action_t = torch.from_numpy(np.clip(action_norm, -1.0, 1.0)).to(device=self._device, dtype=torch.float32)
        min_w = self._vehicle._thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32)
        max_w = self._vehicle._thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32)
        half = 0.5 * (max_w - min_w)
        center = min_w + half
        return action_t * half + center  # (N, 4)

    def _prime_motors(self, env_ids: torch.Tensor | None = None):
        """Set commanded and actual motor state to the midpoint of the vehicle's rotor range."""
        if self._input_ref is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self._n_vehicles, device=self._device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self._device, dtype=torch.long)

        min_w = self._vehicle._thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32)
        max_w = self._vehicle._thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32)
        omega_init = (0.5 * (min_w + max_w)).unsqueeze(0).expand(env_ids.numel(), -1)

        self._input_ref[env_ids] = omega_init
        self._vehicle._thrusters._input_reference[env_ids] = omega_init
        self._vehicle._thrusters._velocity[env_ids] = omega_init
