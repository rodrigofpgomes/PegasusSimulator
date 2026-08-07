"""
non_linear_controller.py
Pegasus batched backend running a geometric (non-linear) quadrotor controller.

This is the analytic counterpart of the learned backends (RaptorBackend / SACBackend):
it exposes the exact same backend interface so it can be dropped into play_multi.py,
but instead of a neural policy it runs the geometric controller of

  [1] J. Pinto, B. J. Guerreiro and R. Cunha, \"Planning Parcel Relay Manoeuvres for
      Quadrotors,\" ICUAS 2021, doi: 10.1109/ICUAS51884.2021.9476757.
  [2] D. Mellinger and V. Kumar, \"Minimum snap trajectory generation and control for
      quadrotors,\" ICRA 2011, doi: 10.1109/ICRA.2011.5980409.

The control target is the point/velocity provided by the ResetManager
(goal_pos, goal_vel); higher-order references (accel, jerk, yaw, yaw_rate) are 0,
so the controller hovers / regulates to the goal. The outer loop produces a desired
thrust u_1 [N] and body torque tau [Nm], which are mapped to per-rotor angular
velocities through the vehicle's own allocation matrix
(force_and_torques_to_velocities), making the backend vehicle-agnostic.

Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
Copyright (c) 2026. BSD-3-Clause.
"""

__all__ = ["NonLinearControllerBackend"]

import numpy as np
import torch

from pegasus.simulator.logic.backends.backend import Backend as _BaseBackend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf


class NonLinearControllerBackend(_BaseBackend):
    """Batched Pegasus backend running a geometric non-linear controller."""

    def __init__(
        self,
        n_vehicles: int,
        reset_manager=None,
        Kp=(10.0, 10.0, 10.0),
        Kd=(8.5, 8.5, 8.5),
        Ki=(1.50, 1.50, 1.50),
        Kr=(3.5, 3.5, 3.5),
        Kw=(0.5, 0.5, 0.5),
        mass: float = 1.5,
        gravity: float = 9.81,
        action_mode: str = "rotor_velocity",
    ):
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

        # Controller gains (stored as plain lists; turned into diagonal matrices in start()).
        self._Kp_cfg = list(Kp)
        self._Kd_cfg = list(Kd)
        self._Ki_cfg = list(Ki)
        self._Kr_cfg = list(Kr)
        self._Kw_cfg = list(Kw)

        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.Kr = None
        self.Kw = None

        self._mass = float(mass)
        self._g = float(gravity)

        # Integral of the position error, per vehicle (N, 3).
        self._int = None
        # Last commanded rotor velocities (kept for debug / recorder compatibility).
        self._last_input_ref = None

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
        self._int = torch.zeros((self._n_vehicles, 3), dtype=torch.float32, device=self._device)

        # Build diagonal gain matrices on the right device.
        self.Kp = torch.diag(torch.tensor(self._Kp_cfg, dtype=torch.float32, device=self._device))
        self.Kd = torch.diag(torch.tensor(self._Kd_cfg, dtype=torch.float32, device=self._device))
        self.Ki = torch.diag(torch.tensor(self._Ki_cfg, dtype=torch.float32, device=self._device))
        self.Kr = torch.diag(torch.tensor(self._Kr_cfg, dtype=torch.float32, device=self._device))
        self.Kw = torch.diag(torch.tensor(self._Kw_cfg, dtype=torch.float32, device=self._device))

        # Prefer the vehicle's own mass if it exposes one; fall back to the configured value.
        veh_mass = getattr(self._vehicle, "mass", None)
        if veh_mass is not None:
            try:
                self._mass = float(veh_mass[0] if hasattr(veh_mass, "__len__") else veh_mass)
            except (TypeError, ValueError):
                pass

        self._received_first_state = False
        self._vehicle.set_input_mode(self._action_mode)
        self._prime_motors()

    def stop(self):
        pass

    def reset(self):
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._state_cache.zero_()
            self._int.zero_()
            self._prime_motors()

        self._received_first_state = False

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if not self._received_first_state:
            return

        # --- current state (world frame, ENU; angular velocity in body frame) ---
        p = self._state_cache[:, 0:3]                         # (N, 3)
        v = self._state_cache[:, 3:6]                         # (N, 3)
        quat_wxyz = self._state_cache[:, 6:10]                # (N, 4)  w, x, y, z
        w = self._state_cache[:, 10:13]                       # (N, 3)  body frame
        R = quaternion_to_matrix(quat_wxyz)                   # (N, 3, 3)

        # --- references from the reset manager (point/velocity tracking) ---
        p_ref = self._goal_tensor(self.reset_manager.goal_pos, p)   # (N, 3)
        v_ref = self._goal_tensor(self.reset_manager.goal_vel, v)   # (N, 3)
        a_ref = torch.zeros_like(p_ref)
        j_ref = torch.zeros_like(p_ref)
        yaw_ref = torch.zeros((self._n_vehicles,), dtype=torch.float32, device=self._device)
        yaw_rate_ref = torch.zeros_like(yaw_ref)

        # --- outer loop: desired force ---
        ep = p - p_ref
        ev = v - v_ref
        self._int = self._int + ep * dt
        ei = self._int

        gravity_comp = torch.tensor([0.0, 0.0, self._mass * self._g],
                                    dtype=torch.float32, device=self._device).unsqueeze(0)
        F_des = -(ep @ self.Kp.T) - (ev @ self.Kd.T) - (ei @ self.Ki.T) + gravity_comp + self._mass * a_ref

        # Body z-axis (3rd column of R) and total thrust along it.
        Z_B = R[:, :, 2]
        u_1 = torch.sum(F_des * Z_B, dim=1)                   # (N,)

        # --- desired attitude ---
        Z_b_des = F_des / torch.linalg.norm(F_des, dim=1, keepdim=True).clamp_min(1e-6)
        X_c_des = torch.stack((torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref)), dim=1)
        Z_cross_Xc = torch.cross(Z_b_des, X_c_des, dim=1)
        Y_b_des = Z_cross_Xc / torch.linalg.norm(Z_cross_Xc, dim=1, keepdim=True).clamp_min(1e-6)
        X_b_des = torch.cross(Y_b_des, Z_b_des, dim=1)
        R_des = torch.stack((X_b_des, Y_b_des, Z_b_des), dim=2)

        # Attitude error (body frame).
        e_R_mat = torch.matmul(R_des.transpose(1, 2), R) - torch.matmul(R.transpose(1, 2), R_des)
        e_R = 0.5 * self.vee_batch(e_R_mat)

        # --- desired body rates (feed-forward from jerk) ---
        proj = torch.sum(Z_b_des * j_ref, dim=1, keepdim=True)
        hw = (self._mass / torch.clamp(u_1, min=1e-6)).unsqueeze(1) * (j_ref - proj * Z_b_des)
        w_des = torch.stack(
            (-torch.sum(hw * Y_b_des, dim=1),
             torch.sum(hw * X_b_des, dim=1),
             yaw_rate_ref * Z_b_des[:, 2]),
            dim=1,
        )
        e_w = w - w_des

        # --- inner loop: body torque ---
        tau = -(e_R @ self.Kr.T) - (e_w @ self.Kw.T)

        # --- allocation: (thrust, torque) -> per-rotor angular velocities ---
        self._input_ref = self._vehicle.force_and_torques_to_velocities(u_1, tau)
        self._last_input_ref = self._input_ref

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
        if linear_velocity is not None:
            self._state_cache[env_ids, 3:6] = linear_velocity.to(self._device, dtype=torch.float32)
        else:
            self._state_cache[env_ids, 3:6] = 0.0
        self._state_cache[env_ids, 6:10] = attitudes.to(self._device, dtype=torch.float32)
        if angular_velocity is not None:
            self._state_cache[env_ids, 10:13] = angular_velocity.to(self._device, dtype=torch.float32)
        else:
            self._state_cache[env_ids, 10:13] = 0.0

        # Reset the integrator for the environments that were just reset.
        self._int[env_ids] = 0.0
        self._prime_motors(env_ids=env_ids)

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
        # This controller does not produce normalized [-1, 1] actions; expose None so
        # recorders that log action_norm simply skip it.
        return None

    # ------------------------------------------------------------------
    # Goal marker (visual reference cube)
    # ------------------------------------------------------------------

    def create_goal_marker(self, root_path: str = "/World/GoalMarkers", size: float = 0.15, color: tuple = (0.0, 0.4, 1.0)):
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

    def _goal_tensor(self, goal, like: torch.Tensor) -> torch.Tensor:
        """Coerce a goal (tensor / array / list / None) into a (N, 3) torch tensor."""
        if goal is None:
            return torch.zeros_like(like)
        if isinstance(goal, torch.Tensor):
            g = goal.to(device=self._device, dtype=torch.float32)
        else:
            g = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=self._device)
        if g.ndim == 1:
            g = g.unsqueeze(0).expand(self._n_vehicles, -1)
        return g

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

    @staticmethod
    def vee_batch(S: torch.Tensor) -> torch.Tensor:
        """so(3) -> R^3 vee map for a batch of skew-symmetric matrices."""
        return torch.stack((-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]), dim=1)
