"""
foundation_policy_pre_controller.py
Pegasus batched backend using the RAPTOR pre-training teacher policy
(MLP, 26-dim obs, trained on a single fixed-dynamics quadrotor via SAC).

Observation layout (26 dims) — pre_training/environment.h:
  [0:3]   pos_error (pos_w - goal_pos_w)
  [3:12]  R_flat (row-major rotation matrix)
  [12:15] vel_error (vel_w - goal_vel_w)
  [15:18] ang_vel_b
  [18:22] last_action_norm   (action history, managed inside SingleQuadPolicy)
  [22:26] rotor_speeds_norm  (actual motor speeds after delay, managed inside SingleQuadPolicy)
"""

__all__ = ["PreTrainBackend"]

import itertools
import sys
from pathlib import Path
import numpy as np
import torch

from pegasus.simulator.logic.backends.backend import Backend as _BaseBackend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

# Allow import from utils/ folder itself
_UTILS_DIR = str(Path(__file__).resolve().parent)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from single_quad_policy import SingleQuadPolicy

# RAPTOR Iris rotor XY positions (FLU, arm = 0.25/sqrt(2) m)
_RAPTOR_ROTOR_XY = (
    ( 0.176777, -0.176777),
    (-0.176777, -0.176777),
    (-0.176777,  0.176777),
    ( 0.176777,  0.176777),
)


class PreTrainBackend(_BaseBackend):
    """Batched Pegasus backend running the single-quad pre-training teacher policy."""

    def __init__(
        self,
        checkpoint_path: str,
        n_vehicles: int,
        reset_manager=None,
        action_mode: str = "rotor_velocity",
        omega_min: float = 0.0,
        omega_max: float = 1100.0,
        motor_tau_rising: float = 0.04,
        motor_tau_falling: float = 0.04,
        dt: float = 0.01,
    ):
        super().__init__(config=None)
        self._checkpoint_path = checkpoint_path
        self._n_vehicles = int(n_vehicles)
        self._action_mode = action_mode
        self.reset_manager = reset_manager

        self._omega_min = omega_min
        self._omega_max = omega_max
        self._motor_tau_rising  = motor_tau_rising
        self._motor_tau_falling = motor_tau_falling
        self._dt = dt

        self._vehicle  = None
        self._device   = None

        self._input_ref   = None
        self._state_cache = None
        self._received_first_state = False

        self._policies: list[SingleQuadPolicy] = []
        self._raptor_to_pegasus = None

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

        self._input_ref   = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)

        self._policies = [
            SingleQuadPolicy(
                self._checkpoint_path,
                omega_min=self._omega_min,
                omega_max=self._omega_max,
                motor_tau_rising=self._motor_tau_rising,
                motor_tau_falling=self._motor_tau_falling,
                dt=self._dt,
            )
            for _ in range(self._n_vehicles)
        ]
        for p in self._policies:
            p.reset()

        self._received_first_state = False
        self._vehicle.set_input_mode(self._action_mode)
        self._raptor_to_pegasus = self._infer_rotor_order()
        self._prime_motors()

    def stop(self):
        pass

    def reset(self):
        if self._input_ref is not None:
            self._input_ref.zero_()
            self._state_cache.zero_()

        for p in self._policies:
            p.reset()

        self._prime_motors()
        self._received_first_state = False

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if not self._received_first_state or not self._policies:
            return
        if self._action_mode != "rotor_velocity":
            raise RuntimeError("PreTrainBackend requires action_mode='rotor_velocity'.")

        pos_w     = self._state_cache[:, 0:3]
        vel_w     = self._state_cache[:, 3:6]
        quat      = self._state_cache[:, 6:10]    # w,x,y,z
        ang_vel_b = self._state_cache[:, 10:13]

        goal_pos = self.reset_manager.goal_pos
        goal_vel = self.reset_manager.goal_vel
        if isinstance(goal_pos, torch.Tensor):
            goal_pos = goal_pos.cpu().numpy()
        else:
            goal_pos = np.asarray(goal_pos, dtype=np.float32)
        if isinstance(goal_vel, torch.Tensor):
            goal_vel = goal_vel.cpu().numpy()
        else:
            goal_vel = np.asarray(goal_vel, dtype=np.float32)

        pos_error_np = pos_w.cpu().numpy() - goal_pos           # (N, 3)
        vel_error_np = vel_w.cpu().numpy() - goal_vel           # (N, 3)
        R_flat_np    = quaternion_to_matrix(quat).reshape(self._n_vehicles, 9).cpu().numpy()
        ang_vel_np   = ang_vel_b.cpu().numpy()

        omega_cmds = np.empty((self._n_vehicles, 4), dtype=np.float32)
        for i, policy in enumerate(self._policies):
            action_norm = policy.step(
                pos_error_np[i],
                R_flat_np[i],
                vel_error_np[i],
                ang_vel_np[i],
            )
            omega_cmds[i] = policy.action_to_omega(action_norm)

        omega_t = torch.from_numpy(omega_cmds).to(device=self._device, dtype=torch.float32)

        omega_pegasus = torch.empty_like(omega_t)
        omega_pegasus[:, self._raptor_to_pegasus] = omega_t
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
        self._state_cache[env_ids, 0:3]   = positions.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 3:6]   = linear_velocity.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 6:10]  = attitudes.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 10:13] = angular_velocity.to(self._device, dtype=torch.float32)

        for i in env_ids.cpu().tolist():
            self._policies[i].reset()

        self._prime_motors(env_ids=env_ids)
        self._received_first_state = True

    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        n, parts = self._n_vehicles, self._vehicle.parts_per_vehicle
        z = torch.zeros((n, parts, 3), dtype=torch.float32, device=self._device)
        return z, z

    def get_state(self) -> torch.Tensor:
        return self._state_cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_rotor_order(self) -> torch.Tensor:
        identity = torch.arange(4, dtype=torch.long, device=self._device)
        rotor_positions = self._vehicle._rotor_positions_body
        if rotor_positions is None:
            return identity

        pegasus_xy = rotor_positions[0, :, :2].to(dtype=torch.float32, device=self._device)
        if pegasus_xy.shape != (4, 2) or not torch.isfinite(pegasus_xy).all():
            return identity

        raptor_xy = torch.tensor(_RAPTOR_ROTOR_XY, dtype=torch.float32, device=self._device)
        peg_norm = pegasus_xy / pegasus_xy.abs().max().clamp_min(1e-6)
        rap_norm = raptor_xy  / raptor_xy.abs().max().clamp_min(1e-6)

        best_perm, best_cost = list(range(4)), float("inf")
        for perm in itertools.permutations(range(4)):
            cost = torch.sum((peg_norm[list(perm)] - rap_norm) ** 2).item()
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)

        return torch.tensor(best_perm, dtype=torch.long, device=self._device)

    def _prime_motors(self, env_ids: torch.Tensor | None = None):
        if self._input_ref is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self._n_vehicles, device=self._device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self._device, dtype=torch.long)

        omega_init = torch.full(
            (env_ids.numel(), 4),
            0.5 * (self._omega_min + self._omega_max),
            dtype=torch.float32, device=self._device,
        )
        self._input_ref[env_ids] = omega_init
        self._vehicle._thrusters._input_reference[env_ids] = omega_init
        self._vehicle._thrusters._velocity[env_ids] = omega_init
