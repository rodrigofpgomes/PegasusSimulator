"""
foundation_policy_pre_controller.py
Pegasus batched backend using the RAPTOR pre-training teacher policy
(MLP, 26-dim obs, trained on a single fixed-dynamics quadrotor via SAC).

This file MERGES the former single_quad_policy.py MLP wrapper directly into the
backend. Because the backend owns the vehicle, the rotor-speed observation
channel is read straight from the thrust curve (vehicle._thrusters._velocity)
instead of an internal motor model, and the teacher MLP is evaluated for all N
vehicles in a single vectorized forward pass (the checkpoint weights are shared
by every vehicle).

Observation layout (26 dims) -- pre_training/environment.h:
  [0:3]   pos_error (pos_w - goal_pos_w)
  [3:12]  R_flat (row-major rotation matrix)
  [12:15] vel_error (vel_w - goal_vel_w)
  [15:18] ang_vel_b
  [18:22] last_action_norm   (action history, normalized [-1, 1])
  [22:26] rotor_speeds_norm  (actual motor speeds, normalized [-1, 1])
           preferred source: REAL simulator rotor state (thrusters._velocity);
           fallback: internal first-order motor model.
           norm = (omega - omega_min) / (omega_max - omega_min) * 2 - 1

Action (4 dims), normalized [-1, 1]:
  omega = (action + 1) / 2 * (omega_max - omega_min) + omega_min
"""

__all__ = ["PreTrainBackend"]

import itertools
import numpy as np
import torch
import h5py

from pegasus.simulator.logic.backends.backend import Backend as _BaseBackend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix

# RAPTOR Iris rotor XY positions (FLU, arm = 0.25/sqrt(2) m)
_RAPTOR_ROTOR_XY = (
    ( 0.176777, -0.176777),
    (-0.176777, -0.176777),
    (-0.176777,  0.176777),
    ( 0.176777,  0.176777),
)


def _relu(x):
    return np.maximum(0.0, x)


class PreTrainBackend(_BaseBackend):
    """Batched Pegasus backend running the single-quad pre-training teacher policy.

    The teacher MLP (loaded once from an HDF5 checkpoint) is evaluated for all
    vehicles in one vectorized forward pass. The rotor-speed observation channel
    is taken from the vehicle thrust curve when available, falling back to an
    internal first-order motor model otherwise.
    """

    OBS_DIM = 26
    ACT_DIM = 4

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

        self._raptor_to_pegasus = None

        # Teacher MLP weights (loaded once; shared by all vehicles).
        self._load_weights(checkpoint_path)

        # Per-vehicle policy state (vectorized; allocated in start()).
        self._last_action_norm = None   # (N, 4) action history
        self._omega_actual     = None   # (N, 4) internal motor-model fallback

    # ------------------------------------------------------------------
    # Teacher MLP (formerly SingleQuadPolicy)
    # ------------------------------------------------------------------

    def _load_weights(self, path: str):
        """Load MLP weights from a RAPTOR pre-training HDF5 checkpoint.

        Architecture:
          input_layer:  (64, 26) weights + (1, 64) biases -> RELU
          hidden_layer: (64, 64) weights + (1, 64) biases -> RELU
          output_layer: (8,  64) weights + (1, 8)  biases -> identity
            output[0:4] = mean (tanh-squashed action)
            output[4:8] = log_std (unused at inference)
        """
        with h5py.File(path, "r") as f:
            layers = f["actor/layers/0"]
            self.W0 = layers["input_layer/weights/parameters"][()].T    # (26, 64)
            self.b0 = layers["input_layer/biases/parameters"][()].squeeze()     # (64,)
            self.W1 = layers["hidden_layer_0/weights/parameters"][()].T  # (64, 64)
            self.b1 = layers["hidden_layer_0/biases/parameters"][()].squeeze()  # (64,)
            self.W2 = layers["output_layer/weights/parameters"][()].T    # (64, 8)
            self.b2 = layers["output_layer/biases/parameters"][()].squeeze()    # (8,)

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        """Vectorized deterministic actor forward pass.

        obs: (N, 26) -> action_norm: (N, 4) in [-1, 1] (tanh-squashed mean).
        """
        h = _relu(obs @ self.W0 + self.b0)
        h = _relu(h @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2                   # (N, 8)
        return np.tanh(out[:, : self.ACT_DIM]).astype(np.float32)

    def _action_to_omega(self, action_norm: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to omega [rad/s]."""
        a = np.clip(action_norm, -1.0, 1.0)
        return ((a + 1.0) * 0.5 * (self._omega_max - self._omega_min) + self._omega_min).astype(np.float32)

    def _read_rotor_speeds_norm(self) -> np.ndarray:
        """Rotor-speed observation in RAPTOR order, normalized to [-1, 1].

        Preferred source is the actual thrust-curve rotor speed
        (vehicle._thrusters._velocity); falls back to the internal first-order
        motor model when the thrusters are unavailable.
        """
        rotor_vel = self._vehicle._thrusters._velocity.to(self._device, dtype=torch.float32)   # (N, 4) Pegasus order
        rotor_vel_policy = rotor_vel[:, self._raptor_to_pegasus]      # -> RAPTOR order
        rotor_norm = (rotor_vel_policy - self._omega_min) / (self._omega_max - self._omega_min) * 2.0 - 1.0
        
        return rotor_norm.cpu().numpy().astype(np.float32)

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

        mid = 0.5 * (self._omega_min + self._omega_max)
        self._last_action_norm = np.zeros((self._n_vehicles, self.ACT_DIM), dtype=np.float32)
        self._omega_actual     = np.full((self._n_vehicles, self.ACT_DIM), mid, dtype=np.float32)

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

        if self._last_action_norm is not None:
            self._last_action_norm[:] = 0.0
            self._omega_actual[:] = 0.5 * (self._omega_min + self._omega_max)

        self._prime_motors()
        self._received_first_state = False

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if not self._received_first_state or self._last_action_norm is None:
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

        pos_error_np = (pos_w.cpu().numpy() - goal_pos).astype(np.float32)   # (N, 3)
        vel_error_np = (vel_w.cpu().numpy() - goal_vel).astype(np.float32)   # (N, 3)
        R_flat_np    = quaternion_to_matrix(quat).reshape(self._n_vehicles, 9).cpu().numpy().astype(np.float32)
        ang_vel_np   = ang_vel_b.cpu().numpy().astype(np.float32)

        # Rotor-speed observation from the REAL simulator state (matches
        # rl-tools `state.rpm`), already reordered into RAPTOR rotor order.
        rotor_norm_np = self._read_rotor_speeds_norm()                      # (N, 4)

        # Assemble the full 26-dim observation batch.
        obs = np.concatenate([pos_error_np, R_flat_np, vel_error_np, ang_vel_np, self._last_action_norm, rotor_norm_np], axis=1).astype(np.float32) # (N, 26)
        
        assert obs.shape == (self._n_vehicles, self.OBS_DIM), \
            f"Expected obs (N, {self.OBS_DIM}), got {obs.shape}"

        action_norm = self._forward(obs)                                    # (N, 4)

        # Store action history.
        self._last_action_norm = action_norm.copy()

        # RAPTOR-order omega commands -> Pegasus rotor order.
        omega_cmds = self._action_to_omega(action_norm)                     # (N, 4) RAPTOR order
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

        # Reset per-env policy state (action history + internal motor model).
        if self._last_action_norm is not None:
            idx = env_ids.cpu().numpy()
            self._last_action_norm[idx] = 0.0
            self._omega_actual[idx] = 0.5 * (self._omega_min + self._omega_max)

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
