"""
single_quad_policy.py
Wrapper for the RAPTOR teacher policy trained on a single fixed-dynamics quadrotor.

Observation layout (26 dims) — matches pre_training/environment.h:
  [0:3]   pos_error (pos_w - goal_pos_w)
  [3:12]  R_flat (row-major rotation matrix, 9 values)
  [12:15] vel_error (vel_w - goal_vel_w)
  [15:18] ang_vel_b
  [18:22] last_action_norm  (action history, normalized [-1,1])
  [22:26] rotor_speeds_norm (actual motor speeds, normalized [-1,1])
           norm = (omega - omega_min) / (omega_max - omega_min) * 2 - 1

Action (4 dims) — normalized [-1,1]:
  mapped to omega via: omega = (action + 1) / 2 * (omega_max - omega_min) + omega_min

Motor delay simulation (first-order):
  omega_actual[t] = omega_actual[t-1] + (omega_cmd[t-1] - omega_actual[t-1]) * dt / tau

Usage:
    policy = SingleQuadPolicy(checkpoint_path)
    policy.reset()
    for step in range(N):
        action_norm = policy.step(pos_error, R_flat, vel_error, ang_vel_b)
        omega_cmd = policy.action_to_omega(action_norm)
"""

import numpy as np
import h5py



def _relu(x):
    return np.maximum(0.0, x)


def _tanh(x):
    return np.tanh(x)


class SingleQuadPolicy:
    """MLP teacher policy loaded from a RAPTOR pre-training HDF5 checkpoint."""

    OBS_DIM = 26
    ACT_DIM = 4

    def __init__(self, checkpoint_path: str, omega_min: float, omega_max: float, motor_tau_rising: float, motor_tau_falling: float, dt: float):
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.tau_rising  = motor_tau_rising
        self.tau_falling = motor_tau_falling
        self.dt = dt

        self._load_weights(checkpoint_path)

        self._last_action_norm = np.zeros(self.ACT_DIM, dtype=np.float32)
        self._omega_actual = np.full(self.ACT_DIM, 0.5 * (omega_min + omega_max), dtype=np.float32)

    # ------------------------------------------------------------------

    def _load_weights(self, path: str):
        """Load MLP weights from HDF5.

        Architecture (from checkpoint):
          input_layer:  (64, 26)  weights + (1, 64)  biases  -> RELU
          hidden_layer: (64, 64)  weights + (1, 64)  biases  -> RELU
          output_layer: (8,  64)  weights + (1, 8)   biases  -> identity
            output[0:4] = mean (tanh-squashed action)
            output[4:8] = log_std (not used at inference)
        """
        with h5py.File(path, 'r') as f:
            layers = f['actor/layers/0']
            self.W0 = layers['input_layer/weights/parameters'][()].T   # (26, 64)
            self.b0 = layers['input_layer/biases/parameters'][()].squeeze()   # (64,)
            self.W1 = layers['hidden_layer_0/weights/parameters'][()].T # (64, 64)
            self.b1 = layers['hidden_layer_0/biases/parameters'][()].squeeze()  # (64,)
            self.W2 = layers['output_layer/weights/parameters'][()].T  # (64, 8)
            self.b2 = layers['output_layer/biases/parameters'][()].squeeze()   # (8,)

    # ------------------------------------------------------------------

    def reset(self):
        self._last_action_norm[:] = 0.0
        self._omega_actual[:] = 0.5 * (self.omega_min + self.omega_max)

    # ------------------------------------------------------------------

    def step(self, pos_error: np.ndarray, R_flat: np.ndarray, vel_error: np.ndarray, ang_vel_b: np.ndarray) -> np.ndarray:
        """Run one policy step. Returns action_norm in [-1, 1] (shape (4,)).

        Args:
            pos_error: (3,) pos_w - goal_pos_w
            R_flat:    (9,) row-major rotation matrix
            vel_error: (3,) vel_w - goal_vel_w
            ang_vel_b: (3,) angular velocity in body frame (rad/s)
        """
        # Normalize actual rotor speeds to [-1, 1]
        rotor_speeds_norm = (self._omega_actual - self.omega_min) / \
                            (self.omega_max - self.omega_min) * 2.0 - 1.0

        obs = np.concatenate([
            pos_error.astype(np.float32),
            R_flat.astype(np.float32),
            vel_error.astype(np.float32),
            ang_vel_b.astype(np.float32),
            self._last_action_norm,
            rotor_speeds_norm.astype(np.float32),
        ]).astype(np.float32)

        assert obs.shape == (self.OBS_DIM,), f"Expected obs dim {self.OBS_DIM}, got {obs.shape}"

        # MLP forward pass
        h = _relu(obs @ self.W0 + self.b0)
        h = _relu(h @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2        # (8,)

        # Take only the mean (first 4); apply tanh squash (SampleAndSquash deterministic)
        action_norm = np.tanh(out[:self.ACT_DIM]).astype(np.float32)

        # Update motor delay state
        self._update_motor_delay(action_norm)

        # Store for next obs
        self._last_action_norm = action_norm.copy()

        return action_norm

    # ------------------------------------------------------------------

    def action_to_omega(self, action_norm: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to omega [rad/s]."""
        a = np.clip(action_norm, -1.0, 1.0)
        return ((a + 1.0) * 0.5 * (self.omega_max - self.omega_min) + self.omega_min).astype(np.float32)

    def omega_to_action(self, omega: np.ndarray) -> np.ndarray:
        """Convert omega [rad/s] to normalized action [-1, 1]."""
        return ((omega - self.omega_min) / (self.omega_max - self.omega_min) * 2.0 - 1.0).astype(np.float32)

    # ------------------------------------------------------------------

    def _update_motor_delay(self, action_norm: np.ndarray):
        """First-order motor delay: separate tau for rising/falling."""
        omega_cmd = self.action_to_omega(action_norm)
        rising  = omega_cmd > self._omega_actual
        tau = np.where(rising, self.tau_rising, self.tau_falling)
        alpha = self.dt / tau
        self._omega_actual += alpha * (omega_cmd - self._omega_actual)
        self._omega_actual = np.clip(self._omega_actual, self.omega_min, self.omega_max)
