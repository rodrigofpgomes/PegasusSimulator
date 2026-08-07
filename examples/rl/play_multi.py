#!/usr/bin/env python
"""
| File: play_multi.py
| Description: Unified multi-vehicle evaluation (final play). Spawns any subset of
|              controllers in a shared world for side-by-side comparison:
|                - RAPTOR foundation policy on Iris (--raptor) and optionally Crazyflie (--crazyflie)
|                - RAPTOR pre-training foundation policy on Iris (--pre_checkpoint)
|                - SAC-trained policy on Iris (--sac_checkpoint)
|              No RL training. Resets are driven by a fixed episode duration.
|              Supports static goal and lemniscate (figure-8) trajectory tracking, and
|              optional CSV recording consumed by dashboard_final.py.
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
"""

import sys
import csv
import json
import math
import argparse
import importlib
import importlib.util
from pathlib import Path

import torch
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent.parent
PEGASUS_EXT_DIR = PROJECT_ROOT / "extensions" / "pegasus.simulator"

for p in [str(RL_DIR), str(PROJECT_ROOT), str(PEGASUS_EXT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------
# Lemniscate (figure-8) trajectory
# ---------------------------------------------------------------------

class LemniscateTrajectory:
    """Bernoulli lemniscate (figure-8) reference trajectory in the XY plane.

    Parametric form (Gerono lemniscate):
        x(t) = cx + A * sin(w*t)
        y(t) = cy + A * sin(w*t) * cos(w*t)  = cy + (A/2) * sin(2*w*t)
        z(t) = z_ref   (constant altitude)

    Velocity:
        dx/dt = A*w * cos(w*t)
        dy/dt =   A*w * cos(2*w*t)
        dz/dt = 0

    The per-env phase offset makes each environment track a different point
    on the curve so that they don't all crash at the same moment.

    Args:
        n_envs:    Number of parallel environments.
        amplitude: Half-width of the figure-8 [m].
        period:    Time for one full loop [s].
        z_ref:     Reference altitude [m].
        center:    (N,3) tensor with the XY center for each env (from init_pos).
        device:    Torch device.
        phase_offset_per_env: If True spread env start phases uniformly over [0, 2π).
    """

    def __init__(
        self,
        n_envs: int,
        amplitude: float,
        period: float,
        z_ref: float,
        center: torch.Tensor,
        device: str,
        phase_offset_per_env: bool = True,
    ):
        self.n_envs = n_envs
        self.A = float(amplitude)
        self.w = 2.0 * math.pi / float(period)  # angular frequency [rad/s]
        self.z_ref = float(z_ref)
        self.device = device

        # Per-env XY center (N,3), kept on device
        self.center = center.to(device=device, dtype=torch.float32).clone()
        self.center[:, 2] = self.z_ref  # override z

        # Per-env time accumulator [s]
        self._t = torch.zeros(n_envs, dtype=torch.float32, device=device)

        if phase_offset_per_env and n_envs > 1:
            # Spread phases uniformly so envs are not synchronized
            phases = torch.linspace(0.0, 2.0 * math.pi, n_envs + 1, device=device)[:-1]
            # Convert phase offset to equivalent time offset
            self._t = phases / self.w

    def reset(self, env_ids: torch.Tensor):
        """Reset per-env timers for the given env IDs."""
        self._t[env_ids] = 0.0

    def step(self, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance time by dt and return (pos, vel, acc) each (N, 3)."""
        self._t += dt
        return self._eval(self._t)

    def at_time(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate at an explicit time tensor (N,)."""
        return self._eval(t)

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wt = self.w * t                             # (N,)
        sin_wt  = torch.sin(wt)
        cos_wt  = torch.cos(wt)
        cos_2wt = torch.cos(2.0 * wt)

        pos = self.center.clone()
        pos[:, 0] += self.A * sin_wt
        pos[:, 1] += 0.5 * self.A * torch.sin(2.0 * wt)
        # z already set to z_ref in center

        vel = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        vel[:, 0] = self.A * self.w * cos_wt
        vel[:, 1] = self.A * self.w * cos_2wt

        acc = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        acc[:, 0] = -self.A * self.w**2 * sin_wt
        acc[:, 1] = -2.0 * self.A * self.w**2 * sin_wt * cos_wt
        # vz = 0

        return pos, vel, acc

    @property
    def current_time(self) -> torch.Tensor:
        return self._t


# ---------------------------------------------------------------------
# Recorder utils
# ---------------------------------------------------------------------
def _to_np_ref(value, n_envs: int):
    if value is None:
        return np.full((n_envs, 3), np.nan, dtype=np.float32)

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] < 3:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

        arr = arr[:3][None, :]
        arr = np.repeat(arr, n_envs, axis=0)

    elif arr.ndim == 2:
        if arr.shape[1] < 3:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

        arr = arr[:, :3]

        if arr.shape[0] == 1:
            arr = np.repeat(arr, n_envs, axis=0)
        elif arr.shape[0] >= n_envs:
            arr = arr[:n_envs]
        else:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

    else:
        return np.full((n_envs, 3), np.nan, dtype=np.float32)

    return arr


def make_reference_provider(reset_manager, n_envs: int):
    def provider():
        if reset_manager is None:
            return {
                "position": np.full((n_envs, 3), np.nan, dtype=np.float32),
                "velocity": np.full((n_envs, 3), np.nan, dtype=np.float32),
            }

        ref_pos = _to_np_ref(reset_manager.goal_pos, n_envs)
        ref_vel = np.zeros((n_envs, 3), dtype=np.float32)

        return {
            "position": ref_pos,
            "velocity": ref_vel,
        }

    return provider


# ---------------------------------------------------------------------
# Attitude-error utilities  (trace(I - R_d^T R))
# ---------------------------------------------------------------------
def _desired_attitude_from_accel(a_des, yaw = 0) -> np.ndarray:
    """Build the desired rotation matrix R_d from a desired acceleration.

    Geometric / differential-flatness construction with the yaw fixed to 0,
    identical to the convention used in non_linear_controller.py and
    lqr_controller_batch.py:

        z_b_des = a_des / ||a_des||
        x_c_des = [cos(yaw), sin(yaw), 0] = [1, 0, 0]
        y_b_des = (z_b_des x x_c_des) / ||.||
        x_b_des = y_b_des x z_b_des
        R_d     = [x_b_des | y_b_des | z_b_des]   (columns)

    Returns (N,3,3).
    """
    a = np.asarray(a_des, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    n = a.shape[0]

    z_b = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)

    x_c = np.zeros((n, 3), dtype=np.float64)
    x_c[:, 0] = np.cos(yaw)
    x_c[:, 1] = np.sin(yaw)

    y_b = np.cross(z_b, x_c)
    y_norm = np.linalg.norm(y_b, axis=1, keepdims=True)
    
    x_b = np.cross(y_b, z_b)

    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, :, 0] = x_b
    R[:, :, 1] = y_b
    R[:, :, 2] = z_b
    return R


def _attitude_error_trace(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Attitude error Psi = trace(I - R_d^T R), returned per-vehicle (N,).

    trace(R_d^T R) equals the Frobenius inner product sum_ij R_d[i,j] * R[i,j],
    so Psi = 3 - sum(R_d * R).
    """
    return 3.0 - np.sum(R_des * R, axis=(1, 2))


class EpisodeTrajectoryRecorder:
    def __init__(
        self,
        vehicles: dict,
        n_envs: int,
        step_dt: float,
        out_dir: str,
        record_every: int = 1,
        reset_manager=None,
        trajectory=None,
        n_rotors: int = 4,
        use_tensorboard: bool = False,
    ):
        self.vehicles = vehicles
        self.n_envs = n_envs
        self.step_dt = float(step_dt)
        self.record_every = max(1, int(record_every))
        self.reset_manager = reset_manager
        self.trajectory = trajectory

        self.n_rotors = int(n_rotors)

        # --- Desired-attitude (R_d) construction parameters (yaw fixed to 0) ---
        # a_des = a_ref + Kp*(p_ref - p) + Kd*(v_ref - v) + g*e3
        # Kp/Kd/mass mirror the non_linear_controller defaults (Kp=10, Kd=8.5,
        # m=1.5) expressed per unit mass, so the desired thrust direction (and
        # hence R_d) matches that geometric controller. The feed-forward variant
        # drops the PD term (a_des = a_ref + g*e3).
        self.att_gravity = 9.81
        self.att_mass = 1.5
        self.att_Kp = np.array([2.0, 2.0, 2.0], dtype=np.float64) / self.att_mass
        self.att_Kd = np.array([1.2, 1.2, 1.2], dtype=np.float64) / self.att_mass

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_step = np.zeros(n_envs, dtype=np.int64)
        self.completed_episodes = np.zeros(n_envs, dtype=np.int64)

        self.traj_file = open(self.out_dir / "trajectories.csv", "w", newline="")
        self.ep_file = open(self.out_dir / "episodes.csv", "w", newline="")

        rotor_cols = []
        for group in ("cmd_w", "cmd_rpm", "actual_w", "actual_rpm",
                    "l2f_action", "l2f_cmd_rpm", "l2f_current_rpm"):
            rotor_cols += [f"{group}_{i}" for i in range(self.n_rotors)]

        self.traj_writer = csv.DictWriter(
            self.traj_file,
            fieldnames=[
                "controller",
                "env_id",
                "episode_id",
                "episode_step",
                "global_step",
                "t",

                "x", "y", "z",
                "vx", "vy", "vz",
                "speed",

                "goal_x", "goal_y", "goal_z",
                "ref_vx", "ref_vy", "ref_vz",
                "pos_error",
                "z_error",

                "att_err_ff", "att_err_pd",

                *rotor_cols,

                "fx_body", "fy_body", "fz_body",
                "body_force_norm",

                "tx_body", "ty_body", "tz_body",
                "body_torque_norm",
            ],
        )
        self.traj_writer.writeheader()

        self.ep_writer = csv.DictWriter(
            self.ep_file,
            fieldnames=[
                "env_id",
                "episode_id",
                "end_global_step",
                "length_steps",
                "duration_s",
                "reason",
            ],
        )
        self.ep_writer.writeheader()

        self.tb = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))

    def _get_goals_numpy(self):
        if self.reset_manager is None:
            return np.full((self.n_envs, 3), np.nan, dtype=np.float32)

        return _to_np_ref(self.reset_manager.goal_pos, self.n_envs)

    def _get_body_force_torque_numpy(self, vehicle):
        body_index = vehicle.body_index
        last_forces = vehicle._last_forces_local
        last_torques = vehicle._last_torques_local

        if last_forces is None:
            body_force = np.full((self.n_envs, 3), np.nan, dtype=np.float32)
        else:
            body_force = last_forces[:, body_index, :].detach().cpu().numpy()

        if last_torques is None:
            body_torque = np.full((self.n_envs, 3), np.nan, dtype=np.float32)
        else:
            body_torque = last_torques[:, body_index, :].detach().cpu().numpy()

        return body_force, body_torque

    def _get_first_backend(self, vehicle):
        backends = vehicle._backends
        if not backends:
            return None
        return backends[0]

    def _get_rotor_input_numpy(self, vehicle):
        nan = np.full((self.n_envs, self.n_rotors), np.nan, dtype=np.float32)

        backend = self._get_first_backend(vehicle)

        cmd_w = backend.input_reference() if backend is not None else None

        if cmd_w is None:
            cmd_w_np = nan.copy()
        else:
            if isinstance(cmd_w, torch.Tensor):
                cmd_w_np = cmd_w.detach().cpu().numpy()
            else:
                cmd_w_np = np.asarray(cmd_w, dtype=np.float32)

            if cmd_w_np.ndim != 2 or cmd_w_np.shape[1] < self.n_rotors:
                cmd_w_np = nan.copy()
            else:
                cmd_w_np = cmd_w_np[:self.n_envs, :self.n_rotors]

        actual_w = vehicle._thrusters.velocity

        if actual_w is None:
            actual_w_np = nan.copy()
        else:
            if isinstance(actual_w, torch.Tensor):
                actual_w_np = actual_w.detach().cpu().numpy()
            else:
                actual_w_np = np.asarray(actual_w, dtype=np.float32)

            if actual_w_np.ndim != 2 or actual_w_np.shape[1] < self.n_rotors:
                actual_w_np = nan.copy()
            else:
                actual_w_np = actual_w_np[:self.n_envs, :self.n_rotors]

        return cmd_w_np, actual_w_np

    def _get_l2f_debug_numpy(self, vehicle):
        nan = np.full((self.n_envs, self.n_rotors), np.nan, dtype=np.float32)
        backend = self._get_first_backend(vehicle)

        def read_attr(name):
            if backend is None or not hasattr(backend, name):
                return nan.copy()
            value = getattr(backend, name)
            if value is None:
                return nan.copy()
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < self.n_rotors:
                return nan.copy()
            return arr[:self.n_envs, :self.n_rotors]

        return (
            read_attr("last_action_norm_l2f"),
            read_attr("last_rpm_cmd_l2f"),
            read_attr("last_current_rpm_l2f"),
        )

    def _get_ref_vel_numpy(self):
        """Returns reference velocity (N,3) from trajectory, or zeros for static goal."""
        if self.trajectory is not None:
            _, ref_vel, _ = self.trajectory.at_time(self.trajectory.current_time)
            return ref_vel.cpu().numpy()
        return np.zeros((self.n_envs, 3), dtype=np.float32)

    def _get_ref_acc_numpy(self):
        """Returns reference acceleration (N,3) from trajectory, or zeros for static goal."""
        if self.trajectory is not None:
            _, _, ref_acc = self.trajectory.at_time(self.trajectory.current_time)
            return ref_acc.detach().cpu().numpy()
        return np.zeros((self.n_envs, 3), dtype=np.float32)

    def _compute_attitude_errors(self, state, goals, ref_vel, ref_acc):
        """Attitude error trace(I - R_d^T R), with R_d built from the desired
        acceleration and yaw fixed to 0. Returns (att_err_ff, att_err_pd), (N,).

          att_err_ff : R_d from the feed-forward reference only
                       (a_des = a_ref + g*e3)   -> 'sem PD'
          att_err_pd : R_d from the full outer loop
                       (a_des = a_ref + Kp*(p_ref-p) + Kd*(v_ref-v) + g*e3) -> 'com PD'

        For a static hover goal (a_ref = 0) the feed-forward R_d reduces to the
        identity (level attitude), so att_err_ff measures how far the vehicle is
        tilted away from level flight.
        """
        R = quaternion_to_matrix(state.attitude).detach().cpu().numpy()  # (N,3,3) world<-body

        pos = state.position.detach().cpu().numpy().astype(np.float64)
        vel = state.linear_velocity.detach().cpu().numpy().astype(np.float64)

        p_ref = np.asarray(goals, dtype=np.float64)
        v_ref = np.asarray(ref_vel, dtype=np.float64)
        a_ref = np.asarray(ref_acc, dtype=np.float64)

        g_vec = np.array([0.0, 0.0, self.att_gravity], dtype=np.float64)

        # Feed-forward desired acceleration (no position/velocity feedback).
        a_des_ff = a_ref + g_vec
        # Full outer-loop desired acceleration (PD feedback + feed-forward).
        a_des_pd = a_ref - self.att_Kp * (pos - p_ref) + self.att_Kd * (vel - v_ref) + g_vec

        R_des_ff = _desired_attitude_from_accel(a_des_ff)
        R_des_pd = _desired_attitude_from_accel(a_des_pd)

        att_err_ff = _attitude_error_trace(R, R_des_ff)
        att_err_pd = _attitude_error_trace(R, R_des_pd)
        return att_err_ff, att_err_pd

    def sample(self):
        if self.global_step % self.record_every != 0:
            return

        goals = self._get_goals_numpy()
        ref_vel = self._get_ref_vel_numpy()
        ref_acc = self._get_ref_acc_numpy()

        for controller_name, vehicle in self.vehicles.items():
            if vehicle is None:
                continue

            state = vehicle.state

            pos = state.position.detach().cpu().numpy()

            vel = state.linear_velocity.detach().cpu().numpy()
            speed = np.linalg.norm(vel, axis=1)

            body_force, body_torque = self._get_body_force_torque_numpy(vehicle)
            cmd_w, actual_w = self._get_rotor_input_numpy(vehicle)
            l2f_action, l2f_cmd_rpm, l2f_current_rpm = self._get_l2f_debug_numpy(vehicle)

            cmd_rpm = cmd_w * 60.0 / (2.0 * np.pi)
            actual_rpm = actual_w * 60.0 / (2.0 * np.pi)

            pos_error = np.linalg.norm(pos - goals, axis=1)
            z_error = goals[:, 2] - pos[:, 2]

            att_err_ff, att_err_pd = self._compute_attitude_errors(
                state, goals, ref_vel, ref_acc
            )

            body_force_norm = np.linalg.norm(body_force, axis=1)
            body_torque_norm = np.linalg.norm(body_torque, axis=1)

            for env_id in range(self.n_envs):
                row = {
                    "controller": controller_name,
                    "env_id": int(env_id),
                    "episode_id": int(self.episode_id[env_id]),
                    "episode_step": int(self.episode_step[env_id]),
                    "global_step": int(self.global_step),
                    "t": float(self.episode_step[env_id] * self.step_dt),

                    "x": float(pos[env_id, 0]),
                    "y": float(pos[env_id, 1]),
                    "z": float(pos[env_id, 2]),

                    "vx": float(vel[env_id, 0]),
                    "vy": float(vel[env_id, 1]),
                    "vz": float(vel[env_id, 2]),
                    "speed": float(speed[env_id]),

                    "goal_x": float(goals[env_id, 0]),
                    "goal_y": float(goals[env_id, 1]),
                    "goal_z": float(goals[env_id, 2]),
                    "ref_vx": float(ref_vel[env_id, 0]),
                    "ref_vy": float(ref_vel[env_id, 1]),
                    "ref_vz": float(ref_vel[env_id, 2]),
                    "pos_error": float(pos_error[env_id]),
                    "z_error": float(z_error[env_id]),

                    "att_err_ff": float(att_err_ff[env_id]),
                    "att_err_pd": float(att_err_pd[env_id]),

                    "fx_body": float(body_force[env_id, 0]),
                    "fy_body": float(body_force[env_id, 1]),
                    "fz_body": float(body_force[env_id, 2]),
                    "body_force_norm": float(body_force_norm[env_id]),

                    "tx_body": float(body_torque[env_id, 0]),
                    "ty_body": float(body_torque[env_id, 1]),
                    "tz_body": float(body_torque[env_id, 2]),
                    "body_torque_norm": float(body_torque_norm[env_id]),
                }

                for i in range(self.n_rotors):
                    row[f"cmd_w_{i}"]          = float(cmd_w[env_id, i])
                    row[f"cmd_rpm_{i}"]        = float(cmd_rpm[env_id, i])
                    row[f"actual_w_{i}"]       = float(actual_w[env_id, i])
                    row[f"actual_rpm_{i}"]     = float(actual_rpm[env_id, i])
                    row[f"l2f_action_{i}"]     = float(l2f_action[env_id, i])
                    row[f"l2f_cmd_rpm_{i}"]    = float(l2f_cmd_rpm[env_id, i])
                    row[f"l2f_current_rpm_{i}"]= float(l2f_current_rpm[env_id, i])

                self.traj_writer.writerow(row)

                if self.tb is not None:
                    prefix = f"{controller_name}/env_{env_id}"
                    self.tb.add_scalar(f"{prefix}/z", row["z"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/speed", row["speed"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/pos_error", row["pos_error"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/cmd_rpm_0", row["cmd_rpm_0"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/actual_rpm_0", row["actual_rpm_0"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fx_body", row["fx_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fy_body", row["fy_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fz_body", row["fz_body"], self.global_step)

    def advance(self, terminated, truncated):
        self.global_step += 1
        self.episode_step += 1

        terminated = terminated.detach().view(-1).cpu().numpy().astype(bool)
        truncated = truncated.detach().view(-1).cpu().numpy().astype(bool)
        done = np.logical_or(terminated, truncated)

        for env_id in np.where(done)[0]:
            self.completed_episodes[env_id] += 1

            if terminated[env_id]:
                reason = "terminated"
            elif truncated[env_id]:
                reason = "truncated"
            else:
                reason = "done"

            self.ep_writer.writerow({
                "env_id": int(env_id),
                "episode_id": int(self.episode_id[env_id]),
                "end_global_step": int(self.global_step),
                "length_steps": int(self.episode_step[env_id]),
                "duration_s": float(self.episode_step[env_id] * self.step_dt),
                "reason": reason,
            })

            self.episode_id[env_id] += 1
            self.episode_step[env_id] = 0

        self.traj_file.flush()
        self.ep_file.flush()

        if self.tb is not None:
            self.tb.flush()

    def completed(self, target_episodes_per_env: int) -> bool:
        if target_episodes_per_env <= 0:
            return False
        return np.all(self.completed_episodes >= target_episodes_per_env)

    def close(self):
        self.traj_file.close()
        self.ep_file.close()

        if self.tb is not None:
            self.tb.close()


# ---------------------------------------------------------------------
# Isaac Sim imports
# ---------------------------------------------------------------------
from isaacsim import SimulationApp


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", default=False, action="store_true")
    p.add_argument("--episode_duration", type=float, default=10.0, help="Episode length in seconds")

    # ---- Global vehicle model. ALL instantiated controllers use this same
    #      airframe (any per-entry vehicle/cfg in the manifest is ignored). ----
    p.add_argument("--vehicle", choices=["iris", "crazyflie", "shuttle", "shuttle_glider"], default="iris",
                   help="Vehicle model used for ALL instantiated controllers.")

    # ---- Manifest-driven controllers (preferred). A JSON file describing the
    #      vehicles/controllers to instantiate. When provided, it overrides the
    #      individual --raptor/--crazyflie/--*_checkpoint flags below. ----
    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON manifest of vehicles/controllers to spawn "
                        "(overrides the individual controller flags).")

    # ---- Controller selection (enable any subset; at least one is required) ----
    p.add_argument("--raptor", action="store_true",
                   help="Add a RAPTOR foundation-policy controlled Iris.")
    p.add_argument("--crazyflie", action="store_true",
                   help="Also add a RAPTOR-controlled Crazyflie (requires --raptor).")

    p.add_argument("--goal_xy_range", type=float, nargs=2, default=None,
                   metavar=("LOW", "HIGH"), help="XY goal randomization range around spawn [m]. If omitted, goal = spawn position.")
    p.add_argument("--goal_z_range",  type=float, nargs=2, default=None,
                   metavar=("LOW", "HIGH"), help="Z goal randomization range [m].")

    # Trajectory tracking
    p.add_argument("--trajectory", choices=["none", "lemniscate"], default="none",
                   help="Reference trajectory type. 'none' = static goal.")
    p.add_argument("--traj_amplitude", type=float, default=1.5,
                   help="Lemniscate half-width [m] (default 1.5 m, as in RAPTOR paper).")
    p.add_argument("--traj_period", type=float, default=8.0,
                   help="Time for one full lemniscate loop [s] (default 8 s).")
    p.add_argument("--traj_z", type=float, default=1.5,
                   help="Reference altitude for trajectory [m] (default 1.5 m).")

    p.add_argument("--sac_checkpoint", type=str, default=None,
                   help="Path to a skrl SAC checkpoint (.pt) to evaluate. If set, adds a SAC-controlled Iris vehicle.")
    p.add_argument("--sac_vehicle", choices=["Iris_White", "Crazyflie_L2F"], default="Iris_White",
                   help="Vehicle USD to use for the SAC controller (default: Iris_White).")

    p.add_argument("--pre_checkpoint", type=str, default=None,
                   help="Path to a RAPTOR pre-training HDF5 checkpoint to evaluate. If set, adds a pre-train-controlled Iris vehicle.")

    p.add_argument("--record", default=False, action="store_true")
    p.add_argument("--record_dir", default="play_records")
    p.add_argument("--record_every", type=int, default=1)
    p.add_argument("--num_episodes_per_env", type=int, default=0)

    return p.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})


# ---------------------------------------------------------------------
# Post-SimulationApp imports
# ---------------------------------------------------------------------
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.transforms import quaternion_to_matrix
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import ResetManager, GoalCfg


def load_raptor_backend_class():
    path = RL_DIR / "utils" / "foundation_policy_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"RaptorBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("foundation_policy_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "RaptorBackend"):
        raise AttributeError(f"RaptorBackend not found in {path}")

    return module.RaptorBackend


def load_sac_backend_class():
    path = RL_DIR / "utils" / "sac_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"SACBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("sac_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module.SACBackend


def load_pretrain_backend_class():
    path = RL_DIR / "utils" / "foundation_policy_pre_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"PreTrainBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("foundation_policy_pre_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module.PreTrainBackend


def load_skrl_backend_class():
    path = RL_DIR / "utils" / "skrl_agent_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"SkrlAgentBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("skrl_agent_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "SkrlAgentBackend"):
        raise AttributeError(f"SkrlAgentBackend not found in {path}")

    return module.SkrlAgentBackend


def load_nonlinear_backend_class():
    path = RL_DIR / "utils" / "non_linear_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"NonLinearControllerBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("non_linear_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "NonLinearControllerBackend"):
        raise AttributeError(f"NonLinearControllerBackend not found in {path}")

    return module.NonLinearControllerBackend



def make_iris_cfg():
    # Iris real parameters (from PX4/Gazebo iris.sdf):
    #   kf = 8.54858e-6, moment_constant = 0.016
    #   km = moment_constant * kf = 1.368e-7
    # This gives cm = km/kf = 0.016, within RAPTOR training range [0.005, 0.05].
    # The Pegasus default km=1e-6 is incorrect and gives cm=0.117, far out of distribution.
    km = 0.016 * 8.54858e-6  # = 1.368e-7
    return {
        "num_rotors": 4,
        "rotor_constant": [8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6],
        "rolling_moment_coefficient": [km, km, km, km],
        "rot_dir": [-1, -1, 1, 1],
        "min_rotor_velocity": [0.0, 0.0, 0.0, 0.0],
        "max_rotor_velocity": [1100.0, 1100.0, 1100.0, 1100.0],
        #"motor_time_constant": [0.04, 0.04, 0.04, 0.04],
        "motor_time_constant": [0.00, 0.00, 0.00, 0.00],
    }


def make_crazyflie_cfg():
    # L2F Crazyflie parameters (learning_to_fly/parameters/dynamics/crazy_flie.h).
    return {
        "num_rotors": 4,
        "rotor_constant": [
            2.8815744627880866e-8,
            2.8815744627880866e-8,
            2.8815744627880866e-8,
            2.8815744627880866e-8,
        ],
        "rolling_moment_coefficient": [
            1.7187300725171608e-10,
            1.7187300725171608e-10,
            1.7187300725171608e-10,
            1.7187300725171608e-10,
        ],
        "rot_dir": [-1, 1, -1, 1],
        "min_rotor_velocity": [0.0, 0.0, 0.0, 0.0],
        "max_rotor_velocity": [
            2272.6281256068564,
            2272.6281256068564,
            2272.6281256068564,
            2272.6281256068564,
        ],
        "motor_time_constant": [0.15, 0.15, 0.15, 0.15],
    }

def make_shuttle_cfg():
    # Shuttle real parameters (from PX4/Gazebo shuttle.sdf):
    return {
        "num_rotors": 4,
        "rotor_constant": [1.709716e-05, 1.709716e-05, 1.709716e-05, 1.709716e-05],
        "rolling_moment_coefficient": [1e-06, 1e-06, 1e-06, 1e-06],
        "rot_dir": [-1, -1, 1, 1],
        "min_rotor_velocity": [0, 0, 0, 0],                             # rad/s
        "max_rotor_velocity": [1400, 1400, 1400, 1400],                 # rad/s
        "motor_time_constant": [0.008, 0.008, 0.008, 0.008],            # s
    }

def make_shuttle_glider_cfg():
    # Shuttle real parameters (from PX4/Gazebo shuttle.sdf):
    return {
    "num_rotors": 5,
    "rotor_constant":             [1.709716e-05, 1.709716e-05, 1.709716e-05, 1.709716e-05, 8.54858e-06],
    "rolling_moment_coefficient": [1e-06, 1e-06, 1e-06, 1e-06, 0.0],
    "rot_dir":                    [-1, -1, 1, 1, 1],
    "min_rotor_velocity":         [0, 0, 0, 0, 0],
    "max_rotor_velocity":         [1400, 1400, 1400, 1400, 3500],
    "motor_time_constant":        [0.008, 0.008, 0.008, 0.008, 0.0125],
}


# ---------------------------------------------------------------------
# Controller registry (manifest-driven instantiation)
# ---------------------------------------------------------------------
# Physical-vehicle key -> function returning its MultirotorBatch parameter dict.
# Add new airframes here.
VEHICLE_CFGS = {
    "iris": make_iris_cfg,
    "crazyflie": make_crazyflie_cfg,
    "shuttle": make_shuttle_cfg,
    "shuttle_glider": make_shuttle_glider_cfg,
}

# Physical-vehicle key -> USD asset key in ROBOTS. The same model is used for
# every controller so that all instantiated vehicles are identical.
VEHICLE_USD = {
    "iris": "Iris",
    "crazyflie": "Crazyflie_L2F",
    "shuttle": "Shuttle",
    "shuttle_glider": "Shuttle_glider",
}


def _build_raptor_backend(entry, n_envs, device, physics_dt):
    RaptorBackend = load_raptor_backend_class()
    return RaptorBackend(
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity"),
    )


def _build_pretrain_backend(entry, n_envs, device, physics_dt):
    PreTrainBackend = load_pretrain_backend_class()
    if not entry.get("checkpoint"):
        raise SystemExit(f"[config] controller '{entry.get('name')}' (pretrain) requires a 'checkpoint'.")
    return PreTrainBackend(
        checkpoint_path=entry["checkpoint"],
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity"),
        omega_min=entry.get("omega_min", 0.0),
        omega_max=entry.get("omega_max", 1100.0),
        motor_tau_rising=entry.get("motor_tau_rising", 0.04),
        motor_tau_falling=entry.get("motor_tau_falling", 0.04),
        dt=physics_dt,
    )


def _build_sac_backend(entry, n_envs, device, physics_dt):
    SACBackend = load_sac_backend_class()
    if not entry.get("checkpoint"):
        raise SystemExit(f"[config] controller '{entry.get('name')}' (sac) requires a 'checkpoint'.")
    return SACBackend(
        checkpoint_path=entry["checkpoint"],
        obs_dim=entry.get("obs_dim", 26),
        act_dim=entry.get("act_dim", 4),
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity_direct"),
        device=device,
    )


def _build_nonlinear_backend(entry, n_envs, device, physics_dt):
    NonLinearControllerBackend = load_nonlinear_backend_class()
    kwargs = dict(
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity"),
    )
    # Optional geometric-controller gains / dynamics overrides from the manifest.
    for key in ("Kp", "Kd", "Ki", "Kr", "Kw", "mass", "gravity"):
        if key in entry:
            kwargs[key] = entry[key]
    return NonLinearControllerBackend(**kwargs)


def _make_skrl_builder(algo):
    """Return a manifest builder for a skrl agent controller (PPO/SAC).

    The controller is run EXACTLY like play.py: the agent config is loaded from
    tasks/<task>/agents/<algo>_cfg.py, the models + agent are built, the
    checkpoint is restored with agent.load() (which also restores any state
    preprocessor, e.g. PPO's RunningStandardScaler), and inference uses
    agent.act() in eval mode. Each entry must provide 'checkpoint' and 'task';
    'preset' defaults to 'isaac_lab'.
    """
    def _build(entry, n_envs, device, physics_dt):
        SkrlAgentBackend = load_skrl_backend_class()
        if not entry.get("checkpoint"):
            raise SystemExit(f"[config] controller '{entry.get('name')}' ({algo}/skrl) requires a 'checkpoint'.")
        if not entry.get("task"):
            raise SystemExit(
                f"[config] controller '{entry.get('name')}' ({algo}/skrl) requires a 'task' "
                f"(skrl agent config is loaded from tasks/<task>/agents/{algo}_cfg.py, like play.py)."
            )
        return SkrlAgentBackend(
            checkpoint_path=entry["checkpoint"],
            task=entry["task"],
            algo=algo,
            preset=entry.get("preset", "isaac_lab"),
            obs_dim=entry.get("obs_dim", 26),
            act_dim=entry.get("act_dim", 4),
            n_vehicles=n_envs,
            reset_manager=None,
            action_mode=entry.get("action_mode", "rotor_velocity_direct"),
            device=device,
        )
    return _build


# Controller type -> backend builder. Single source of truth; add new types here.
#   "raptor"   : RAPTOR foundation policy           (utils/foundation_policy_controller.py)
#   "pretrain" : RAPTOR pre-training foundation pol. (utils/foundation_policy_pre_controller.py)
#   "sac"      : standalone hand-rolled SAC actor    (utils/sac_controller.py)
#   "ppo"      : full skrl PPO agent, like play.py    (utils/skrl_agent_controller.py)
#   "sac_skrl" : full skrl SAC agent, like play.py    (utils/skrl_agent_controller.py)
#   "nonlinear": geometric Mellinger-Kumar controller (utils/non_linear_controller.py)
CONTROLLER_REGISTRY = {
    "raptor":    _build_raptor_backend,
    "pretrain":  _build_pretrain_backend,
    "sac":       _build_sac_backend,
    "ppo":       _make_skrl_builder("ppo"),
    "sac_skrl":  _make_skrl_builder("sac"),
    "nonlinear": _build_nonlinear_backend,
}


def load_manifest(path):
    """Load and validate a JSON manifest of controllers.

    Accepts either {"vehicles": [ ... ]} or a bare list [ ... ]. An optional
    top-level "vehicle" (iris|crazyflie) selects the airframe used for ALL
    controllers; if absent, the --vehicle CLI flag is used.

    Each entry: name (unique), type (raptor|pretrain|sac|ppo|sac_skrl), checkpoint
    (path or null), plus optional per-controller overrides consumed by the
    builders. The skrl types (ppo, sac_skrl) additionally require 'task' and
    accept an optional 'preset' (default 'isaac_lab'), mirroring play.py.
    Any per-entry vehicle/cfg field is ignored: every vehicle is instantiated
    identically from the single global vehicle choice.

    Returns (vehicles, global_vehicle); global_vehicle may be None.
    """
    with open(path, "r") as f:
        manifest = json.load(f)

    if isinstance(manifest, dict):
        vehicles = manifest.get("vehicles")
        global_vehicle = manifest.get("vehicle")
    else:
        vehicles = manifest
        global_vehicle = None

    if not vehicles:
        raise SystemExit(f"[config] manifest '{path}' has no 'vehicles' entries.")
    if global_vehicle is not None and global_vehicle not in VEHICLE_CFGS:
        raise SystemExit(f"[config] manifest 'vehicle' is '{global_vehicle}'; "
                         f"must be one of {sorted(VEHICLE_CFGS)}.")

    seen = set()
    for i, entry in enumerate(vehicles):
        name = entry.get("name")
        ctype = entry.get("type")
        if not name:
            raise SystemExit(f"[config] vehicle #{i} is missing 'name'.")
        if name in seen:
            raise SystemExit(f"[config] duplicate controller name '{name}'.")
        seen.add(name)
        if ctype not in CONTROLLER_REGISTRY:
            raise SystemExit(f"[config] controller '{name}' has unknown type '{ctype}'. "
                             f"Known types: {sorted(CONTROLLER_REGISTRY)}.")
    return vehicles, global_vehicle


def main():
    device = args.device
    n_envs = args.n_envs
    physics_dt = 0.01  # 100 Hz
    episode_steps = int(args.episode_duration / physics_dt)

    # Validate controller selection.
    manifest_vehicles, manifest_vehicle_choice = (
        load_manifest(args.config) if args.config else (None, None))
    if manifest_vehicles is None:
        if not (args.raptor or args.pre_checkpoint or args.sac_checkpoint):
            raise SystemExit("Select at least one controller: --config, --raptor, --pre_checkpoint and/or --sac_checkpoint.")
        if args.crazyflie and not args.raptor:
            raise SystemExit("--crazyflie requires --raptor.")

    # Global vehicle model: every controller is instantiated with this same
    # airframe. A manifest top-level "vehicle" wins over the --vehicle CLI flag.
    vehicle_choice = (manifest_vehicle_choice or args.vehicle)
    if vehicle_choice not in VEHICLE_USD:
        raise SystemExit(f"[config] unknown vehicle '{vehicle_choice}'; "
                         f"must be one of {sorted(VEHICLE_USD)}.")
    usd_key = VEHICLE_USD[vehicle_choice]
    cfg_fn  = VEHICLE_CFGS[vehicle_choice]
    print(f"[config] vehicle model for ALL controllers: {vehicle_choice} "
          f"(USD '{usd_key}')", flush=True)

    pg = PegasusInterface()
    pg.set_world_settings(physics_dt=physics_dt, rendering_dt=physics_dt, device=device)
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    if "cuda" in device:
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # Load only the backend classes needed for the selected controllers.
    RaptorBackend   = load_raptor_backend_class()    if args.raptor         else None
    PreTrainBackend = load_pretrain_backend_class()  if args.pre_checkpoint else None
    SACBackend      = load_sac_backend_class()        if args.sac_checkpoint else None

    # Each controller is an independent vehicle batch sharing the same world.
    # The first vehicle created defines the spawn grid; the rest are aligned to it
    # so that every controller starts from identical initial conditions.
    controllers = {}        # name -> vehicle (insertion order preserved)
    backends_order = []     # backends in creation order
    _anchor = {"vehicle": None}
    _next_id = {"value": 1}

    def _spawn(name, usd_key, backend, cfg_fn):
        """Instantiate one controller's vehicle batch and register it."""
        cfg = MultirotorBatchConfig(cfg=cfg_fn(), n_vehicles=n_envs)
        cfg.backends = [backend]
        kwargs = dict(
            stage_prefix=f"/World/{name}",
            usd_file=ROBOTS[usd_key],
            vehicle_batch_id=_next_id["value"],
            n_vehicles=n_envs,
            config=cfg,
        )
        if _anchor["vehicle"] is None:
            kwargs["spacing"] = 2.5
        else:
            kwargs["init_pos"] = _anchor["vehicle"]._init_pos
            kwargs["init_orientation"] = _anchor["vehicle"]._init_orientation
        vehicle = MultirotorBatch(**kwargs)
        # Disable collisions for this vehicle batch: the controllers are spawned
        # overlapping in space for comparison, so PhysX contacts between them (or
        # with the ground) must not perturb the recorded trajectories.
        vehicle.disable_collisions()
        if _anchor["vehicle"] is None:
            _anchor["vehicle"] = vehicle
        _next_id["value"] += 1
        controllers[name] = vehicle
        backends_order.append(backend)
        return vehicle

    if manifest_vehicles is not None:
        # Manifest-driven: instantiate exactly the controllers described in JSON,
        # all using the same global vehicle model.
        for entry in manifest_vehicles:
            backend = CONTROLLER_REGISTRY[entry["type"]](entry, n_envs, device, physics_dt)
            _spawn(entry["name"], usd_key, backend, cfg_fn)
            print(f"[config] spawned '{entry['name']}' (type={entry['type']}, "
                  f"vehicle={vehicle_choice}, checkpoint={entry.get('checkpoint')})", flush=True)
    else:
        if args.raptor:
            _spawn("raptor_iris", usd_key,
                   RaptorBackend(n_vehicles=n_envs, reset_manager=None, action_mode="rotor_velocity"),
                   cfg_fn)
            if args.crazyflie:
                _spawn("raptor_cf", usd_key,
                       RaptorBackend(n_vehicles=n_envs, reset_manager=None, action_mode="rotor_velocity"),
                       cfg_fn)

        if args.pre_checkpoint:
            _spawn("pretrain_iris", usd_key,
                   PreTrainBackend(
                       checkpoint_path=args.pre_checkpoint,
                       n_vehicles=n_envs,
                       reset_manager=None,
                       action_mode="rotor_velocity",
                       omega_min=0.0,
                       omega_max=1100.0,
                       motor_tau_rising=0.04,
                       motor_tau_falling=0.04,
                       dt=physics_dt,
                   ),
                   cfg_fn)

        if args.sac_checkpoint:
            _spawn("sac_iris", usd_key,
                   SACBackend(
                       checkpoint_path=args.sac_checkpoint,
                       obs_dim=26,
                       act_dim=4,
                       n_vehicles=n_envs,
                       reset_manager=None,
                       action_mode="rotor_velocity_direct",
                       device=device,
                   ),
                   cfg_fn)

    # -----------------------------------------------------------------
    # Start simulation
    # -----------------------------------------------------------------
    print("\n[Sim] Calling world.reset()...", flush=True)
    world.reset()
    print("[Sim] world.reset() done.", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    goal_cfg = None
    if args.goal_xy_range is not None or args.goal_z_range is not None:
        goal_cfg = GoalCfg(
            goal_pos_xy_range=args.goal_xy_range if args.goal_xy_range is not None else [-2.0, 2.0],
            goal_pos_z_range=args.goal_z_range   if args.goal_z_range  is not None else [0.5, 1.5],
        )

    vehicles_list = list(controllers.values())

    reset_manager = ResetManager(
        vehicles=vehicles_list,
        device=device,
        goal_cfg=goal_cfg,
    )

    for backend in backends_order:
        backend.setup(reset_manager)

    if goal_cfg is not None:
        reset_manager.set_goal_cfg(goal_cfg)
        print(
            f"[Sim] Re-applied CLI goal_cfg: "
            f"xy={goal_cfg.goal_pos_xy_range}, z={goal_cfg.goal_pos_z_range}",
            flush=True,
        )

    # The goal marker is only provided by the RAPTOR backend; use the first backend
    # that supports it (if any) and keep a handle to update it during the loop.
    marker_backend = next((b for b in backends_order if hasattr(b, "create_goal_marker")), None)

    # -----------------------------------------------------------------
    # Trajectory (lemniscate or static goal)
    # -----------------------------------------------------------------
    trajectory = None
    use_lemniscate = (args.trajectory == "lemniscate")

    if use_lemniscate:
        # Center each env on its spawn position; override Z to traj_z.
        center = _anchor["vehicle"]._init_pos.clone().to(device=device, dtype=torch.float32)
        trajectory = LemniscateTrajectory(
            n_envs=n_envs,
            amplitude=args.traj_amplitude,
            period=args.traj_period,
            z_ref=args.traj_z,
            center=center,
            device=device,
        )
        print(
            f"[Sim] Lemniscate trajectory: A={args.traj_amplitude} m, "
            f"T={args.traj_period} s, z={args.traj_z} m",
            flush=True,
        )
    else:
        print("[Sim] Static goal mode.", flush=True)

    # Goal marker — one cube at the reference position, shared across vehicles.
    if marker_backend is not None:
        marker_backend.create_goal_marker(root_path="/World/GoalMarker", size=0.15, color=(1.0, 0.0, 0.0))

    def _update_goal(traj_obj):
        """Push the current reference position into reset_manager and the goal marker."""
        pos, vel, acc = traj_obj.at_time(traj_obj.current_time)
        reset_manager._goal_pos[:] = pos
        reset_manager._goal_vel[:] = vel
        reset_manager._goal_acc[:] = acc

    def _draw_lemniscate_viewport(traj_obj):
        """Draw the lemniscate curve(s) in the Isaac Sim viewport using debug lines.
        Called once after world.reset(); headless mode skips this silently.
        Color matches the matplotlib/Plotly default blue used in the paper plots.
        """
        if traj_obj is None or args.headless:
            return

        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()

        # Matplotlib default blue: #1f77b4  -> (31, 119, 180)
        color = (31 / 255, 119 / 255, 180 / 255, 1.0)
        # Parametrise directly over phase angle: wt ∈ [0, 2π] closes the curve
        # regardless of the chosen period T.
        n_samples = 200
        wt = torch.linspace(0.0, 2.0 * math.pi, n_samples + 1, device=device)  # +1 so last == first

        # Use env 0 centre
        cx = float(traj_obj.center[0, 0])
        cy = float(traj_obj.center[0, 1])
        cz = float(traj_obj.z_ref)

        xs = cx + traj_obj.A * torch.sin(wt)
        ys = cy + 0.5 * traj_obj.A * torch.sin(2.0 * wt)
        pts = [(float(xs[i]), float(ys[i]), cz) for i in range(n_samples + 1)]

        draw.clear_lines()
        draw.draw_lines_spline(pts, color, 5, False)
        print("[draw] Lemniscate drawn in viewport.", flush=True)

    # -----------------------------------------------------------------
    # Recorder
    # -----------------------------------------------------------------
    recorder = None

    if args.record:
        record_vehicles = dict(controllers)

        n_rotors = int(cfg_fn().get("num_rotors", 4))

        recorder = EpisodeTrajectoryRecorder(
            vehicles=record_vehicles,
            n_envs=n_envs,
            step_dt=physics_dt,
            out_dir=args.record_dir,
            record_every=args.record_every,
            reset_manager=reset_manager,
            trajectory=trajectory,
            n_rotors=n_rotors,
            use_tensorboard=False,
        )

    # -----------------------------------------------------------------
    # Physics loop
    # -----------------------------------------------------------------
    step = 0

    reset_manager.reset_all()
    _draw_lemniscate_viewport(trajectory)
    

    if trajectory is not None:
        _update_goal(trajectory)

    if marker_backend is not None:
        marker_backend.update_goal_marker(reset_manager.goal_pos)

    try:
        while simulation_app.is_running():
            if recorder is not None:
                recorder.sample()

            # --- Sincronizar o marcador com o frame que world.step vai RENDERIZAR ---
            # world.step integra o veículo até t+dt e desenha-o nessa posição.
            # O marcador, porém, só era avançado DEPOIS do passo (mais abaixo),
            # pelo que ficava sempre um passo de física atrasado e o drone parecia
            # ir à frente da caixa vermelha. Aqui pré-colocamos o marcador na
            # referência de t+dt, alinhado com o veículo pós-passo.
            # NB: mexe só no prim visual; reset_manager.goal_pos (controlador +
            # recorder) continua a ser avançado apenas após o world.step.
            if marker_backend is not None:
                if trajectory is not None:
                    next_pos, _, _ = trajectory.at_time(trajectory.current_time + physics_dt)
                    marker_backend.update_goal_marker(next_pos)
                else:
                    marker_backend.update_goal_marker(reset_manager.goal_pos)

            world.step(render=not args.headless)
            step += 1

            # Avança o tempo da trajetória e publica a nova referência que o
            # controlador e o recorder vão usar na PRÓXIMA iteração.
            if trajectory is not None:
                trajectory.step(physics_dt)
                _update_goal(trajectory)

            if step % episode_steps == 0:
                reset_manager.reset_all()
                for backend in backends_order:
                    backend.reset()

                # Reset trajectory phase for all envs on episode end
                if trajectory is not None:
                    all_ids = torch.arange(n_envs, device=device)
                    trajectory.reset(all_ids)
                    _update_goal(trajectory)

                if marker_backend is not None:
                    marker_backend.update_goal_marker(reset_manager.goal_pos)

                if recorder is not None:
                    done = torch.ones(n_envs, dtype=torch.bool)
                    recorder.advance(done, ~done)

                    if recorder.completed(args.num_episodes_per_env):
                        print(f"Finished {args.num_episodes_per_env} episodes per env. Exiting.", flush=True)
                        break

            elif recorder is not None:
                recorder.advance(torch.zeros(n_envs, dtype=torch.bool), torch.zeros(n_envs, dtype=torch.bool))

    finally:
        if recorder is not None:
            recorder.close()

        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()