"""
| File: trajectory.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Track-frame reference generator for the Shuttle + EasyGlider.
|
| Design (see thesis page "Metodologia de treino do veiculo conjunto"):
|   - References are parameterised by an AIRSPEED schedule V0 -> V_target and a
|     horizontal heading psi, not by position noise. V and psi are what set
|     q_bar, alpha and beta, i.e. the physics that matters for the hybrid.
|   - Forward-biased: motion is always along +x of the (yaw psi) track frame.
|     No backwards flight and NO ping-pong replay (unlike the RAPTOR/Langevin
|     generator used for the pure quadrotor).
|   - C2-smooth: a cosine/smoothstep S-ramp blends V0 -> V_target so that
|     position, velocity and acceleration references are mutually consistent.
|   - Feasibility-bounded: the ramp duration is stretched so the peak along-track
|     acceleration never exceeds the level's thrust-limited a_max.
|   - Anchored to the reset state: p_ref(0) = centre and v_ref(0) = V0 * dir(psi),
|     so a "cruise start" (V0 = V_target) begins exactly on the trajectory.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class TrackReferenceGenerator:

    HOVER = 0
    STRAIGHT = 1
    TURN = 2

    def __init__(
        self,
        num_envs: int,
        episode_steps: int,
        dt: float,
        device: str,
        levels: tuple[Any, ...],
        ramp_duration: float = 2.5,
        a_max: float = 6.0,
    ):
        self.num_envs = int(num_envs)
        self.episode_steps = int(episode_steps)
        self.dt = float(dt)
        self.device = device
        self.levels = levels
        self.ramp_duration = max(float(ramp_duration), 1e-4)
        self.a_max = max(float(a_max), 1e-3)

        self.pos_traj = torch.zeros(self.num_envs, self.episode_steps, 3, device=self.device)
        self.vel_traj = torch.zeros_like(self.pos_traj)
        self.acc_traj = torch.zeros_like(self.pos_traj)

        self.env_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_modes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.speed0 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.speed_target = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._t = torch.arange(self.episode_steps, device=self.device, dtype=torch.float32) * self.dt

    # ------------------------------------------------------------------
    # Longitudinal airspeed profile: C2 blend V0 -> V_target over T_ramp.
    # ------------------------------------------------------------------
    def _speed_profile(self, speed0: torch.Tensor, speed_target: torch.Tensor):
        """Return along-track (s, v, a), each (n, episode_steps).

        v(t) = V0 + dV * S(tau),  S(tau) = 3 tau^2 - 2 tau^3,  tau = t/T in [0,1]
        The ramp duration T is per-env, stretched so peak |a| <= a_max.
        """
        t = self._t.unsqueeze(0)                                  # (1, K)
        V0 = speed0.unsqueeze(1)                                  # (n, 1)
        dV = (speed_target - speed0).unsqueeze(1)                 # (n, 1)

        # Peak accel of the smoothstep is 1.5 |dV| / T -> stretch T if needed.
        T_need = 1.5 * dV.abs() / self.a_max
        T = torch.clamp_min(T_need, self.ramp_duration)           # (n, 1)

        tau = torch.clamp(t / T, 0.0, 1.0)
        S = tau * tau * (3.0 - 2.0 * tau)                         # smoothstep
        dS = 6.0 * tau * (1.0 - tau)                              # dS/dtau
        v = V0 + dV * S
        a = torch.where(t < T, dV * dS / T, torch.zeros_like(S))

        # s(t) = V0 t + dV * T * (tau^3 - 0.5 tau^4)  for t <= T; linear after.
        s_ramp = V0 * t + dV * T * (tau ** 3 - 0.5 * tau ** 4)
        s_endramp = V0 * T + 0.5 * dV * T                         # s at tau=1
        s_after = s_endramp + speed_target.unsqueeze(1) * (t - T)
        s = torch.where(t < T, s_ramp, s_after)
        return s, v, a

    # ------------------------------------------------------------------
    def reset(
        self,
        env_ids: torch.Tensor,
        centers: torch.Tensor,
        levels: torch.Tensor,
        heading: torch.Tensor,
        speed0: torch.Tensor,
        speed_target: torch.Tensor,
        modes: torch.Tensor,
        turn_radius: torch.Tensor,
        turn_dir: torch.Tensor,
        vertical_amp: torch.Tensor,
        vertical_period: torch.Tensor,
    ):
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        centers = centers.to(device=self.device, dtype=torch.float32)

        self.env_levels[env_ids] = levels.to(dtype=torch.long, device=self.device)
        self.env_modes[env_ids] = modes.to(dtype=torch.long, device=self.device)
        self.env_heading[env_ids] = heading.to(dtype=torch.float32, device=self.device)
        self.speed0[env_ids] = speed0.to(dtype=torch.float32, device=self.device)
        self.speed_target[env_ids] = speed_target.to(dtype=torch.float32, device=self.device)

        c = centers[env_ids]                                     # (n, 3)
        psi = heading.to(self.device)
        cpsi = torch.cos(psi).unsqueeze(1)                       # (n,1)
        spsi = torch.sin(psi).unsqueeze(1)

        s, v, a = self._speed_profile(speed0.to(self.device), speed_target.to(self.device))

        # Default = hover at the centre.
        self.pos_traj[env_ids] = c.unsqueeze(1)
        self.vel_traj[env_ids] = 0.0
        self.acc_traj[env_ids] = 0.0

        straight = modes.to(self.device) == self.STRAIGHT
        turn = modes.to(self.device) == self.TURN

        # ---- vertical component (mild, both straight and turn) ----
        t = self._t.unsqueeze(0)
        amp = vertical_amp.to(self.device).unsqueeze(1)
        omega_z = (2.0 * math.pi / vertical_period.to(self.device).clamp_min(1e-3)).unsqueeze(1)
        z_off = 0.5 * amp * (1.0 - torch.cos(omega_z * t))
        vz = 0.5 * amp * omega_z * torch.sin(omega_z * t)
        az = 0.5 * amp * omega_z.square() * torch.cos(omega_z * t)

        # ---- STRAIGHT ----
        if straight.any():
            idx = straight.nonzero(as_tuple=False).squeeze(1)
            self.pos_traj[env_ids[idx], :, 0] = c[idx, 0:1] + s[idx] * cpsi[idx]
            self.pos_traj[env_ids[idx], :, 1] = c[idx, 1:2] + s[idx] * spsi[idx]
            self.pos_traj[env_ids[idx], :, 2] = c[idx, 2:3] + z_off[idx]
            self.vel_traj[env_ids[idx], :, 0] = v[idx] * cpsi[idx]
            self.vel_traj[env_ids[idx], :, 1] = v[idx] * spsi[idx]
            self.vel_traj[env_ids[idx], :, 2] = vz[idx]
            self.acc_traj[env_ids[idx], :, 0] = a[idx] * cpsi[idx]
            self.acc_traj[env_ids[idx], :, 1] = a[idx] * spsi[idx]
            self.acc_traj[env_ids[idx], :, 2] = az[idx]

        # ---- TURN: constant-radius arc, parameterised by arc length s(t) ----
        if turn.any():
            idx = turn.nonzero(as_tuple=False).squeeze(1)
            R = turn_radius.to(self.device)[idx].unsqueeze(1).clamp_min(1e-3)
            d = turn_dir.to(self.device)[idx].unsqueeze(1)
            theta = s[idx] / R
            th_dot = v[idx] / R
            th_ddot = a[idx] / R
            st, ct = torch.sin(theta), torch.cos(theta)

            # Arc tangent to the local +x (track) axis at t=0.
            x_l = R * st
            y_l = d * R * (1.0 - ct)
            vx_l = R * ct * th_dot
            vy_l = d * R * st * th_dot
            ax_l = R * (-st * th_dot.square() + ct * th_ddot)
            ay_l = d * R * (ct * th_dot.square() + st * th_ddot)

            # Rotate the whole arc from the track frame to the world by psi.
            cp, sp = cpsi[idx], spsi[idx]
            self.pos_traj[env_ids[idx], :, 0] = c[idx, 0:1] + x_l * cp - y_l * sp
            self.pos_traj[env_ids[idx], :, 1] = c[idx, 1:2] + x_l * sp + y_l * cp
            self.pos_traj[env_ids[idx], :, 2] = c[idx, 2:3] + z_off[idx]
            self.vel_traj[env_ids[idx], :, 0] = vx_l * cp - vy_l * sp
            self.vel_traj[env_ids[idx], :, 1] = vx_l * sp + vy_l * cp
            self.vel_traj[env_ids[idx], :, 2] = vz[idx]
            self.acc_traj[env_ids[idx], :, 0] = ax_l * cp - ay_l * sp
            self.acc_traj[env_ids[idx], :, 1] = ax_l * sp + ay_l * cp
            self.acc_traj[env_ids[idx], :, 2] = az[idx]

    def current(self, env_ids: torch.Tensor | None = None, step_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if step_ids is None:
            raise ValueError("TrackReferenceGenerator.current requires step_ids")
        step_ids = step_ids.to(dtype=torch.long, device=self.device).clamp(0, self.episode_steps - 1)
        return (
            self.pos_traj[env_ids, step_ids],
            self.vel_traj[env_ids, step_ids],
            self.acc_traj[env_ids, step_ids],
        )
