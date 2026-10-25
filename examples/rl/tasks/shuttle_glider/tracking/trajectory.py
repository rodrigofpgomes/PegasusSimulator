"""
| File: trajectory.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Segment reference of the Shuttle-Glider tracking task: one smooth speed change, one turn (curvature
| roll-in, hold and roll-out) and one vertical-speed pulse per episode, all optional.
|
| Every flight regime is a choice of these parameters (hover: zero speed; transition: 0 -> V or V -> 0; cruise; turn;
| climb/descent), so a single generator covers the whole envelope without a curriculum. Each change follows the C2
| smoothstep S(tau) = 3 tau^2 - 2 tau^3, whose derivative is zero at both ends, so the reference acceleration is
| continuous and zero at t = 0 (the trimmed reset is an equilibrium of the first reference sample).
|
|   horizontal speed  V(t)   = V0 + (V1 - V0) S((t - t_v) / T_v)
|   curvature         k(t)   = k [S((t - t_k) / T_k) - S((t - t_ko) / T_k)]      (k > 0: left turn)
|   heading           psi(t) = psi0 + int V k dt,  psi_dot = V k
|   vertical speed    v_z(t) = v_z [S((t - t_z) / T_z) - S((t - t_zo) / T_z)]
|   position          integral of the velocity (trapezoidal, on the control grid)
|
| The heading is defined at every speed: at rest it is psi0, the direction of the path that follows or precedes the
| hover, so the hover heading objective is the same as in forward flight.
"""

from __future__ import annotations

import torch


def smoothstep(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """C2 smoothstep S(tau) clamped to [0, 1] and its derivative dS/dtau."""
    tau = tau.clamp(0.0, 1.0)
    return tau * tau * (3.0 - 2.0 * tau), 6.0 * tau * (1.0 - tau)


class SegmentReference:
    """Per-episode position, velocity, acceleration, heading and heading rate on the control grid."""

    PARAMS = ("V0", "V1", "t_v", "T_v", "kappa", "t_k", "T_k", "t_ko", "vz", "t_z", "T_z", "t_zo", "psi0")

    def __init__(self, num_envs: int, episode_steps: int, dt: float, device: str):
        self.num_envs = num_envs
        self.steps = int(episode_steps)
        self.dt = float(dt)
        self.device = device
        self._t = torch.arange(self.steps, device=device, dtype=torch.float32) * self.dt
        shape = (num_envs, self.steps)
        self.pos = torch.zeros(shape + (3,), device=device)
        self.vel = torch.zeros_like(self.pos)
        self.acc = torch.zeros_like(self.pos)
        self.heading = torch.zeros(shape, device=device)
        self.heading_rate = torch.zeros(shape, device=device)

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor, p: dict[str, torch.Tensor]):
        """Build the reference of env_ids from the parameter tensors p[name] (one value per env, see PARAMS)."""
        if env_ids.numel() == 0:
            return
        col = lambda name: p[name].to(self.device, torch.float32).unsqueeze(1)
        t = self._t.unsqueeze(0)

        S_v, dS_v = smoothstep((t - col("t_v")) / col("T_v"))
        V = col("V0") + (col("V1") - col("V0")) * S_v
        V_dot = (col("V1") - col("V0")) * dS_v / col("T_v")

        S_in, dS_in = smoothstep((t - col("t_k")) / col("T_k"))
        S_out, dS_out = smoothstep((t - col("t_ko")) / col("T_k"))
        kappa = col("kappa") * (S_in - S_out)
        psi_dot = V * kappa
        # Trapezoidal heading integral from psi0
        psi = col("psi0") + torch.cat((torch.zeros_like(psi_dot[:, :1]),
                                       torch.cumsum(0.5 * (psi_dot[:, 1:] + psi_dot[:, :-1]) * self.dt, dim=1)), dim=1)

        S_z, dS_z = smoothstep((t - col("t_z")) / col("T_z"))
        S_zo, dS_zo = smoothstep((t - col("t_zo")) / col("T_z"))
        vz = col("vz") * (S_z - S_zo)
        az = col("vz") * (dS_z - dS_zo) / col("T_z")

        c, s = torch.cos(psi), torch.sin(psi)
        vel = torch.stack((V * c, V * s, vz), dim=-1)
        # d/dt (V [c, s]) = V_dot [c, s] + V psi_dot [-s, c]
        acc = torch.stack((V_dot * c - V * psi_dot * s, V_dot * s + V * psi_dot * c, az), dim=-1)
        disp = torch.cat((torch.zeros_like(vel[:, :1]),
                          torch.cumsum(0.5 * (vel[:, 1:] + vel[:, :-1]) * self.dt, dim=1)), dim=1)

        self.pos[env_ids] = centers[env_ids].unsqueeze(1) + disp
        self.vel[env_ids] = vel
        self.acc[env_ids] = acc
        self.heading[env_ids] = psi
        self.heading_rate[env_ids] = psi_dot

    def sample(self, env_ids: torch.Tensor, step_ids: torch.Tensor):
        """Reference (pos, vel, acc, heading, heading_rate) of env_ids at their current steps (clamped to the end)."""
        k = step_ids.clamp(0, self.steps - 1)
        return (self.pos[env_ids, k], self.vel[env_ids, k], self.acc[env_ids, k],
                self.heading[env_ids, k], self.heading_rate[env_ids, k])
