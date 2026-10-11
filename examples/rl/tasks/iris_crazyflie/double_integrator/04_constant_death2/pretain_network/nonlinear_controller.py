#!/usr/bin/env python
"""
| File: nonlinear_controller.py
| Description: Batched translational teacher for the ep, ev -> acceleration task.

This is the translational outer loop extracted from the nonlinear quadrotor
controller. The current RL task does not expose attitude to the policy and does
not ask the policy to output thrust/torques, so only the translational law is
identifiable and relevant here.
"""

from __future__ import annotations

from typing import Sequence

import torch


class NonlinearControllerTeacher:
    """
    Teacher law for the current environment.

    Environment errors:
        ep = p_ref - p
        ev = v_ref - v

    Controller:
        a_raw = (Kp * ep + Kd * ev) / m

    Executable teacher action:
        a_cmd = clamp(a_raw, -max_acceleration, max_acceleration)

    With max_acceleration = 1.0, a_cmd is numerically identical to the action
    expected by the current environment.
    """

    def __init__(
        self,
        mass: float = 1.5,
        Kp: Sequence[float] = (10.0, 10.0, 10.0),
        Kd: Sequence[float] = (8.5, 8.5, 8.5),
        max_acceleration: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.m = float(mass)
        self.max_acceleration = float(max_acceleration)

        if self.m <= 0.0:
            raise ValueError("mass must be > 0")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be > 0")

        self.Kp = torch.diag(torch.tensor(Kp, dtype=torch.float32, device=self.device))
        self.Kd = torch.diag(torch.tensor(Kd, dtype=torch.float32, device=self.device))

    def compute_raw(self, ep: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
        """Return the unsaturated desired inertial acceleration [m/s^2]."""
        ep = ep.to(device=self.device, dtype=torch.float32)
        ev = ev.to(device=self.device, dtype=torch.float32)

        if ep.ndim != 2 or ep.shape[1] != 3:
            raise ValueError(f"ep must have shape (N, 3), got {tuple(ep.shape)}")
        if ev.shape != ep.shape:
            raise ValueError(f"ev must have shape {tuple(ep.shape)}, got {tuple(ev.shape)}")

        return (ep @ self.Kp.T + ev @ self.Kd.T) / self.m

    def compute(self, ep: torch.Tensor, ev: torch.Tensor):
        """
        Return:
            a_cmd:      executable/saturated action, shape (N, 3)
            a_raw:      unsaturated controller acceleration, shape (N, 3)
            saturated:  True if any axis required clipping, shape (N,)
        """
        a_raw = self.compute_raw(ep, ev)

        a_cmd = torch.clamp(
            a_raw,
            min=-self.max_acceleration,
            max=self.max_acceleration,
        )

        saturated = torch.any(
            torch.abs(a_raw) > self.max_acceleration,
            dim=1,
        )

        return a_cmd, a_raw, saturated

    def compute_from_observation(self, obs: torch.Tensor):
        """
        obs = [ep_x, ep_y, ep_z, ev_x, ev_y, ev_z]
        """
        obs = obs.to(device=self.device, dtype=torch.float32)

        if obs.ndim != 2 or obs.shape[1] != 6:
            raise ValueError(f"obs must have shape (N, 6), got {tuple(obs.shape)}")

        ep = obs[:, 0:3]
        ev = obs[:, 3:6]

        return self.compute(ep, ev)
