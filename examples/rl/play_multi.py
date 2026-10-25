#!/usr/bin/env python
"""
| File: play_multi.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: Evaluate controllers in one world under shared references (reference protocol).
|
| Every controller drives its own batch of the same vehicle (--vehicle or the
| manifest "vehicle"), from the same initial state, while one analytic reference
| (see "Reference trajectories") is fed to all of them. Use play.py instead for
| statistics over a task's own training distribution.
|
| Controllers come from a JSON manifest (--config, see evaluation_log.validate_manifest):
|   - skrl PPO/PPO2/SAC checkpoints with their task; "run_metadata" restores the
|     training configuration (action scaling, vehicle parameters, sim_dt/decimation)
|   - RAPTOR foundation / pre-training policies, LQR, geometric controller, PX4
| --raptor and --pre_checkpoint build a manifest without --config.
|
| Timing: one loop iteration advances one physics step of --physics-dt (by
| default the skrl tasks' sim_dt; all must agree). Each skrl backend computes a
| new action every task decimation steps and holds it in between, which is the
| PegasusEnv.step schedule used in training. Graphical mode renders that same
| physics step; headless mode advances it without rendering.
|
| Vehicles: shuttle (hover, static goals, lemniscate), shuttle_glider2 (hover
| and forward-flight references) and easyglider (forward-flight references
| only, starting in flight with --init_speed). --init_speed starts every vehicle
| at that speed along +x with the task's trim when it has one; shuttle_glider2 starts at rest
| use the task's hover trim (rotors spinning at the hover speeds), as in training.
|
| Episodes last --episode_duration; the whole comparison stops at the first
| failed vehicle (non-finite state or position error above --failure-distance).
| EvaluationLog records each interval and checks its physics callback count.
| Manifest entries can also attach PX4 SITL to skrl vehicles for
| estimated-state observations (one PX4 instance per px4.vehicle_id, estimate
| log px4_estimate_<name>.jsonl when there are several) or rate-and-thrust control.
| A px4_position entry evaluates PX4's own position controller in a separate
| one-vehicle batch beside the other controllers.
"""

# Standard library imports
import sys
import csv
import glob
import json
import math
from copy import deepcopy
import argparse
import importlib
import importlib.util
import re
import traceback
from collections import deque
from pathlib import Path

import torch
import numpy as np


# =============================================================================
# Paths
# =============================================================================
RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent.parent
PEGASUS_EXT_DIR = PROJECT_ROOT / "extensions" / "pegasus.simulator"

for p in [str(RL_DIR / "utils"), str(RL_DIR), str(PROJECT_ROOT), str(PEGASUS_EXT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation_log import AERO_PULLER_INDEX, FIXED_WING, SKRL_TYPES, VEHICLES, validate_manifest


# =============================================================================
# Reference trajectories
# =============================================================================
# All references are analytic in ENU world coordinates and start at the spawn
# point (center). Each one returns position, velocity and acceleration, so the
# observation of tracking tasks receives a mutually consistent reference.
#
#   none         static goal (hover)                     Shuttle, Shuttle_glider2
#   lemniscate   horizontal figure eight                 Shuttle, Shuttle_glider2 (low speed)
#   straight     level flight along +x                   Shuttle_glider2, EasyGlider
#   longitudinal straight flight with climb/descent      Shuttle_glider2, EasyGlider
#   turn         constant-radius level turn              Shuttle_glider2, EasyGlider
#   helix        constant-radius turn with climb rate    Shuttle_glider2, EasyGlider
#   mission      hover -> transition -> cruise -> turn -> climb/descent -> deceleration -> hover
#                (evaluation of a complete flight; starts at the task's hover trim)   Shuttle_glider2
#
# Forward references change speed with the C2 smoothstep of the shuttle_glider
# training generator (TrackReferenceGenerator): the ramp lasts ramp_time, or
# longer so that the peak along-track acceleration stays below a_max.

def smoothstep_ramp(t: torch.Tensor, v0: float, v1: float, ramp_time: float, a_max: float):
    """Return distance, speed and acceleration of a C2 speed change v0 -> v1.

    v(t) = v0 + (v1 - v0) S(t/T), S(u) = 3u^2 - 2u^3, with
    T = max(ramp_time, 1.5 |v1 - v0| / a_max), since the peak of dS/du is 1.5.
    Acceleration is zero at both ends, so a start at v0 = v1 is exactly steady.
    """
    dv = float(v1) - float(v0)
    T = max(float(ramp_time), 1.5 * abs(dv) / max(float(a_max), 1e-6), 1e-4)
    u = torch.clamp(t / T, 0.0, 1.0)
    v = v0 + dv * u * u * (3.0 - 2.0 * u)
    a = torch.where(t < T, dv * 6.0 * u * (1.0 - u) / T, torch.zeros_like(t))
    s = torch.where(t < T, v0 * t + dv * T * (u ** 3 - 0.5 * u ** 4), v0 * T + 0.5 * dv * T + v1 * (t - T))
    return s, v, a


class TimedTrajectory:
    """Base class for analytic position, velocity and acceleration references."""

    def __init__(self, n_envs: int, center: torch.Tensor, device: str):
        """Initialize independent clocks around the requested path center."""
        self.n_envs = int(n_envs)
        self.device = device
        self.center = center.to(device=device, dtype=torch.float32).clone()
        self._t = torch.zeros(self.n_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor):
        """Restart the trajectory clock for the selected environments."""
        self._t[env_ids] = 0.0

    def step(self, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance every trajectory clock by one physics step."""
        self._t += float(dt)
        return self._eval(self._t)

    def at_time(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the reference without changing its current clock."""
        return self._eval(t.to(device=self.device, dtype=torch.float32))

    def peak_acceleration(self, duration: float, dt: float) -> float:
        """Largest reference acceleration norm [m/s^2] over one episode."""
        peak = 0.0
        for t in torch.arange(0.0, float(duration) + 0.5 * dt, float(dt), device=self.device):
            _, _, acc = self.at_time(torch.full((self.n_envs,), float(t), device=self.device))
            peak = max(peak, float(torch.linalg.vector_norm(acc, dim=1).max()))
        return peak

    @property
    def current_time(self) -> torch.Tensor:
        """Return the elapsed physics time for each environment."""
        return self._t

    def _altitude_profile(self, t: torch.Tensor, z_target: float, z_ramp_tau: float):
        """Smooth second-order altitude ramp from spawn z to z_target.

        The profile is a critically damped unit-step response:
            h(t) = 1 - (1 + t/tau) exp(-t/tau)

        Therefore z(0) equals the spawn altitude and vz(0)=0, avoiding both an
        instantaneous vertical position error and a vertical-velocity step.
        """
        z0 = self.center[:, 2]
        dz = float(z_target) - z0

        if float(z_ramp_tau) <= 0.0:
            z = torch.full_like(t, float(z_target))
            vz = torch.zeros_like(t)
            az = torch.zeros_like(t)
            return z, vz, az

        tau = float(z_ramp_tau)

        u = t / tau
        exp_term = torch.exp(-u)
        h = 1.0 - (1.0 + u) * exp_term
        h_dot = (t / (tau * tau)) * exp_term
        h_ddot = ((1.0 / (tau * tau)) - (t / (tau * tau * tau))) * exp_term
        z = z0 + dz * h
        vz = dz * h_dot
        az = dz * h_ddot
        return z, vz, az

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return position, velocity and acceleration at the requested time."""
        raise NotImplementedError


class LemniscateTrajectory(TimedTrajectory):
    """Lemniscate in the horizontal plane (xy).

    Peak accelerations are A w^2 along x and 2 A w^2 along y (w = 2 pi / period).
    With several environments, each one starts at a different phase, so its
    initial reference differs from the spawn point.
    """

    def __init__(self, n_envs: int, amplitude: float, period: float, z_ref: float,
                 z_ramp_tau: float, center: torch.Tensor, device: str, phase_offset_per_env: bool = True,
                 speed_ramp_s: float = 0.0):
        """Set the figure-eight dimensions, optional per-environment phases and optional speed ramp.

        With speed_ramp_s > 0 the angular speed rises from rest to w as 0.5 w (1 - cos(pi t / r)), like the
        evaluation reference of the direct_rotor tasks (lem_ramp_s), so the reference starts at rest.
        """
        super().__init__(n_envs, center, device)
        self.speed_ramp_s = float(speed_ramp_s)
        self.A = float(amplitude)
        self.w = 2.0 * math.pi / float(period)
        self.z_ref = float(z_ref)
        self.z_ramp_tau = max(float(z_ramp_tau), 1e-4)

        self._phase_time_offset = torch.zeros(self.n_envs, dtype=torch.float32, device=device)
        if phase_offset_per_env and self.n_envs > 1:
            phases = torch.linspace(0.0, 2.0 * math.pi, self.n_envs + 1, device=device)[:-1]
            self._phase_time_offset = phases / self.w

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the horizontal figure eight and smooth altitude ramp."""
        if self.speed_ramp_s > 0.0:
            r, w = self.speed_ramp_s, self.w
            ramp = t < r
            phase = torch.where(ramp, w * (t / 2 - r / (2 * math.pi) * torch.sin(math.pi * t / r)), w * (t - r / 2))
            sd = torch.where(ramp, 0.5 * w * (1 - torch.cos(math.pi * t / r)), torch.full_like(t, w))
            sdd = torch.where(ramp, 0.5 * w * (math.pi / r) * torch.sin(math.pi * t / r), torch.zeros_like(t))
            wt = phase + self.w * self._phase_time_offset
        else:
            sd, sdd = torch.full_like(t, self.w), torch.zeros_like(t)
            wt = self.w * (t + self._phase_time_offset)
        sin_wt = torch.sin(wt)
        cos_wt = torch.cos(wt)
        sin_2wt = torch.sin(2.0 * wt)
        cos_2wt = torch.cos(2.0 * wt)

        pos = self.center.clone()
        pos[:, 0] += self.A * sin_wt
        pos[:, 1] += 0.5 * self.A * sin_2wt

        vel = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        vel[:, 0] = self.A * sd * cos_wt
        vel[:, 1] = self.A * sd * cos_2wt

        acc = torch.zeros_like(vel)
        acc[:, 0] = self.A * (-sin_wt * sd**2 + cos_wt * sdd)
        acc[:, 1] = self.A * (-2.0 * sin_2wt * sd**2 + cos_2wt * sdd)

        pos[:, 2], vel[:, 2], acc[:, 2] = self._altitude_profile(t, self.z_ref, self.z_ramp_tau)
        return pos, vel, acc


class ForwardFlightTrajectory(TimedTrajectory):
    """Straight level flight along world +x, from speed0 to speed."""

    def __init__(self, n_envs: int, speed: float, z_ref: float, ramp_time: float, a_max: float,
                 z_ramp_tau: float, center: torch.Tensor, device: str, speed0: float = 0.0):
        """Set the initial and target speeds, the speed ramp and the altitude ramp."""
        super().__init__(n_envs, center, device)
        self.speed = float(speed)
        self.speed0 = float(speed0)
        self.z_ref = float(z_ref)
        self.ramp_time = float(ramp_time)
        self.a_max = float(a_max)
        self.z_ramp_tau = max(float(z_ramp_tau), 1e-4)

    def _forward_profile(self, t: torch.Tensor):
        """Return along-track distance, speed and acceleration."""
        return smoothstep_ramp(t, self.speed0, self.speed, self.ramp_time, self.a_max)

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate straight flight with speed and altitude ramps."""
        s, v, a = self._forward_profile(t)
        pos = self.center.clone()
        pos[:, 0] += s

        vel = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        vel[:, 0] = v

        acc = torch.zeros_like(vel)
        acc[:, 0] = a

        pos[:, 2], vel[:, 2], acc[:, 2] = self._altitude_profile(t, self.z_ref, self.z_ramp_tau)
        return pos, vel, acc


class LongitudinalFlightTrajectory(ForwardFlightTrajectory):
    """Forward flight with periodic climb/descent in the x-z plane.

    The vertical term 0.5 Az (1 - cos(wz t)) is the one used by the shuttle_glider
    training generator; its peak vertical acceleration is 0.5 Az wz^2.
    """

    def __init__(self, n_envs: int, speed: float, z_ref: float, z_amplitude: float, z_period: float,
                 ramp_time: float, a_max: float, z_ramp_tau: float, center: torch.Tensor, device: str,
                 speed0: float = 0.0):
        """Set the periodic vertical excursion on top of forward flight."""
        super().__init__(n_envs, speed, z_ref, ramp_time, a_max, z_ramp_tau, center, device, speed0)
        self.z_amplitude = float(z_amplitude)
        self.z_w = 2.0 * math.pi / float(z_period)

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add a periodic climb and descent to the straight flight reference."""
        pos, vel, acc = super()._eval(t)
        wt = self.z_w * t
        pos[:, 2] += 0.5 * self.z_amplitude * (1.0 - torch.cos(wt))
        vel[:, 2] += 0.5 * self.z_amplitude * self.z_w * torch.sin(wt)
        acc[:, 2] += 0.5 * self.z_amplitude * self.z_w**2 * torch.cos(wt)
        return pos, vel, acc


class LevelTurnTrajectory(ForwardFlightTrajectory):
    """Constant-radius horizontal turn, tangent to +x at the start.

    Steady centripetal acceleration is speed^2 / radius.
    """

    def __init__(self, n_envs: int, speed: float, radius: float, z_ref: float, ramp_time: float, a_max: float,
                 z_ramp_tau: float, center: torch.Tensor, device: str, direction: float = 1.0, speed0: float = 0.0):
        """Set the turn radius and direction around the starting point."""
        super().__init__(n_envs, speed, z_ref, ramp_time, a_max, z_ramp_tau, center, device, speed0)
        self.radius = max(abs(float(radius)), 1e-3)
        self.direction = 1.0 if float(direction) >= 0.0 else -1.0

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a horizontal turn with tangential and radial acceleration."""
        s, v, a_t = self._forward_profile(t)
        theta = s / self.radius
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        pos = self.center.clone()
        pos[:, 0] += self.radius * sin_th
        pos[:, 1] += self.direction * self.radius * (1.0 - cos_th)

        vel = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        vel[:, 0] = v * cos_th
        vel[:, 1] = self.direction * v * sin_th

        centripetal = v.square() / self.radius
        acc = torch.zeros_like(vel)
        acc[:, 0] = a_t * cos_th - centripetal * sin_th
        acc[:, 1] = self.direction * (a_t * sin_th + centripetal * cos_th)

        pos[:, 2], vel[:, 2], acc[:, 2] = self._altitude_profile(t, self.z_ref, self.z_ramp_tau)
        return pos, vel, acc


class HelicalTurnTrajectory(LevelTurnTrajectory):
    """Constant-radius turn plus a climb rate reached with the same smooth ramp."""

    def __init__(self, n_envs: int, speed: float, radius: float, z_ref: float, climb_rate: float,
                 ramp_time: float, a_max: float, z_ramp_tau: float, center: torch.Tensor, device: str,
                 direction: float = 1.0, speed0: float = 0.0):
        """Set the climb rate in addition to the horizontal turn parameters."""
        super().__init__(n_envs, speed, radius, z_ref, ramp_time, a_max, z_ramp_tau, center, device,
                         direction, speed0)
        self.climb_rate = float(climb_rate)

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add a smooth climb to the level turn reference."""
        pos, vel, acc = super()._eval(t)
        dz, vz, az = smoothstep_ramp(t, 0.0, self.climb_rate, self.ramp_time, self.a_max)

        pos[:, 2] += dz
        vel[:, 2] += vz
        acc[:, 2] += az
        return pos, vel, acc


class MissionTrajectory(TimedTrajectory):
    """Complete flight from hover through every regime and back to hover (evaluation only).

    profile "level": hover, transition 0 -> V (smoothstep ramp), straight cruise, turn entry (curvature ramped
    0 -> 1/R), steady turn, turn exit, straight cruise, climb and descent (0.5 Az (1 - cos) over one period Tz),
    deceleration V -> 0, final hover.
    cruise_climb > 0 (profile "level"): a "climb" phase at cruise speed after the transition raises the reference by
    cruise_climb (smoothstep, peak vertical speed cruise_climb_rate), and a "descent" phase before the deceleration
    lowers it back, so the mission starts and ends in hover at the lower altitude z_ref.
    profile "helix": the same without the straight climb/descent; during the turn the vertical speed rises to
    +climb_rate, reverses to -climb_rate at the middle of the turn and returns to zero at its end (antisymmetric,
    so the altitude returns to the cruise altitude): first half a climbing helix, second half a descending one.
    Speed, curvature and vertical speed are scheduled in time and the path is integrated on a fine grid, so position,
    velocity and acceleration are mutually consistent and continuous (a = dv/dt T + v^2 kappa N).
    The phase table (name, start, end) is in self.phases.
    """

    def __init__(self, n_envs: int, speed: float, radius: float, z_ref: float, z_amplitude: float, z_period: float,
                 ramp_time: float, a_max: float, z_ramp_tau: float, center: torch.Tensor, device: str,
                 direction: float = 1.0, turn_angle_deg: float = 180.0, hover_s: float = 3.0, cruise_s: float = 5.0,
                 curvature_ramp_s: float = 2.0, final_hover_s: float = 5.0, dt: float = 0.002, speed0: float = 0.0,
                 profile: str = "level", climb_rate: float = 0.3, decel_a_max: float | None = None,
                 cruise_climb: float = 0.0, cruise_climb_rate: float = 1.0):
        """Build the phase schedule and integrate the horizontal path.

        decel_a_max: peak deceleration of the final V -> 0 ramp (default a_max). The Shuttle-Glider cannot brake with
        the puller; its 6-DOF trim admits about 0.7 m/s^2 at 14 m/s and 1.2 m/s^2 at 12 m/s (shuttle_glider/tracking).
        """
        super().__init__(n_envs, center, device)
        V, R = float(speed), float(radius)
        if V <= 0.0:
            raise ValueError("mission needs --traj_speed > 0")
        if R < V * V / float(a_max):
            raise ValueError(f"mission turn radius {R:.1f} m is below V^2/a_max = {V * V / float(a_max):.1f} m")
        if float(speed0) != 0.0:
            raise ValueError("mission starts in hover: use --init_speed 0")
        self.z_ref, self.z_ramp_tau = float(z_ref), max(float(z_ramp_tau), 1e-4)
        self.z_amplitude, self.z_period = float(z_amplitude), float(z_period)
        T_v = max(float(ramp_time), 1.5 * V / max(float(a_max), 1e-6))     # speed ramp duration (as smoothstep_ramp)
        T_d = max(float(ramp_time), 1.5 * V / max(float(a_max if decel_a_max is None else decel_a_max), 1e-6))
        c = float(curvature_ramp_s)
        hold = math.radians(float(turn_angle_deg)) * R / V - c        # heading turned during each ramp: V c / (2 R)
        if hold < 0.0:
            raise ValueError("mission turn angle too small for the curvature ramps")
        if profile not in ("level", "helix"):
            raise ValueError(f"mission profile must be level or helix, got {profile!r}")
        self.profile = profile
        dz = float(cruise_climb)
        T_c = 1.5 * abs(dz) / max(float(cruise_climb_rate), 1e-6)     # smoothstep: peak rate 1.5 dz / T
        if dz != 0.0 and profile != "level":
            raise ValueError("cruise_climb is only available with the level mission profile")
        if profile == "level" and dz != 0.0:
            names = ("hover", "transition", "climb", "cruise", "turn entry", "turn", "turn exit", "cruise 2",
                     "climb/descent", "descent", "deceleration", "final hover")
            lengths = (float(hover_s), T_v, T_c, float(cruise_s), c, hold, c, 2.0, self.z_period, T_c, T_d,
                       float(final_hover_s))
        elif profile == "level":
            names = ("hover", "transition", "cruise", "turn entry", "turn", "turn exit", "cruise 2", "climb/descent",
                     "deceleration", "final hover")
            lengths = (float(hover_s), T_v, float(cruise_s), c, hold, c, 2.0, self.z_period, T_d, float(final_hover_s))
        else:
            names = ("hover", "transition", "cruise", "turn entry", "helix climb", "helix descent", "turn exit",
                     "cruise 2", "deceleration", "final hover")
            lengths = (float(hover_s), T_v, float(cruise_s), c, 0.5 * hold, 0.5 * hold, c, 2.0, T_d,
                       float(final_hover_s))
        edges = np.concatenate([[0.0], np.cumsum(lengths)])
        self.phases = [(n, float(edges[i]), float(edges[i + 1])) for i, n in enumerate(names)]
        self.duration = float(edges[-1])
        e = dict(zip(names, edges[:-1]))           # phase start times
        self._climb = ((float(e["climb/descent"]), float(e["climb/descent"]) + self.z_period) if profile == "level"
                       else (math.inf, math.inf))

        def ramp(t, t0, T):
            u = np.clip((t - t0) / T, 0.0, 1.0)
            return u * u * (3 - 2 * u), np.where((t > t0) & (t < t0 + T), 6 * u * (1 - u) / T, 0.0)

        self.dt = float(dt)
        tg = np.arange(0.0, self.duration + 2 * self.dt, self.dt)
        s_acc, ds_acc = ramp(tg, e["transition"], T_v)
        s_dec, ds_dec = ramp(tg, e["deceleration"], T_d)
        v = V * (s_acc - s_dec)
        a_t = V * (ds_acc - ds_dec)
        exit_start = edges[names.index("turn exit")]
        k_in, _ = ramp(tg, e["turn entry"], c)
        k_out, _ = ramp(tg, exit_start, c)
        kappa = float(direction) * (k_in - k_out) / R
        dpsi = v * kappa
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (dpsi[1:] + dpsi[:-1]) * self.dt)])
        vx, vy = v * np.cos(psi), v * np.sin(psi)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * self.dt)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * self.dt)])
        an = v * v * kappa
        ax = a_t * np.cos(psi) - an * np.sin(psi)
        ay = a_t * np.sin(psi) + an * np.cos(psi)
        if profile == "helix":
            # Vertical speed over the turn (entry start to exit end): 0 -> +w, +w -> -w at the middle, -w -> 0
            t0, t1 = e["turn entry"], exit_start + c
            Tr = min(float(ramp_time), 0.25 * (t1 - t0))
            w = float(climb_rate)
            up, dup = ramp(tg, t0, Tr)
            rev, drev = ramp(tg, 0.5 * (t0 + t1) - Tr, 2 * Tr)
            down, ddown = ramp(tg, t1 - Tr, Tr)
            vz = w * (up - 2 * rev + down)
            az = w * (dup - 2 * drev + ddown)
            z = np.concatenate([[0.0], np.cumsum(0.5 * (vz[1:] + vz[:-1]) * self.dt)])
        elif dz != 0.0:
            up, dup = ramp(tg, e["climb"], T_c)
            down, ddown = ramp(tg, e["descent"], T_c)
            z, vz = dz * (up - down), dz * (dup - ddown)
            az = np.gradient(vz, self.dt)
        else:
            z = vz = az = np.zeros_like(tg)
        f = lambda arr: torch.as_tensor(np.stack(arr, axis=1), dtype=torch.float32, device=device)
        self._grid = (f([x, y, z]), f([vx, vy, vz]), f([ax, ay, az]))

    def _eval(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate the precomputed horizontal path and add the altitude ramp and the climb/descent."""
        pos_g, vel_g, acc_g = self._grid
        x = (t / self.dt).clamp(0.0, pos_g.shape[0] - 1.0)
        i = x.floor().long().clamp(max=pos_g.shape[0] - 2)     # float32 rounding near the end of a long grid
        w = (x - i).clamp(0.0, 1.0).unsqueeze(1)
        lerp = lambda g: g[i] * (1 - w) + g[i + 1] * w
        pos = self.center.clone()
        p, v, a = lerp(pos_g), lerp(vel_g), lerp(acc_g)
        pos[:, 0:2] += p[:, 0:2]
        vel = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=self.device)
        acc = torch.zeros_like(vel)
        vel[:, 0:2] = v[:, 0:2]
        acc[:, 0:2] = a[:, 0:2]
        pos[:, 2], vel[:, 2], acc[:, 2] = self._altitude_profile(t, self.z_ref, self.z_ramp_tau)
        pos[:, 2] += p[:, 2]
        vel[:, 2] += v[:, 2]
        acc[:, 2] += a[:, 2]
        t0, t1 = self._climb
        inside = (t >= t0) & (t <= t1)
        wz = 2.0 * math.pi / self.z_period
        tc = (t - t0).clamp(min=0.0)
        pos[:, 2] += torch.where(inside, 0.5 * self.z_amplitude * (1 - torch.cos(wz * tc)), torch.zeros_like(t))
        vel[:, 2] += torch.where(inside, 0.5 * self.z_amplitude * wz * torch.sin(wz * tc), torch.zeros_like(t))
        acc[:, 2] += torch.where(inside, 0.5 * self.z_amplitude * wz * wz * torch.cos(wz * tc), torch.zeros_like(t))
        return pos, vel, acc


# =============================================================================
# Recorder utils
# =============================================================================
def _to_np_ref(value, n_envs: int):
    """Convert a position reference to an (N, 3) NumPy array for CSV output."""
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


# =============================================================================
# Attitude-error utilities  (trace(I - R_d^T R))
# =============================================================================
def _desired_attitude_from_accel(a_des, yaw = 0) -> np.ndarray:
    """Build the desired rotation matrix R_d from a desired acceleration.

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
    singular = y_norm[:, 0] < 1e-8
    
    if np.any(singular):
        x_c_alt = np.zeros((np.sum(singular), 3), dtype=np.float64)
        x_c_alt[:, 1] = 1.0

        y_b[singular] = np.cross(z_b[singular], x_c_alt)

        y_norm = np.linalg.norm(y_b, axis=1, keepdims=True)
    
    y_b = y_b / np.clip(y_norm, 1e-9, None)
    
    x_b = np.cross(y_b, z_b)
    
    x_b = x_b / np.clip(np.linalg.norm(x_b, axis=1, keepdims=True), 1e-9, None)

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
    """Write synchronized state, control and reference traces for each vehicle.

    The optional CSV stream (--record) is separate
    from the always-on EvaluationLog. One row per controller, vehicle and
    recorded physics step: the state at time t, the reference at the same t and
    the command applied over the previous interval. Episode counters advance
    once per physics step because play_multi owns that loop directly, without a
    PegasusEnv or a skrl wrapper.

    att_err_* is a multirotor diagnostic (desired thrust axis from the reference
    acceleration). Vehicles with aerodynamic surfaces add heading, attitude, body
    velocity, aerodynamic wrench, surface and puller columns.
    """

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
        vehicle_type: str = "iris",
        use_tensorboard: bool = False,
    ):
        """Create output files and metric columns for the selected airframe."""
        
        self.vehicles = vehicles
        self.n_envs = n_envs
        self.step_dt = float(step_dt)
        self.record_every = max(1, int(record_every))
        self.reset_manager = reset_manager
        self.trajectory = trajectory
        
        # Puller rotor index for vehicles with aerodynamic surfaces, else None
        self.puller_index = AERO_PULLER_INDEX.get(vehicle_type)
        self.is_aero = self.puller_index is not None

        self.n_rotors = int(n_rotors)

        # Build diagnostic desired attitudes from feed-forward acceleration or
        # the position and velocity errors. These gains define a logged metric;
        # they do not change the vehicle controller or its physical inputs.
        self.att_gravity = 9.81
        self.att_mass = 1.5
        self.att_Kp = np.array([2.0, 2.0, 2.0], dtype=np.float64) / self.att_mass
        self.att_Kd = np.array([1.2, 1.2, 1.2], dtype=np.float64) / self.att_mass

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_step = np.zeros(n_envs, dtype=np.int64)

        self.traj_file = open(self.out_dir / "trajectories.csv", "x", newline="")
        self.ep_file = open(self.out_dir / "episodes.csv", "x", newline="")

        rotor_cols = []
        for group in ("cmd_w", "actual_w"):
            rotor_cols += [f"{group}_{i}" for i in range(self.n_rotors)]
            
        trajectory_fields = [
            "controller",
            "env_id",
            "episode_id",
            "episode_step",
            "global_step",
            "t",
            "x", "y", "z",
            "vx", "vy", "vz",
            "speed",
            "p_body", "q_body", "r_body",
            "goal_x", "goal_y", "goal_z",
            "ref_vx", "ref_vy", "ref_vz",
            "ref_ax", "ref_ay", "ref_az",
            "pos_error",
            "z_error",
            "att_err_ff",
            "att_err_pd",
            *rotor_cols,
            "fx_body", "fy_body", "fz_body",
            "body_force_norm",
            "tx_body", "ty_body", "tz_body",
            "body_torque_norm",
        ]
        
        self.glider_metric_cols = []
        self.surface_metric_cols = []

        if self.is_aero:
            self.glider_metric_cols = [
                "heading_deg",
                "heading_ref_deg",
                "heading_error_deg",
                "heading_error_abs_deg",
                "heading_alignment",
                "heading_valid",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "tilt_angle_deg",
                "v_forward_body",
                "v_lateral_body",
                "beta_kinematic_deg",
                "puller_thrust",
                "puller_thrust_along_ref",
                "puller_thrust_useful",
                "puller_v_induced_est",
                "puller_slipstream_dv_est",
                "aero_fx_body",
                "aero_fy_body",
                "aero_fz_body",
                "aero_force_norm",
                "aero_tx_body",
                "aero_ty_body",
                "aero_tz_body",
                "aero_torque_norm",
                "elevator_deg",
                "aileron_deg",
                "rudder_deg",
            ]
            
            trajectory_fields += self.glider_metric_cols

        self.traj_writer = csv.DictWriter(
            self.traj_file,
            fieldnames=trajectory_fields,
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
        """Read the shared position goal for all controller batches."""
        if self.reset_manager is None:
            return np.full((self.n_envs, 3), np.nan, dtype=np.float32)

        return _to_np_ref(self.reset_manager.goal_pos, self.n_envs)

    def _get_body_force_torque_numpy(self, vehicle):
        """Read the last body-local force and torque applied to the main body."""
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
        """Return the backend that supplies this vehicle's control commands."""
        backends = vehicle._backends
        if not backends:
            return None
        return backends[0]

    def _get_rotor_input_numpy(self, vehicle):
        """Read commanded and physical rotor speeds for the current sample."""
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

    def _get_control_surfaces_numpy(self, vehicle):
        """Return applied surface deflections [elevator, aileron, rudder] in degrees."""

        if not self.is_aero:
            return np.zeros((self.n_envs, 3), dtype=np.float32)

        return torch.rad2deg(vehicle.surface_deflections).detach().cpu().numpy()

    def _get_aero_wrench_numpy(self, vehicle):
        """Return the aerodynamic wrench applied to the merged body."""

        nan = np.full((self.n_envs, 3), np.nan, dtype=np.float32)
        wrench = getattr(vehicle, "_aero_wrench_local", None)
        if wrench is None:
            return nan.copy(), nan.copy()
        force, torque = wrench

        return force.detach().cpu().numpy(), torque.detach().cpu().numpy()
    

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
                       (a_des = a_ref + g*e3)
          att_err_pd : R_d from the full outer loop
                       (a_des = a_ref + Kp*(p_ref-p) + Kd*(v_ref-v) + g*e3)

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
        a_des_pd = a_ref + self.att_Kp * (p_ref - pos) + self.att_Kd * (v_ref - vel) + g_vec

        R_des_ff = _desired_attitude_from_accel(a_des_ff)
        R_des_pd = _desired_attitude_from_accel(a_des_pd)

        att_err_ff = _attitude_error_trace(R, R_des_ff)
        att_err_pd = _attitude_error_trace(R, R_des_pd)
        return att_err_ff, att_err_pd
    
    def _compute_aero_metrics(self, vehicle, state, goals: np.ndarray, ref_vel: np.ndarray, actual_w: np.ndarray):
        """Compute heading/attitude alignment with the reference and puller metrics.

        The desired heading is the horizontal reference velocity when it exceeds
        0.15 m/s, otherwise the direction to the goal. Puller thrust uses the
        model rotor constant and the actual rotor speed; the induced velocity and
        slipstream are actuator-disk estimates, not simulated quantities.
        """

        n = self.n_envs
        nan = np.full(n, np.nan, dtype=np.float64)

        if actual_w.ndim != 2 or actual_w.shape[1] <= self.puller_index:
            return {key: nan.copy() for key in self.glider_metric_cols}

        # R maps vectors from body to world.
        R = quaternion_to_matrix(state.attitude).detach().cpu().numpy().astype(np.float64)

        # ZYX Euler angles: R = Rz(yaw) Ry(pitch) Rx(roll), FLU body in ENU world.
        # Rotation about +y (left) is positive nose DOWN: a nose-up attitude has pitch_deg < 0.
        pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
        roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])

        pos_w = state.position.detach().cpu().numpy().astype(np.float64)

        vel_w = state.linear_velocity.detach().cpu().numpy().astype(np.float64)
        
        R_t = np.transpose(R, (0, 2, 1))                # (N,3,3)
        vel_b = (R_t @ vel_w[..., None]).squeeze(-1)    # (N,3)
        
        ref_vel = np.asarray(ref_vel, dtype=np.float64)

        goals = np.asarray(goals, dtype=np.float64)

        # Body X axis expressed in world coordinates
        forward_w = R[:, :, 0]
        forward_xy = forward_w[:, :2]

        forward_norm = np.linalg.norm(forward_xy, axis=1, keepdims=True)

        forward_hat = forward_xy / np.clip(forward_norm, 1e-9, None)

        # Desired horizontal direction
        # Use trajectory tangent whenever a reference velocity exists.
        # Otherwise, use the horizontal direction towards the goal.
        ref_vel_xy = ref_vel[:, :2]

        ref_speed_xy = np.linalg.norm(ref_vel_xy, axis=1)

        to_goal_xy = goals[:, :2] - pos_w[:, :2]

        desired_raw = np.where((ref_speed_xy > 0.15)[:, None], ref_vel_xy, to_goal_xy)

        desired_norm = np.linalg.norm(desired_raw, axis=1, keepdims=True)

        heading_valid = (desired_norm[:, 0] > 0.05) & (forward_norm[:, 0] > 1e-6)

        desired_hat = desired_raw / np.clip(desired_norm, 1e-9, None)

        # Heading and alignment
        heading = np.arctan2(forward_hat[:, 1], forward_hat[:, 0])

        heading_ref = np.arctan2(desired_hat[:, 1], desired_hat[:, 0])

        # Signed error: desired heading minus current heading.
        heading_error = np.arctan2(np.sin(heading_ref - heading), np.cos(heading_ref - heading))

        heading_alignment = np.sum(forward_hat * desired_hat, axis=1)
        heading_alignment = np.clip(heading_alignment, -1.0, 1.0)

        # Metrics without a valid desired direction should not enter aggregate heading statistics.
        heading_ref = np.where(heading_valid, heading_ref, np.nan)
        heading_error = np.where(heading_valid, heading_error, np.nan)
        heading_alignment = np.where(heading_valid, heading_alignment, np.nan)

        # Tilt: angle between Z_body and Z_world
        cos_tilt = np.clip(R[:, 2, 2], -1.0, 1.0)
        tilt_angle = np.arccos(cos_tilt)

        v_forward_body = vel_b[:, 0]
        v_lateral_body = vel_b[:, 1]

        # This is a kinematic sideslip proxy. It does not include wind.
        body_speed_xy = np.linalg.norm(vel_b[:, :2], axis=1)
        beta_kinematic = np.arctan2(vel_b[:, 1], vel_b[:, 0])

        beta_kinematic = np.where(body_speed_xy > 0.05, beta_kinematic, np.nan)

        # Puller thrust [N] from the actual rotor speed [rad/s] and the model rotor constant
        kf_p = float(vehicle._thrusters._rotor_constant.reshape(-1)[self.puller_index])
        omega_p = actual_w[:, self.puller_index].astype(np.float64)

        puller_thrust = kf_p * omega_p**2

        # Signed projection on the trajectory tangent.
        puller_along_ref = puller_thrust * heading_alignment

        # Only the positive/progressive component.
        puller_useful = np.maximum(puller_along_ref, 0.0)

        # Actuator-disk slipstream estimate
        rho = 1.225
        prop_radius = 0.13
        disk_area = np.pi * prop_radius**2

        # Existing flow along the rotor axis.
        v_axial_model = np.maximum(v_forward_body, 0.0)

        v_induced_p = 0.5 * (np.sqrt(v_axial_model**2 + 2.0 * puller_thrust / (rho * disk_area + 1e-9)) - v_axial_model)

        wake_factor = 1.5
        coverage_factor = 0.3

        delta_v_slipstream = wake_factor * coverage_factor * v_induced_p

        return {
            "heading_deg": np.rad2deg(heading),
            "heading_ref_deg": np.rad2deg(heading_ref),
            "heading_error_deg": np.rad2deg(heading_error),
            "heading_error_abs_deg": np.abs(
                np.rad2deg(heading_error)
            ),
            "heading_alignment": heading_alignment,
            "heading_valid": heading_valid.astype(np.float64),
            "roll_deg": np.rad2deg(roll),
            "pitch_deg": np.rad2deg(pitch),
            "yaw_deg": np.rad2deg(yaw),
            "tilt_angle_deg": np.rad2deg(tilt_angle),
            "v_forward_body": v_forward_body,
            "v_lateral_body": v_lateral_body,
            "beta_kinematic_deg": np.rad2deg(beta_kinematic),
            "puller_thrust": puller_thrust,
            "puller_thrust_along_ref": puller_along_ref,
            "puller_thrust_useful": puller_useful,
            "puller_v_induced_est": v_induced_p,
            "puller_slipstream_dv_est": delta_v_slipstream,
        }

    def sample(self):
        """Record the state, command and reference before the next physics step."""
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
            omega_b = state.angular_velocity.detach().cpu().numpy().reshape(-1, 3)   # FLU body rates [rad/s]

            body_force, body_torque = self._get_body_force_torque_numpy(vehicle)
            cmd_w, actual_w = self._get_rotor_input_numpy(vehicle)
            control_surfaces_deg = self._get_control_surfaces_numpy(vehicle)

            glider_metrics = None

            if self.is_aero:
                aero_force, aero_torque = self._get_aero_wrench_numpy(vehicle)
                glider_metrics = self._compute_aero_metrics(
                    vehicle=vehicle,
                    state=state,
                    goals=goals,
                    ref_vel=ref_vel,
                    actual_w=actual_w,
                )
                glider_metrics.update({
                    "aero_fx_body": aero_force[:, 0],
                    "aero_fy_body": aero_force[:, 1],
                    "aero_fz_body": aero_force[:, 2],
                    "aero_force_norm": np.linalg.norm(aero_force, axis=1),
                    "aero_tx_body": aero_torque[:, 0],
                    "aero_ty_body": aero_torque[:, 1],
                    "aero_tz_body": aero_torque[:, 2],
                    "aero_torque_norm": np.linalg.norm(aero_torque, axis=1),
                    "elevator_deg": control_surfaces_deg[:, 0],
                    "aileron_deg": control_surfaces_deg[:, 1],
                    "rudder_deg": control_surfaces_deg[:, 2],
                })
                


            pos_error = np.linalg.norm(pos - goals, axis=1)
            z_error = goals[:, 2] - pos[:, 2]

            att_err_ff, att_err_pd = self._compute_attitude_errors(state, goals, ref_vel, ref_acc)

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
                    "p_body": float(omega_b[env_id, 0]),
                    "q_body": float(omega_b[env_id, 1]),
                    "r_body": float(omega_b[env_id, 2]),

                    "goal_x": float(goals[env_id, 0]),
                    "goal_y": float(goals[env_id, 1]),
                    "goal_z": float(goals[env_id, 2]),
                    "ref_vx": float(ref_vel[env_id, 0]),
                    "ref_vy": float(ref_vel[env_id, 1]),
                    "ref_vz": float(ref_vel[env_id, 2]),
                    "ref_ax": float(ref_acc[env_id, 0]),
                    "ref_ay": float(ref_acc[env_id, 1]),
                    "ref_az": float(ref_acc[env_id, 2]),
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
                    row[f"cmd_w_{i}"] = float(cmd_w[env_id, i])
                    row[f"actual_w_{i}"] = float(actual_w[env_id, i])
                    
                if self.is_aero:
                    for key in self.glider_metric_cols:
                        row[key] = float(glider_metrics[key][env_id])

                    row["elevator_deg"] = float(control_surfaces_deg[env_id, 0])
                    row["aileron_deg"] = float(control_surfaces_deg[env_id, 1])
                    row["rudder_deg"] = float(control_surfaces_deg[env_id, 2])

                self.traj_writer.writerow(row)

                if self.tb is not None:
                    prefix = f"{controller_name}/env_{env_id}"
                    self.tb.add_scalar(f"{prefix}/z", row["z"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/speed", row["speed"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/pos_error", row["pos_error"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/cmd_w_0", row["cmd_w_0"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/actual_w_0", row["actual_w_0"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fx_body", row["fx_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fy_body", row["fy_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fz_body", row["fz_body"], self.global_step)
                    if self.is_aero:
                        self.tb.add_scalar(f"{prefix}/elevator_deg", row["elevator_deg"], self.global_step)
                        self.tb.add_scalar(f"{prefix}/aileron_deg", row["aileron_deg"], self.global_step)
                        self.tb.add_scalar(f"{prefix}/rudder_deg", row["rudder_deg"], self.global_step)

    def advance(self, terminated, truncated):
        """Advance one physics step and write any completed episode rows."""
        self.global_step += 1
        self.episode_step += 1

        terminated = terminated.detach().view(-1).cpu().numpy().astype(bool)
        truncated = truncated.detach().view(-1).cpu().numpy().astype(bool)
        done = np.logical_or(terminated, truncated)

        for env_id in np.where(done)[0]:
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

    def close(self):
        """Flush and close the CSV files and optional TensorBoard writer."""
        self.traj_file.close()
        self.ep_file.close()

        if self.tb is not None:
            self.tb.close()


# =============================================================================
# Command line and SimulationApp startup
# =============================================================================
from isaacsim import SimulationApp


def parse_args():
    """Parse vehicle, controller, reference and evaluation output settings."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--physics-dt", type=float, default=None,
                   help="Physics step [s]; defaults to the skrl tasks' training sim_dt (0.01 without skrl).")
    p.add_argument("--spawn-altitude", type=float, default=1.5)
    p.add_argument("--environment", default="Flat Plane",
                   help="World to load: a key of pegasus.simulator.params.SIMULATION_ENVIRONMENTS (e.g. 'Flat Plane') "
                        "or a path to a .usd/.usda file.")
    p.add_argument("--failure-distance", type=float, default=10.0)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", default=False, action="store_true")
    p.add_argument("--follow-camera", choices=["off", "chase", "side"], default="off",
                   help="Viewport camera following env-0's first vehicle: 'chase' (behind and above) or 'side'; "
                        "C toggles between the two (examples/rl/utils/follow_cameras.py).")
    p.add_argument("--cam-back", type=float, default=None, help="Follow camera 'chase': distance behind the vehicle [m] (default 7).")
    p.add_argument("--cam-up", type=float, default=None, help="Follow camera 'chase': height above the vehicle [m] (default 3).")
    p.add_argument("--cam-side", type=float, default=None,
                   help="Follow camera 'side': lateral offset [m], > 0 left of the heading, < 0 right (default 14).")
    p.add_argument("--cam-side-up", type=float, default=None, help="Follow camera 'side': height above the vehicle [m] (default 1.5).")
    p.add_argument("--cam-focal", type=float, default=None, help="Follow cameras: focal length [mm] (default 18.1 chase, 24 side).")
    p.add_argument("--prop-glider", action="append", default=[], metavar="X,Y,Z,YAW_DEG",
                   help="Add a static, physics-free Shuttle Glider (visual prop) with its origin at X,Y,Z [m] and the given yaw [deg]. "
                        "Repeat for several. Rest on a flat surface: origin z = surface z + 0.425.")
    p.add_argument("--prop-shuttle", action="append", default=[], metavar="X,Y,Z,YAW_DEG",
                   help="Like --prop-glider but with the Shuttle alone (multirotor, no glider wing); its origin is at its base.")
    p.add_argument("--screenshot", default=None, metavar="PATH",
                   help="With --pause-at: save a high-resolution PNG of the viewport (scene, reference line and the telemetry overlay "
                        "composed on top) when the simulation pauses; also PATH_scene.png without the overlay.")
    p.add_argument("--screenshot-size", default="3840x2160", metavar="WxH", help="Resolution of --screenshot.")
    p.add_argument("--screenshot-spp", type=int, default=256,
                   help="Samples per pixel of the path-traced --screenshot (0: keep the current render mode, no path tracing).")
    p.add_argument("--pause-at", type=float, default=None,
                   help="Pause the simulation at this time [s] and keep the viewport live (orbit freely, take a screenshot).")
    p.add_argument("--cam-target", choices=["first", "all"], default="first",
                   help="Follow cameras: env-0's first vehicle, or the centre of all controllers' env-0 vehicles.")
    p.add_argument("--cam-fit", type=float, default=0.0,
                   help="With --cam-target all: largest fraction of the frame the vehicles may span; the side camera backs "
                        "off to keep them inside (0: fixed distance).")
    p.add_argument("--cam-shift", type=float, default=0.0,
                   help="Follow cameras: image shift, fraction of the frame height (> 0 places the followed point above the "
                        "centre, leaving the bottom free for telemetry).")
    p.add_argument("--trail", type=float, default=0.0, metavar="SECONDS",
                   help="Draw the last SECONDS of each controller's env-0 path in its manifest 'color' (hex, e.g. '#ff3b30'). "
                        "A manifest 'color' also tints that controller's glider wing.")
    p.add_argument("--goal-marker-color", default="#ff0000", help="Colour (hex) of the cube at the current reference position.")
    p.add_argument("--line-width", type=float, default=2.0, help="Width of the reference line [px]; trails are 1.5x wider.")
    p.add_argument("--video-dir", default=None, metavar="DIR",
                   help="Record the follow camera (needs --follow-camera) as numbered JPEG frames in DIR plus frames.csv; "
                        "works with --headless (examples/rl/utils/video_capture.py). Compose the telemetry and encode with "
                        "scripts/shuttle_glider/video/compose_mission_video.py.")
    p.add_argument("--video-size", default="3840x2160", metavar="WxH", help="Resolution of --video-dir frames.")
    p.add_argument("--video-fps", type=float, default=25.0,
                   help="Frame rate of --video-dir (simulated time); 1/fps must be a multiple of the physics step.")
    p.add_argument("--video-subframes", type=int, default=4,
                   help="Renders per frame with physics stopped (real-time anti-aliasing converges).")
    p.add_argument("--video-spp", type=int, default=0, help="> 0: path trace every frame with this many samples per pixel.")
    p.add_argument("--video-quality", type=int, default=95, help="JPEG quality of the frames.")
    p.add_argument("--telemetry-overlay", default=False, action="store_true",
                   help="Scrolling plots (speed, ||ep||, rotors, control surfaces) in the top-right of the viewport (examples/rl/utils/telemetry_overlay.py).")
    p.add_argument("--episode_duration", type=float, default=10.0, help="Episode length in seconds")

    # ---- Global vehicle model. ALL instantiated controllers use this same
    #      airframe (any per-entry vehicle/cfg in the manifest is ignored). ----
    p.add_argument("--vehicle", choices=sorted(VEHICLES), default="iris",
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
    p.add_argument("--trajectory", choices=["none", "lemniscate", "straight", "forward", "longitudinal", "turn", "helix", "mission"], default="none",
                   help="Reference: static goal, figure-8, straight flight, x-z climb/descent, level turn or helical turn.")
    p.add_argument("--traj_amplitude", type=float, default=1.5, help="Lemniscate half-width [m].")
    p.add_argument("--traj_period", type=float, default=8.0, help="Lemniscate period [s].")
    p.add_argument("--traj_z", type=float, default=1.5, help="Reference/base altitude [m].")
    p.add_argument("--traj_speed", type=float, default=1.5, help="Forward reference speed [m/s].")
    p.add_argument("--traj_radius", type=float, default=10.0, help="Turn radius for turn/helix [m].")
    p.add_argument("--traj_z_amplitude", type=float, default=0.30, help="Vertical excursion amplitude parameter for longitudinal trajectory [m].")
    p.add_argument("--traj_z_period", type=float, default=8.0, help="Vertical oscillation period for longitudinal trajectory [s].")
    p.add_argument("--traj_climb_rate", type=float, default=0.15, help="Asymptotic vertical climb rate for helix [m/s].")
    p.add_argument("--traj_ramp_time", type=float, default=2.5,
                   help="Minimum duration of the smooth speed/climb ramp [s] (shuttle_glider training: 2.5 s).")
    p.add_argument("--traj_a_max", type=float, default=6.0,
                   help="Peak along-track acceleration of the speed ramp [m/s^2] (shuttle_glider training: 6).")
    p.add_argument("--init_speed", type=float, default=0.0,
                   help="Initial speed along +x [m/s]; forward references start at it. Required for easyglider.")
    p.add_argument("--traj_lemniscate_ramp", type=float, default=0.0, help="Lemniscate only: time [s] to ramp the angular speed up from rest (0: full speed from t=0; the direct_rotor evaluation uses 2).")
    p.add_argument("--traj_z_ramp_tau", type=float, default=2.5, help="Smooth altitude-ramp time constant from spawn z to traj_z [s].")
    p.add_argument("--traj_turn_direction", choices=["left", "right"], default="left", help="Turn direction for turn/helix.")
    p.add_argument("--mission_turn_angle", type=float, default=180.0, help="Mission: heading change of the turn [deg].")
    p.add_argument("--mission_hover", type=float, default=3.0, help="Mission: initial hover [s].")
    p.add_argument("--mission_cruise", type=float, default=5.0, help="Mission: straight cruise before the turn [s].")
    p.add_argument("--mission_climb", type=float, default=0.0,
                   help="Mission (level): climb [m] at cruise speed after the transition and descent back before the "
                        "deceleration, so hover is flown at --traj_z and cruise at --traj_z + this (0: level).")
    p.add_argument("--mission_climb_rate", type=float, default=1.0,
                   help="Mission: peak vertical speed of the cruise climb and descent [m/s].")
    p.add_argument("--mission_curvature_ramp", type=float, default=2.0, help="Mission: turn entry/exit curvature ramp [s].")
    p.add_argument("--mission_final_hover", type=float, default=5.0, help="Mission: final hover [s].")
    p.add_argument("--mission_decel_amax", type=float, default=None,
                   help="Mission: peak deceleration of the final V -> 0 ramp [m/s^2] (default --traj_a_max; "
                        "Shuttle-Glider: 1.0, the puller cannot brake).")
    p.add_argument("--mission_profile", choices=["level", "helix"], default="level",
                   help="Mission: level (180 deg turn + straight climb/descent) or helix (climbing then descending "
                        "turn at +-traj_climb_rate; use --mission_turn_angle 720 for one loop up and one down).")

    p.add_argument("--pre_checkpoint", type=str, default=None,
                   help="Path to a RAPTOR pre-training HDF5 checkpoint to evaluate. If set, adds a pre-train-controlled Iris vehicle.")

    p.add_argument("--record", default=False, action="store_true")
    p.add_argument("--record_dir", default=None,
                   help="Exact run directory; defaults to logs/evaluations/<mode>/<test>/<timestamp>.")
    p.add_argument("--record_every", type=int, default=1)
    p.add_argument("--num_episodes_per_env", type=int, default=1)

    return p.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})


# =============================================================================
# Imports that require an initialized SimulationApp
# =============================================================================
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.vehicles.shuttle_glider_batch2 import ShuttleGliderBatch2, ShuttleGliderBatchConfig2
from pegasus.simulator.logic.vehicles.glider_batch import GliderBatch, GliderBatchConfig
from pegasus.simulator.logic.vehicles.vehicle_physics import get_vehicle_physics
from pegasus.simulator.logic.transforms import quaternion_to_matrix
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import ResetManager, GoalCfg
from task_loader import apply_env_overrides, control_timing, load_env


def load_raptor_backend_class():
    """Load the external RAPTOR controller from this example's utils folder."""
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




def load_pretrain_backend_class():
    """Load the RAPTOR pretraining controller from this example's utils folder."""
    path = RL_DIR / "utils" / "foundation_policy_pre_controller.py"

    if not path.exists():
        raise FileNotFoundError(f"PreTrainBackend not found: {path}")

    spec = importlib.util.spec_from_file_location("foundation_policy_pre_controller", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module.PreTrainBackend


def load_skrl_backend_class():
    """Load the skrl checkpoint controller used by manifest entries."""
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
    """Load the geometric controller used by manifest entries."""
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



# =============================================================================
# Controller registry (manifest-driven instantiation)
# =============================================================================
# Physical-vehicle key -> asset and physics preset key: evaluation_log.VEHICLES.
# The selected task may override the baseline physics; all controllers then
# share that exact override.


def _build_raptor_backend(entry, n_envs, device, physics_dt):
    """Create the external RAPTOR policy backend."""
    RaptorBackend = load_raptor_backend_class()
    return RaptorBackend(
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity"),
    )


def _build_pretrain_backend(entry, n_envs, device, physics_dt):
    """Create the RAPTOR pretraining backend from its checkpoint and settings."""
    PreTrainBackend = load_pretrain_backend_class()
    if entry.get("policy_config"):
        with open(entry["policy_config"]) as stream:
            external = json.load(stream)
        entry = {**external, **entry}
    if not entry.get("checkpoint"):
        raise SystemExit(f"[config] controller '{entry.get('name')}' (pretrain) requires a 'checkpoint'.")
    return PreTrainBackend(
        checkpoint_path=entry["checkpoint"],
        n_vehicles=n_envs,
        reset_manager=None,
        action_mode=entry.get("action_mode", "rotor_velocity"),
        omega_min=entry.get("omega_min", 0.0),
        omega_max=entry.get("omega_max", 1100.0),
    )




def _build_nonlinear_backend(entry, n_envs, device, physics_dt):
    """Create a geometric controller with optional gains from the manifest."""
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


def _resolve_environment(name: str) -> str:
    """USD path of --environment: a SIMULATION_ENVIRONMENTS key, or a .usd/.usda file path."""
    if name in SIMULATION_ENVIRONMENTS:
        return SIMULATION_ENVIRONMENTS[name]
    path = Path(name).expanduser().resolve()
    if path.is_file():
        return str(path)
    raise ValueError(f"--environment {name!r} is neither a file nor one of: {', '.join(sorted(SIMULATION_ENVIRONMENTS))}")


def _build_px4_position_backend(entry, n_envs, device, physics_dt):
    """Build a dedicated PX4 position controller for one Shuttle vehicle."""
    from pegasus.simulator.logic.backends.px4_position_backend import PX4PositionBackend

    return PX4PositionBackend(entry["px4"])


def _make_skrl_builder(algo):
    """Return a manifest builder for a skrl agent controller (PPO/SAC).

    The controller is run EXACTLY like play.py: the agent config is loaded from
    tasks/<task>/agents/<algo>_cfg.py, the models + agent are built, the
    checkpoint is restored with agent.load() (which also restores any state
    preprocessor, e.g. PPO's RunningStandardScaler), and inference uses the
    deterministic policy mean. Each entry must provide 'checkpoint' and 'task';
    'preset' defaults to 'default' or the only preset. 'run_metadata' (the run's
    run_metadata.json) and 'env_cfg' ({field: value}) restore the training
    task configuration; without them the task defaults are used.
    """
    def _build(entry, n_envs, device, physics_dt):
        """Restore one skrl checkpoint for use as a vehicle backend."""
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
            preset=entry.get("preset"),
            obs_dim=entry.get("obs_dim"),
            act_dim=entry.get("act_dim"),
            n_vehicles=n_envs,
            reset_manager=None,
            action_mode=entry.get("action_mode", "rotor_velocity_direct"),
            device=device,
            px4=entry.get("px4"),
            run_metadata=entry.get("run_metadata"),
            env_cfg_overrides=entry.get("env_cfg"),
        )
    return _build


def resolve_lqr(entry, physics_dt):
    """Return (K 3x6, P or None, decimation, description) of an "lqr" manifest entry.

    The gains come from exactly one of:
      "gains": {"K": [[...6], [...6], [...6]]}        explicit gain (legacy top-level "K" also accepted)
      "gains": {"kp": 2.0, "kd": 1.5}                  PD law, scalars or per-axis triples
      "gains": {"q_pos": 15, "q_vel": 0.5, "r": 1, "gamma": 0.99}   discounted LQR at the control period
      "gains": {"task": "<task id>", "run_metadata": "...", "gamma": 0.99}
                                                        the task's q_pos/q_vel/r_u (or lqr_q_*) and step_dt
    "control_dt" sets the outer-loop period (default: the task's step_dt, else
    the physics step); it must be a multiple of the physics step. The LQR is
    designed for that period, so its command is held between decisions.
    """
    from lqr_design import closed_loop_poles, discounted_dlqr, pd_gains  # examples/rl/utils is on sys.path

    gains = dict(entry.get("gains") or {})
    if "K" in entry:
        gains.setdefault("K", entry["K"])
    control_dt = entry.get("control_dt")
    P = None
    if "task" in gains:
        task_entry = {"task": gains["task"], "run_metadata": gains.get("run_metadata"), "env_cfg": gains.get("env_cfg")}
        cfg = resolve_task_cfg(task_entry)
        control_dt = control_dt or cfg.sim_dt * cfg.decimation
        gamma = gains.get("gamma", 0.99)
        if all(hasattr(cfg, key) for key in ("q_pos", "q_vel", "r_u")):
            # Shuttle acceleration tasks: the reward weights the state after the step, scaled by dt,
            # which is the standard cost with Q/gamma (see the task's _compute_discounted_dlqr)
            q_pos, q_vel, r = cfg.q_pos * control_dt / gamma, cfg.q_vel * control_dt / gamma, cfg.r_u * control_dt
            K, P = discounted_dlqr(control_dt, q_pos, q_vel, r, gamma)
            P = P - np.diag([q_pos] * 3 + [q_vel] * 3)
            source = f"task {gains['task']} (q_pos={cfg.q_pos}, q_vel={cfg.q_vel}, r={cfg.r_u}, state after step)"
        elif all(hasattr(cfg, key) for key in ("lqr_q_ep", "lqr_q_ev", "lqr_r_u")):
            K, P = discounted_dlqr(control_dt, cfg.lqr_q_ep, cfg.lqr_q_ev, cfg.lqr_r_u, gamma)
            source = f"task {gains['task']} (q_pos={cfg.lqr_q_ep}, q_vel={cfg.lqr_q_ev}, r={cfg.lqr_r_u})"
        else:
            raise ValueError(f"Task {gains['task']} defines neither q_pos/q_vel/r_u nor lqr_q_ep/lqr_q_ev/lqr_r_u weights")
    elif "q_pos" in gains:
        control_dt = control_dt or physics_dt
        K, P = discounted_dlqr(control_dt, gains["q_pos"], gains["q_vel"], gains["r"], gains.get("gamma", 1.0))
        source = f"LQR q_pos={gains['q_pos']}, q_vel={gains['q_vel']}, r={gains['r']}, gamma={gains.get('gamma', 1.0)}"
    elif "kp" in gains:
        K, source = pd_gains(gains["kp"], gains["kd"]), f"PD kp={gains['kp']}, kd={gains['kd']}"
    elif "K" in gains:
        K, source = np.asarray(gains["K"], dtype=float), "explicit K"
        if entry.get("P") is not None:
            P = np.asarray(entry["P"], dtype=float)
    else:
        raise ValueError(f"LQR '{entry.get('name')}' needs gains: K, kp/kd, q_pos/q_vel/r or task")

    control_dt = float(control_dt or physics_dt)
    decimation = int(round(control_dt / physics_dt))
    if decimation < 1 or abs(decimation * physics_dt - control_dt) > 1e-9:
        raise ValueError(f"LQR control_dt {control_dt} must be a multiple of the physics step {physics_dt}")
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 6) or not np.isfinite(K).all():
        raise ValueError("LQR gain K must be a finite 3x6 matrix")
    if np.abs(closed_loop_poles(K, control_dt)).max() >= 1.0:
        print(f"[config] WARNING LQR '{entry.get('name')}': closed loop unstable for the double integrator "
              f"at dt={control_dt}", flush=True)
    return K, P, decimation, f"{source}, control_dt={control_dt}"


def _build_lqr_backend(entry, n_envs, device, physics_dt):
    """Create the LQR backend from the gains resolved by resolve_lqr."""
    # Loaded by path: another top-level "utils" package shadows examples/rl/utils in Isaac Sim
    spec = importlib.util.spec_from_file_location("lqr_controller_batch", str(RL_DIR / "utils" / "lqr_controller_batch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    LQRBackend = module.LQRBackend
    K, P, decimation, description = resolve_lqr(entry, physics_dt)
    print(f"[config] LQR '{entry['name']}': {description}\n  K = {np.round(K, 4).tolist()}", flush=True)
    return LQRBackend(
        n_vehicles=n_envs,
        P=None if P is None else torch.tensor(P, dtype=torch.float32, device=device),
        K=torch.tensor(K, dtype=torch.float32, device=device),
        action_mode=entry.get("action_mode", "rotor_velocity"),
        decimation=decimation,
        accel_limit=entry.get("accel_limit"),
        angular_damping=entry.get("angular_damping"),
    )


# Controller type -> backend builder. Single source of truth; add new types here.
#   "raptor"   : RAPTOR foundation policy           (utils/foundation_policy_controller.py)
#   "pretrain" : RAPTOR pre-training foundation pol. (utils/foundation_policy_pre_controller.py)
#   "lqr"      : double-integrator LQR, explicit K/P   (utils/lqr_controller_batch.py)
#   "ppo"/"ppo2": full skrl PPO agent, like play.py   (utils/skrl_agent_controller.py)
#   "sac_skrl" : full skrl SAC agent, like play.py    (utils/skrl_agent_controller.py)
#   "nonlinear": geometric Mellinger-Kumar controller (utils/non_linear_controller.py)
#   "px4_position": PX4 SITL position controller, Shuttle only
CONTROLLER_REGISTRY = {
    "lqr": _build_lqr_backend,
    "raptor":    _build_raptor_backend,
    "pretrain":  _build_pretrain_backend,
    "ppo":       _make_skrl_builder("ppo"),
    "ppo2":      _make_skrl_builder("ppo2"),
    "sac_skrl":  _make_skrl_builder("sac"),
    "nonlinear": _build_nonlinear_backend,
    "px4_position": _build_px4_position_backend,
}
assert set(CONTROLLER_REGISTRY) == {"lqr", "raptor", "pretrain", "nonlinear", "px4_position", *SKRL_TYPES}


def load_manifest(path):
    """Load and validate a JSON manifest of controllers.

    See evaluation_log.validate_manifest for the format. Every controller is
    instantiated on the single global vehicle: the manifest "vehicle" if present,
    otherwise --vehicle. A "checkpoint" or "run_metadata" path may contain glob
    wildcards (e.g. ".../train/s03/*_PPO2/checkpoints/agent_120000.pt"); it resolves
    to the last match in sorted order, i.e. the most recent timestamped run.

    Returns (vehicles, global_vehicle); global_vehicle may be None.
    """
    with open(path, "r") as f:
        manifest = json.load(f)
    for entry in (manifest if isinstance(manifest, list) else manifest.get("vehicles", [])):
        for key in ("checkpoint", "run_metadata"):
            value = entry.get(key)
            if isinstance(value, str) and any(c in value for c in "*?["):
                matches = sorted(glob.glob(value))
                if not matches:
                    raise SystemExit(f"[config] {path}: no file matches {key} '{value}'")
                entry[key] = matches[-1]
    try:
        vehicles, global_vehicle = validate_manifest(manifest)
    except ValueError as error:
        raise SystemExit(f"[config] {path}: {error}")
    return vehicles, global_vehicle


def resolve_task_cfg(entry):
    """Return the task config of a skrl entry with its training snapshot and overrides applied."""
    _, cfg = load_env(entry["task"])
    apply_env_overrides(cfg, entry.get("run_metadata"), values=entry.get("env_cfg"))
    return cfg


def main():
    """Spawn the selected controllers and evaluate them in one physics world.

    Each loop iteration calls world.step exactly once. Controllers advance on
    their own schedule, while the shared reference and audit advance at the
    physics rate. Rendering accompanies the same step in graphical mode.
    """
    device = args.device
    n_envs = args.n_envs

    if args.n_envs < 1:
        raise ValueError("A positive batch size is required")
    if args.headless and args.num_episodes_per_env <= 0:
        raise ValueError("Headless evaluation requires a finite episode count")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.trajectory == "forward":
        args.trajectory = "straight"

    # Validate controller selection.
    manifest_vehicles, manifest_vehicle_choice = (
        load_manifest(args.config) if args.config else (None, None))
    if manifest_vehicles is None:
        manifest_vehicles = []
        if args.raptor:
            manifest_vehicles.append({"name": "raptor", "type": "raptor"})
        if args.pre_checkpoint:
            manifest_vehicles.append({"name": "pretrain", "type": "pretrain", "checkpoint": args.pre_checkpoint})
        if not manifest_vehicles:
            raise SystemExit("Select --config, --raptor or --pre_checkpoint")
        if args.crazyflie:
            args.vehicle = "crazyflie"

    # Global vehicle model: every controller is instantiated with this same
    # airframe. A manifest top-level "vehicle" wins over the --vehicle CLI flag.
    vehicle_choice = (manifest_vehicle_choice or args.vehicle)
    if vehicle_choice not in VEHICLES:
        raise SystemExit(f"[config] unknown vehicle '{vehicle_choice}'; "
                         f"must be one of {sorted(VEHICLES)}.")
    usd_key = VEHICLES[vehicle_choice]
    if usd_key not in ROBOTS:
        raise ValueError(f"No USD asset is registered for {usd_key}")

    # References must match what the vehicle can fly: a fixed wing cannot hover.
    moving = args.trajectory in ("straight", "longitudinal", "turn", "helix")
    if args.init_speed < 0 or (args.init_speed > 0 and not moving):
        raise ValueError("--init_speed > 0 requires a straight, longitudinal, turn or helix reference")
    if vehicle_choice in FIXED_WING and (not moving or args.init_speed <= 0):
        raise ValueError(f"{vehicle_choice} is a fixed wing: use a forward-flight reference and --init_speed > 0")

    # Training configuration of every skrl controller (snapshot + overrides).
    task_cfgs = {entry["name"]: resolve_task_cfg(entry) for entry in manifest_vehicles if entry["type"] in SKRL_TYPES}
    task_dts = {cfg.sim_dt for cfg in task_cfgs.values()}
    if len(task_dts) > 1:
        raise ValueError(f"skrl tasks were trained with different sim_dt {sorted(task_dts)}; evaluate them separately")
    physics_dt = args.physics_dt if args.physics_dt is not None else (task_dts.pop() if task_dts else 0.01)
    args.physics_dt = physics_dt
    if physics_dt <= 0 or args.episode_duration < physics_dt:
        raise ValueError("Positive dt and an episode at least one step long are required")
    episode_steps = int(round(args.episode_duration / physics_dt))

    cfg_fn = lambda: get_vehicle_physics(usd_key)
    selected_physics = None
    px4_entries = [entry for entry in manifest_vehicles if entry.get("px4") is not None]
    if px4_entries:
        # Several skrl controllers may each attach their own PX4 SITL estimator (one PX4 instance per vehicle_id, separate
        # ports and lockstep links); a PX4 position controller runs alone.
        if n_envs != 1 or args.num_episodes_per_env != 1:
            raise ValueError("PX4 SITL evaluation requires one vehicle per controller and one episode")
        if len(px4_entries) > 1 and any(entry["type"] not in SKRL_TYPES for entry in px4_entries):
            raise ValueError("Several PX4 vehicles are supported for skrl controllers with the PX4 estimator only")
        px4_ids = [int(entry["px4"].get("vehicle_id", 0)) for entry in px4_entries]
        if len(set(px4_ids)) != len(px4_ids):
            raise ValueError(f"PX4 vehicles need distinct px4.vehicle_id (got {px4_ids})")
        if px4_entries[0]["type"] not in (*SKRL_TYPES, "px4_position"):
            raise ValueError("PX4 SITL requires a skrl controller or px4_position")
        if px4_entries[0]["type"] == "px4_position":
            if vehicle_choice != "shuttle":
                raise ValueError("PX4 position control currently supports the Shuttle (airframe 10050_none_shuttle)")

    # Control rate of each controller [Hz]; stored in metadata.json.
    timing = {}
    for entry in manifest_vehicles:
        kind = entry["type"]
        if vehicle_choice in AERO_PULLER_INDEX and kind not in SKRL_TYPES:
            raise ValueError(f"{kind} implements four-rotor control, incompatible with {vehicle_choice}")
        if kind == "pretrain" and vehicle_choice != "iris":
            raise ValueError("The maintained pretrain integration is scoped to the external Iris policy")
        if kind in ("raptor", "pretrain") and abs(physics_dt - 0.01) > 1e-9:
            raise ValueError("External policy adapter requires 100 Hz; another frequency requires contract validation")
        if kind in SKRL_TYPES:
            task_cfg = task_cfgs[entry["name"]]
            if task_cfg.vehicle != usd_key:
                raise ValueError(f"Task vehicle {task_cfg.vehicle} differs from requested {usd_key}")
            # The policy holds each action for decimation physics steps of sim_dt, as in
            # training; this is only the training rate if sim_dt is the training sim_dt.
            if abs(task_cfg.sim_dt - physics_dt) > 1e-9:
                raise ValueError(f"Evaluation physics dt {physics_dt} differs from the task's training sim_dt "
                                 f"{task_cfg.sim_dt}; control decimation alone is insufficient")
            if not entry.get("run_metadata"):
                print(f"[config] WARNING '{entry['name']}': no run_metadata, task defaults assumed to match training",
                      flush=True)
            timing[entry["name"]] = control_timing(task_cfg)
            cfg = getattr(task_cfg, "vehicle_physics_cfg", None)
            if cfg is None:
                cfg = get_vehicle_physics(usd_key)
            if selected_physics is not None and cfg != selected_physics:
                raise ValueError("Controllers request different physics; evaluate these conditions in separate runs")
            selected_physics = cfg
        else:
            # LQR holds its command for its own decimation; the other controllers act every step.
            decimation = resolve_lqr(entry, physics_dt)[2] if kind == "lqr" else 1
            timing[entry["name"]] = {"sim_dt": physics_dt, "decimation": decimation,
                                     "physics_hz": 1.0 / physics_dt, "control_hz": 1.0 / (physics_dt * decimation)}
        print(f"[timing] '{entry['name']}': policy {timing[entry['name']]['control_hz']:.1f} Hz, "
              f"physics {1.0 / physics_dt:.1f} Hz", flush=True)
    if selected_physics is not None:
        cfg_fn = lambda: deepcopy(selected_physics)

    print(f"[config] vehicle model for ALL controllers: {vehicle_choice} "
          f"(USD '{usd_key}')", flush=True)

    pg = PegasusInterface()
    # The loop owns individual physics steps; match the render interval to them.
    pg.set_world_settings(physics_dt=physics_dt, rendering_dt=physics_dt, device=device)
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    pg.load_environment(_resolve_environment(args.environment))

    if args.environment not in SIMULATION_ENVIRONMENTS:
        # A custom world (.usd/.usda file) has its own ground: the viewport grid at z=0 would show through the water and
        # in terrain dips, looking like dashed lines.
        import carb.settings
        for _key in ("/app/viewport/grid/enabled", "/persistent/app/viewport/grid/enabled"):
            carb.settings.get_settings().set(_key, False)

    if args.environment in SIMULATION_ENVIRONMENTS:
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            position=np.array([1.0, 1.0, 1.0]),
            attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
        )
    # A custom world (.usd/.usda file) brings its own lights; adding this dome on top over-exposes it.

    if "cuda" in device:
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # Each controller is an independent vehicle batch sharing the same world.
    # The first vehicle created defines the spawn grid; the rest are aligned to it
    # so that every controller starts from identical initial conditions.
    controllers = {}        # name -> vehicle (insertion order preserved)
    backends_order = []     # backends in creation order
    _anchor = {"vehicle": None}
    _next_id = {"value": 1}

    def _spawn(name, usd_key, backend, cfg_fn):
        """Instantiate one controller's vehicle batch and register it."""

        physics_cfg = cfg_fn()

        if vehicle_choice == "shuttle_glider2":
            vehicle_cfg = ShuttleGliderBatchConfig2(cfg=physics_cfg, n_vehicles=n_envs)
            VehicleClass = ShuttleGliderBatch2

        elif vehicle_choice == "easyglider":
            vehicle_cfg = GliderBatchConfig(n_vehicles=n_envs, thrust_curve_cfg=physics_cfg)
            VehicleClass = GliderBatch

        else:
            vehicle_cfg = MultirotorBatchConfig(cfg=physics_cfg, n_vehicles=n_envs)
            VehicleClass = MultirotorBatch

        vehicle_cfg.backends = [backend]
        if getattr(backend, "_px4_backend", None) is not None:
            # The first backend remains the command source. Sensor delivery
            # goes to the secondary PX4 backend after the physical state update.
            vehicle_cfg.backends.append(backend._px4_backend)
            vehicle_cfg.sensors = (backend._px4_sensors,)
        elif getattr(backend, "_px4_sensors", None) is not None:
            # A dedicated PX4 controller is the first command backend itself.
            vehicle_cfg.sensors = (backend._px4_sensors,)

        kwargs = dict(
            stage_prefix=f"/World/{name}",
            usd_file=ROBOTS[usd_key],
            vehicle_batch_id=_next_id["value"],
            n_vehicles=n_envs,
            config=vehicle_cfg,
        )

        if _anchor["vehicle"] is None:
            kwargs["spacing"] = 2.5
            initial = torch.zeros((n_envs, 3), device=device)
            initial[:, 0] = torch.arange(n_envs, device=device) * 2.5
            initial[:, 2] = args.spawn_altitude
            kwargs["init_pos"] = initial
        else:
            kwargs["init_pos"] = _anchor["vehicle"]._init_pos
            kwargs["init_orientation"] = _anchor["vehicle"]._init_orientation

        vehicle = VehicleClass(**kwargs)

        if hasattr(vehicle, "disable_collisions"):
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
    # =============================================================================
    # Start simulation
    # =============================================================================
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

    reset_manager.set_goal_cfg(goal_cfg)
    reset_manager.init_state_cfg = None
    if goal_cfg is not None:
        print(
            f"[Sim] Re-applied CLI goal_cfg: "
            f"xy={goal_cfg.goal_pos_xy_range}, z={goal_cfg.goal_pos_z_range}",
            flush=True,
        )

    # Use one backend to draw the shared reference marker for every controller.
    marker_backend = next((b for b in backends_order if hasattr(b, "create_goal_marker")), None)

    # =============================================================================
    # Reference trajectory
    # =============================================================================
    trajectory = None
    center = _anchor["vehicle"]._init_pos.clone().to(device=device, dtype=torch.float32)
    direction = 1.0 if args.traj_turn_direction == "left" else -1.0
    ramp = dict(ramp_time=args.traj_ramp_time, a_max=args.traj_a_max, z_ramp_tau=args.traj_z_ramp_tau,
                center=center, device=device, speed0=args.init_speed)

    if args.trajectory == "lemniscate":
        trajectory = LemniscateTrajectory(n_envs=n_envs, amplitude=args.traj_amplitude, period=args.traj_period,
                                          z_ref=args.traj_z, z_ramp_tau=args.traj_z_ramp_tau, center=center, device=device,
                                          speed_ramp_s=args.traj_lemniscate_ramp)
        print(f"[Sim] Lemniscate: A={args.traj_amplitude:.2f} m, T={args.traj_period:.2f} s, z={args.traj_z:.2f} m", flush=True)
    elif args.trajectory == "straight":
        trajectory = ForwardFlightTrajectory(n_envs=n_envs, speed=args.traj_speed, z_ref=args.traj_z, **ramp)
        print(f"[Sim] Straight flight: V0={args.init_speed:.2f} -> V={args.traj_speed:.2f} m/s, z={args.traj_z:.2f} m, "
              f"ramp>={args.traj_ramp_time:.2f} s, a_max={args.traj_a_max:.2f} m/s^2", flush=True)
    elif args.trajectory == "longitudinal":
        trajectory = LongitudinalFlightTrajectory(n_envs=n_envs, speed=args.traj_speed, z_ref=args.traj_z,
                                                  z_amplitude=args.traj_z_amplitude, z_period=args.traj_z_period, **ramp)
        print(f"[Sim] Longitudinal x-z: V0={args.init_speed:.2f} -> V={args.traj_speed:.2f} m/s, Az={args.traj_z_amplitude:.2f} m, "
              f"Tz={args.traj_z_period:.2f} s, z_target={args.traj_z:.2f} m, tau_z={args.traj_z_ramp_tau:.2f} s", flush=True)
    elif args.trajectory == "turn":
        trajectory = LevelTurnTrajectory(n_envs=n_envs, speed=args.traj_speed, radius=args.traj_radius,
                                         z_ref=args.traj_z, direction=direction, **ramp)
        print(f"[Sim] Level turn: V={args.traj_speed:.2f} m/s, R={args.traj_radius:.2f} m, "
              f"direction={args.traj_turn_direction}, z_target={args.traj_z:.2f} m, tau_z={args.traj_z_ramp_tau:.2f} s", flush=True)
    elif args.trajectory == "helix":
        trajectory = HelicalTurnTrajectory(n_envs=n_envs, speed=args.traj_speed, radius=args.traj_radius,
                                           z_ref=args.traj_z, climb_rate=args.traj_climb_rate,
                                           direction=direction, **ramp)
        print(f"[Sim] Helical turn: V={args.traj_speed:.2f} m/s, R={args.traj_radius:.2f} m, "
              f"Vz={args.traj_climb_rate:.2f} m/s, direction={args.traj_turn_direction}", flush=True)
    elif args.trajectory == "mission":
        trajectory = MissionTrajectory(n_envs=n_envs, speed=args.traj_speed, radius=args.traj_radius, z_ref=args.traj_z,
                                       z_amplitude=args.traj_z_amplitude, z_period=args.traj_z_period,
                                       direction=direction, turn_angle_deg=args.mission_turn_angle,
                                       hover_s=args.mission_hover, cruise_s=args.mission_cruise,
                                       curvature_ramp_s=args.mission_curvature_ramp,
                                       final_hover_s=args.mission_final_hover, profile=args.mission_profile,
                                       climb_rate=args.traj_climb_rate, decel_a_max=args.mission_decel_amax,
                                       cruise_climb=args.mission_climb, cruise_climb_rate=args.mission_climb_rate, **ramp)
        print(f"[Sim] Mission ({args.mission_profile}): V={args.traj_speed:.2f} m/s, R={args.traj_radius:.2f} m ({args.traj_turn_direction}, "
              f"{args.mission_turn_angle:.0f} deg), "
              + (f"climb rate +-{args.traj_climb_rate:.2f} m/s, " if args.mission_profile == "helix" else
                 f"Az={args.traj_z_amplitude:.2f} m, Tz={args.traj_z_period:.2f} s, ") +
              (f"cruise climb {args.mission_climb:.1f} m at <= {args.mission_climb_rate:.2f} m/s, "
               if args.mission_climb else "") +
              f"total {trajectory.duration:.1f} s", flush=True)
        for name, t0, t1 in trajectory.phases:
            print(f"[Sim]   {t0:6.2f} - {t1:6.2f} s  {name}", flush=True)
        if args.episode_duration < trajectory.duration:
            print(f"[Sim] WARNING: --episode_duration {args.episode_duration:.1f} s ends before the mission "
                  f"({trajectory.duration:.1f} s)", flush=True)
    else:
        print("[Sim] Static goal mode.", flush=True)

    # Peak reference acceleration: compare with what the vehicle was trained to fly.
    reference_peak_acc = (trajectory.peak_acceleration(args.episode_duration, physics_dt)
                          if trajectory is not None else 0.0)
    if trajectory is not None:
        print(f"[Sim] Peak reference acceleration: {reference_peak_acc:.2f} m/s^2", flush=True)

    # Draw one shared goal marker at the reference position.
    if marker_backend is not None:
        marker_rgb = tuple(int(args.goal_marker_color.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        marker_backend.create_goal_marker(root_path="/World/GoalMarker", size=0.15, color=marker_rgb)

    def _update_goal(traj_obj):
        """Push the current reference position into reset_manager and the goal marker."""
        pos, vel, acc = traj_obj.at_time(traj_obj.current_time)
        reset_manager._goal_pos[:] = pos
        reset_manager._goal_vel[:] = vel
        reset_manager._goal_acc[:] = acc

    ref_lines = {}  # debug_draw arguments of the reference, kept to redraw it with the trails

    def _draw_trajectory_viewport(traj_obj):
        """Draw env-0 reference path over one episode in the Isaac Sim viewport."""
        if traj_obj is None or (args.headless and not args.video_dir):
            return

        from isaacsim.util.debug_draw import _debug_draw
        from speed_colormap import parula
        draw = _debug_draw.acquire_debug_draw_interface()

        # Thin segments coloured by the reference speed (parula, 0..cruise speed).
        n_samples = 1200
        duration = min(float(args.episode_duration), float(getattr(traj_obj, "duration", args.episode_duration)))
        vmax = float(args.traj_speed) if args.traj_speed > 0 else 12.0
        points, speeds = [], []
        for t_sample in torch.linspace(0.0, duration, n_samples, device=device):
            t_vec = torch.full((n_envs,), float(t_sample), dtype=torch.float32, device=device)
            pos, vel, _ = traj_obj.at_time(t_vec)
            points.append(tuple(float(c) for c in pos[0]))
            speeds.append(float(vel[0].norm()))

        ref_lines.update(start=points[:-1], end=points[1:], colors=[parula(s, 0.0, vmax) for s in speeds[:-1]],
                         widths=[float(args.line_width)] * (n_samples - 1))
        draw.clear_lines()
        draw.draw_lines(ref_lines["start"], ref_lines["end"], ref_lines["colors"], ref_lines["widths"])
        print(f"[draw] {args.trajectory} reference drawn in viewport.", flush=True)

    # =============================================================================
    # Recorder
    # =============================================================================
    from evaluation_log import EvaluationLog, evaluation_directory

    run_dir = (Path(args.record_dir) if args.record_dir else
               evaluation_directory(RL_DIR / "logs" / "evaluations", vehicle_choice, manifest_vehicles))
    args.record_dir = str(run_dir)
    recorder = None
    phases = getattr(trajectory, "phases", None)
    if phases is not None:                    # mission: phase table for the plots
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "mission_phases.json").write_text(json.dumps(
            [{"phase": n, "t_start": t0, "t_end": t1} for n, t0, t1 in phases], indent=2))

    if args.record:
        record_vehicles = dict(controllers)

        first_vehicle = next(iter(controllers.values()))
        n_rotors = int(first_vehicle._thrusters._num_rotors)

        recorder = EpisodeTrajectoryRecorder(
            vehicles=record_vehicles,
            n_envs=n_envs,
            step_dt=physics_dt,
            out_dir=str(run_dir),
            record_every=args.record_every,
            reset_manager=reset_manager,
            trajectory=trajectory,
            n_rotors=n_rotors,
            vehicle_type=vehicle_choice,
            use_tensorboard=False,
        )

    # =============================================================================
    # Physics loop
    # =============================================================================
    step = 0

    # In-flight start: the first controller whose task has a trim supplies it, as
    # in the shuttle_glider cruise starts; otherwise level attitude and stopped rotors.
    cruise = None
    # Shuttle_glider2 tasks are trained from trimmed states: a start at rest uses the task's hover trim (rotors at
    # the hover speeds) instead of the Shuttle protocol of stopped rotors with a mid-throttle command.
    if args.init_speed > 0 or vehicle_choice == "shuttle_glider2":
        cruise = next((c for c in (getattr(b, "cruise_start", lambda v: None)(args.init_speed) for b in backends_order)
                       if c is not None), None)
        if cruise is None:
            print("[Sim] WARNING: no task trim available; in-flight start is level with rotors stopped", flush=True)
            velocity = torch.zeros((n_envs, 3), device=device)
            velocity[:, 0] = args.init_speed
            cruise = {"linear_velocity": velocity}
        else:
            print(f"[Sim] {'In-flight' if args.init_speed > 0 else 'Hover'} start at {args.init_speed:.2f} m/s, "
                  f"trim alpha {cruise['alpha_trim_deg']:.2f} deg", flush=True)

    def _reset_episode():
        """Reset every vehicle to the shared initial state and restart policy histories."""
        if cruise is None:
            reset_manager.reset_all()
        else:
            reset_manager.reset_envs(torch.arange(n_envs, device=device), randomize_goals=goal_cfg is not None,
                                     init_overrides=cruise)
        for backend in backends_order:
            if hasattr(backend, "after_evaluation_reset"):
                backend.after_evaluation_reset()

    audit_args = dict(vars(args), reference_peak_acceleration_m_s2=reference_peak_acc,
                      alpha_trim_deg=(cruise or {}).get("alpha_trim_deg"),
                      env_cfg_changes={b_name: getattr(b, "override_log", {}).get("changed")
                                       for b_name, b in zip(controllers, backends_order)})
    audit_log = EvaluationLog(run_dir, audit_args, manifest_vehicles, cfg_fn(), vehicle_choice, timing=timing)

    _reset_episode()
    _draw_trajectory_viewport(trajectory)

    if args.prop_glider or args.prop_shuttle:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        prop_stage = stage_utils.get_current_stage()
        prop_specs = ([("Glider", "Shuttle_glider2", s) for s in args.prop_glider]
                      + [("Shuttle", "Shuttle", s) for s in args.prop_shuttle])
        for i, (prop_kind, prop_asset, spec) in enumerate(prop_specs):
            px, py, pz, yaw = (float(v) for v in spec.split(","))
            # the asset root carries its own xform ops: the pose goes on a parent Xform
            xf = UsdGeom.Xformable(prop_stage.DefinePrim(f"/World/Props/{prop_kind}_{i}", "Xform"))
            xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
            xf.AddRotateZOp().Set(yaw)
            prop = prop_stage.DefinePrim(f"/World/Props/{prop_kind}_{i}/vehicle", "Xform")
            prop.GetReferences().AddReference(ROBOTS[prop_asset])
            for q in Usd.PrimRange(prop):       # visual only: no bodies, colliders, joints or articulation
                if q.IsA(UsdPhysics.Joint):
                    q.SetActive(False)
                    continue
                for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI,
                            UsdPhysics.MeshCollisionAPI, UsdPhysics.ArticulationRootAPI):
                    if q.HasAPI(api):
                        q.RemoveAPI(api)
            print(f"[props] {prop_kind.lower()} {i} at ({px:.1f}, {py:.1f}, {pz:.2f}) yaw {yaw:.0f} deg", flush=True)

    # Manifest colours: tint each controller's glider wing and draw its trail in that colour.
    hex_rgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    vehicle_colors = {e["name"]: hex_rgb(e["color"]) for e in (manifest_vehicles or []) if e.get("color")}
    if vehicle_colors:
        from pxr import Gf as _Gf, UsdGeom as _UsdGeom
        tint_stage = stage_utils.get_current_stage()
        for c_name, rgb in vehicle_colors.items():
            root = re.compile(rf"/World/{re.escape(c_name)}(_batch_\d+)?/")  # vehicle batches live at /World/<name>_batch_<id>
            meshes = [q for q in tint_stage.Traverse()
                      if root.match(str(q.GetPath())) and str(q.GetPath()).endswith("/glider/mesh")]
            for q in meshes:
                _UsdGeom.Mesh(q).GetDisplayColorAttr().Set([_Gf.Vec3f(*rgb)])
            print(f"[color] '{c_name}' {rgb}: {len(meshes)} wing mesh(es) tinted", flush=True)

    follow_cams = None
    if args.video_dir and args.follow_camera == "off":
        raise ValueError("--video-dir records the follow camera: add --follow-camera chase|side")
    video_size = tuple(int(v) for v in args.video_size.lower().split("x"))
    if args.follow_camera != "off" and (not args.headless or args.video_dir):
        from follow_cameras import FollowCameras
        cam_kw = {k: v for k, v in dict(back=args.cam_back, up=args.cam_up, side=args.cam_side,
                                        side_up=args.cam_side_up, focal=args.cam_focal).items() if v is not None}
        follow_cams = FollowCameras(start=args.follow_camera, fit=args.cam_fit if args.cam_target == "all" else 0.0,
                                    shift=args.cam_shift, aspect=video_size[0] / video_size[1], **cam_kw)

    trails = {c_name: deque(maxlen=max(2, int(round(args.trail / 0.05)))) for c_name in vehicle_colors} if args.trail > 0 else {}

    video = None
    video_every = 1
    if args.video_dir:
        from video_capture import VideoCapture
        video_every = int(round(1.0 / (args.video_fps * physics_dt)))
        if video_every < 1 or abs(video_every * physics_dt * args.video_fps - 1.0) > 1e-6:
            raise ValueError(f"--video-fps {args.video_fps}: 1/fps must be a multiple of the physics step {physics_dt}")
        video = VideoCapture(args.video_dir, follow_cams._path[follow_cams.mode], size=video_size,
                             quality=args.video_quality, subframes=args.video_subframes, spp=args.video_spp)

    def _draw_lines_with_trails():
        """Redraw the reference and each controller's trail (debug_draw has one shared line buffer)."""
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        if ref_lines:
            draw.draw_lines(ref_lines["start"], ref_lines["end"], ref_lines["colors"], ref_lines["widths"])
        for c_name, pts in trails.items():
            if len(pts) > 1:
                pts = list(pts)
                draw.draw_lines(pts[:-1], pts[1:], [vehicle_colors[c_name] + (1.0,)] * (len(pts) - 1),
                                [1.5 * float(args.line_width)] * (len(pts) - 1))

    telemetry = None
    if args.telemetry_overlay and not args.headless:
        from telemetry_overlay import TelemetryOverlay
        telemetry = TelemetryOverlay()


    if trajectory is not None:
        _update_goal(trajectory)

    if marker_backend is not None:
        marker_backend.update_goal_marker(reset_manager.goal_pos)

    try:
        n_px4 = sum(getattr(b, "_px4_config", None) is not None for b in backends_order)
        for b_name, backend in zip(controllers, backends_order):
            if getattr(backend, "_px4_config", None) is not None:
                print(f"[PX4] Preparing SITL backend of '{b_name}' before evaluation.", flush=True)
                path = run_dir / ("px4_estimate.jsonl" if n_px4 == 1 else f"px4_estimate_{b_name}.jsonl")
                backend.prepare_px4(physics_dt, world.render if not args.headless else (lambda: None), path)
                print("[PX4] SITL backend ready.", flush=True)

        # A headless Kit app can report is_running=False after setup even though
        # physics stepping remains available. The episode count owns termination.
        while args.headless or simulation_app.is_running():
            if recorder is not None:
                recorder.sample()

            # The next frame draws the vehicle at t + dt. Move only the visual
            # marker to that reference before stepping; update the controller's
            # goal after physics so it remains synchronized with the recorder.
            if marker_backend is not None:
                if trajectory is not None:
                    next_pos, _, _ = trajectory.at_time(trajectory.current_time + physics_dt)
                    marker_backend.update_goal_marker(next_pos)
                else:
                    marker_backend.update_goal_marker(reset_manager.goal_pos)

            # The audit verifies one post-physics callback for this interval.
            audit_log.begin_interval(controllers)
            world.step(render=not args.headless)
            step += 1

            if follow_cams is not None:
                cam_state = _anchor["vehicle"].state
                cam_points = None
                if args.cam_target == "all":
                    cam_points = np.stack([v.state.position[0].detach().cpu().numpy() for v in controllers.values()])
                follow_cams.update(cam_state.position[0].detach().cpu().numpy(),
                                   cam_state.linear_velocity[0].detach().cpu().numpy(), physics_dt, points=cam_points)

            if trails and step % max(1, int(round(0.05 / physics_dt))) == 0:
                for c_name in trails:
                    trails[c_name].append(tuple(float(c) for c in controllers[c_name].state.position[0].detach().cpu()))

            if video is not None and step % video_every == 0:
                if trails:
                    _draw_lines_with_trails()
                video.capture(world.render, step, step * physics_dt)

            if telemetry is not None:
                tel_vehicle = _anchor["vehicle"]
                tel_state = tel_vehicle.state
                tel_pos = tel_state.position[0].detach().cpu()
                tel_goal = reset_manager.goal_pos[0].detach().cpu().to(tel_pos.dtype)
                tel_ep = tel_pos - tel_goal
                tel_vel = tel_state.linear_velocity[0].detach().cpu()
                tel_w = tel_state.angular_velocity.detach().cpu().reshape(-1, 3)[0]   # FLU body rates [rad/s]

                def _comp(key, names, vec, floor, ref=None):
                    # one panel per component, side by side in a row, each with its own colour
                    return [dict(key=f"{key}{i}", title=n, values=[float(vec[i])], fmt="{:6.1f}", floor=floor,
                                 symmetric=ref is None, autorange=ref is not None, color=i,
                                 ref=None if ref is None else [float(ref[i])]) for i, n in enumerate(names)]

                tel_rows = [
                    [dict(key="speed", title="V [m/s]", values=[float(tel_vel.norm())], floor=1.0),
                     dict(key="ep", title="||ep|| [m]", values=[float(tel_ep.norm())], floor=0.5)],
                    _comp("p", ("px [m]", "py [m]", "pz [m]"), tel_pos, 2.0, ref=tel_goal),
                    _comp("v", ("vx [m/s]", "vy [m/s]", "vz [m/s]"), tel_vel, 2.0),
                    _comp("w", ("wx [rad/s]", "wy [rad/s]", "wz [rad/s]"), tel_w, 0.5),
                ]
                tel_last = []
                tel_rotors = tel_vehicle._thrusters.velocity
                if tel_rotors is not None:
                    tel_rotors = 100.0 * (tel_rotors[0].detach().cpu() / tel_vehicle._thrusters.max_rotor_velocity.detach().cpu())
                    tel_last.append(dict(key="rotors", title="rotors [% max]", values=tel_rotors.tolist(),
                                         fmt="{:3.0f}", ymax=100.0))
                if getattr(tel_vehicle, "surface_deflections", None) is not None:
                    tel_surf = torch.rad2deg(tel_vehicle.surface_deflections[0].detach().cpu())
                    tel_last.append(dict(key="surfaces", title="elev/ail/rud [deg]", values=tel_surf.tolist(),
                                         fmt="{:5.1f}", floor=5.0, symmetric=True))
                if tel_last:
                    tel_rows.append(tel_last)
                telemetry.update(step * physics_dt, tel_rows)

            if args.pause_at is not None and not args.headless and step * physics_dt >= args.pause_at:
                print(f"[pause] t = {step * physics_dt:.2f} s: simulation paused, viewport live (close the window to exit).", flush=True)
                world.pause()
                if args.screenshot:
                    from cover_screenshot import take_screenshot
                    shot_size = tuple(int(v) for v in args.screenshot_size.lower().split("x"))
                    take_screenshot(simulation_app, args.screenshot, size=shot_size, pathtrace=args.screenshot_spp > 0,
                                    spp=args.screenshot_spp, overlay=telemetry)
                while simulation_app.is_running():
                    simulation_app.update()
                break

            if trajectory is not None:
                trajectory.step(physics_dt)
                _update_goal(trajectory)

            failed = audit_log.sample(controllers, reset_manager, step, physics_dt, args.failure_distance)
            timed_out = step % episode_steps == 0

            # Count this physics step before a failure or final timeout exits.
            if recorder is not None:
                done = torch.full((n_envs,), bool(failed or timed_out), dtype=torch.bool)
                clear = torch.zeros_like(done)
                recorder.advance(done if failed else clear, done if timed_out and not failed else clear)

            if failed:
                audit_log.end_episode(step * physics_dt, "comparison_stopped_on_failure")
                print("Evaluation stopped at the first failed condition; all comparison data retained.")
                break
            if timed_out:
                audit_log.end_episode(step * physics_dt, "timeout")
                if args.num_episodes_per_env > 0 and step // episode_steps >= args.num_episodes_per_env:
                    break
                for backend in backends_order:
                    backend.reset()
                _reset_episode()

                # Reset trajectory phase for all envs on episode end
                if trajectory is not None:
                    all_ids = torch.arange(n_envs, device=device)
                    trajectory.reset(all_ids)
                    _update_goal(trajectory)

                if marker_backend is not None:
                    marker_backend.update_goal_marker(reset_manager.goal_pos)

    except Exception:
        traceback.print_exc()
        raise
    finally:
        for backend in backends_order:
            if getattr(backend, "_px4_config", None) is not None:
                backend.stop()
        audit_log.close()
        if video is not None:
            video.close()
        for vehicle in controllers.values():
            vehicle.close()
        if recorder is not None:
            recorder.close()

        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
