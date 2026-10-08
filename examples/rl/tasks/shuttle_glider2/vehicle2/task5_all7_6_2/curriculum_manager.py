"""
| File: curriculum_manager.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Performance-based curriculum for the Shuttle + EasyGlider.
|
| Each level defines, in one place: initial-state randomisation bounds, the
| cruise-start probability, the airspeed / turn distribution of the reference,
| a nose-up trim angle-of-attack, the aerodynamic reward-shaping weights, and
| the progression thresholds (evaluated only on current-level episodes).
|
| The RL agent is UNAWARE of this class; it only reshapes the task distribution
| and the reset randomisation used by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CurriculumLevel:
    name: str

    # Initial-state randomisation (perturbation / hover-start envs)
    max_position: tuple[float, float, float]
    max_linear_velocity: float
    max_angular_velocity: float
    max_angle_deg: float

    # Reference distribution
    cruise_prob: float          # fraction of moving envs that start AT cruise
    speed_min: float
    speed_max: float
    turn_probability: float
    turn_radius_min: float
    turn_radius_max: float
    vertical_amp: float         # mild altitude oscillation amplitude [m]
    vertical_period: float      # [s]
    alpha_trim_deg: float       # nose-up trim AoA used for cruise starts

    # Aerodynamic reward shaping
    w_heading: float
    w_beta: float
    w_alpha: float
    w_quad_effort: float

    # Progression thresholds on raw tracking error [m] and [m/s]
    pos_error_threshold: float
    vel_error_threshold: float
    success_rate_threshold: float

    # Anti-forgetting rehearsal (continual-learning mixture). With probability
    # `rehearsal_prob` the reference speed is drawn from the FULL mastered envelope
    # [rehearsal_min, speed_max] instead of the current focus band
    # [speed_min, speed_max]. This keeps the bulk of samples on the new regime
    # (so the new task is still learned) while a minority revisits slower speeds
    # (so earlier regimes and the accelerate/decelerate transitions are not lost).
    rehearsal_prob: float = 0.25
    rehearsal_min: float = 0.0


# Speeds ramp up to the ~12 m/s wing-offload regime; alpha_trim and the aero
# shaping weights grow with speed; turns only appear once forward flight is solid.
DEFAULT_LEVELS = (
    CurriculumLevel(name="hover",       max_position=(0.02, 0.02, 0.01), max_linear_velocity=0.05, max_angular_velocity=0.05, max_angle_deg=2.0,
                    cruise_prob=0.00, speed_min=0.0, speed_max=0.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=0.0,
                    w_heading=0.00, w_beta=0.00, w_alpha=0.00, w_quad_effort=0.00,
                    pos_error_threshold=0.15, vel_error_threshold=0.20, success_rate_threshold=0.95),
    CurriculumLevel(name="fwd_2",       max_position=(0.05, 0.05, 0.03), max_linear_velocity=0.10, max_angular_velocity=0.10, max_angle_deg=5.0,
                    cruise_prob=0.30, speed_min=2.0, speed_max=3.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=3.0,
                    w_heading=0.10, w_beta=0.05, w_alpha=0.02, w_quad_effort=0.01,
                    pos_error_threshold=0.20, vel_error_threshold=0.30, success_rate_threshold=0.93),
    CurriculumLevel(name="fwd_4",       max_position=(0.08, 0.08, 0.04), max_linear_velocity=0.20, max_angular_velocity=0.15, max_angle_deg=8.0,
                    cruise_prob=0.40, speed_min=3.5, speed_max=5.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=5.0,
                    w_heading=0.20, w_beta=0.10, w_alpha=0.03, w_quad_effort=0.02,
                    pos_error_threshold=0.25, vel_error_threshold=0.40, success_rate_threshold=0.92),
    CurriculumLevel(name="fwd_6",       max_position=(0.12, 0.12, 0.06), max_linear_velocity=0.30, max_angular_velocity=0.20, max_angle_deg=10.0,
                    cruise_prob=0.50, speed_min=5.0, speed_max=7.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=7.0,
                    w_heading=0.30, w_beta=0.15, w_alpha=0.05, w_quad_effort=0.03,
                    pos_error_threshold=0.30, vel_error_threshold=0.50, success_rate_threshold=0.90),
    CurriculumLevel(name="fwd_8",       max_position=(0.15, 0.15, 0.08), max_linear_velocity=0.40, max_angular_velocity=0.25, max_angle_deg=12.0,
                    cruise_prob=0.60, speed_min=7.0, speed_max=9.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=9.0,
                    w_heading=0.40, w_beta=0.20, w_alpha=0.06, w_quad_effort=0.08,
                    pos_error_threshold=0.35, vel_error_threshold=0.60, success_rate_threshold=0.90),
    CurriculumLevel(name="fwd_10",      max_position=(0.20, 0.20, 0.10), max_linear_velocity=0.50, max_angular_velocity=0.30, max_angle_deg=14.0,
                    cruise_prob=0.70, speed_min=9.0, speed_max=11.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.0, vertical_period=1.0, alpha_trim_deg=11.0,
                    w_heading=0.50, w_beta=0.25, w_alpha=0.08, w_quad_effort=0.10,
                    pos_error_threshold=0.45, vel_error_threshold=0.75, success_rate_threshold=0.88),
    CurriculumLevel(name="fwd_12_vert", max_position=(0.25, 0.25, 0.12), max_linear_velocity=0.60, max_angular_velocity=0.35, max_angle_deg=15.0,
                    cruise_prob=0.70, speed_min=10.0, speed_max=13.0, turn_probability=0.00, turn_radius_min=0.0, turn_radius_max=0.0,
                    vertical_amp=0.5, vertical_period=10.0, alpha_trim_deg=12.0,
                    w_heading=0.55, w_beta=0.30, w_alpha=0.10, w_quad_effort=0.12,
                    pos_error_threshold=0.55, vel_error_threshold=0.90, success_rate_threshold=0.88),
    CurriculumLevel(name="wide_turns",  max_position=(0.25, 0.25, 0.12), max_linear_velocity=0.60, max_angular_velocity=0.40, max_angle_deg=18.0,
                    cruise_prob=0.60, speed_min=8.0, speed_max=11.0, turn_probability=0.40, turn_radius_min=30.0, turn_radius_max=50.0,
                    vertical_amp=0.4, vertical_period=10.0, alpha_trim_deg=10.0,
                    w_heading=0.60, w_beta=0.35, w_alpha=0.10, w_quad_effort=0.12,
                    pos_error_threshold=0.65, vel_error_threshold=1.00, success_rate_threshold=0.86),
    CurriculumLevel(name="turns",       max_position=(0.30, 0.30, 0.15), max_linear_velocity=0.70, max_angular_velocity=0.50, max_angle_deg=22.0,
                    cruise_prob=0.60, speed_min=9.0, speed_max=12.0, turn_probability=0.60, turn_radius_min=20.0, turn_radius_max=35.0,
                    vertical_amp=0.4, vertical_period=10.0, alpha_trim_deg=11.0,
                    w_heading=0.65, w_beta=0.40, w_alpha=0.12, w_quad_effort=0.12,
                    pos_error_threshold=0.75, vel_error_threshold=1.20, success_rate_threshold=0.85),
    # -------------------------------------------------------------------------
    # Terminal CONSOLIDATION level. Once "turns" is mastered the focus band is
    # WIDENED to the entire mastered envelope (hover -> full speed, straight +
    # turns + mild climbs) so the final policy is jointly optimised over
    # everything it has seen, instead of over-fitting the hardest regime. This
    # is a stationary joint-training phase: no level follows it, so its
    # thresholds are used ONLY for the good_windows log, never for promotion.
    # Aero weights are the top-level values; forward_gate scales them down at
    # low speed automatically, so a single set is correct across the band.
    CurriculumLevel(name="consolidate", max_position=(0.30, 0.30, 0.15), max_linear_velocity=0.70, max_angular_velocity=0.50, max_angle_deg=22.0,
                    cruise_prob=0.50, speed_min=0.0, speed_max=13.0, turn_probability=0.45, turn_radius_min=20.0, turn_radius_max=60.0,
                    vertical_amp=0.4, vertical_period=10.0, alpha_trim_deg=8.0,
                    w_heading=0.65, w_beta=0.40, w_alpha=0.12, w_quad_effort=0.08,
                    pos_error_threshold=0.60, vel_error_threshold=1.00, success_rate_threshold=0.88,
                    rehearsal_prob=0.25, rehearsal_min=0.0),
)


class CurriculumManager:
    """Performance-based curriculum shared by all parallel environments."""

    def __init__(
        self,
        device: str,
        levels: tuple[CurriculumLevel, ...] = DEFAULT_LEVELS,
        start_level: int = 0,
        evaluation_episodes: int = 2048,
        required_consecutive_windows: int = 3,
        previous_level_prob: float = 0.20,
        current_level_prob: float = 0.80,
        next_level_prob: float = 0.00,
    ):
        if not levels:
            raise ValueError("Curriculum requires at least one level")
        if evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be > 0")
        if required_consecutive_windows <= 0:
            raise ValueError("required_consecutive_windows must be > 0")

        probs = previous_level_prob + current_level_prob + next_level_prob
        if abs(probs - 1.0) > 1e-6:
            raise ValueError("previous/current/next probabilities must sum to 1")

        self.device = device
        self.levels = levels
        self.max_level = len(levels) - 1
        self.current_level = int(max(0, min(start_level, self.max_level)))
        self.evaluation_episodes = int(evaluation_episodes)
        self.required_consecutive_windows = int(required_consecutive_windows)

        self.previous_level_prob = float(previous_level_prob)
        self.current_level_prob = float(current_level_prob)
        self.next_level_prob = float(next_level_prob)

        self._good_windows = 0
        self._window_episodes = 0
        self._window_pos_error_sum = 0.0
        self._window_vel_error_sum = 0.0
        self._window_success_sum = 0.0

        self.last_window_pos_error = float("nan")
        self.last_window_vel_error = float("nan")
        self.last_window_success_rate = float("nan")
        self.last_level_changed = False

    @property
    def level(self) -> CurriculumLevel:
        return self.levels[self.current_level]

    @property
    def good_windows(self) -> int:
        return self._good_windows

    def sample_levels(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if self.current_level == 0 and self.max_level == 0:
            return torch.zeros(n, dtype=torch.long, device=self.device)

        p_prev = self.previous_level_prob if self.current_level > 0 else 0.0
        p_next = self.next_level_prob if self.current_level < self.max_level else 0.0
        p_curr = 1.0 - p_prev - p_next

        u = torch.rand(n, device=self.device)
        sampled = torch.full((n,), self.current_level, dtype=torch.long, device=self.device)
        if p_prev > 0.0:
            sampled[u < p_prev] = self.current_level - 1
        if p_next > 0.0:
            sampled[u >= (p_prev + p_curr)] = self.current_level + 1
        return sampled

    def observe_episodes(self, mean_pos_error: torch.Tensor, mean_vel_error: torch.Tensor, success: torch.Tensor) -> bool:
        if mean_pos_error.numel() == 0:
            self.last_level_changed = False
            return False

        mean_pos_error = mean_pos_error.detach().float().reshape(-1)
        mean_vel_error = mean_vel_error.detach().float().reshape(-1)
        success = success.detach().float().reshape(-1)

        n = mean_pos_error.numel()
        self._window_episodes += n
        self._window_pos_error_sum += mean_pos_error.sum().item()
        self._window_vel_error_sum += mean_vel_error.sum().item()
        self._window_success_sum += success.sum().item()
        self.last_level_changed = False

        if self._window_episodes < self.evaluation_episodes:
            return False

        self.last_window_pos_error = self._window_pos_error_sum / self._window_episodes
        self.last_window_vel_error = self._window_vel_error_sum / self._window_episodes
        self.last_window_success_rate = self._window_success_sum / self._window_episodes

        spec = self.level
        passed = (
            self.last_window_pos_error <= spec.pos_error_threshold
            and self.last_window_vel_error <= spec.vel_error_threshold
            and self.last_window_success_rate >= spec.success_rate_threshold
        )
        self._good_windows = self._good_windows + 1 if passed else 0

        if self._good_windows >= self.required_consecutive_windows and self.current_level < self.max_level:
            self.current_level += 1
            self._good_windows = 0
            self.last_level_changed = True

        self._window_episodes = 0
        self._window_pos_error_sum = 0.0
        self._window_vel_error_sum = 0.0
        self._window_success_sum = 0.0
        return self.last_level_changed
