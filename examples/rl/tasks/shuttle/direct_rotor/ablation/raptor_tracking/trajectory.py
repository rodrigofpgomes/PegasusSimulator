import torch

class RaptorLikeTrajectory:
    """Sample a fixed or isotropic Langevin reference for each vehicle.

    The Langevin branch follows rl-tools l2f (70_post_integration.h): a damped
    oscillator driven by Euler-Maruyama noise, whose velocity is low-pass
    filtered with weight alpha and integrated into the reference position.
    The episode counter selects the reference sample, which moves forward for
    the whole episode. Each reset draws a new reference. The reference acceleration
    is the finite difference of the filtered velocity (zero for a fixed reference).
    """

    def __init__(
        self,
        num_envs: int,
        episode_steps: int,
        dt: float,
        device: str,
        gamma: float = 1.0,
        omega: float = 2.0,
        sigma: float = 0.5,
        alpha: float = 0.01,
        mixture_langevin_prob: float = 0.5,
    ):
        self.num_envs = num_envs
        # One extra sample covers the terminal observation at the time limit
        self.episode_steps = episode_steps + 1
        self.dt = dt
        self.device = device

        self.gamma = gamma
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.mixture_langevin_prob = mixture_langevin_prob

        self.pos_traj = torch.zeros(num_envs, self.episode_steps, 3, device=device)
        self.vel_traj = torch.zeros(num_envs, self.episode_steps, 3, device=device)
        self.acc_traj = torch.zeros(num_envs, self.episode_steps, 3, device=device)

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """Draw new references around the spawn positions for selected vehicles."""
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        use_langevin = torch.rand(n, device=self.device) < self.mixture_langevin_prob

        # Vehicles outside the Langevin mixture track their fixed spawn point
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0
        self.acc_traj[env_ids] = 0.0

        if not use_langevin.any():
            return

        langevin_ids = env_ids[use_langevin]
        m = langevin_ids.numel()

        center = centers[langevin_ids]

        # Raw oscillator state and filtered reference, relative to the spawn point
        x_raw = torch.zeros(m, 3, device=self.device)
        v_raw = torch.zeros(m, 3, device=self.device)
        x_ref = torch.zeros(m, 3, device=self.device)
        v_ref = torch.zeros(m, 3, device=self.device)

        # The first sample starts at the vehicle spawn position
        self.pos_traj[langevin_ids, 0] = center
        self.vel_traj[langevin_ids, 0] = 0.0

        sqrt_dt = self.dt ** 0.5

        for t in range(1, self.episode_steps):
            # dv = (-gamma*v - omega^2*x) dt + sigma dW,  dW ~ N(0, dt)
            dW = sqrt_dt * torch.randn(m, 3, device=self.device)
            v_raw = v_raw + (-self.gamma * v_raw - self.omega ** 2 * x_raw) * self.dt + self.sigma * dW
            x_raw = x_raw + v_raw * self.dt

            # Exponential smoothing of the velocity, integrated into the reference
            v_new = self.alpha * v_raw + (1.0 - self.alpha) * v_ref
            self.acc_traj[langevin_ids, t] = (v_new - v_ref) / self.dt
            v_ref = v_new
            x_ref = x_ref + v_ref * self.dt

            self.pos_traj[langevin_ids, t] = center + x_ref
            self.vel_traj[langevin_ids, t] = v_ref

    def current(self, env_ids: torch.Tensor, step_ids: torch.Tensor):
        """Return the reference samples at the episode steps."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        index = step_ids.to(dtype=torch.long, device=self.device).clamp(0, self.episode_steps - 1)

        pos = self.pos_traj[env_ids, index]
        vel = self.vel_traj[env_ids, index]
        acc = self.acc_traj[env_ids, index]

        return pos, vel, acc


class LemniscateTrajectory:
    """Figure-eight reference around each vehicle's spawn point, for evaluation (same interface as above).

    The path is p(s) = c + A (sin s, 0.5 sin 2s, 0) with the phase s(t) driven by an angular speed omega that is
    ramped up from rest over ramp_s seconds, so the reference starts at the spawn point with zero velocity and
    acceleration. omega is set from the requested peak acceleration of the steady figure eight,
    a_peak = A omega^2 max_s |f''(s)| with f'' = (-sin s, -2 sin 2s), i.e. max |f''| = 2.12. Position, velocity and
    acceleration are analytic, the same for every vehicle apart from the spawn point.
    """

    def __init__(self, num_envs: int, episode_steps: int, dt: float, device: str,
                 amplitude: float = 1.0, peak_acc: float = 2.0, ramp_s: float = 2.0):
        self.num_envs = num_envs
        self.episode_steps = episode_steps + 1      # one extra sample covers the terminal observation
        self.dt, self.device = dt, device
        grid = torch.linspace(0.0, 2.0 * torch.pi, 20001, dtype=torch.float64)
        f2_max = torch.sqrt(torch.sin(grid) ** 2 + (2.0 * torch.sin(2.0 * grid)) ** 2).max().item()
        self.omega = (peak_acc / (amplitude * f2_max)) ** 0.5
        self.amplitude, self.peak_acc, self.ramp_s = amplitude, peak_acc, ramp_s

        t = torch.arange(self.episode_steps, dtype=torch.float64) * dt
        w, r = self.omega, max(ramp_s, 1e-6)
        ramp = t < r
        # phase and its derivatives: speed rises from 0 to omega as 0.5 omega (1 - cos(pi t / r)), then stays at omega
        s = torch.where(ramp, w * (t / 2 - r / (2 * torch.pi) * torch.sin(torch.pi * t / r)), w * (t - r / 2))
        sd = torch.where(ramp, 0.5 * w * (1 - torch.cos(torch.pi * t / r)), torch.full_like(t, w))
        sdd = torch.where(ramp, 0.5 * w * (torch.pi / r) * torch.sin(torch.pi * t / r), torch.zeros_like(t))
        A, zero = amplitude, torch.zeros_like(t)
        pos = A * torch.stack([torch.sin(s), 0.5 * torch.sin(2 * s), zero], dim=1)
        vel = A * torch.stack([torch.cos(s) * sd, torch.cos(2 * s) * sd, zero], dim=1)
        acc = A * torch.stack([-torch.sin(s) * sd ** 2 + torch.cos(s) * sdd,
                               -2 * torch.sin(2 * s) * sd ** 2 + torch.cos(2 * s) * sdd, zero], dim=1)
        self.rel_pos, self.vel_traj, self.acc_traj = (x.float().to(device) for x in (pos, vel, acc))
        self.pos_traj = torch.zeros(num_envs, self.episode_steps, 3, device=device)

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """Place the figure eight around the spawn points of the selected vehicles."""
        if env_ids.numel() == 0:
            return
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1) + self.rel_pos.unsqueeze(0)

    def current(self, env_ids: torch.Tensor, step_ids: torch.Tensor):
        """Return the reference samples at the episode steps."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        index = step_ids.to(dtype=torch.long, device=self.device).clamp(0, self.episode_steps - 1)
        return self.pos_traj[env_ids, index], self.vel_traj[index], self.acc_traj[index]
