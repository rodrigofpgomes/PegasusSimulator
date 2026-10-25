import torch

class RaptorLikeTrajectory:
    """Sample a fixed or isotropic Langevin reference for each vehicle.

    The Langevin branch follows rl-tools l2f (70_post_integration.h): a damped
    oscillator driven by Euler-Maruyama noise, whose velocity is low-pass
    filtered with weight alpha and integrated into the reference position.
    The episode counter selects the reference sample, which moves forward for
    the whole episode. Each reset draws a new reference.
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

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """Draw new references around the spawn positions for selected vehicles."""
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        use_langevin = torch.rand(n, device=self.device) < self.mixture_langevin_prob

        # Vehicles outside the Langevin mixture track their fixed spawn point
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0

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
            v_ref = self.alpha * v_raw + (1.0 - self.alpha) * v_ref
            x_ref = x_ref + v_ref * self.dt

            self.pos_traj[langevin_ids, t] = center + x_ref
            self.vel_traj[langevin_ids, t] = v_ref

    def current(self, env_ids: torch.Tensor, step_ids: torch.Tensor):
        """Return the reference samples at the episode steps."""
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        index = step_ids.to(dtype=torch.long, device=self.device).clamp(0, self.episode_steps - 1)

        pos = self.pos_traj[env_ids, index]
        vel = self.vel_traj[env_ids, index]

        return pos, vel
