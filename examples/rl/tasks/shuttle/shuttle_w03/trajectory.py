import torch

class RaptorLikeTrajectory:
    """
    Aproxima a trajetória RAPTOR:
      - mixture 50/50: null trajectory ou Langevin-like
      - duração 500 steps
      - replay ping-pong: forward, depois backward com vel invertida
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
        #alpha: float = 0.01,
        mixture_langevin_prob: float = 0.5,
    ):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.dt = dt
        self.device = device

        self.gamma = gamma
        self.omega = omega
        self.sigma = sigma
        #self.alpha = alpha
        self.mixture_langevin_prob = mixture_langevin_prob

        self.pos_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.vel_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.acc_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.use_langevin = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """
        centers: (num_envs, 3), normalmente vehicle._init_pos.
        """
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        self.step_counter[env_ids] = 0

        use_langevin = torch.rand(n, device=self.device) < self.mixture_langevin_prob
        self.use_langevin[env_ids] = use_langevin

        # default/null trajectory: referência fixa no spawn
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0
        self.acc_traj[env_ids] = 0.0

        if not use_langevin.any():
            return

        langevin_ids = env_ids[use_langevin]
        m = langevin_ids.numel()

        x = centers[langevin_ids].clone()
        v = torch.zeros(m, 3, device=self.device)

        # Guarda step 0
        self.pos_traj[langevin_ids, 0] = x
        self.vel_traj[langevin_ids, 0] = v

        # Langevin-like second-order process
        # x_dot = v
        # v_dot = -gamma*v - omega^2*(x-center) + sigma*noise
        center = centers[langevin_ids]

        for t in range(1, self.episode_steps):
            noise = torch.randn(m, 3, device=self.device)

            acc = (
                -self.gamma * v
                - (self.omega ** 2) * (x - center)
                + self.sigma * noise
            )

            v = v + self.dt * acc
            x = x + self.dt * v

            self.pos_traj[langevin_ids, t] = x
            self.vel_traj[langevin_ids, t] = v
            self.acc_traj[langevin_ids, t] = acc

        self.acc_traj[langevin_ids, 0] = self.acc_traj[langevin_ids, 1]

    def current(self, env_ids: torch.Tensor | None = None, step_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        if step_ids is None:
            full_step = self.step_counter[env_ids]
        else:
            full_step = step_ids.to(dtype=torch.long, device=self.device)

        interval = full_step // self.episode_steps
        progress = full_step % self.episode_steps

        forward = (interval % 2) == 0
        index = torch.where(forward, progress, self.episode_steps - progress - 1)

        pos = self.pos_traj[env_ids, index]
        vel = self.vel_traj[env_ids, index]
        acc = self.acc_traj[env_ids, index]

        vel = torch.where(forward.unsqueeze(1), vel, -vel)

        return pos, vel, acc

    def advance(self):
        self.step_counter += 1