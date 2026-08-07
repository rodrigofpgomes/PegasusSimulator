import torch

import math


class LissajousTrajectory:
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
        period: float,
        z_ref: float,
        amplitude: float,
        center: torch.Tensor,
        mixture_langevin_prob: float = 0.5,
    ):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.dt = dt
        self.device = device

        self.A = float(amplitude)
        self.w = 2.0 * math.pi / float(period)  # angular frequency [rad/s]
        self.z_ref = float(z_ref)

        self.mixture_langevin_prob = mixture_langevin_prob

        self.pos_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.vel_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.use_traj = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """
        centers: (num_envs, 3), normalmente vehicle._init_pos.
        """
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        self.step_counter[env_ids] = 0

        use_traj = torch.rand(n, device=self.device) < self.mixture_langevin_prob
        self.use_traj[env_ids] = use_traj

        # default/null trajectory: referência fixa no spawn
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0

        if not use_traj.any():
            return

        traj_ids = env_ids[use_traj]
        m = traj_ids.numel()

        x = centers[traj_ids].clone()
        v = torch.zeros(m, 3, device=self.device)

        # Guarda step 0
        self.pos_traj[traj_ids, 0] = x
        self.vel_traj[traj_ids, 0] = v

        center = centers[traj_ids]

        phi = torch.rand(m, device=self.device) * (2 * torch.pi) # random phase offset

        for t in range(1, self.episode_steps):
            
            wt = torch.tensor(self.w * t * self.dt, device=self.device)
            sin_wt  = torch.sin(wt)
            cos_wt  = torch.cos(wt)
            cos_2wt = torch.cos(2.0 * wt)

            pos = center.clone()
            pos[:, 0] += self.A * sin_wt
            pos[:, 1] += 0.5 * self.A * torch.sin(2.0 * wt)
            # z already set to z_ref in center

            vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            vel[:, 0] = self.A * self.w * cos_wt
            vel[:, 1] = self.A * self.w * cos_2wt
            vel[:, 2] = 0.0

            self.pos_traj[traj_ids, t] = pos
            self.vel_traj[traj_ids, t] = vel

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

        vel = torch.where(forward.unsqueeze(1), vel, -vel)

        return pos, vel

    def advance(self):
        self.step_counter += 1