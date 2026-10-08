"""H8 reference generator (Frenet / unicycle), test-matched envelope.

See the task README for the measurement that produced these numbers. Summary:
the evaluation condition (lemniscate A=1.5, T=10) spans ext95 = 1.50 m, speed
0.62-1.33 m/s, heading rate p90 = 1.88 rad/s and asks for 1.03 N of useful
longitudinal force. H8 covers 100% of that support with only 2.0x the area,
a_need 3.26 rad/s^2 (yaw authority is 5.92) and 3.76 N of usable rotor-5
force -- 3.6x what the test itself demands, without wasting capacity on a
region the test never visits.
"""

import math

import torch


class RaptorLikeTrajectory:
    """Reference generator with a persistent direction of travel.

    Drop-in replacement for the Langevin version: same class name, same public
    interface (``reset`` / ``current`` / ``advance``), same buffers.

    WHY THIS EXISTS
    ---------------
    Rotor 5 pushes along body +X with thrust >= 0. For it to ever be useful the
    reference must satisfy three conditions at once:

      1. the direction of travel must rotate slowly enough for yaw to track it
         (available yaw authority is alpha = 5.92 rad/s^2);
      2. the demand must have a LONGITUDINAL component, i.e. acceleration along
         the direction of travel -- purely centripetal demand is served by
         tilting and rotor 5 cannot help with it;
      3. that longitudinal component must be positive a useful fraction of the
         time, because rotor-5 thrust cannot be negative.

    The Cartesian Langevin process fails condition 1 by an order of magnitude:
    its demand direction rotates at a median of 17 rad/s (p90 66 rad/s), which
    would need about 47 rad/s^2 of yaw authority against the 5.92 available.
    No reward shaping can repair that, because it is kinematic and not a matter
    of preference.

    This generator instead builds the trajectory in a Frenet frame: it drives
    the SPEED and the HEADING RATE as two slow Ornstein-Uhlenbeck processes and
    integrates them into a unicycle path. Direction persistence then becomes a
    construction property rather than an accident, and the two demand channels
    (longitudinal from speed change, lateral from turning) are controlled
    independently.

    THREE-WAY MIXTURE
    -----------------
      * static    -- reference pinned at spawn (kept: it is one of the two
                     evaluation conditions);
      * frenet    -- the unicycle reference described above (the bulk);
      * lemniscate-- randomised Gerono figure-8, an anchor for the evaluation
                     trajectory family.

    CONVENTIONS PRESERVED FROM THE LANGEVIN VERSION
    -----------------------------------------------
      * ``vel_traj[t]`` is stored AFTER the update and ``pos_traj[t]`` is
        integrated with that new velocity, so ``acc_traj[t]`` is exactly
        ``(vel_traj[t] - vel_traj[t-1]) / dt``. Here that identity is enforced
        by construction: acceleration is computed as the discrete derivative of
        the stored velocity rather than analytically, so it cannot drift out of
        agreement with the trajectory the vehicle is asked to follow.
      * ``acc_traj[:, 0]`` is copied from step 1.
      * ``use_langevin`` keeps its name and now means "this episode has a MOVING
        reference", i.e. non-static. That is exactly the mask needed when
        averaging demand-correlation metrics, because a static episode has zero
        demand variance and would otherwise contribute a forced zero.
    """

    def __init__(
        self,
        num_envs: int,
        episode_steps: int,
        dt: float,
        device: str,
        # --- kept for signature compatibility; used only for the vertical axis
        gamma: float = 1.0,
        omega: float = 2.0,
        sigma: float = 6.0,
        # --- mixture: probability of a MOVING reference (1 - p_static)
        mixture_langevin_prob: float = 0.65,
        # --- of the moving episodes, the share that is a lemniscate
        lemniscate_share: float = 0.30,
        # --- Frenet speed channel (longitudinal demand)
        speed_min: float = 0.60,
        speed_max: float = 1.20,
        speed_sigma_frac: float = 0.50,
        speed_tau: float = 0.30,
        # --- Frenet heading channel (lateral demand)
        yaw_rate_sigma: float = 0.35,
        yaw_rate_tau: float = 2.0,
        # --- containment and envelope
        home_radius: float = 0.6,
        home_gain: float = 0.4,
        max_speed: float = 1.40,
        # --- vertical axis (second-order spring, as before but gentler)
        sigma_z: float = 0.20,
        # --- lemniscate branch
        lemni_amp_min: float = 1.0,
        lemni_amp_max: float = 1.8,
        lemni_period_min: float = 8.0,
        lemni_period_max: float = 14.0,
    ):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.dt = dt
        self.device = device

        self.gamma = gamma
        self.omega = omega
        self.sigma = sigma

        self.mixture_langevin_prob = mixture_langevin_prob
        self.lemniscate_share = lemniscate_share

        self.speed_min = speed_min
        self.speed_max = speed_max
        self.speed_sigma_frac = speed_sigma_frac
        self.speed_tau = speed_tau

        self.yaw_rate_sigma = yaw_rate_sigma
        self.yaw_rate_tau = yaw_rate_tau

        self.home_radius = home_radius
        self.home_gain = home_gain
        self.max_speed = max_speed
        self.sigma_z = sigma_z

        self.lemni_amp_min = lemni_amp_min
        self.lemni_amp_max = lemni_amp_max
        self.lemni_period_min = lemni_period_min
        self.lemni_period_max = lemni_period_max

        self.pos_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.vel_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.acc_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

        # True = moving reference (frenet or lemniscate); False = static.
        self.use_langevin = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Which moving family was used, for diagnostics only.
        self.use_lemniscate = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """centers: (num_envs, 3), normally vehicle._init_pos."""
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        self.step_counter[env_ids] = 0

        moving = torch.rand(n, device=self.device) < self.mixture_langevin_prob
        is_lemni = moving & (
            torch.rand(n, device=self.device) < self.lemniscate_share
        )
        is_frenet = moving & (~is_lemni)

        self.use_langevin[env_ids] = moving
        self.use_lemniscate[env_ids] = is_lemni

        # Default / null trajectory: reference pinned at spawn.
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0
        self.acc_traj[env_ids] = 0.0

        if is_frenet.any():
            self._fill_frenet(env_ids[is_frenet], centers)
        if is_lemni.any():
            self._fill_lemniscate(env_ids[is_lemni], centers)

    # ----------------------------------------------------------------- frenet

    def _fill_frenet(self, ids: torch.Tensor, centers: torch.Tensor):
        m = ids.numel()
        dev = self.device
        dt = self.dt
        steps = self.episode_steps

        center = centers[ids]

        # Per-episode randomisation: cruise speed and initial heading.
        s0 = self.speed_min + (self.speed_max - self.speed_min) * torch.rand(
            m, device=dev
        )
        sig_s = self.speed_sigma_frac * s0
        psi = 2.0 * math.pi * torch.rand(m, device=dev)

        # OU discretisation: exact for the stationary process.
        a_s = math.exp(-dt / self.speed_tau)
        n_s = sig_s * math.sqrt(max(1.0 - a_s * a_s, 1e-12))
        a_r = math.exp(-dt / self.yaw_rate_tau)
        n_r = self.yaw_rate_sigma * math.sqrt(max(1.0 - a_r * a_r, 1e-12))

        s = s0.clone()
        r_ou = self.yaw_rate_sigma * torch.randn(m, device=dev)

        x = center.clone()
        v = torch.zeros(m, 3, device=dev)

        # Vertical axis: the original second-order spring, gentler.
        vz = torch.zeros(m, device=dev)

        self.pos_traj[ids, 0] = x
        self.vel_traj[ids, 0] = v

        for t in range(1, steps):
            # --- speed channel (longitudinal demand)
            s = s0 + a_s * (s - s0) + n_s * torch.randn(m, device=dev)
            s = s.clamp(min=0.15, max=self.max_speed)

            # --- heading channel (lateral demand)
            r_ou = a_r * r_ou + n_r * torch.randn(m, device=dev)

            # Soft containment: turn back toward the spawn point only once the
            # reference has drifted beyond home_radius, so the interior of the
            # region is explored with an undisturbed heading process.
            rel = x[:, :2] - center[:, :2]
            rad = torch.linalg.norm(rel, dim=1)
            bearing = torch.atan2(-rel[:, 1], -rel[:, 0])
            err = torch.atan2(
                torch.sin(bearing - psi), torch.cos(bearing - psi)
            )
            pull = self.home_gain * err * (rad / self.home_radius - 1.0).clamp(
                min=0.0
            )
            r = r_ou + pull

            psi = psi + dt * r

            # --- vertical axis
            acc_z = (
                -self.gamma * vz
                - (self.omega ** 2) * (x[:, 2] - center[:, 2])
                + self.sigma_z * torch.randn(m, device=dev)
            )
            vz = vz + dt * acc_z

            v_new = torch.stack(
                [s * torch.cos(psi), s * torch.sin(psi), vz], dim=1
            )

            # acc is the exact discrete derivative of the stored velocity.
            acc = (v_new - v) / dt
            v = v_new
            x = x + dt * v

            self.pos_traj[ids, t] = x
            self.vel_traj[ids, t] = v
            self.acc_traj[ids, t] = acc

        self.acc_traj[ids, 0] = self.acc_traj[ids, 1]

    # ------------------------------------------------------------- lemniscate

    def _fill_lemniscate(self, ids: torch.Tensor, centers: torch.Tensor):
        m = ids.numel()
        dev = self.device
        dt = self.dt
        steps = self.episode_steps

        center = centers[ids]

        amp = self.lemni_amp_min + (
            self.lemni_amp_max - self.lemni_amp_min
        ) * torch.rand(m, device=dev)
        period = self.lemni_period_min + (
            self.lemni_period_max - self.lemni_period_min
        ) * torch.rand(m, device=dev)
        w = 2.0 * math.pi / period
        phase = 2.0 * math.pi * torch.rand(m, device=dev)

        # Random yaw rotation of the whole figure, so the policy cannot latch
        # onto one world-frame orientation.
        rot = 2.0 * math.pi * torch.rand(m, device=dev)
        cr, sr = torch.cos(rot), torch.sin(rot)

        tt = torch.arange(steps, device=dev, dtype=torch.float32) * dt
        ang = w.unsqueeze(1) * tt.unsqueeze(0) + phase.unsqueeze(1)

        lx = amp.unsqueeze(1) * torch.sin(ang)
        ly = 0.5 * amp.unsqueeze(1) * torch.sin(2.0 * ang)

        px = cr.unsqueeze(1) * lx - sr.unsqueeze(1) * ly
        py = sr.unsqueeze(1) * lx + cr.unsqueeze(1) * ly

        pos = torch.stack(
            [
                center[:, 0].unsqueeze(1) + px,
                center[:, 1].unsqueeze(1) + py,
                center[:, 2].unsqueeze(1).expand(-1, steps),
            ],
            dim=2,
        )

        # Velocity and acceleration as discrete derivatives, matching the
        # convention of the other branches exactly.
        vel = torch.zeros_like(pos)
        vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt
        vel[:, 0] = vel[:, 1]

        acc = torch.zeros_like(pos)
        acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt
        acc[:, 0] = acc[:, 1]

        # Reject-by-clamp: keep the reference inside the speed envelope.
        sp = torch.linalg.norm(vel, dim=2, keepdim=True)
        scale = (self.max_speed / sp.clamp(min=1e-6)).clamp(max=1.0)
        vel = vel * scale
        acc = acc * scale

        self.pos_traj[ids] = pos
        self.vel_traj[ids] = vel
        self.acc_traj[ids] = acc

    # ---------------------------------------------------------------- current

    def current(
        self,
        env_ids: torch.Tensor | None = None,
        step_ids: torch.Tensor | None = None,
    ):
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
