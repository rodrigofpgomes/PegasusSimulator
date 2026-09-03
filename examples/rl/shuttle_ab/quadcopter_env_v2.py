"""
| File: quadcopter_env_v2.py
| Env instrumentado para as experiencias A/B/C do 5.o rotor do shuttle_glider.
|
| Baseado no teu quadcopter_env.py. Nao muda a arquitetura: muda o que estava a
| impedir o rotor 5 de ter valor e o que estava a matar o gradiente no env de
| heading. Tudo o que e' novo esta' atras de flags, e os defaults reproduzem o
| baseline de 28D que ja' funciona.
|
| Flags principais (ver QuadcopterEnvCfg):
|   rotor5_mode  : "active" | "dead" | "removed"      -> ablacao limpa
|   obs_layout   : "world_error" | "body_heading"
|   reward_mode  : "baseline" | "heading"
|   clip_mode    : "legacy" | "shaping_only"          -> a correcao do clip
|
| Presets prontos no fim do ficheiro: cfg_E1(), cfg_E3().
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from pegasus.simulator.logic.rl.base_env import PegasusEnv, PegasusEnvCfg
from pegasus.simulator.logic.rl.reset_manager import InitStateCfg
from pegasus.simulator.logic.transforms import quaternion_to_matrix

from .eval_recorder import EpisodeRecorder
from .trajectory_v2 import RaptorLikeTrajectory


# ---------------------------------------------------------------------------
# Configuracao fisica
# ---------------------------------------------------------------------------
def shuttle_glider_physics_cfg(max_w5: float = 3500.0) -> dict:
    """Fabrica da config fisica. NOTA: rotor_axes_body e' explicito.

    O teu env baseline de 28D nao tinha esta chave. Se o simulador assumiu
    [0,0,1] por omissao, o rotor 5 estava a empurrar para CIMA a' frente do CoM
    (105 N + momento de pitch enorme) e a policy zera-lo era racional.
    """
    return {
        "num_rotors": 5,
        "rotor_constant": [1.709716e-05] * 4 + [8.54858e-06],
        "rolling_moment_coefficient": [1e-06] * 4 + [0.0],
        "rot_dir": [-1, -1, 1, 1, 1],
        "min_rotor_velocity": [0, 0, 0, 0, 0],
        "max_rotor_velocity": [1400, 1400, 1400, 1400, max_w5],
        "motor_time_constant": [0.008] * 4 + [0.0125],
        "rotor_axes_body": [[0.0, 0.0, 1.0]] * 4 + [[1.0, 0.0, 0.0]],
    }


VERTICAL_TAIL_PHYSICS_CFG = {
    "S_vtail": 0.022078,
    "CL_beta": 3.0,
    "CL_max": 0.8,
    "CD0": 0.03,
    "induced_k": 0.10,
    "CD_crossflow": 1.0,
    "prop_radius": 0.13,
    "wake_factor": 1.5,
    "coverage_factor": 0.3,
    "p_com": [-0.06945947, 0.0, 0.10253066],
    "p_vtail": [-0.848256, 0.0, -0.093542],
    "rho": 1.225,
    "forward_gate_v0": 0.5,
    "forward_gate_k": 0.15,
    "min_airspeed": 0.2,
}


class QuadcopterEnvCfg(PegasusEnvCfg):
    # -- simulacao
    sim_dt = 0.01
    decimation = 1
    episode_length_s = 5.0

    # -- espacos (derivados em __post_init__)
    action_space = 5
    observation_space = 28

    physics_cfg = shuttle_glider_physics_cfg()
    vertical_tail_physics_cfg = VERTICAL_TAIL_PHYSICS_CFG

    # ------------------------------------------------------------------
    # EXPERIMENTOS
    # ------------------------------------------------------------------
    # "active"  : 5 acoes, rotor 5 controlado pela policy.
    # "dead"    : 5 acoes, canal 5 forcado a -1 ANTES da reward e da obs.
    #             (a tua versao antiga zerava so' o omega no _apply_action,
    #              deixando 3 canais parasitas: custo d_action fantasma, ruido
    #              realimentado na obs, e sumidouro de entropia no SAC.)
    # "removed" : 4 acoes. A policy nao tem a dimensao morta. CONTROLO LIMPO.
    rotor5_mode = "active"

    obs_layout = "world_error"      # "world_error" | "body_heading"
    reward_mode = "baseline"        # "baseline" | "heading"
    use_heading_obs = None          # None -> segue reward_mode == "heading"

    # ------------------------------------------------------------------
    # A CORRECAO DO CLIP
    # ------------------------------------------------------------------
    # "legacy"       : cost = clip(w_pos*pos + tudo o resto, max=cost_clip).
    #                  Reproduz o teu env exatamente. Com os termos de heading
    #                  somados, o clip saturava e d(reward)/d(pos_error) = 0.
    # "shaping_only" : pos_cost FORA do clip (ja' e' limitado pela terminacao,
    #                  <= sqrt(3) normalizado); so' o shaping e' limitado.
    clip_mode = "legacy"
    cost_clip = 3.0                 # usado em legacy
    shaping_clip = 1.0              # usado em shaping_only

    # -- pesos
    constant = 1.5
    termination_penalty = 200.0
    w_pos = 1.0
    w_vel = 0.3
    w_d_action = 1.0                # usado em reward_mode="baseline"
    w_d_action_main = 1.0
    w_d_action_5 = 0.10
    w_heading = 0.5
    w_lateral = 0.15
    w_backward = 0.10
    w_tilt = 0.10
    w_power = 0.0                   # custo de potencia (sum w^3 normalizado)

    # -- heading por look-ahead (substitui o vel_ref instantaneo)
    # "goal_vel"            : p_ahead = goal_pos + goal_vel * T. Funciona
    #                         IGUAL no treino e na avaliacao. No lemniscate
    #                         goal_vel e a derivada ANALITICA (direcao exata,
    #                         sem ruido); no goal estatico goal_vel = 0, logo
    #                         la_dist = 0 e o heading desliga-se sozinho pela
    #                         histerese -- que e o comportamento correto.
    # "trajectory_lookahead": so funciona com a trajetoria interna (treino).
    heading_source = "goal_vel"
    heading_lookahead_time = 0.25           # s, usado por "goal_vel"
    heading_lookahead_steps = 25            # 0.25 s @ 100 Hz, "trajectory_lookahead"
    heading_lookahead_from = "vehicle"      # "vehicle" (pure pursuit) | "reference"
    heading_dir_filter_tau = 0.30           # s, EMA na direcao de referencia
    heading_on_dist = 0.15                  # m, histerese: liga
    heading_off_dist = 0.08                 # m, histerese: desliga
    w_heading_warmup_start = 0              # passos globais
    w_heading_warmup_ramp = 200_000         # passos globais

    # -- normalizacao / limites
    max_pos_error_per_axis = 1.0
    max_lin_vel_per_axis = 2.0
    max_ang_vel_per_axis = 35.0
    min_upright = -0.17
    obs_pos_error_limit = 0.3
    obs_vel_error_limit = 0.5
    clamp_observations_in_train = False     # False mantem o A/B honesto

    # -- trajetoria
    trajectory_mixture_langevin_prob = 0.5
    langevin_gamma = 1.0
    langevin_omega = 2.0
    langevin_sigma = 0.5
    trajectory_eval_seed = None             # fixa as referencias na avaliacao

    # O SkrlAgentBackend (utils/skrl_agent_controller.py:172) poe isto a
    # False antes de instanciar o env. Nesse caso NAO ha trajetoria interna:
    # a referencia vem de fora, escrita em reset_manager._goal_* por
    # _update_goal() do play_multi.py (lemniscate ou goal estatico).
    use_raptor_trajectory = True

    goal_pos_xy_range = [-0.5, 0.5]
    goal_pos_z_range = None                 # derivado do spawn_z

    # -- aerodinamica
    use_vertical_tail = False
    use_puller_slipstream = False
    wind_y_mps = 0.0

    # -- registo
    eval_csv_path = None
    run_tag = "A_active"
    seed_tag = 0

    init_state_cfg = InitStateCfg(max_angle_deg=90.0, guidance_prob=0.1)

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()

        assert self.rotor5_mode in ("active", "dead", "removed")
        assert self.obs_layout in ("world_error", "body_heading")
        assert self.reward_mode in ("baseline", "heading")
        assert self.clip_mode in ("legacy", "shaping_only")

        # numero de canais que a policy realmente controla
        self.action_space = 4 if self.rotor5_mode == "removed" else 5

        if self.use_heading_obs is None:
            self.use_heading_obs = self.reward_mode == "heading"

        a = self.action_space
        if self.obs_layout == "world_error":
            #  pos_err(3) + vel_err(3) + R(9) + ang_b(3) + hist(a) + speeds(a)
            obs = 18 + 2 * a
        else:
            #  pos_err_b(3) + vel_b(3) + vel_ref_b(3) + R(9) + ang_b(3)
            obs = 21 + 2 * a
        if self.use_heading_obs:
            obs += 3            # cos, sin, valid
        self.observation_space = obs


class QuadcopterEnv(PegasusEnv):
    cfg: QuadcopterEnvCfg

    # =====================================================================
    # setup
    # =====================================================================
    def setup(self):
        super().setup()

        n = self.num_envs
        dev = self.device
        a = self.cfg.action_space

        thr = self.backend._vehicle._thrusters
        self._min_w = torch.as_tensor(thr.min_rotor_velocity, device=dev, dtype=torch.float32)
        self._max_w = torch.as_tensor(thr.max_rotor_velocity, device=dev, dtype=torch.float32)
        self._kf = torch.as_tensor(thr._rotor_constant, device=dev, dtype=torch.float32)
        self._num_rotors = int(self._max_w.numel())

        # comando fisico completo (5 canais) mesmo quando a policy tem 4
        self._cmd5 = torch.full((n, self._num_rotors), -1.0, device=dev)
        self._action = torch.zeros(n, a, device=dev)
        self._last_action = torch.zeros(n, a, device=dev)
        self._action_history_obs = torch.zeros(n, a, device=dev)

        # heading
        self._ref_dir_filt = torch.zeros(n, 2, device=dev)
        self._heading_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._global_step = 0

        # acumuladores de metricas por episodio
        self._m_count = torch.zeros(n, device=dev)
        self._m_pos_sq = torch.zeros(n, device=dev)
        self._m_vel_sq = torch.zeros(n, device=dev)
        self._m_thrust5 = torch.zeros(n, device=dev)
        self._m_thrust_tot = torch.zeros(n, device=dev)
        self._m_clip = torch.zeros(n, device=dev)
        self._m_hvalid = torch.zeros(n, device=dev)
        self._m_tilt = torch.zeros(n, device=dev)
        self._m_power = torch.zeros(n, device=dev)
        self._m_a5 = torch.zeros(n, device=dev)
        self._m_a5_sq = torch.zeros(n, device=dev)
        self._m_tail = torch.zeros(n, device=dev)
        self._m_reward = torch.zeros(n, device=dev)

        # trajetoria
        self._trajectory = RaptorLikeTrajectory(
            num_envs=n,
            episode_steps=int(self.cfg.episode_length_s / self.cfg.sim_dt),
            device=dev,
            dt=self.cfg.sim_dt,
            mixture_langevin_prob=self.cfg.trajectory_mixture_langevin_prob,
            gamma=self.cfg.langevin_gamma,
            omega=self.cfg.langevin_omega,
            sigma=self.cfg.langevin_sigma,
        )
        if self.cfg.trajectory_eval_seed is not None:
            # mesmas referencias em A, B e C -> comparacao emparelhada
            self._trajectory.set_generator(self.cfg.trajectory_eval_seed)

        if not self.cfg.use_raptor_trajectory:
            # AVALIACAO: quem manda e o play_multi.py.
            self._trajectory = None
            print("[env] referencia EXTERNA (reset_manager.goal_*)")

        self._recorder = None
        if self.cfg.eval_csv_path:
            self._recorder = EpisodeRecorder(
                self.cfg.eval_csv_path, self.cfg.run_tag, self.cfg.seed_tag
            )

        self.assert_physics_sane()

    # =====================================================================
    # PASSO 0: validar a fisica antes de gastar GPU
    # =====================================================================
    def assert_physics_sane(self):
        axes = getattr(self.backend._vehicle._thrusters, "_rotor_axes_body", None)
        print("[diag] rotor_axes_body =", None if axes is None else axes.tolist())
        print("[diag] rotor_positions_body =",
              getattr(self.backend._vehicle, "_rotor_positions_body", None))
        w5_max = float(self._max_w[-1])
        kf5 = float(self._kf[-1])
        w5_neutral = 0.5 * (float(self._min_w[-1]) + w5_max)
        print(f"[diag] thrust5 @max = {kf5 * w5_max ** 2:.2f} N "
              f"| @neutro(a=0) = {kf5 * w5_neutral ** 2:.2f} N")
        print(f"[diag] thrust_quad @max (cada) = {float(self._kf[0]) * float(self._max_w[0]) ** 2:.2f} N")
        if axes is not None:
            ax5 = axes[-1] if axes.dim() == 2 else axes
            if abs(float(ax5[0])) < 0.9:
                print("[diag] !!! AVISO: o eixo do rotor 5 nao aponta em +x do corpo.")
                print("[diag] !!! Para o treino e verifica a config fisica primeiro.")

    # =====================================================================
    # helpers
    # =====================================================================
    def _to_omega(self, cmd: torch.Tensor) -> torch.Tensor:
        """acao em [-1,1] -> velocidade de rotor."""
        return self._min_w + 0.5 * (cmd + 1.0) * (self._max_w - self._min_w)

    def _rotor_speeds_norm(self) -> torch.Tensor:
        w = self.backend._vehicle._thrusters._velocity
        rs = (w - self._min_w) / (self._max_w - self._min_w) * 2.0 - 1.0
        return rs[:, : self.cfg.action_space]

    # =====================================================================
    # Referencia: interna (treino) ou externa (play_multi.py)
    # =====================================================================
    # CONTRATO UNICO: tudo o que le a referencia le reset_manager.goal_*.
    # Quem a escreve muda -- _sync_trajectory_reference() no treino,
    # _update_goal() do play_multi.py na avaliacao -- mas o env nao precisa
    # de saber qual. Era exatamente isto que o teu v4_obs_tail_norm2 ja fazia
    # (linhas 290-291, 357-358, 445-446).
    @property
    def _goal_pos(self) -> torch.Tensor:
        return self.reset_manager.goal_pos

    @property
    def _goal_vel(self) -> torch.Tensor:
        return self.reset_manager.goal_vel

    @property
    def _goal_acc(self) -> torch.Tensor:
        return self.reset_manager.goal_acc

    def _heading_terms(self, pos, quat, R):
        """Direcao de referencia por look-ahead, filtrada, com histerese.

        Devolve (cos_err, sin_err, valid_float, la_dist).

        Porque nao usar vel_ref diretamente: com gamma=1, omega=2, sigma=0.5 o
        Langevin da' sigma_v ~ 0.35 m/s e sigma_x ~ 0.18 m -- a direcao de
        vel_ref e' quase puro ruido. Pior: o replay ping-pong nega a velocidade
        a meio do episodio, gerando flips de 180 graus instantaneos. O alvo era
        fisicamente inatingivel.
        """
        cfg = self.cfg
        step = self.episode_length_buf
        env_ids = torch.arange(self.num_envs, device=self.device)

        if cfg.heading_source == "trajectory_lookahead" and self._trajectory is not None:
            p_ahead = self._trajectory.lookahead(
                env_ids, self._trajectory.step_counter[env_ids],
                cfg.heading_lookahead_steps,
            )
        else:
            # Ver heading_source no cfg. Uma linha, tres casos cobertos.
            p_ahead = self._goal_pos + self._goal_vel * cfg.heading_lookahead_time
        anchor = pos if cfg.heading_lookahead_from == "vehicle" else self._goal_pos
        delta_xy = (p_ahead - anchor)[:, :2]
        la_dist = torch.linalg.norm(delta_xy, dim=-1)

        # histerese: evita alvos de heading a saltar quando a referencia para
        on = la_dist > cfg.heading_on_dist
        off = la_dist < cfg.heading_off_dist
        self._heading_active = torch.where(
            on, torch.ones_like(self._heading_active),
            torch.where(off, torch.zeros_like(self._heading_active), self._heading_active),
        )

        dir_raw = delta_xy / la_dist.clamp(min=1e-6).unsqueeze(-1)
        alpha = float(cfg.sim_dt / max(cfg.heading_dir_filter_tau, cfg.sim_dt))
        upd = self._heading_active.unsqueeze(-1)
        self._ref_dir_filt = torch.where(
            upd,
            (1.0 - alpha) * self._ref_dir_filt + alpha * dir_raw,
            self._ref_dir_filt,
        )
        nrm = torch.linalg.norm(self._ref_dir_filt, dim=-1, keepdim=True)
        ref_dir = self._ref_dir_filt / nrm.clamp(min=1e-6)

        nose = R[:, :2, 0]                       # eixo +x do corpo em xy do mundo
        nose = nose / torch.linalg.norm(nose, dim=-1, keepdim=True).clamp(min=1e-6)

        cos_e = (nose * ref_dir).sum(-1)
        sin_e = nose[:, 0] * ref_dir[:, 1] - nose[:, 1] * ref_dir[:, 0]
        valid = self._heading_active.float() * (nrm.squeeze(-1) > 0.5).float()
        return cos_e * valid, sin_e * valid, valid, la_dist

    # =====================================================================
    # pre physics / apply
    # =====================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clone().clamp(-1.0, 1.0)

        # ABLACAO LIMPA: forcar ANTES da reward e da observacao. Isto e' o que
        # elimina o custo d_action fantasma e a realimentacao de ruido.
        if self.cfg.rotor5_mode == "dead":
            a[:, 4] = -1.0

        self._action = a

        # comando fisico de 5 canais
        if self.cfg.rotor5_mode == "removed":
            self._cmd5[:, :4] = a
            self._cmd5[:, 4] = -1.0
        else:
            self._cmd5[:] = a

        # A trajetoria interna avanca UMA vez por passo, aqui. Na avaliacao
        # _trajectory e None e quem avanca e trajectory.step(physics_dt) do
        # play_multi.py, seguido de _update_goal().
        if self._trajectory is not None:
            self._trajectory.advance()
            self._sync_trajectory_reference()

        self._global_step += 1

    def _apply_action(self):
        omega = self._to_omega(self._cmd5)
        self.backend._input_reference[:] = omega

        if self.cfg.use_vertical_tail:
            force_b, torque_b = self.vertical_tail_wrench()
            self._m_tail += torch.linalg.norm(force_b, dim=-1)
            self.backend.set_external_forces_and_torques(
                forces=force_b,
                torques=torque_b,
                body_ids=self.backend._vehicle.body_index,
                is_global=False,     # wrench calculado no frame do CORPO
            )

    # =====================================================================
    # aerodinamica da deriva vertical
    # =====================================================================
    def vertical_tail_wrench(self):
        """Forca/torque no frame do CORPO.

        Nota de dimensionamento: com max_lin_vel_per_axis = 2 m/s,
        q = 0.5*1.225*2^2 = 2.45 Pa, logo F ~ 2.45*0.0221*0.8 ~ 0.04 N, i.e.
        0.15 % do peso (~30 N). Foi por isso que as tuas tres versoes
        aerodinamicas deram o mesmo resultado: a alteracao esta' abaixo do ruido.
        """
        c = self.cfg.vertical_tail_physics_cfg
        dev = self.device
        state = self.backend.get_state()
        vel_w = state[:, 3:6]
        quat = state[:, 6:10]
        R = quaternion_to_matrix(quat)

        wind_w = torch.zeros_like(vel_w)
        wind_w[:, 1] = self.cfg.wind_y_mps
        air_w = vel_w - wind_w
        air_b = torch.bmm(R.transpose(1, 2), air_w.unsqueeze(-1)).squeeze(-1)

        vx, vy = air_b[:, 0], air_b[:, 1]
        if self.cfg.use_puller_slipstream:
            vx = self.puller_slipstream(vx, c)

        speed_xy = torch.sqrt(vx * vx + vy * vy)
        valid = (speed_xy >= c["min_airspeed"]).float()
        gate = torch.sigmoid((vx - c["forward_gate_v0"]) / c["forward_gate_k"])

        beta = torch.atan2(-vy, vx.clamp(min=1e-3))
        CL = (c["CL_beta"] * beta).clamp(-c["CL_max"], c["CL_max"])
        CD = c["CD0"] + c["induced_k"] * CL * CL

        q = 0.5 * c["rho"] * speed_xy * speed_xy
        L = q * c["S_vtail"] * CL * gate * valid
        D = q * c["S_vtail"] * CD * gate * valid

        force_b = torch.zeros(self.num_envs, 3, device=dev)
        force_b[:, 0] = -D
        force_b[:, 1] = -L

        arm = (torch.as_tensor(c["p_vtail"], device=dev)
               - torch.as_tensor(c["p_com"], device=dev)).unsqueeze(0)
        torque_b = torch.cross(arm.expand_as(force_b), force_b, dim=-1)
        return force_b, torque_b

    @staticmethod
    def puller_slipstream(vx: torch.Tensor, c: dict) -> torch.Tensor:
        """Aumento de vx na esteira do rotor tractor (modelo de disco)."""
        return vx * (1.0 + c["wake_factor"] * c["coverage_factor"])

    # =====================================================================
    # observacoes
    # =====================================================================
    def _get_observations(self):
        cfg = self.cfg
        state = self.backend.get_state()
        pos, vel_w, quat, ang_b = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        R = quaternion_to_matrix(quat)

        pos_err_w = self._goal_pos - pos
        vel_err_w = self._goal_vel - vel_w

        clamp_now = cfg.test_mode or cfg.clamp_observations_in_train
        if clamp_now:
            pos_err_w = pos_err_w.clamp(-cfg.obs_pos_error_limit, cfg.obs_pos_error_limit)
            vel_err_w = vel_err_w.clamp(-cfg.obs_vel_error_limit, cfg.obs_vel_error_limit)

        hist = self._action_history_obs
        rs = self._rotor_speeds_norm()

        if cfg.obs_layout == "world_error":
            parts = [pos_err_w, vel_err_w, R.reshape(self.num_envs, 9), ang_b, hist, rs]
        else:
            Rt = R.transpose(1, 2)
            pos_err_b = torch.bmm(Rt, pos_err_w.unsqueeze(-1)).squeeze(-1)
            vel_b = torch.bmm(Rt, vel_w.unsqueeze(-1)).squeeze(-1)
            vel_ref_b = torch.bmm(Rt, self._goal_vel.unsqueeze(-1)).squeeze(-1)
            parts = [
                pos_err_b / cfg.max_pos_error_per_axis,
                vel_b / cfg.max_lin_vel_per_axis,
                vel_ref_b / cfg.max_lin_vel_per_axis,
                R.reshape(self.num_envs, 9),
                ang_b / cfg.max_ang_vel_per_axis,
                hist,
                rs,
            ]

        if cfg.use_heading_obs:
            cos_e, sin_e, valid, _ = self._heading_terms(pos, quat, R)
            parts += [cos_e.unsqueeze(-1), sin_e.unsqueeze(-1), valid.unsqueeze(-1)]

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    # =====================================================================
    # recompensa  <-- O PATCH
    # =====================================================================
    def _get_rewards(self):
        cfg = self.cfg
        state = self.backend.get_state()
        pos, vel_w, quat, ang_b = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        R = quaternion_to_matrix(quat)
        Rt = R.transpose(1, 2)

        pos_err_w = self._goal_pos - pos
        vel_err_w = self._goal_vel - vel_w
        pos_err_b = torch.bmm(Rt, pos_err_w.unsqueeze(-1)).squeeze(-1)

        pos_cost = torch.linalg.norm(pos_err_b / cfg.max_pos_error_per_axis, dim=-1)
        vel_cost = torch.linalg.norm(vel_err_w / cfg.max_lin_vel_per_axis, dim=-1)

        d_action = self._action - self._last_action

        if cfg.reward_mode == "baseline":
            shaping = cfg.w_vel * vel_cost + cfg.w_d_action * torch.linalg.norm(d_action, dim=-1)
            heading_cost = torch.zeros_like(pos_cost)
            valid = torch.zeros_like(pos_cost)
        else:
            cos_e, sin_e, valid, _ = self._heading_terms(pos, quat, R)
            # custo em [0,1], zero quando o nariz aponta para o look-ahead
            heading_cost = 0.5 * (1.0 - cos_e) * valid

            vel_b = torch.bmm(Rt, vel_w.unsqueeze(-1)).squeeze(-1)
            lateral = (vel_b[:, 1].abs() / cfg.max_lin_vel_per_axis) * valid
            backward = (-vel_b[:, 0]).clamp(min=0.0) / cfg.max_lin_vel_per_axis
            tilt = (1.0 - R[:, 2, 2]).clamp(min=0.0)

            dm = torch.linalg.norm(d_action[:, :4], dim=-1)
            d5 = d_action[:, 4].abs() if cfg.action_space == 5 else torch.zeros_like(dm)

            # RAMPA: aprende a voar primeiro, so' depois a apontar o nariz.
            ramp = 0.0
            if cfg.w_heading_warmup_ramp > 0:
                ramp = (self._global_step - cfg.w_heading_warmup_start) / cfg.w_heading_warmup_ramp
                ramp = min(max(ramp, 0.0), 1.0)
            else:
                ramp = 1.0
            w_head_eff = cfg.w_heading * ramp

            shaping = (
                cfg.w_vel * vel_cost
                + w_head_eff * heading_cost
                + cfg.w_lateral * lateral
                + cfg.w_backward * backward
                + cfg.w_tilt * tilt
                + cfg.w_d_action_main * dm
                + cfg.w_d_action_5 * d5
            )

        if cfg.w_power > 0.0:
            w = self.backend._vehicle._thrusters._velocity
            power = (w / self._max_w).pow(3.0).sum(-1)
            shaping = shaping + cfg.w_power * power

        # ------------------------------------------------------------------
        # O BUG E A CORRECAO
        # ------------------------------------------------------------------
        if cfg.clip_mode == "legacy":
            cost = (cfg.w_pos * pos_cost + shaping).clamp(max=cfg.cost_clip)
            clipped = (cfg.w_pos * pos_cost + shaping) > cfg.cost_clip
        else:
            # pos_cost FICA FORA DO CLIP. Ja' e' naturalmente limitado pela
            # terminacao (|pos_err| <= max_pos_error_per_axis por eixo, logo
            # a norma normalizada <= sqrt(3)). Assim d(reward)/d(pos_error)
            # nunca e' zero -- e era este zero que te impedia de voar para
            # pontos ligeiramente fora da zona de treino.
            shaping_c = shaping.clamp(max=cfg.shaping_clip)
            cost = cfg.w_pos * pos_cost + shaping_c
            clipped = shaping > cfg.shaping_clip

        reward = cfg.constant - cost

        died, _ = self._get_dones()
        reward = torch.where(died, reward - cfg.termination_penalty, reward)

        # ------------------------------------------------------------------
        # metricas
        # ------------------------------------------------------------------
        with torch.no_grad():
            w = self.backend._vehicle._thrusters._velocity
            thrust = self._kf * w * w
            self._m_count += 1.0
            self._m_pos_sq += (pos_err_w * pos_err_w).sum(-1)
            self._m_vel_sq += (vel_err_w * vel_err_w).sum(-1)
            self._m_thrust5 += thrust[:, 4] if self._num_rotors > 4 else 0.0
            self._m_thrust_tot += thrust.sum(-1)
            self._m_clip += clipped.float()
            self._m_hvalid += valid
            self._m_tilt += R[:, 2, 2]
            self._m_power += (w / self._max_w).pow(3.0).sum(-1)
            if cfg.action_space == 5:
                self._m_a5 += self._action[:, 4]
                self._m_a5_sq += self._action[:, 4] ** 2
            self._m_reward += reward

        self._last_action = self._action.clone()
        self._action_history_obs = self._action.clone()
        return reward

    # =====================================================================
    # terminacao
    # =====================================================================
    def _get_dones(self):
        cfg = self.cfg
        state = self.backend.get_state()
        pos, vel_w, quat, ang_b = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        R = quaternion_to_matrix(quat)

        pos_err = self._goal_pos - pos
        out_pos = (pos_err.abs() > cfg.max_pos_error_per_axis).any(-1)
        out_vel = (vel_w.abs() > cfg.max_lin_vel_per_axis).any(-1)
        out_ang = (ang_b.abs() > cfg.max_ang_vel_per_axis).any(-1)
        flipped = R[:, 2, 2] < cfg.min_upright

        died = out_pos | out_vel | out_ang | flipped
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return died, timeout

    # =====================================================================
    # reset
    # =====================================================================
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return

        with torch.no_grad():
            cnt = self._m_count[env_ids].clamp(min=1.0)
            rmse_pos = torch.sqrt(self._m_pos_sq[env_ids] / cnt)
            rmse_vel = torch.sqrt(self._m_vel_sq[env_ids] / cnt)
            t5_mean = self._m_thrust5[env_ids] / cnt
            t5_frac = self._m_thrust5[env_ids] / self._m_thrust_tot[env_ids].clamp(min=1e-6)
            clip_frac = self._m_clip[env_ids] / cnt
            hvalid = self._m_hvalid[env_ids] / cnt
            tilt = self._m_tilt[env_ids] / cnt
            power = self._m_power[env_ids] / cnt
            a5_mean = self._m_a5[env_ids] / cnt
            a5_std = (self._m_a5_sq[env_ids] / cnt - a5_mean ** 2).clamp(min=0.0).sqrt()
            tail = self._m_tail[env_ids] / cnt
            died, _ = self._get_dones()

            self.extras.setdefault("log", {})
            self.extras["log"].update({
                "Metrics/rmse_pos": rmse_pos.mean().item(),
                "Metrics/rmse_vel": rmse_vel.mean().item(),
                "Metrics/thrust5_mean_N": t5_mean.mean().item(),
                "Metrics/thrust5_impulse_frac": t5_frac.mean().item(),
                "Metrics/clip_frac": clip_frac.mean().item(),
                "Metrics/heading_valid_frac": hvalid.mean().item(),
                "Metrics/tilt_zz_mean": tilt.mean().item(),
                "Metrics/power_mean": power.mean().item(),
                "Metrics/a5_mean": a5_mean.mean().item(),
                "Metrics/a5_std": a5_std.mean().item(),
                "Metrics/tail_force_mean_N": tail.mean().item(),
            })

            if self._recorder is not None:
                self._recorder.add(
                    env_ids=env_ids,
                    rmse_pos=rmse_pos,
                    rmse_vel=rmse_vel,
                    traj_langevin=(
                        self._trajectory.is_langevin(env_ids)
                        if self._trajectory is not None
                        else torch.zeros(len(env_ids), dtype=torch.bool,
                                         device=self.device)
                    ),
                    terminated=died[env_ids],
                    steps=cnt,
                    thrust5_mean=t5_mean,
                    thrust5_frac=t5_frac,
                    power_mean=power,
                    total_reward=self._m_reward[env_ids],
                )

            for buf in (self._m_count, self._m_pos_sq, self._m_vel_sq,
                        self._m_thrust5, self._m_thrust_tot, self._m_clip,
                        self._m_hvalid, self._m_tilt, self._m_power,
                        self._m_a5, self._m_a5_sq, self._m_tail, self._m_reward):
                buf[env_ids] = 0.0

        super()._reset_idx(env_ids)

        # Reset CONSISTENTE. No teu env, _last_action ficava a 0 e
        # _action_history_obs ficava a _reset_rotor_norm -> primeiro d_action
        # artificialmente enorme no primeiro passo de cada episodio.
        rn = self.backend._vehicle._thrusters._reset_rotor_norm
        rn = torch.as_tensor(rn, device=self.device, dtype=torch.float32)
        rn = rn.expand(len(env_ids), -1)[:, : self.cfg.action_space]
        self._last_action[env_ids] = rn
        self._action_history_obs[env_ids] = rn
        self._action[env_ids] = rn

        self._ref_dir_filt[env_ids] = 0.0
        self._heading_active[env_ids] = False

        self._sync_trajectory_reference(env_ids, regenerate=True)

    def _sync_trajectory_reference(self, env_ids=None, regenerate: bool = False):
        """Publica a referencia INTERNA em reset_manager.goal_*.

        No-op quando use_raptor_trajectory=False: nessa altura os buffers ja
        foram escritos por _update_goal() do play_multi.py e sobrepo-los
        destruiria a referencia de teste.
        """
        if self._trajectory is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if regenerate:
            centers = self.backend._vehicle._init_pos[env_ids]
            self._trajectory.reset(env_ids, centers)

        pos_ref, vel_ref, acc_ref = self._trajectory.current(env_ids)
        self.reset_manager._goal_pos[env_ids] = pos_ref
        self.reset_manager._goal_vel[env_ids] = vel_ref
        self.reset_manager._goal_acc[env_ids] = acc_ref


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def cfg_E1(condition: str, seed: int = 0, csv: str | None = "results_E1.csv",
           eval_mode: bool = False) -> QuadcopterEnvCfg:
    """E1 -- ablacao limpa. condition in {A_active, B_dead, C_removed}."""
    mode = {"A_active": "active", "B_dead": "dead", "C_removed": "removed"}[condition]
    cfg = QuadcopterEnvCfg()
    cfg.rotor5_mode = mode
    cfg.obs_layout = "world_error"
    cfg.reward_mode = "baseline"
    cfg.clip_mode = "legacy"          # igual ao teu baseline que funciona
    cfg.run_tag = condition
    cfg.seed_tag = seed
    cfg.eval_csv_path = csv
    if eval_mode:
        # NOTA: em play_multi.py o SkrlAgentBackend forca test_mode=False e
        # use_raptor_trajectory=False. eval_mode so serve para avaliar com o
        # harness de TREINO. Para o teu protocolo real (--trajectory
        # none|lemniscate) nao precisas disto: o lemniscate e analitico e
        # determinista, logo A/B/C ja veem referencias identicas.
        cfg.test_mode = True
        cfg.trajectory_eval_seed = 12345   # MESMAS referencias em A, B e C
    cfg.__post_init__()
    return cfg


def cfg_E3(seed: int = 0, w_heading: float = 0.5, csv: str | None = "results_E3.csv",
           eval_mode: bool = False) -> QuadcopterEnvCfg:
    """E3 -- env de heading reparado. Corre primeiro com w_heading=0.0."""
    cfg = QuadcopterEnvCfg()
    cfg.rotor5_mode = "active"
    cfg.obs_layout = "body_heading"
    cfg.reward_mode = "heading"
    cfg.clip_mode = "shaping_only"    # a correcao do clip
    cfg.w_heading = w_heading
    cfg.use_heading_obs = True
    cfg.trajectory_mixture_langevin_prob = 0.5   # de volta ao curriculum
    cfg.physics_cfg = shuttle_glider_physics_cfg(3500.0)
    cfg.run_tag = f"E3_head{w_heading:g}"
    cfg.seed_tag = seed
    cfg.eval_csv_path = csv
    if eval_mode:
        cfg.test_mode = True
        cfg.trajectory_eval_seed = 12345
    cfg.__post_init__()
    return cfg
