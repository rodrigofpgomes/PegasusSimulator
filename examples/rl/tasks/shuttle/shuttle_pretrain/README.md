# Shuttle — shuttle_pretrain

Port do pré-treino RAPTOR-like para o veículo `Shuttle`.

Descrição no código: Single-vehicle hover task replicating the RAPTOR pre-training setup.

## Configuração principal
- `observation_space` = `26`
- `action_space` = `4`
- `state_space` = `0`
- `sim_dt` = `0.01`
- `decimation` = `1`
- `episode_length_s` = `5.0`
- `action_mode` = `"rotor_velocity_direct"`
- `vehicle` = `"Shuttle"`
- `w_pos` = `1.0`
- `w_ori` = `0.1`
- `w_d_action` = `1.0`
- `constant` = `1.5`
- `termination_penalty` = `100.0`
- `max_pos_error_per_axis` = `1.0`
- `max_lin_vel_per_axis` = `2.0`
- `max_ang_vel_per_axis` = `35.0`
- `use_raptor_trajectory` = `True`
- `trajectory_mixture_langevin_prob` = `0.5`
- `langevin_gamma` = `1.0`
- `langevin_omega` = `2.0`
- `langevin_sigma` = `0.5`
- `langevin_alpha` = `0.01`
- `test_mode` = `False`
- `clamp_observations_in_test` = `True`
- `obs_pos_error_limit` = `0.3`
- `obs_vel_error_limit` = `0.5`

## Observações
- Dimensão configurada: `26`.
- Vetor `policy`: `[pos_error, R_flat, vel_error, ang_b, self._action_history_obs, rotor_speeds_norm]`.
- Em `test_mode`, `pos_error` é limitado por `obs_pos_error_limit` e `vel_error` por `obs_vel_error_limit`.

## Ações e clamps
- `action` = `actions.clamp(-1.0, 1.0)`
- A ação é limitada a `[-1, 1]` antes de ser aplicada.
- A ação normalizada é mapeada para velocidades de rotor entre `min_rotor_velocity` e `max_rotor_velocity`.
- Limites dos rotores: min `[0, 0, 0, 0]`, max `[1400, 1400, 1400, 1400]`.

Clamps relevantes no código:
- `action = actions.clamp(-1.0, 1.0)`
- `if self.cfg.test_mode and self.cfg.clamp_observations_in_test:`
- `pos_error = pos_error.clamp(-self.cfg.obs_pos_error_limit, self.cfg.obs_pos_error_limit)`
- `vel_error = vel_error.clamp(-self.cfg.obs_vel_error_limit, self.cfg.obs_vel_error_limit)`
- `q = quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp_min(1e-8)`
- `acos_arg = torch.clamp(1.0 - qz_abs, -1.0, 1.0)`

## Reward, custos e terminação
- `ori_cost` = `2.0 * torch.acos(acos_arg)`
- `pos_cost` = `torch.linalg.norm(pos_error, dim=1)`
- `d_action_cost` = `torch.linalg.norm(d_action, dim=1)`
- `cost` = `self.cfg.w_pos * pos_cost + self.cfg.w_ori * ori_cost + self.cfg.w_d_action * d_action_cost`
- `reward` = `self.cfg.constant - cost`
- `died` = `(pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)`
- `died` = `died |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)`
- `died` = `died |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)`
- `reward[died]` = `-self.cfg.termination_penalty`

Terminação/truncatura:
- `terminated` = `(pos_error.abs() > self.cfg.max_pos_error_per_axis).any(dim=1)`
- `terminated` = `terminated |= (vel_w.abs() > self.cfg.max_lin_vel_per_axis).any(dim=1)`
- `terminated` = `terminated |= (ang_b.abs() > self.cfg.max_ang_vel_per_axis).any(dim=1)`
- `truncated` = `self.episode_length_buf >= self.max_episode_length - 1`

## Trajetória / referência
- `trajectory.py`: mistura null/Langevin-like (`trajectory_mixture_langevin_prob`) com replay ping-pong; devolve `pos, vel`.

## Modelo e treino
- `agents/sac_cfg.py` — SAC: ator Gaussiano `64-64 ReLU` com tanh + correção de log-prob; críticos Q `256-256 ReLU`. Parâmetros principais: `batch_size` = `128`, `gradient_steps` = `1`, `discount_factor` = `0.99`, `polyak` = `0.005`, `actor_learning_rate` = `3e-4`, `critic_learning_rate` = `3e-4`, `entropy_learning_rate` = `1e-4`, `target_entropy` = `-2.0`, `initial_entropy_value` = `0.5`, `policy_delay` = `2`. Preset: `timesteps` = `500_000`, `memory_size` = `500_000`, `seed` = `10`.

## Observações de implementação
- Em morte, o reward normal é substituído por `-termination_penalty` (`100.0`).
