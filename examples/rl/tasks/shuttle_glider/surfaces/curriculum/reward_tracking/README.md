# shuttle_glider2/vehicle_final/task5_v1_tracking

**Objetivo:** Tracking com gerador de referência configurado abaixo. Variante mantida; a ordem cronológica entre nomes não foi inferida.

Veículo: `'Shuttle_glider2'`. Classe: `QuadcopterEnv`; configuração: `QuadcopterEnvCfg` em `quadcopter_env.py`.

## Observação

Dimensão declarada: **40**. Ordem extraída da concatenação final do método. As dimensões marcadas “ver código” exigem o smoke test; o espaço declarado não prova a forma produzida.

| Ordem | Canal/expressão | Dimensão |
| --- | --- | --- |
| 1 | `pos_err_b` | 3 |
| 2 | `vel_err_b` | 3 |
| 3 | `alpha` | 1 |
| 4 | `beta` | 1 |
| 5 | `airspeed` | 1 |
| 6 | `goal_acc_b` | 3 |
| 7 | `R_flat` | 9 |
| 8 | `ang` | 3 |
| 9 | `self._action_history_obs` | 8 |
| 10 | `rotor_speeds_norm` | 5 |
| 11 | `surfaces_norm` | 3 |

Referenciais/normalização: as transformações abaixo são as expressões ativas do método; ENU mundial e FLU corpo nas interfaces da framework. Matrizes de erro/relativas seguem a multiplicação específica da tarefa; escala numérica não altera o referencial.

```python
R_flat = R.reshape(self.num_envs, 9)  # measured absolute attitude
rotor_speeds_norm = 2.0 * (thrusters._velocity - min_w) / (max_w - min_w).clamp_min(1e-6) - 1.0
pos_err_b = pos_err_b / self.cfg.obs_scale_pos
vel_err_b = vel_err_b / self.cfg.obs_scale_vel
vel_body = vel_body / self.cfg.obs_scale_vel
goal_vel_b = goal_vel_b / self.cfg.obs_scale_vel
goal_acc_b = goal_acc_b / self.cfg.obs_scale_acc
alpha = alpha / self.cfg.obs_scale_angle
beta = beta / self.cfg.obs_scale_angle
airspeed = airspeed / self.cfg.obs_scale_vel
ang = ang_b / self.cfg.obs_scale_rate
airspeed = airspeed / 15.0
goal_acc_b, R_flat, ang, self._action_history_obs,
```

## Ação

Dimensão: **8**; intervalo da policy `[-1,1]`; modo `'rotor_velocity_direct'`.
Ordem: rotores da configuração; no shuttle-glider, `[rotor_0, rotor_1, rotor_2, rotor_3, puller]`, seguidos por `[elevator, aileron, rudder]` apenas nas variantes de 8 ações. `omega = min + (a+1)/2*(max-min)`; velocidades em rad/s. Superfícies em radianos, com limites e eventual rate limit definidos em `_apply_action`.

```python
elevator_target = torch.where(elevator_action >= 0.0, elevator_action * self.cfg.delta_e_max, elevator_action * (-self.cfg.delta_e_min))
aileron_target = torch.where(aileron_action >= 0.0, aileron_action * self.cfg.delta_a_max, aileron_action * (-self.cfg.delta_a_min))
rudder_target = torch.where(rudder_action >= 0.0, rudder_action * self.cfg.delta_r_max, rudder_action * (-self.cfg.delta_r_min))
target = torch.stack([elevator_target, aileron_target, rudder_target], dim=-1)
```

## Reward

Termos e pesos base abaixo; expressão completa, gates, reduções e retornos em `_get_rewards` e `reward_terms.py`.

| Peso/escala base | Expressão |
| --- | --- |
| `w_pos` | `1.0` |
| `w_vel` | `0.3` |
| `w_quad_effort` | `0.04` |
| `w_puller_effort` | `0.005` |
| `w_d_rotor` | `0.2` |
| `w_surf` | `0.003` |
| `w_surf_rate` | `0.02` |
| `w_hover_puller_effort` | `0.04` |
| `w_hover_ang_rate` | `0.03` |
| `w_beta` | `0.02` |
| `w_alpha` | `0.03` |
| `w_rotor_margin` | `0.01` |
| `w_elevator_margin` | `0.01` |
| `w_ctrl_offload` | `0.01` |
| `w_yaw_offload` | `0.005` |
| `constant` | `1.5` |
| `termination_penalty` | `200.0` |
| `cost_clip` | `6.0` |

Curriculum de distribuição, com pesos de reward estacionários em `QuadcopterEnvCfg`; os gates e termos ativos dependem de `reward_variant` e `reward_terms.py`.

## Referência, reset e terminação

`sim_dt=0.01`, `decimation=1`, episódio `8.0 s`. O timeout foi corrigido para ocorrer após `max_episode_length` ações; anteriormente ocorria um passo antes.

| Parâmetro | Valor/expressão |
| --- | --- |
| `aero_on_speed` | `4.0` |
| `aero_full_speed` | `8.0` |
| `trajectory_generator` | `True` |
| `trajectory_ramp_duration` | `2.5` |
| `trajectory_a_max` | `6.0` |
| `trajectory_radial_acc_max` | `6.0` |
| `curriculum_enabled` | `True` |
| `curriculum_start_level` | `0` |
| `curriculum_evaluation_episodes` | `2048` |
| `curriculum_required_consecutive_windows` | `3` |
| `curriculum_previous_level_prob` | `0.2` |
| `curriculum_current_level_prob` | `0.8` |
| `curriculum_next_level_prob` | `0.0` |
| `goal_pos_xy_range` | `None` |
| `goal_pos_z_range` | `None` |
| `trim_elevator_on_speed` | `6.0` |
| `trim_elevator_full_speed` | `10.0` |

Reset longitudinal partilhado em `pegasus.simulator.logic.rl.aligned_trim`: revisão `aligned-com-review-v2`. Calcula forças e pitch com os braços/massa carregados e limita o ângulo de ataque inicial; não é trim 6-DOF de curva nem convergência de treino demonstrada.

Reset e falhas: `_reset_idx`, `_get_dones`; contratos completos por ID em `docs/review/contracts_after.json`.

## Algoritmos e hiperparâmetros

A existência de uma configuração indica suporte configurado, não prova que tenha sido usada num treino. Não foi validado nenhum checkpoint no simulador. Valores abaixo são expressões atribuídas no ficheiro; defaults herdados do SKRL dependem da versão instalada.

### PPO2 — `agents/ppo2_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 120000, 'seed': 42}}` |
| `cfg['rollouts']` | `24` |
| `cfg['learning_epochs']` | `5` |
| `cfg['mini_batches']` | `4` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['lambda']` | `0.95` |
| `cfg['ratio_clip']` | `0.2` |
| `cfg['value_clip']` | `0.2` |
| `cfg['clip_predicted_values']` | `False` |
| `cfg['use_clipped_value_loss']` | `True` |
| `cfg['value_loss_scale']` | `1.0` |
| `cfg['entropy_loss_scale']` | `0.0` |
| `cfg['learning_rate']` | `0.0005` |
| `cfg['grad_norm_clip']` | `1.0` |
| `cfg['schedule']` | `'adaptive'` |
| `cfg['desired_kl']` | `0.01` |
| `cfg['min_learning_rate']` | `1e-05` |
| `cfg['max_learning_rate']` | `0.01` |
| `cfg['learning_rate_factor']` | `1.5` |
| `cfg['learning_rate_scheduler']` | `None` |
| `cfg['learning_rate_scheduler_kwargs']` | `{}` |
| `cfg['state_preprocessor']` | `None` |
| `cfg['state_preprocessor_kwargs']` | `{}` |
| `cfg['value_preprocessor']` | `None` |
| `cfg['value_preprocessor_kwargs']` | `{}` |
| `cfg['rewards_shaper']` | `None` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `0` |
| `cfg['kl_threshold']` | `0.0` |
| `cfg['time_limit_bootstrap']` | `True` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': 'quadcopter_direct', 'experiment_name': '', 'write_interval': 24, 'checkpoint_interval': 24 * 50}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))
```

### PPO — `agents/ppo_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 24 * 200, 'seed': 42}, 'seeded': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 24 * 1500 * 4096, 'seed': 42}}` |
| `cfg['rollouts']` | `24` |
| `cfg['learning_epochs']` | `5` |
| `cfg['mini_batches']` | `4` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['lambda']` | `0.95` |
| `cfg['learning_rate']` | `0.0005` |
| `cfg['learning_rate_scheduler']` | `KLAdaptiveLR` |
| `cfg['learning_rate_scheduler_kwargs']` | `{'kl_threshold': 0.016}` |
| `cfg['grad_norm_clip']` | `1.0` |
| `cfg['ratio_clip']` | `0.2` |
| `cfg['value_clip']` | `0.2` |
| `cfg['clip_predicted_values']` | `True` |
| `cfg['entropy_loss_scale']` | `0.0` |
| `cfg['value_loss_scale']` | `1.0` |
| `cfg['state_preprocessor']` | `None` |
| `cfg['state_preprocessor_kwargs']` | `{}` |
| `cfg['value_preprocessor']` | `RunningStandardScaler` |
| `cfg['value_preprocessor_kwargs']` | `{'size': 1}` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `0` |
| `cfg['kl_threshold']` | `0.0` |
| `cfg['rewards_shaper']` | `_reward_shaper` |
| `cfg['time_limit_bootstrap']` | `False` |
| `cfg['experiment']` | `{'directory': 'quadcopter_direct', 'experiment_name': '', 'write_interval': 'auto', 'checkpoint_interval': 'auto'}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU())
nn.Linear(256, self.num_actions)
nn.Linear(256, 1)
```

### SAC — `agents/sac_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `cfg['gradient_steps']` | `1` |
| `cfg['batch_size']` | `512` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['polyak']` | `0.005` |
| `cfg['actor_learning_rate']` | `0.0003` |
| `cfg['critic_learning_rate']` | `0.0003` |
| `cfg['entropy_learning_rate']` | `0.0001` |
| `cfg['learn_entropy']` | `True` |
| `cfg['initial_entropy_value']` | `0.5` |
| `cfg['target_entropy']` | `-float(5)` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `10000` |
| `cfg['grad_norm_clip']` | `0` |
| `cfg['state_preprocessor']` | `None` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 1000, 'checkpoint_interval': 10000, 'store_separately': False, 'wandb': False, 'wandb_kwargs': {}}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, self.num_actions * 2))
nn.Sequential(nn.Linear(self.num_observations + self.num_actions, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
```

Validação v2: forma e finitude executadas em CPU com backend tensorial substituto (N=1 e N=3), incluindo isolamento do reset parcial. Não equivale a criação/voo no Isaac Sim.
