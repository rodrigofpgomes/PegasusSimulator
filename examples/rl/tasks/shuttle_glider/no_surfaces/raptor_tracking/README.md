# ShuttleGlider without surface control

This task is adapted from `sg_f2_rep3` in the experimental copy. It will be
trained again; its previous checkpoints and results are not used. The policy
commands the four Shuttle rotors and the puller. The three surface commands
sent to the vehicle are zero, so the surfaces remain neutral. There is no
curriculum in this task.

The motor coefficients, speed limits and response times now come from
`get_vehicle_physics("Shuttle_glider2")`. This changes the physical experiment
relative to the source variant and makes it comparable with new curriculum
tasks that use the same preset. The remaining sections describe the inherited
observation, reward and trajectory implementation; they do not report new
training results.

## Source analysis

**Objetivo:** Tracking com gerador de referência configurado abaixo. Variante mantida; a ordem cronológica entre nomes não foi inferida.

Veículo: `'Shuttle_glider2'`. Classe: `QuadcopterEnv`; configuração: `QuadcopterEnvCfg` em `quadcopter_env.py`.

## Observação

Dimensão declarada: **34**. Ordem extraída da concatenação final do método. As dimensões marcadas “ver código” exigem o smoke test; o espaço declarado não prova a forma produzida.

| Ordem | Canal/expressão | Dimensão |
| --- | --- | --- |
| 1 | `pos_error` | 3 |
| 2 | `vel_error` | 3 |
| 3 | `R_flat` | 9 |
| 4 | `ang_b` | 3 |
| 5 | `self._action_history_obs` | 5 |
| 6 | `rotor_speeds_norm` | 5 |
| 7 | `acc_ref_b` | 3 |
| 8 | `vel_ref_b` | 3 |

Referenciais/normalização: as transformações abaixo são as expressões ativas do método; ENU mundial e FLU corpo nas interfaces da framework. Matrizes de erro/relativas seguem a multiplicação específica da tarefa; escala numérica não altera o referencial.

```python
pos_error = pos_error.clamp(-self.cfg.obs_pos_error_limit, self.cfg.obs_pos_error_limit)
vel_error = vel_error.clamp(-self.cfg.obs_vel_error_limit, self.cfg.obs_vel_error_limit)
R = quaternion_to_matrix(quat)
R_flat = R.reshape(self.num_envs, 9)
rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0
obs = torch.cat([pos_error, vel_error, R_flat, ang_b, self._action_history_obs, rotor_speeds_norm, acc_ref_b, vel_ref_b], dim=1)
```

## Ação

Dimensão: **5**; intervalo da policy `[-1,1]`; modo `'rotor_velocity_direct'`.
Ordem: rotores da configuração; no shuttle-glider, `[rotor_0, rotor_1, rotor_2, rotor_3, puller]`, seguidos por `[elevator, aileron, rudder]` apenas nas variantes de 8 ações. `omega = min + (a+1)/2*(max-min)`; velocidades em rad/s. Superfícies em radianos, com limites e eventual rate limit definidos em `_apply_action`.

```python
# Mapeamento afim para os limites dos rotores; ver _apply_action.
```

## Reward

Termos e pesos base abaixo; expressão completa, gates, reduções e retornos em `_get_rewards`.

| Peso/escala base | Expressão |
| --- | --- |
| `w_pos` | `1.0` |
| `w_vel` | `0.3` |
| `w_d_action` | `1.0` |
| `w_d_action_5` | `0.1` |
| `w_align` | `0.0` |
| `w_lateral` | `0.0` |
| `w_use` | `1.0` |
| `w_level` | `0.25` |
| `constant` | `1.5` |
| `termination_penalty` | `200.0` |
| `cost_clip` | `3.0` |

Sem `curriculum_manager.py` local. Ver configuração e `_get_rewards` para os pesos ativos.

## Referência, reset e terminação

`sim_dt=0.01`, `decimation=1`, episódio `5.0 s`. O timeout foi corrigido para ocorrer após `max_episode_length` ações; anteriormente ocorria um passo antes.

| Parâmetro | Valor/expressão |
| --- | --- |
| `use_raptor_trajectory` | `True` |
| `trajectory_mixture_langevin_prob` | `0.65` |
| `langevin_gamma` | `1.0` |
| `langevin_omega` | `2.0` |
| `langevin_sigma` | `0.5` |
| `goal_pos_xy_range` | `None` |
| `goal_pos_z_range` | `None` |


Reset e falhas: `_reset_idx`, `_get_dones`; contratos completos por ID em `docs/review/contracts_after.json`.

## Algoritmos e hiperparâmetros

A existência de uma configuração indica suporte configurado, não prova que tenha sido usada num treino. Não foi validado nenhum checkpoint no simulador. Valores abaixo são expressões atribuídas no ficheiro; defaults herdados do SKRL dependem da versão instalada.

### PPO2 — `agents/ppo2_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 24000, 'seed': 42}}` |
| `cfg['rollouts']` | `16` |
| `cfg['learning_epochs']` | `8` |
| `cfg['mini_batches']` | `8` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['lambda']` | `0.95` |
| `cfg['ratio_clip']` | `0.2` |
| `cfg['value_clip']` | `0.2` |
| `cfg['clip_predicted_values']` | `False` |
| `cfg['use_clipped_value_loss']` | `True` |
| `cfg['value_loss_scale']` | `2.0` |
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
| `cfg['time_limit_bootstrap']` | `False` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': 'quadcopter_direct', 'experiment_name': '', 'write_interval': 24, 'checkpoint_interval': 24 * 50}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
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
| `cfg['state_preprocessor']` | `RunningStandardScaler` |
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
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
nn.Linear(64, self.num_actions)
nn.Linear(64, 1)
```

### SAC — `agents/sac_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 300000, 'memory_size': 256, 'seed': 42}}` |
| `cfg['gradient_steps']` | `1` |
| `cfg['batch_size']` | `512` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['polyak']` | `0.005` |
| `cfg['actor_learning_rate']` | `0.0003` |
| `cfg['critic_learning_rate']` | `0.0003` |
| `cfg['entropy_learning_rate']` | `0.0003` |
| `cfg['learning_rate_scheduler']` | `None` |
| `cfg['learning_rate_scheduler_kwargs']` | `{}` |
| `cfg['state_preprocessor']` | `RunningStandardScaler` |
| `cfg['state_preprocessor_kwargs']` | `{'size': None}` |
| `cfg['random_timesteps']` | `8` |
| `cfg['learning_starts']` | `8` |
| `cfg['grad_norm_clip']` | `1.0` |
| `cfg['learn_entropy']` | `True` |
| `cfg['initial_entropy_value']` | `0.2` |
| `cfg['target_entropy']` | `None` |
| `cfg['rewards_shaper']` | `None` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 1000, 'checkpoint_interval': 5000, 'store_separately': False, 'wandb': False, 'wandb_kwargs': {}}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
```

Validação v2: forma e finitude executadas em CPU com backend tensorial substituto (N=1 e N=3), incluindo isolamento do reset parcial. Não equivale a criação/voo no Isaac Sim.
