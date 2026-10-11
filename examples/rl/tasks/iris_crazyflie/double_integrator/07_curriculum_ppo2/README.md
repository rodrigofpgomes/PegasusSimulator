# double_integrator/07_curriculum_ppo2

**Objetivo:** Regulação/controlo com referência estática configurada pela tarefa. Variante mantida; a ordem cronológica entre nomes não foi inferida.

Veículo: `'Iris'`. Classe: `QuadcopterEnv`; configuração: `QuadcopterEnvCfg` em `quadcopter_env.py`.

## Observação

Dimensão declarada: **6**. Ordem extraída da concatenação final do método. As dimensões marcadas “ver código” exigem o smoke test; o espaço declarado não prova a forma produzida.

| Ordem | Canal/expressão | Dimensão |
| --- | --- | --- |
| 1 | `ep` | 3 |
| 2 | `ev` | 3 |

Referenciais/normalização: as transformações abaixo são as expressões ativas do método; ENU mundial e FLU corpo nas interfaces da framework. Matrizes de erro/relativas seguem a multiplicação específica da tarefa; escala numérica não altera o referencial.

```python
ev = torch.zeros(self.num_envs, 3, device=self.device) - quaternion_apply(quat, lin_vel_b)
```

## Ação

Dimensão: **3**; intervalo da policy `[-1,1]`; modo `'direct_force'`.
Transformação física: ver `_apply_action` e `action_mode`. A dimensão 4 pode representar thrust e três torques, não quatro motores. A dimensão 3 representa comando translacional nas variantes double-integrator.

```python
"""Converts clamped actions to forces and torques and applies them."""
torques = torch.zeros((self.num_envs, self.parts_per_vehicle, 3), device=self.device)
g_force = torch.tensor([0.0, 0.0, self._robot_weight], device=self.device).unsqueeze(0)
torques[:, self._body_index, :] = - self.cfg.angular_damping * ang_vel_b
self.backend.set_forces_and_torques(forces, torques)
```

## Reward

Termos e pesos base abaixo; expressão completa, gates, reduções e retornos em `_get_rewards`.

| Peso/escala base | Expressão |
| --- | --- |

Sem `curriculum_manager.py` local. Ver configuração e `_get_rewards` para os pesos ativos.

## Referência, reset e terminação

`sim_dt=0.01`, `decimation=2`, episódio `10.0 s`. O timeout foi corrigido para ocorrer após `max_episode_length` ações; anteriormente ocorria um passo antes.

| Parâmetro | Valor/expressão |
| --- | --- |
| `goal_pos_xy_range` | `field(default_factory=lambda: [-2.0, 2.0])` |
| `goal_pos_z_range` | `field(default_factory=lambda: [0.5, 1.5])` |


Reset e falhas: `_reset_idx`, `_get_dones`; contratos completos por ID em `docs/review/contracts_after.json`.

## Algoritmos e hiperparâmetros

A existência de uma configuração indica suporte configurado, não prova que tenha sido usada num treino. Não foi validado nenhum checkpoint no simulador. Valores abaixo são expressões atribuídas no ficheiro; defaults herdados do SKRL dependem da versão instalada.

### PPO2 — `agents/ppo2_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 24 * 600, 'seed': None}}` |
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
| `cfg['state_preprocessor_kwargs']` | `{'size': None}` |
| `cfg['value_preprocessor']` | `RunningStandardScaler` |
| `cfg['value_preprocessor_kwargs']` | `{'size': 1}` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `0` |
| `cfg['kl_threshold']` | `0.0` |
| `cfg['rewards_shaper_scale']` | `1.0` |
| `cfg['time_limit_bootstrap']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 24, 'checkpoint_interval': 24 * 10 * 5}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
```

### PPO — `agents/ppo_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 24 * 600, 'seed': None}}` |
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
| `cfg['state_preprocessor_kwargs']` | `{'size': None}` |
| `cfg['value_preprocessor']` | `RunningStandardScaler` |
| `cfg['value_preprocessor_kwargs']` | `{'size': 1}` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `0` |
| `cfg['kl_threshold']` | `0.0` |
| `cfg['rewards_shaper_scale']` | `1.0` |
| `cfg['time_limit_bootstrap']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 24, 'checkpoint_interval': 24 * 10 * 5}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
```

### SAC — `agents/sac_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _default_models, 'cfg': _make_cfg(), 'timesteps': 50000, 'memory_size': 50000, 'seed': None}}` |
| `cfg['gradient_steps']` | `1` |
| `cfg['batch_size']` | `256` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['polyak']` | `0.005` |
| `cfg['actor_learning_rate']` | `0.0005` |
| `cfg['critic_learning_rate']` | `0.0005` |
| `cfg['entropy_learning_rate']` | `0.0005` |
| `cfg['learning_rate_scheduler']` | `None` |
| `cfg['learning_rate_scheduler_kwargs']` | `{}` |
| `cfg['state_preprocessor']` | `RunningStandardScaler` |
| `cfg['state_preprocessor_kwargs']` | `{'size': None}` |
| `cfg['random_timesteps']` | `5000` |
| `cfg['learning_starts']` | `5000` |
| `cfg['grad_norm_clip']` | `1.0` |
| `cfg['learn_entropy']` | `True` |
| `cfg['initial_entropy_value']` | `0.2` |
| `cfg['target_entropy']` | `None` |
| `cfg['rewards_shaper']` | `lambda rewards, *args: rewards * 0.01` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 1000, 'checkpoint_interval': 10000, 'store_separately': False, 'wandb': False, 'wandb_kwargs': {}}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, self.num_actions))
nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
```

Validação v2: forma e finitude executadas em CPU com backend tensorial substituto (N=1 e N=3), incluindo isolamento do reset parcial. Não equivale a criação/voo no Isaac Sim.
