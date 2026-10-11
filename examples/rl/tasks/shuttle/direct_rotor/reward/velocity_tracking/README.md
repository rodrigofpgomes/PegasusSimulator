# Velocity Tracking

Adapted from `raptor_pretrain_folder/raptor_pretrain4_5_w_v` for a new Shuttle training run. The current task uses
`vehicle="Shuttle"`, four direct rotor-speed actions and
`get_vehicle_physics("Shuttle")`. The source notes below describe the
historical implementation; any Iris vehicle or old motor-parameter references
in those notes do not describe this adapted physical setup.

## Source notes

# raptor_pretrain_folder/raptor_pretrain4_5_w_v

**Objetivo:** Tracking com gerador de referência configurado abaixo. Variante mantida; a ordem cronológica entre nomes não foi inferida.

Veículo: `'Iris'`. Classe: `QuadcopterEnv`; configuração: `QuadcopterEnvCfg` em `quadcopter_env.py`.

## Observação

Dimensão declarada: **26**. Ordem extraída da concatenação final do método. As dimensões marcadas “ver código” exigem o smoke test; o espaço declarado não prova a forma produzida.

| Ordem | Canal/expressão | Dimensão |
| --- | --- | --- |
| 1 | `pos_error` | 3 |
| 2 | `vel_error` | 3 |
| 3 | `R_err_flat` | 9 |
| 4 | `ang_b` | 3 |
| 5 | `self._action_history_obs` | 4 |
| 6 | `rotor_speeds_norm` | 4 |

Referenciais/normalização: as transformações abaixo são as expressões ativas do método; ENU mundial e FLU corpo nas interfaces da framework. Matrizes de erro/relativas seguem a multiplicação específica da tarefa; escala numérica não altera o referencial.

```python
pos_error = pos_error.clamp(-self.cfg.obs_pos_error_limit, self.cfg.obs_pos_error_limit)
vel_error = vel_error.clamp(-self.cfg.obs_vel_error_limit, self.cfg.obs_vel_error_limit)
R = quaternion_to_matrix(quat)
R_err = torch.bmm(Rd_ref.transpose(1, 2), R)  # Rdᵀ R
R_err_flat = R_err.reshape(self.num_envs, 9)
cos_b3 = torch.sum(b3 * b3_des, dim=1).clamp(-1.0, 1.0)
rotor_speeds_norm = (rpm - min_w) / (max_w - min_w) * 2.0 - 1.0
obs = torch.cat([pos_error, vel_error, R_err_flat, ang_b, self._action_history_obs, rotor_speeds_norm], dim=1)
```

## Ação

Dimensão: **4**; intervalo da policy `[-1,1]`; modo `'rotor_velocity_direct'`.
Ordem: rotores da configuração; no shuttle-glider, `[rotor_0, rotor_1, rotor_2, rotor_3, puller]`, seguidos por `[elevator, aileron, rudder]` apenas nas variantes de 8 ações. `omega = min + (a+1)/2*(max-min)`; velocidades em rad/s. Superfícies em radianos, com limites e eventual rate limit definidos em `_apply_action`.

```python
# Mapeamento afim para os limites dos rotores; ver _apply_action.
```

## Reward

Termos e pesos base abaixo; expressão completa, gates, reduções e retornos em `_get_rewards`.

| Peso/escala base | Expressão |
| --- | --- |
| `w_pos` | `1.0` |
| `w_vel` | `0.2` |
| `w_ori` | `0.1` |
| `w_d_action` | `1.0` |
| `constant` | `1.5` |
| `termination_penalty` | `100.0` |

Sem `curriculum_manager.py` local. Ver configuração e `_get_rewards` para os pesos ativos.

## Referência, reset e terminação

`sim_dt=0.01`, `decimation=1`, episódio `5.0 s`. O timeout foi corrigido para ocorrer após `max_episode_length` ações; anteriormente ocorria um passo antes.

| Parâmetro | Valor/expressão |
| --- | --- |
| `use_raptor_trajectory` | `True` |
| `trajectory_mixture_langevin_prob` | `0.5` |
| `langevin_gamma` | `1.0` |
| `langevin_omega` | `2.0` |
| `langevin_sigma` | `0.5` |
| `goal_pos_xy_range` | `None` |
| `goal_pos_z_range` | `None` |


Reset e falhas: `_reset_idx`, `_get_dones`; contratos completos por ID em `docs/review/contracts_after.json`.

## Algoritmos e hiperparâmetros

A existência de uma configuração indica suporte configurado, não prova que tenha sido usada num treino. Não foi validado nenhum checkpoint no simulador. Valores abaixo são expressões atribuídas no ficheiro; defaults herdados do SKRL dependem da versão instalada.

### SAC — `agents/sac_cfg.py`

| Parâmetro | Expressão no ficheiro |
| --- | --- |
| `PRESETS` | `{'isaac_lab': {'models': _make_models, 'cfg': _make_cfg(), 'timesteps': 1000000, 'memory_size': 1000000, 'seed': 10}}` |
| `cfg['gradient_steps']` | `1` |
| `cfg['batch_size']` | `128` |
| `cfg['discount_factor']` | `0.99` |
| `cfg['polyak']` | `0.005` |
| `cfg['actor_learning_rate']` | `0.0003` |
| `cfg['critic_learning_rate']` | `0.0003` |
| `cfg['entropy_learning_rate']` | `0.0001` |
| `cfg['learning_rate_scheduler']` | `None` |
| `cfg['learning_rate_scheduler_kwargs']` | `{}` |
| `cfg['learn_entropy']` | `True` |
| `cfg['initial_entropy_value']` | `0.5` |
| `cfg['target_entropy']` | `-2.0` |
| `cfg['policy_delay']` | `2` |
| `cfg['bootstrap_timeouts']` | `True` |
| `cfg['random_timesteps']` | `0` |
| `cfg['learning_starts']` | `10000` |
| `cfg['grad_norm_clip']` | `0` |
| `cfg['weight_decay']` | `0.0` |
| `cfg['state_preprocessor']` | `None` |
| `cfg['state_preprocessor_kwargs']` | `{}` |
| `cfg['mixed_precision']` | `False` |
| `cfg['experiment']` | `{'directory': '', 'experiment_name': '', 'write_interval': 1000, 'checkpoint_interval': 10000, 'store_separately': False, 'wandb': False, 'wandb_kwargs': {}}` |

Redes explicitamente declaradas:

```python
nn.Sequential(nn.Linear(self.num_observations, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.num_actions * 2))
nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
```

Validação v2: forma e finitude executadas em CPU com backend tensorial substituto (N=1 e N=3), incluindo isolamento do reset parcial. Não equivale a criação/voo no Isaac Sim.
