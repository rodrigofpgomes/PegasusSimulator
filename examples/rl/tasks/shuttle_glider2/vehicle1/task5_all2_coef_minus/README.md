# curriculum4 — Shuttle + EasyGlider (etapa 1: só rotores)

Reformulação do treino RL para o veículo conjunto (quadrotor + glider). Substitui
o gerador RAPTOR/Langevin (bom para o quad, inadequado aqui) por seguimento de
trajetória parametrizado por **airspeed + heading**, com inicialização física
(**cruise starts**) e observação/recompensa aerodinâmicas.

## Ficheiros
- `sg_geometry.py` — helpers puros de quaternião/AoA/sideslip (sem dependência do
  simulador; validados offline: `alpha = -pitch`, `beta = 0` em cruzeiro horizontal).
- `trajectory.py` — `TrackReferenceGenerator`: referências forward-biased, C², com
  rampa de airspeed limitada por `a_max` (feasibility), retas e curvas por heading,
  vertical suave, **sem ping-pong**. `v_ref(0)=V0` para casar com o cruise start.
- `curriculum_manager.py` — níveis (hover → 2..13 m/s → curvas) com bounds de reset,
  `cruise_prob`, distribuição de velocidade/curvas, `alpha_trim` e pesos aerodinâmicos.
- `reset_manager.py` — **drop-in retrocompatível** do teu ResetManager: adiciona
  `init_overrides` para injetar cruise starts (atitude de trim, velocidade em body-x,
  rotores em trim). Com `init_overrides=None` é idêntico ao original.
- `quadcopter_env.py` — ambiente: obs **34-D** em body-frame (`pos_err`, `vel_err`,
  `alpha,beta,Va`, `goal_acc`, `R`, `ang`, `last_action`, `rotores`; sem posição
  absoluta, sem `v_body`/`goal_vel` redundantes) com **normalização estática por
  escala física** (toggle `use_static_obs_norm`); reward com pos/vel/Δação + heading
  + beta + **alpha (margem de stall)** + quad_effort; cruise starts.
- `agents/ppo_cfg.py`, `agents/sac_cfg.py` — redes [256,256], `state_preprocessor=None`
  (normalização é estática no env; evita a não-estacionaridade do running scaler sob
  curriculum). `make_preset(obs, act, device)`.

## Integração (confirmar no teu harness)
1. **Cap do puller**: `vehicle_physics_cfg["glider_thrust_cfg"]["max_rotor_velocity"] = 3500`
   (era 1100). Muda o significado de `action[4]` → re-treinar/adaptar.
2. **Substituir** o `reset_manager.py` partilhado por este (é superset seguro), OU
   portar o parâmetro `init_overrides` para o teu.
3. `observation_space = 34` (automático via `QuadcopterEnvCfg`).
4. Se usares o teu **PPO2** (RSL-RL clipped value) em vez do `ppo_cfg.py` aqui,
   deixa `state_preprocessor=None` e confia na normalização estática do env
   (`use_static_obs_norm`); o `value_preprocessor` (returns) podes manter.
5. `vehicle_mass` (default 4.8957 kg) e `gravity` são usados no seed de trim; ajusta se
   a massa merged for outra.

## Notas
- Só rotores nesta etapa (`command[:,5:8]=0`); superfícies ficam para a etapa 2.
- Constantes aerodinâmicas para o seed de trim são lidas do `vehicle.aerodynamics`
  em runtime (fallback para os defaults do EasyGlider).
- Validação em simulador ainda é necessária: as partes que dependem de
  Pegasus/Isaac (poses, `set_state`, aero) não são testáveis offline.
