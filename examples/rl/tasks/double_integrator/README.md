# Estudo do duplo integrador / *reward shaping* (hover)

Esta pasta agrupa, por ordem cronológica de desenvolvimento, as fases do estudo do
controlador de *hover* aprendido por RL. Cada subpasta é um *environment* (task)
independente, registado e descoberto pelo `train.py`/`play.py`.

Manteve-se **apenas a versão final de cada fase**; as variantes intermédias estão
documentadas no README de cada fase para referência da tese.

| Fase | Pasta | Ideia central | Espaço obs / ação |
|------|-------|---------------|-------------------|
| 1 | `01_isaac_lab` | Reward base do Isaac Lab (referência de partida) | obs 12 / ação 4 (impulso coletivo + binários) |
| 2 | `02_quadratic` | Erro quadrático negado puro de posição e velocidade | obs 6 / ação 3 (aceleração desejada) |
| 3 | `03_exp` | Reward exponencial `exp(-custo)` para evitar que o drone "aprenda a morrer" | obs 6 / ação 3 |
| 4 | `04_constant_death` | Constante positiva no reward + penalização de morte (−500) | obs 6 / ação 3 |
| 5 | `05_lyapunov_reward` | Função de Lyapunov adicionada ao reward; análise do impacto | obs 6 / ação 3 |
| 6 | `06_lyapunov_learned` | Rede de Lyapunov **aprendida** usada como aproximação no reward | obs 6 / ação 3 |
| 7 | `07_curriculum_ppo2` | `ppo2`: coeficientes do custo variáveis ao longo do treino (*curriculum*) | obs 6 / ação 3 |

> A fase RAPTOR (pré-treino de um único veículo) está em `../raptor_pretrain`,
> por ter um espaço de observação/ação e dinâmica de treino distintos.

## Notas transversais
- **Ação (fases 2–7):** aceleração desejada normalizada `[-1, 1]` reescalada para
  `[-20, 20] m/s²`, aplicada via `RLBackend` no modo de força/binário.
- **Observação (fases 2–7):** `ep = goal_pos - pos` (3) e `ev = -R(quat)·v_body` (3).
- **Terminação:** `z < min_altitude` ou `z > max_altitude` (truncatura por timeout).
- Cada fase tem `agents/` com as configs de policy (`ppo_cfg.py`, `sac_cfg.py`,
  e `ppo2_cfg.py` na fase 7).
