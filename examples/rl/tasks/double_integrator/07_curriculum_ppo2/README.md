# Fase 7 — Curriculum de coeficientes (ppo2)

Tentativa de replicar/melhorar o treino variando os **coeficientes do custo ao longo
do treino** (*curriculum*), através de um algoritmo `ppo2` dedicado
(`agents/ppo2_cfg.py`).

- **Observação / ação:** iguais à fase 2 (obs 6, ação 3).
- **Reward:** soma ponderada `{constant, ep, ev, u}` com pesos
  (`self._reward_weights`) que evoluem por `self._curriculum_stage`. Os pesos são
  registados em `extras['log']['RewardWeights/*']`.

## Variantes (preservadas só nesta documentação)
- **`quadcopter3_4`** — curriculum sem penalização de morte.
- **`quadcopter3_7`** — curriculum com penalização de morte (−500).
- **`quadcopter3_8`** — versão final usada, com fatores `0.5` nos termos do reward
  e sem penalização de morte.
- **`quadcopter3_5` / `quadcopter3_6`** (em backup) — iterações anteriores da mesma
  família (`3_6` com o termo `u` comentado).

Mantém-se aqui a variante final (`quadcopter3_8`).

## Conclusão
A variação dos coeficientes ao longo do treino **não trouxe benefício** prático,
ficando registada para a tese como tentativa explorada.
