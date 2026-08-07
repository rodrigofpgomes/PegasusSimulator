# Fase 4 — Constante positiva + penalização de morte

Abordagem alternativa à fase 3 para o mesmo problema ("aprender a morrer").

- **Observação / ação:** iguais à fase 2 (obs 6, ação 3).
- **Reward:** `reward_alive = -custo · dt + 1.0` (constante positiva de sobrevivência),
  com penalização de morte:
  `reward = -500` quando `terminated` (saída dos limites), caso contrário `reward_alive`.

A constante `+1.0` incentiva a permanência em voo e a penalização `-500` torna a
terminação explicitamente indesejável.
