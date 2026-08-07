# Fase 3 — Reward exponencial

Resposta ao problema de "aprender a morrer" da fase 2.

- **Observação / ação:** iguais à fase 2 (obs 6, ação 3).
- **Reward:** `reward = exp(-custo · dt)`, com
  `custo = scale_ep·‖ep‖² + scale_ev·‖ev‖²`.

Ao mapear o custo para `(0, 1]` através da exponencial, o reward passa a ser sempre
**positivo**, pelo que manter-se vivo e perto do alvo é sempre preferível a terminar
o episódio. Eliminou o comportamento degenerado da fase 2.
