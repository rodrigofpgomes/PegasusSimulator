# Fase 2 — Erro quadrático negado (puro)

Primeiro *environment* de duplo integrador (controlo em aceleração).

- **Observação (6):** `ep = goal_pos - pos` (3), `ev = -R(quat)·v_body` (3).
- **Ação (3):** aceleração desejada `[-1,1] → [-20,20] m/s²`.
- **Reward:** `scale_ep·‖ep‖² + scale_ev·‖ev‖²` (erro quadrático negado, sem termos
  adicionais).

## Observação experimental
Com o reward puramente quadrático negado, **o drone aprende a "morrer"**: como cada
passo acumula custo negativo, terminar o episódio cedo (saindo dos limites) maximiza
o retorno. Este problema motivou as fases 3 e 4 (reward exponencial / constante
positiva + penalização de morte).
