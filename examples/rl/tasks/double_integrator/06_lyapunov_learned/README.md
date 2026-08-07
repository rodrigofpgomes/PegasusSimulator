# Fase 6 — Rede de Lyapunov aprendida

Em vez de uma função de Lyapunov analítica (fase 5), aqui **aprende-se** uma função
de Lyapunov (`lyapunov_network.py`) que é depois usada como aproximação no reward.

- **Observação / ação:** iguais à fase 2 (obs 6, ação 3).
- **`lyapunov_network.py`:** definição da rede que aproxima `V`.
- **`weights/`:** pesos treinados preservados —
  - `best_agent_v7.pt` e `best_agent_v8.pt` (duas sessões de treino; o *environment*
    e a rede são idênticos, apenas mudam os pesos do agente).

## Variantes
As pastas originais `quadcopter7` e `quadcopter8` tinham o **mesmo** código de
*environment* e de rede; só diferiam no checkpoint do agente. Ambos os checkpoints
foram mantidos em `weights/`.
