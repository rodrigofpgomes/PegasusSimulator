# Fase 5 — Função de Lyapunov no reward

Introdução de um termo baseado numa **função de Lyapunov** (analítica) no sinal de
reward, para promover convergência estável para o alvo.

- **Observação / ação:** iguais à fase 2 (obs 6, ação 3).
- **Reward:** combina os termos de erro com um termo de Lyapunov (decaimento de `V`).

## Variantes (preservadas só nesta documentação)
Existiram duas iterações deste *environment*:
- **`quadcopter5`** — versão inicial com o termo de esforço de controlo `u` ativo.
- **`quadcopter6`** — versão usada como final, com o termo `u` comentado (apenas
  erro + Lyapunov).

Mantém-se aqui a variante final (`quadcopter6`).
