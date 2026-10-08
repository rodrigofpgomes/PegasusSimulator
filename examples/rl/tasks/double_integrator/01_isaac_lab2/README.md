# Fase 1 — Reward base do Isaac Lab

Ponto de partida do estudo, adaptado diretamente do *environment* de quadcopter do
Isaac Lab.

- **Observação (12):** `lin_vel_b` (3), `ang_vel_b` (3), `projected_gravity_b` (3),
  `desired_pos_b` (3 - posição do alvo no referencial do corpo).
- **Ação (4):** `action[0]` → impulso coletivo escalado por `thrust_to_weight`;
  `action[1:]` → binários no referencial do corpo escalados por `moment_scale`.
- **Terminação:** `z < 0.1` ou `z > 2.0`.
- **Reward:** formulação original do Isaac Lab (distância ao alvo + termos de
  estabilização).

Serviu de referência antes de migrar para o modelo de **duplo integrador** (controlo
em aceleração, obs 6 / ação 3) explorado nas fases seguintes.
