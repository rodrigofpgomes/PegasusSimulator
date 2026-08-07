# RAPTOR — pré-treino de um único veículo

Fase atual. Reproduz o **pré-treino** do RAPTOR: treina um único quadrotor de
dinâmica fixa, à semelhança da fase de *pre-training* do paper.

- **Observação (26):** estado estendido usado pela policy professora do RAPTOR.
- **Ação (4):** velocidades dos rotores (controlo direto dos motores).
- **Dinâmica de simulação:** `decimation = 1`, 100 Hz, episódios de 500 passos (5 s).
- **`trajectory.py`:** geração de trajetórias de referência (ex.: estática /
  lemniscata) para avaliação.
- **Policy:** SAC (`agents/sac_cfg.py`).

Os *backends* de inferência associados (RAPTOR foundation/teacher) estão em
`../../utils/foundation_policy*.py` e `../../utils/single_quad_policy.py`.
