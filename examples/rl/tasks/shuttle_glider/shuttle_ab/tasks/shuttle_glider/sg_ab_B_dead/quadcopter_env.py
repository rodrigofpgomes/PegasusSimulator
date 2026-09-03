"""
| Task B_dead -- condicao canal 5 forcado a -1 ANTES da reward e da obs
|
| Shim fino. Toda a logica vive em sg_ab_common/env_core.py; aqui so' se
| fixam os defaults do cfg, porque o _load_env() do skrl_agent_controller
| instancia QuadcopterEnvCfg() sem aceitar overrides vindos do manifesto
| JSON. Por isso e' que cada condicao precisa da sua propria pasta.
"""

from tasks.shuttle_glider.sg_ab_common.env_core import (
    QuadcopterEnv,
    QuadcopterEnvCfg as _BaseCfg,
)

__all__ = ["QuadcopterEnv", "QuadcopterEnvCfg"]


class QuadcopterEnvCfg(_BaseCfg):
    # canal 5 forcado a -1 ANTES da reward e da obs
    rotor5_mode = "dead"

    run_tag = "B_dead"
    seed_tag = 0            # muda por corrida de treino, se quiseres
    eval_csv_path = None    # em avaliacao quem grava e' o play_multi.py

    # -> action_space = 5, observation_space = 28
    #    (derivados no __post_init__, nao editar a' mao)
