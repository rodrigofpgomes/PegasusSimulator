"""
| Task A_active -- condicao rotor 5 controlado pela policy -- BASELINE
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
    # rotor 5 controlado pela policy -- BASELINE
    rotor5_mode = "active"

    run_tag = "A_active"
    seed_tag = 0            # muda por corrida de treino, se quiseres
    eval_csv_path = None    # em avaliacao quem grava e' o play_multi.py

    # -> action_space = 5, observation_space = 28
    #    (derivados no __post_init__, nao editar a' mao)
