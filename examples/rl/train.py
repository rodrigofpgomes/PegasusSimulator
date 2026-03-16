"""
train.py — entry point para treino RL no Pegasus Simulator.

Uso:
    python train.py --task hover   --algo ppo
    python train.py --task hover   --algo sac
    python train.py --task trajectory --algo ppo

    # override de hiperparâmetros na linha de comandos
    python train.py --task hover --algo ppo --lr 3e-4 --n_envs 128

    # rede customizada: edita a secção "REDE CUSTOM" abaixo

Localização recomendada:
    examples/rl/train.py

Mapa de imports:
    isaacsim, carb, omni.*
        → instalados com o Isaac Sim

    pegasus.simulator.*
        → extensions/pegasus.simulator/pegasus/simulator/

    tasks.*
        → examples/rl/tasks/  (relativo a este ficheiro)
        adiciona examples/rl/ ao PYTHONPATH antes de correr:
        export PYTHONPATH=$PYTHONPATH:/path/to/examples/rl
"""
import argparse
import sys
import os
import torch
import numpy as np

# ── Isaac Sim — TEM de ser o primeiro import antes de tudo ───
# Sem isto o simulador crasha ao importar qualquer módulo Pegasus.
import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ── Isaac Sim runtime — só disponíveis após SimulationApp() ──
import omni.timeline
from omni.isaac.core.world import World

# utilitário para criar prims USD (luzes, câmara, etc.)
import isaacsim.core.utils.prims as prim_utils

# ── Pegasus — params, veículos, interface ─────────────────────
# extensions/pegasus.simulator/pegasus/simulator/params.py
from pegasus.simulator.params import ROBOTS

# extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotor_batch.py
from pegasus.simulator.logic.vehicles.multirotor_batch import (
    MultirotorBatch,
    MultirotorBatchConfig,
)

# extensions/pegasus.simulator/pegasus/simulator/logic/interface/pegasus_interface.py
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# extensions/pegasus.simulator/pegasus/simulator/logic/transforms.py
from pegasus.simulator.logic.transforms import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

# ── RL engine — dentro do package Pegasus instalado ──────────
# extensions/pegasus.simulator/pegasus/simulator/logic/rl/
from pegasus.simulator.logic.rl import RLBackend, ResetManager, PPORunner, SACRunner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",   default="hover",  choices=["hover", "trajectory"])
    p.add_argument("--algo",   default="ppo",    choices=["ppo", "sac"])
    p.add_argument("--n_envs", type=int,   default=None)
    p.add_argument("--lr",     type=float, default=None)
    p.add_argument("--iters",  type=int,   default=None)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── carrega env + algo cfg consoante a task ───────────────
    if args.task == "hover":
        # examples/rl/tasks/hover/hover_env.py
        from tasks.hover.hover_env import HoverEnv, HoverEnvCfg
        env_cfg = HoverEnvCfg()

        if args.algo == "ppo":
            # examples/rl/tasks/hover/agents/ppo_cfg.py
            from tasks.hover.agents.ppo_cfg import HoverPPOCfg
            algo_cfg = HoverPPOCfg()
        elif args.algo == "sac":
            # examples/rl/tasks/hover/agents/sac_cfg.py
            from tasks.hover.agents.sac_cfg import HoverSACCfg
            algo_cfg = HoverSACCfg()

        EnvClass = HoverEnv

    elif args.task == "trajectory":
        # examples/rl/tasks/trajectory/trajectory_env.py
        from tasks.trajectory.trajectory_env import TrajectoryEnv, TrajectoryEnvCfg
        env_cfg = TrajectoryEnvCfg()

        if args.algo == "ppo":
            # examples/rl/tasks/trajectory/agents/ppo_cfg.py
            from tasks.trajectory.agents.ppo_cfg import TrajectoryPPOCfg
            algo_cfg = TrajectoryPPOCfg()
        elif args.algo == "sac":
            # examples/rl/tasks/trajectory/agents/sac_cfg.py
            from tasks.trajectory.agents.sac_cfg import TrajectorySACCfg
            algo_cfg = TrajectorySACCfg()

        EnvClass = TrajectoryEnv

    # ── overrides da linha de comando ─────────────────────────
    if args.lr     is not None: algo_cfg.learning_rate   = args.lr
    if args.iters  is not None: algo_cfg.max_iterations  = args.iters

    n_envs = args.n_envs or 64

    # ────────────────────────────────────────────────────────────
    # REDE CUSTOM — descomenta e substitui network_factory
    # para usar a tua própria arquitetura
    # ────────────────────────────────────────────────────────────
    # from tasks.hover.networks.custom_net import ResidualActorCritic
    # algo_cfg.network_factory = lambda obs_dim, act_dim: ResidualActorCritic(
    #     obs_dim, act_dim, hidden=256
    # )
    #
    # from pegasus.simulator.logic.rl.networks.lstm import LstmActorCritic
    # algo_cfg.network_factory = lambda obs_dim, act_dim: LstmActorCritic(
    #     obs_dim, act_dim, hidden_size=256, num_layers=1
    # )
    # ────────────────────────────────────────────────────────────

    # ── inicializa simulador ──────────────────────────────────
    pg = PegasusInterface()

    world_settings = dict(pg._world_settings)
    world_settings["device"] = "cuda"
    pg._world = World(**world_settings)
    world     = pg.world

    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={
            "inputs:intensity": 5e3,
            "inputs:color": (1.0, 1.0, 1.0),
        }
    )

    # ── posições iniciais ─────────────────────────────────────
    device = "cuda"
    spacing = 2.5
    

    # ── backend RL ────────────────────────────────────────────
    # action_mode vem da cfg do environment:
    #   "direct_force"   → treino — política aprende [thrust, tx, ty, tz]
    #   "rotor_velocity" → deploy — política aprende velocidades normalizadas
    backend = RLBackend(
        n_vehicles  = n_envs,
        device      = device,
        action_mode = env_cfg.action_mode,
    )

    config = MultirotorBatchConfig(n_vehicles=n_envs)
    config.backends = [backend]

    MultirotorBatch(
        stage_prefix     = "/World/quadrotor",
        usd_file         = ROBOTS["Iris"],
        vehicle_batch_id = 1,
        n_vehicles       = n_envs,
        spacing          = spacing,
        config           = config,
    )

    # ── reset manager ─────────────────────────────────────────
    # ResetManager usa vehicle.init_pos e vehicle.init_orientation
    # que o VehicleBatch guardou durante _spawn_batch()
    reset_mgr = ResetManager(
        vehicle = backend._vehicle,
        device  = device,
    )

    # ── environment ───────────────────────────────────────────
    env = EnvClass(env_cfg, backend, reset_mgr)

    # ── runner ────────────────────────────────────────────────
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    if args.algo == "ppo":
        runner = PPORunner(env, algo_cfg, world=world)
        runner.learn()
    elif args.algo == "sac":
        # SACRunner segue o mesmo padrão
        from pegasus.simulator.logic.rl.runners.sac_runner import SACRunner
        runner = SACRunner(env, algo_cfg, world=world)
        runner.learn()

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()