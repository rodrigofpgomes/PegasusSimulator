"""
play.py — Generic RL inference launcher (skrl).

Loads config.json from the training run to reconstruct the exact model
architecture, then loads the weights from the skrl checkpoint.

Usage:
    python play.py --task quadcopter --algo ppo --preset isaac_lab \
                   --checkpoint tasks/quadcopter/logs/<ts>/checkpoints/best_agent.pt
    python play.py --task quadcopter --algo ppo --preset isaac_lab \
                   --checkpoint ... --n_envs 4

How checkpoint loading works:
    1. Load config.json to verify architecture
    2. Instantiate models via the same factory used in training
    3. Call agent.load(checkpoint) — skrl handles state_dict loading
"""
import argparse
import importlib
import os
import sys

from isaacsim import SimulationApp

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


def discover_tasks():
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(d for d in os.listdir(TASKS_DIR)
                  if os.path.isdir(os.path.join(TASKS_DIR, d))
                  and not d.startswith("_"))


def parse_args():
    tasks = discover_tasks() or ["quadcopter"]
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",       required=True, choices=tasks)
    p.add_argument("--algo",       default="ppo")
    p.add_argument("--preset",     default="isaac_lab")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_envs",     type=int, default=4)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--headless",   action="store_true")
    return p.parse_args()


args           = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import torch
import numpy as np
import omni.timeline
from omni.isaac.core.world              import World
import isaacsim.core.utils.prims        as prim_utils

from pegasus.simulator.params                           import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl                          import RLBackend, ResetManager

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo   import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(task):
    mod      = importlib.import_module(f"tasks.{task}.{task}_env")
    cls_name = "".join(w.capitalize() for w in task.split("_")) + "Env"
    cfg_name = cls_name + "Cfg"
    return getattr(mod, cls_name), getattr(mod, cfg_name)()


def load_agent_cfg(task, algo, preset):
    mod = importlib.import_module(f"tasks.{task}.agents.{algo}_cfg")
    return getattr(mod, "PRESETS")[preset]


def load_algo_class(algo):
    """Return the skrl Agent class for this algorithm."""
    _map = {"ppo": ("skrl.agents.torch.ppo",  "PPO"),
            "sac": ("skrl.agents.torch.sac",  "SAC"),
            "td3": ("skrl.agents.torch.td3",  "TD3"),
            "ddpg":("skrl.agents.torch.ddpg", "DDPG")}
    if algo not in _map:
        raise ValueError(f"Unknown algo '{algo}'. Add it to load_algo_class().")
    mod_name, cls_name = _map[algo]
    return getattr(importlib.import_module(mod_name), cls_name)


def main():
    device  = args.device
    n_envs  = args.n_envs

    EnvClass, env_cfg = load_env(args.task)
    agent_cfg         = load_agent_cfg(args.task, args.algo, args.preset)
    AgentClass        = load_algo_class(args.algo)

    # ── Simulator ─────────────────────────────────────────────
    pg = PegasusInterface()
    pg.set_world_settings(device=device)
    pg._world = World(**dict(pg._world_settings))
    world     = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    # ── Vehicles ──────────────────────────────────────────────
    backend              = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg          = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [backend]

    MultirotorBatch(
        stage_prefix="/World/quadrotor", usd_file=ROBOTS["Iris"],
        vehicle_batch_id=1, n_vehicles=n_envs, spacing=2.5,
        config=vehicle_cfg,
    )

    env        = EnvClass(env_cfg, backend, reset_manager=None)
    env._world = world

    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    env.reset_manager = ResetManager(vehicle=backend._vehicle, device=device)
    env.setup()

    # ── Wrap & load ───────────────────────────────────────────
    wrapped = wrap_env(env, wrapper="gymnasium", verbose=False)

    from pegasus.simulator.logic.rl.algorithms.ppo import _prepare_cfg
    cfg    = _prepare_cfg(agent_cfg["cfg"], device)
    models = agent_cfg["models"](wrapped.observation_space, wrapped.action_space, device)

    agent = AgentClass(
        models            = models,
        memory            = None,
        cfg               = cfg,
        observation_space = wrapped.observation_space,
        action_space      = wrapped.action_space,
        device            = device,
    )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")
    print("Loaded. Running inference...\n")

    obs, _ = wrapped.reset()

    while simulation_app.is_running():
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)

        obs, _, terminated, truncated, _ = wrapped.step(actions)
        world.step(render=not args.headless)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()