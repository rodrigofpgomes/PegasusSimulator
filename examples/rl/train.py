"""
train.py — Generic RL training launcher (skrl backend).

Discovers tasks and algorithms automatically. No hardcoded names.

Usage:
    python train.py --task quadcopter --algo ppo
    python train.py --task quadcopter --algo ppo --preset seeded
    python train.py --task quadcopter --algo ppo --seed 42 --n_envs 512 --headless

Task discovery:
    tasks/<task>/<task>_env.py       → <Task>Env + <Task>EnvCfg
    tasks/<task>/agents/<algo>_cfg.py → PRESETS dict

Algorithm discovery:
    pegasus.simulator.logic.rl.algorithms.<algo>  → train(env, cfg, log_dir, device)

Logs saved to:
    tasks/<task>/logs/<timestamp>/
        config.json   — hyperparameters + model architecture
        <skrl checkpoints>
"""
import argparse
import importlib
import os
import sys

import carb
from isaacsim import SimulationApp

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


def discover_tasks():
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(d for d in os.listdir(TASKS_DIR)
                  if os.path.isdir(os.path.join(TASKS_DIR, d))
                  and not d.startswith("_"))


def discover_algos():
    try:
        import pegasus.simulator.logic.rl.algorithms as pkg
        d = os.path.dirname(pkg.__file__)
        return sorted(f[:-3] for f in os.listdir(d)
                      if f.endswith(".py") and not f.startswith("_"))
    except Exception:
        return ["ppo"]


def parse_args():
    tasks = discover_tasks() or ["quadcopter"]
    algos = discover_algos() or ["ppo"]
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",    required=True, choices=tasks)
    p.add_argument("--algo",    default="ppo", choices=algos)
    p.add_argument("--preset",  default="isaac_lab")
    p.add_argument("--n_envs",  type=int,   default=4096)
    p.add_argument("--seed",    type=int,   default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device",  default="cuda:0")
    return p.parse_args()


args           = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# ── Runtime imports ───────────────────────────────────────────────────
import numpy as np
import omni.timeline
from omni.isaac.core.world              import World
import isaacsim.core.utils.prims        as prim_utils
import isaacsim.core.utils.stage        as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params                           import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl                          import RLBackend, ResetManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(task: str):
    mod      = importlib.import_module(f"tasks.{task}.{task}_env")
    cls_name = "".join(w.capitalize() for w in task.split("_")) + "Env"
    cfg_name = cls_name + "Cfg"
    EnvClass = getattr(mod, cls_name)
    EnvCfg   = getattr(mod, cfg_name)
    return EnvClass, EnvCfg()


def load_agent_cfg(task: str, algo: str, preset: str) -> dict:
    mod     = importlib.import_module(f"tasks.{task}.agents.{algo}_cfg")
    PRESETS = getattr(mod, "PRESETS")
    if preset not in PRESETS:
        raise KeyError(f"Preset '{preset}' not found. Available: {list(PRESETS)}")
    return PRESETS[preset]


def load_train_fn(algo: str):
    mod = importlib.import_module(f"pegasus.simulator.logic.rl.algorithms.{algo}")
    return getattr(mod, "train")


def main():
    EnvClass, env_cfg = load_env(args.task)
    agent_cfg         = load_agent_cfg(args.task, args.algo, args.preset)
    train_fn          = load_train_fn(args.algo)

    # CLI overrides
    if args.seed is not None:
        agent_cfg = dict(agent_cfg)   # copy preset dict
        agent_cfg["seed"] = args.seed

    device = args.device
    n_envs = args.n_envs

    print("\n" + "="*60)
    print("  RL TRAINING (skrl)")
    print("="*60)
    print(f"  Task:      {args.task}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Preset:    {args.preset}")
    print(f"  Seed:      {agent_cfg.get('seed')}")
    print(f"  Timesteps: {agent_cfg['timesteps']}")
    print(f"  Num envs:  {n_envs}")
    print(f"  Headless:  {args.headless}")
    print("="*60 + "\n")

    # ── Simulator ─────────────────────────────────────────────
    pg = PegasusInterface()
    pg.set_world_settings(device=device)
    pg._world = World(**dict(pg._world_settings))
    world     = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    stage = stage_utils.get_current_stage()
    api   = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(24576)

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

    # ── Train ─────────────────────────────────────────────────
    log_dir = os.path.join(TASKS_DIR, args.task)
    train_fn(env=env, agent_cfg=agent_cfg, log_dir=log_dir, device=device)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()