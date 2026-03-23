"""
train.py — Generic RL training launcher.

Discovers available tasks and algorithms automatically.
No hardcoded task or algorithm names.

Usage:
    python train.py --task quadcopter --algo ppo
    python train.py --task quadcopter --algo ppo --preset seeded
    python train.py --task quadcopter --algo ppo --seed 42 --n_envs 512
    python train.py --task quadcopter --algo ppo --headless

Task discovery:
    Looks for folders under tasks/ that contain:
        <task>/<task>_env.py          — env class  (<Task>Env + <Task>EnvCfg)
        <task>/agents/<algo>_cfg.py   — agent config (PPOConfig, SACConfig, ...)
                                        must expose a PRESETS dict

Algorithm discovery:
    Looks for:
        pegasus.simulator.logic.rl.algorithms.<algo>
        with a train(env, agent_cfg, log_dir, device) function
"""
import argparse
import importlib
import os
import sys

import carb
from isaacsim import SimulationApp


# ── task/algo discovery ───────────────────────────────────────────────

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")


def discover_tasks() -> list[str]:
    """Return sorted list of available task names found under tasks/."""
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(
        d for d in os.listdir(TASKS_DIR)
        if os.path.isdir(os.path.join(TASKS_DIR, d))
        and not d.startswith("_")
    )


def _find_algos_dir() -> str | None:
    """Locate the algorithms directory by walking up from this file."""
    # Resolve relative to this script: rl/ -> extensions/pegasus.simulator/...
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # installed as package — importlib will find it
        os.path.join(base, "..", "..", "..", "extensions",
                     "pegasus.simulator", "pegasus", "simulator",
                     "logic", "rl", "algorithms"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isdir(p):
            return p
    # fallback: ask importlib for the installed package path
    try:
        import pegasus.simulator.logic.rl.algorithms as _algos_pkg
        return os.path.dirname(_algos_pkg.__file__)
    except ImportError:
        return None


def discover_algos() -> list[str]:
    """Return sorted list of available algorithm names."""
    algos_dir = _find_algos_dir()
    if algos_dir is None:
        return ["ppo"]
    return sorted(
        f[:-3] for f in os.listdir(algos_dir)
        if f.endswith(".py") and not f.startswith("_")
    )


# ── arg parsing ───────────────────────────────────────────────────────

def parse_args():
    tasks = discover_tasks() or ["quadcopter"]
    algos = discover_algos() or ["ppo"]

    p = argparse.ArgumentParser(
        description="Generic RL training launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",    required=True, choices=tasks,
                   help=f"Task to train. Available: {tasks}")
    p.add_argument("--algo",    default="ppo",  choices=algos,
                   help=f"Algorithm. Available: {algos}")
    p.add_argument("--preset",  default="isaac_lab",
                   help="Config preset (must exist in agents/<algo>_cfg.PRESETS)")
    p.add_argument("--n_envs",  type=int,   default=4096)
    p.add_argument("--seed",    type=int,   default=None,
                   help="Override seed")
    p.add_argument("--lr",      type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--iters",   type=int,   default=None,
                   help="Override max_iterations")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--log_dir", default=os.path.join(TASKS_DIR))
    return p.parse_args()


args           = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# runtime imports
import numpy as np
import omni.timeline
from omni.isaac.core.world               import World
import isaacsim.core.utils.prims         as prim_utils
import isaacsim.core.utils.stage         as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params                            import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor_batch  import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl                          import RLBackend, ResetManager


# ── loaders ───────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(task: str):
    """Import <Task>Env and <Task>EnvCfg from tasks/<task>/<task>_env.py."""
    mod = importlib.import_module(f"tasks.{task}.{task}_env")

    cls_name = "".join(w.capitalize() for w in task.split("_")) + "Env"
    cfg_name = cls_name + "Cfg"

    EnvClass = getattr(mod, cls_name, None)
    EnvCfg   = getattr(mod, cfg_name, None)

    if EnvClass is None or EnvCfg is None:
        raise ImportError(
            f"tasks/{task}/{task}_env.py must define {cls_name} and {cfg_name}"
        )
    return EnvClass, EnvCfg()


def load_agent_cfg(task: str, algo: str, preset: str):
    """Import agent config from tasks/<task>/agents/<algo>_cfg.py."""
    mod     = importlib.import_module(f"tasks.{task}.agents.{algo}_cfg")
    PRESETS = getattr(mod, "PRESETS", {})

    if preset not in PRESETS:
        available = list(PRESETS.keys())
        raise KeyError(
            f"Preset '{preset}' not found for task='{task}' algo='{algo}'. "
            f"Available: {available}"
        )
    return PRESETS[preset]


def load_train_fn(algo: str):
    """Import train() from pegasus...algorithms.<algo>."""
    mod = importlib.import_module(
        f"pegasus.simulator.logic.rl.algorithms.{algo}"
    )
    train_fn = getattr(mod, "train", None)
    if train_fn is None:
        raise ImportError(
            f"pegasus.simulator.logic.rl.algorithms.{algo} must define train()"
        )
    return train_fn


# ── main ─────────────────────────────────────────────────────────────

def main():
    # ── load task + algo ──────────────────────────────────────
    EnvClass, env_cfg = load_env(args.task)
    agent_cfg         = load_agent_cfg(args.task, args.algo, args.preset)
    train_fn          = load_train_fn(args.algo)

    # CLI overrides
    if args.seed  is not None: agent_cfg.seed           = args.seed
    if args.lr    is not None: agent_cfg.learning_rate  = args.lr
    if args.iters is not None: agent_cfg.max_iterations = args.iters

    n_envs  = args.n_envs
    device  = args.device

    print("\n" + "="*60)
    print("  RL TRAINING")
    print("="*60)
    print(f"  Task:           {args.task}")
    print(f"  Algorithm:      {args.algo}")
    print(f"  Preset:         {args.preset}")
    print(f"  Seed:           {getattr(agent_cfg, 'seed', None)}")
    print(f"  Actor class:    {getattr(agent_cfg, 'actor_class', None) or 'default'}")
    print(f"  Num envs:       {n_envs}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Learning rate:  {agent_cfg.learning_rate}")
    print(f"  Headless:       {args.headless}")
    print("="*60 + "\n")

    # ── simulator ─────────────────────────────────────────────
    pg = PegasusInterface()
    world_settings           = dict(pg._world_settings)
    world_settings["device"] = device
    pg._world = World(**world_settings)
    world     = pg.world

    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    stage = stage_utils.get_current_stage()
    scene = stage.GetPrimAtPath("/physicsScene")
    api   = PhysxSchema.PhysxSceneAPI.Apply(scene)
    api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(24576)

    # ── vehicles ──────────────────────────────────────────────
    backend              = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg          = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [backend]

    MultirotorBatch(
        stage_prefix     = "/World/quadrotor",
        usd_file         = ROBOTS["Iris"],
        vehicle_batch_id = 1,
        n_vehicles       = n_envs,
        spacing          = 2.5,
        config           = vehicle_cfg,
    )

    # ── environment ───────────────────────────────────────────
    env        = EnvClass(env_cfg, backend, reset_manager=None)
    env._world = world

    # ── start simulation ──────────────────────────────────────
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    reset_mgr         = ResetManager(vehicle=backend._vehicle, device=device)
    env.reset_manager = reset_mgr
    env.setup()

    # ── train ─────────────────────────────────────────────────
    task_log_dir = os.path.join(args.log_dir, args.task)
    train_fn(
        env       = env,
        agent_cfg = agent_cfg,
        log_dir   = task_log_dir,
        device    = device,
    )

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()