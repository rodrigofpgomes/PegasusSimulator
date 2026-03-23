"""
train_quadcopter_rsl_rl.py — Train Quadcopter with RSL-RL (Isaac Lab exact).

Usage:
    python train_quadcopter_rsl_rl.py
    python train_quadcopter_rsl_rl.py --preset fast --n_envs 512
    python train_quadcopter_rsl_rl.py --preset large --iters 2000 --headless

Checkpoints saved every save_interval iterations (default: 50)
inside logs/<experiment_name>_<timestamp>/.
"""
import argparse
import sys
import os
import numpy as np

# Isaac Sim — must be the very first import
import carb
from isaacsim import SimulationApp



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="isaac_lab_exact",
                   choices=["isaac_lab_exact", "tuned"])
    p.add_argument("--n_envs",        type=int,   default=4096)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--iters",         type=int,   default=None)
    p.add_argument("--save_interval", type=int,   default=None)
    p.add_argument("--headless",      action="store_true")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--log_dir",       default="logs")
    return p.parse_args()


args           = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# runtime imports
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager

import isaacsim.core.utils.stage as stage_utils

from pxr import PhysxSchema

# task imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
from tasks.quadcopter.agents.ppo_cfg import PRESETS

# algorithm (uses rsl_rl internally)
from pegasus.simulator.logic.rl.algorithms.ppo import train as ppo_train


def main():

    agent_cfg = PRESETS[args.preset]

    if args.lr            is not None: agent_cfg.learning_rate = args.lr
    if args.iters         is not None: agent_cfg.max_iterations = args.iters
    if args.save_interval is not None: agent_cfg.save_interval  = args.save_interval

    n_envs  = args.n_envs
    device  = args.device
    env_cfg = QuadcopterEnvCfg()

    print("\n" + "="*60)
    print("  QUADCOPTER PPO — RSL-RL (Isaac Lab exact)")
    print("="*60)
    print(f"  Preset:         {args.preset}")
    print(f"  Num envs:       {n_envs}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Save interval:  {agent_cfg.save_interval}")
    print(f"  Learning rate:  {agent_cfg.learning_rate}")
    print(f"  Steps/env:      {agent_cfg.num_steps_per_env}")
    print(f"  Actor dims:     {agent_cfg.actor_hidden_dims}")
    print(f"  Headless:       {args.headless}")
    print("="*60 + "\n")

    # ── simulator ───────────────────────────────────────────────
    pg = PegasusInterface()
    world_settings = dict(pg._world_settings)
    world_settings["device"] = device
    pg._world = World(**world_settings)
    world     = pg.world

    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    stage = stage_utils.get_current_stage()
    scenePrim = stage.GetPrimAtPath("/physicsScene")  
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scenePrim)
    physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(24576)    

    # ── vehicles ────────────────────────────────────────────────
    # n_vehicles passed here so backend.n_vehicles is available
    # before start() — PegasusEnv.__init__ reads it immediately.
    backend = RLBackend(n_vehicles=n_envs, action_mode="direct_force")

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

    # ── environment (created before start()) ────────────────────
    # backend._vehicle is None here — ResetManager must be created
    # AFTER timeline.play() + world.step() so initialize() has run.
    env = QuadcopterEnv(env_cfg, backend, reset_manager=None)
    env._world = world

    # ── start simulation ────────────────────────────────────────
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # One physics step triggers:
    #   sim_start_stop → MultirotorBatch.start() → initialize()
    #   → backend.start()  (parts_per_vehicle now available)
    #   → POST_PHYSICS_STEP → backend.update_state() (first state)
    world.step(render=False)

    # NOW _vehicle is populated and parts_per_vehicle is valid
    reset_mgr = ResetManager(vehicle=backend._vehicle, device=device)
    env.reset_manager = reset_mgr

    # Allocate episode_length_buf and other device-dependent buffers
    env.setup()

    # ── train ───────────────────────────────────────────────────
    ppo_train(
        env       = env,
        agent_cfg = agent_cfg,
        log_dir   = args.log_dir,
        device    = device,
    )

    # ── cleanup ─────────────────────────────────────────────────
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()