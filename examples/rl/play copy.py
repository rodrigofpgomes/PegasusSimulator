"""
| File: play.py
| Description: Run a trained policy without training (inference only).
| License: BSD-3-Clause.

Usage:
    python play.py --checkpoint logs/quadcopter_ppo_2025-.../model_1500.pt
    python play.py --checkpoint logs/.../model_1500.pt --n_envs 4 --task quadcopter
"""
import argparse
import sys
import os

import carb
from isaacsim import SimulationApp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint from rsl_rl")
    p.add_argument("--task",    default="quadcopter_env",
                   choices=["quadcopter_env"])
    p.add_argument("--n_envs",  type=int,   default=4)
    p.add_argument("--device",  default="cuda")
    return p.parse_args()


args           = parse_args()
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor_batch import (
    MultirotorBatch, MultirotorBatchConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tasks.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg


def main():
    device  = args.device
    n_envs  = args.n_envs
    env_cfg = QuadcopterEnvCfg()

    # ── simulator ──────────────────────────────────────────────
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

    # ── vehicles ───────────────────────────────────────────────
    backend         = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg     = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [backend]

    MultirotorBatch(
        stage_prefix    = "/World/quadrotor",
        usd_file        = ROBOTS["Iris"],
        vehicle_batch_id = 1,
        n_vehicles      = n_envs,
        spacing         = 2.5,
        config          = vehicle_cfg,
    )

    # ── environment (created before start()) ──────────────────
    # reset_manager=None until after world.step() initialises backend
    env        = QuadcopterEnv(env_cfg, backend, reset_manager=None)
    env._world = world

    # ── start simulation ───────────────────────────────────────
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # One step triggers MultirotorBatch.start() → backend.start()
    world.step(render=False)

    # NOW backend._vehicle is populated and parts_per_vehicle is valid
    reset_mgr      = ResetManager(vehicle=backend._vehicle, device=device)
    env.reset_manager = reset_mgr
    env.setup()

    # ── load checkpoint ────────────────────────────────────────
    # rsl_rl saves checkpoints as {"model_state_dict": ..., "iter": ...}
    # The actor_critic is stored in runner.alg.actor_critic
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Reconstruct network from saved config
    # rsl_rl saves the full state dict — we rebuild from the saved policy config
    from rsl_rl.modules import ActorCritic
    obs_dim = env.num_obs
    act_dim = env.num_actions

    # Try to read architecture from checkpoint if available
    actor_hidden  = ckpt.get("actor_hidden_dims",  [64, 64])
    critic_hidden = ckpt.get("critic_hidden_dims", [64, 64])
    activation    = ckpt.get("activation",         "elu")
    noise_std     = ckpt.get("init_noise_std",     1.0)

    policy = ActorCritic(
        num_actor_obs   = obs_dim,
        num_critic_obs  = obs_dim,
        num_actions     = act_dim,
        actor_hidden_dims  = actor_hidden,
        critic_hidden_dims = critic_hidden,
        activation      = activation,
        init_noise_std  = noise_std,
    ).to(device)

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    print("Policy loaded successfully.\n")

    # ── inference loop ─────────────────────────────────────────
    obs, _ = env.reset()
    obs    = obs["policy"]

    print("Running inference. Close the window to stop.\n")
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy.act_inference(obs)

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        world.step(render=True)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()