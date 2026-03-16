"""
play.py — corre uma política treinada sem treino.

Uso:
    python play.py --task hover --algo ppo --checkpoint logs/hover_ppo/checkpoint_1000.pt
"""
import argparse
import torch
import numpy as np

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.transforms import euler_angles_to_matrix, matrix_to_quaternion
from pegasus.simulator.logic.rl import RLBackend, ResetManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="hover", choices=["hover", "trajectory"])
    p.add_argument("--algo",       default="ppo",   choices=["ppo", "sac"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_envs",     type=int, default=4)
    return p.parse_args()


def main():
    args  = parse_args()
    device = "cuda"

    if args.task == "hover":
        from tasks.hover.hover_env import HoverEnv, HoverEnvCfg
        env_cfg  = HoverEnvCfg()
        EnvClass = HoverEnv

    elif args.task == "trajectory":
        from tasks.trajectory.trajectory_env import TrajectoryEnv, TrajectoryEnvCfg
        env_cfg  = TrajectoryEnvCfg()
        EnvClass = TrajectoryEnv

    n_envs = args.n_envs

    # ── simulador ─────────────────────────────────────────────
    pg = PegasusInterface()
    world_settings = dict(pg._world_settings)
    world_settings["device"] = device
    pg._world = World(**world_settings)
    world = pg.world

    spacing  = 2.5
    init_pos = torch.tensor(
        [[i * spacing, 0.0, 1.5] for i in range(n_envs)],
        dtype=torch.float32, device=device,
    )
    init_ori = torch.stack([
        matrix_to_quaternion(
            euler_angles_to_matrix(torch.tensor([0., 0., 0.], device=device), "XYZ")
        ) for _ in range(n_envs)
    ])

    backend = RLBackend(n_vehicles=n_envs, device=device)
    config  = MultirotorBatchConfig(n_vehicles=n_envs)
    config.backends = [backend]
    MultirotorBatch(
        stage_prefix="/World/quadrotor", usd_file=ROBOTS["Iris"],
        vehicle_batch_id=1, n_vehicles=n_envs,
        init_pos=init_pos.tolist(), init_orientation=init_ori.tolist(),
        config=config,
    )

    reset_mgr = ResetManager(backend.vehicle, init_pos, init_ori, device)
    env = EnvClass(env_cfg, backend, reset_mgr)

    # ── carrega checkpoint ────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device)
    algo_cfg = ckpt["cfg"]

    # reconstrói a rede com a factory guardada na cfg
    network = algo_cfg.network_factory(env.num_obs, env.num_actions).to(device)
    network.load_state_dict(ckpt["network"])
    network.eval()

    # ── loop de inferência ────────────────────────────────────
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    obs, _ = env.reset()
    obs = obs["policy"]

    while simulation_app.is_running():
        actions = network.act_inference(obs)
        obs_dict, reward, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        world.step(render=True)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
