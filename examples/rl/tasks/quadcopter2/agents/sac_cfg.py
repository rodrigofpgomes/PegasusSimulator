"""
play.py — Run a trained PPO policy via skrl.
"""
import argparse
import os
import sys
import numpy as np

from isaacsim import SimulationApp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()

args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import torch
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager

from skrl.envs.loaders.torch import wrap_env
from skrl.agents.torch.ppo import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from tasks.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
    from tasks.quadcopter.agents.ppo_cfg import PRESETS, Policy, Value
    
    device = args.device
    n_envs = args.n_envs
    env_cfg = QuadcopterEnvCfg()
    agent_cfg = PRESETS["isaac_lab"]

    # -- Sim Setup --
    pg = PegasusInterface()
    pg.set_world_settings(device=device)
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
    prim_utils.create_prim("/World/Light/DomeLight", "DomeLight", position=np.array([1.0, 1.0, 1.0]))

    backend = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [backend]

    MultirotorBatch(stage_prefix="/World/quadrotor", usd_file=ROBOTS["Iris"], vehicle_batch_id=1, n_vehicles=n_envs, spacing=2.5, config=vehicle_cfg)

    env = QuadcopterEnv(env_cfg, backend, reset_manager=None)
    env._world = world

    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    env.reset_manager = ResetManager(vehicle=backend._vehicle, device=device)
    env.setup()

    # -- SKRL Wrap & Load --
    wrapped_env = wrap_env(env, wrapper="omniverse-isaacgym")

    models = {
        "policy": Policy(wrapped_env.observation_space, wrapped_env.action_space, device),
        "value": Value(wrapped_env.observation_space, wrapped_env.action_space, device)
    }

    # Instanciar o Agente apenas para carregar os pesos
    agent = PPO(models=models, memory=None, cfg=agent_cfg, observation_space=wrapped_env.observation_space, action_space=wrapped_env.action_space, device=device)
    
    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    obs, _ = wrapped_env.reset()

    print("\nRunning inference. Close the window to stop.\n")
    while simulation_app.is_running():
        with torch.no_grad():
            # act() retorna (actions, log_prob, dict)
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
            
        obs, rewards, terminated, truncated, info = wrapped_env.step(actions)
        world.step(render=not args.headless)

    timeline.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()