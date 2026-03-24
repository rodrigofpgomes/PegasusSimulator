"""
| File: play.py
| Description: Run a trained PPO policy (rsl_rl) in inference mode.
| License: BSD-3-Clause.

Usage:
    python play.py --checkpoint logs/quadcopter_direct_2026-03-22_01-26-04/model_650.pt
    python play.py --checkpoint logs/.../model_650.pt --n_envs 4 --device cuda
"""

import argparse
import os
import sys
import numpy as np

from isaacsim import SimulationApp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--task",    default="quadcopter")
    p.add_argument("--preset",  default="isaac_lab",
                   help="Preset name from agents/<algo>_cfg.PRESETS")
    p.add_argument("--n_envs",  type=int,   default=4)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# Runtime imports after SimulationApp
import torch
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution

# Adjust these imports if your project layout differs
sys.path.insert(0, os.path.dirname(__file__))
from tasks.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
from pegasus.simulator.logic.rl.algorithms.wrappers.rsl_rl_wrapper import RslRlVecEnvWrapper


def build_runner_cfg(agent_cfg):
    return {
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "actor": {
            "class_name": MLPModel,
            "hidden_dims": agent_cfg.actor_hidden_dims,
            "activation": agent_cfg.activation,
            "distribution_cfg": {
                "class_name": GaussianDistribution,
                "init_std": agent_cfg.init_noise_std,
            },
        },
        "critic": {
            "class_name": MLPModel,
            "hidden_dims": agent_cfg.critic_hidden_dims,
            "activation": agent_cfg.activation,
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": agent_cfg.clip_param,
            "desired_kl": agent_cfg.desired_kl,
            "entropy_coef": agent_cfg.entropy_coef,
            "gamma": agent_cfg.gamma,
            "lam": agent_cfg.lam,
            "learning_rate": agent_cfg.learning_rate,
            "max_grad_norm": agent_cfg.max_grad_norm,
            "num_learning_epochs": agent_cfg.num_learning_epochs,
            "num_mini_batches": agent_cfg.num_mini_batches,
            "schedule": agent_cfg.schedule,
            "use_clipped_value_loss": agent_cfg.use_clipped_value_loss,
            "value_loss_coef": agent_cfg.value_loss_coef,
            "rnd_cfg": None,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "save_interval": agent_cfg.save_interval,
            "empirical_normalization": False,
        },
        "multi_gpu": {},
    }


def get_inference_policy(runner, device):
    """Handle small API differences across rsl_rl versions."""
    if hasattr(runner, "get_inference_policy"):
        return runner.get_inference_policy(device=device)

    if hasattr(runner, "alg"):
        alg = runner.alg

        if hasattr(alg, "actor"):
            policy = alg.actor
            policy.eval()
            return policy

        if hasattr(alg, "actor_critic"):
            policy = alg.actor_critic
            policy.eval()
            return policy

    raise RuntimeError("Could not extract inference policy from OnPolicyRunner.")


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def move_obs_to_device(obs, device):
    """Move TensorDict/dict/tensor observations to the requested device."""
    if hasattr(obs, "to"):
        return obs.to(device)
    if isinstance(obs, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in obs.items()
        }
    return obs


def act(policy, obs):
    """Run inference with observation moved to the policy device."""
    pdev = module_device(policy)
    obs = move_obs_to_device(obs, pdev)

    if hasattr(policy, "act_inference"):
        return policy.act_inference(obs)
    return policy(obs)


def main():
    device = args.device
    n_envs = args.n_envs

    from tasks.quadcopter.agents.ppo_cfg import PRESETS
    if args.preset not in PRESETS:
        raise KeyError(f"Preset '{args.preset}' not found. Available: {list(PRESETS.keys())}")
    agent_cfg = PRESETS[args.preset]
    env_cfg   = QuadcopterEnvCfg()

    # Simulator
    pg = PegasusInterface()
    pg.set_world_settings(device=device)
    world_settings = dict(pg._world_settings)
    pg._world = World(**world_settings)
    world = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

    #collision_paths = ["/World/Layout/GroundPlane"]
    #world.scene.add_default_ground_plane(z_position=0.0)



    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    # Vehicles
    backend = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [backend]

    MultirotorBatch(
        stage_prefix="/World/quadrotor",
        usd_file=ROBOTS["Iris"],
        vehicle_batch_id=1,
        n_vehicles=n_envs,
        spacing=2.5,
        config=vehicle_cfg
    )

    # Environment
    env = QuadcopterEnv(env_cfg, backend, reset_manager=None)
    env._world = world

    # Start sim
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    reset_mgr = ResetManager(vehicle=backend._vehicle, device=device)
    env.reset_manager = reset_mgr
    env.setup()

    wrapped_env = RslRlVecEnvWrapper(
        env,
        clip_obs=agent_cfg.clip_obs,
        clip_actions=agent_cfg.clip_actions,
    )

    train_cfg_dict = build_runner_cfg(agent_cfg)

    # Build runner exactly like training
    runner = OnPolicyRunner(env=wrapped_env, train_cfg=train_cfg_dict, log_dir=None, device=device)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    try:
        runner.load(args.checkpoint)
    except TypeError:
        # compatibility fallback for versions with named argument
        try:
            runner.load(path=args.checkpoint)
        except TypeError:
            runner.load(resume_path=args.checkpoint)

    policy = get_inference_policy(runner, device=device)
    policy = policy.to(device)
    policy.eval()

    print("Policy loaded successfully.")
    print("policy device:", module_device(policy))
    print("env device:", wrapped_env.device)

    obs, _ = wrapped_env.reset()
    obs = move_obs_to_device(obs, module_device(policy))

    if isinstance(obs, dict):
        if "policy" in obs and hasattr(obs["policy"], "device"):
            print("obs['policy'] device:", obs["policy"].device)
    elif hasattr(obs, "keys") and "policy" in obs.keys():
        print("obs['policy'] device:", obs["policy"].device)

    print("\nRunning inference. Close the window to stop.\n")

    while simulation_app.is_running():
        with torch.no_grad():
            actions = act(policy, obs)
            actions = actions.to(wrapped_env.device)

        obs, rewards, dones, extras = wrapped_env.step(actions)
        obs = move_obs_to_device(obs, module_device(policy))

        world.step(render=not args.headless)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()