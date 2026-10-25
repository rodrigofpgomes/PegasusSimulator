#!/usr/bin/env python
"""
| File: train.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Generic RL training launcher with task and algorithm discovery.
|
| Each policy decision advances cfg.decimation physics steps at cfg.sim_dt.
| During graphical training, skrl requests one separate render after the
| decision. Drawing does not advance the simulation clock.

Usage:
    python train.py --task shuttle/thrust_moment/isaac_lab --algo ppo
    python train.py --task shuttle/direct_rotor/baseline/raptor_tracking --algo ppo --seed 42 --n_envs 512 --headless
"""

# Standard library imports
import os
import sys
import copy
import ast
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_loader import TASKS_DIR, discover_tasks, load_agent_cfg, load_env, resolve_preset

# Isaac Sim imports
from isaacsim import SimulationApp

# =============================================================================
# Task and algorithm discovery
# =============================================================================

def discover_algos():
    """Finds algorithm modules that expose a top-level train function."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    algo_dir = os.path.join(repo_root, "extensions", "pegasus.simulator", "pegasus", "simulator", "logic", "rl", "algorithms")
    
    algos = []
    for f in os.listdir(algo_dir):
        if not f.endswith(".py") or f.startswith("_"):
            continue

        # Helper modules such as transition_contract.py are not trainable algorithms
        with open(os.path.join(algo_dir, f), encoding="utf-8") as source:
            module = ast.parse(source.read(), filename=f)

        if any(isinstance(node, ast.FunctionDef) and node.name == "train" for node in module.body):
            algos.append(f[:-3])

    return sorted(algos)


def parse_args():
    """Parse the training task, agent preset and simulation options."""
    tasks = discover_tasks()
    algos = discover_algos()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",    required=True, choices=tasks)
    p.add_argument("--algo",    default="ppo", choices=algos)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--timesteps", type=int)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to agent checkpoint to resume training")
    p.add_argument("--preset",  default=None, help="Agent preset; defaults to 'default' or the only preset.")
    p.add_argument("--n_envs",  type=int,   default=4096)
    p.add_argument("--seed",    type=int,   default=1)
    p.add_argument("--env-cfg", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a field of the task's env config (repeatable); VALUE is a Python literal.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device",  default="cuda:0")
    return p.parse_args()

args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# =============================================================================
# Imports that require an initialized SimulationApp
# =============================================================================
import torch
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.vehicles.shuttle_glider_batch2 import ShuttleGliderBatch2, ShuttleGliderBatchConfig2
from pegasus.simulator.logic.vehicles.glider_batch import GliderBatch, GliderBatchConfig
from pegasus.simulator.logic.vehicles.vehicle_physics import get_vehicle_physics
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager, GoalCfg


def load_train_fn(algo: str):
    """Loads the training function for the specified algorithm."""
    return getattr(importlib.import_module(f"pegasus.simulator.logic.rl.algorithms.{algo}"), "train")


# =============================================================================
# Training lifecycle
# =============================================================================

def main():
    """Build the task and vehicle, then run training or a short physics smoke test."""
    # Resolve the task before creating its physics model and agent.
    EnvClass, env_cfg = load_env(args.task)
    args.preset = resolve_preset(args.task, args.algo, args.preset)
    agent_cfg = copy.deepcopy(load_agent_cfg(args.task, args.algo, args.preset))
    train_fn = load_train_fn(args.algo)

    # Env-config overrides are recorded in run_metadata.json through the env cfg snapshot.
    for item in args.env_cfg:
        key, sep, raw = item.partition("=")
        if not sep or not hasattr(env_cfg, key):
            raise SystemExit(f"--env-cfg '{item}': expected KEY=VALUE with KEY a field of {type(env_cfg).__name__}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(env_cfg, key, value)
        print(f"[env-cfg] {key} = {value!r}")

    # Apply command-line overrides to a copy of the task's agent preset.
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    agent_cfg["task"] = args.task
    if args.timesteps is not None:
        agent_cfg["timesteps"] = args.timesteps
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

    from skrl.utils import set_seed
    set_seed(agent_cfg["seed"])  # Seed before environment setup and curriculum sampling.

    # One policy decision spans decimation physics steps; rendering is separate.
    pg = PegasusInterface()
    pg.set_world_settings(
        physics_dt=env_cfg.sim_dt,
        rendering_dt=env_cfg.sim_dt * env_cfg.decimation,
        device=device,
    )
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    if "cuda" in device:
        # Increase GPU Found Lost Aggregate Pairs Capacity
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # Attach the RL backend to the task's selected vehicle model.
    backend = RLBackend(
        n_vehicles=n_envs,
        action_mode=env_cfg.action_mode,
        device=device,
        goal_markers_enabled=not args.headless,
    )
    # A task may record experimental physics; otherwise use the vehicle baseline.
    physics_cfg = getattr(env_cfg, "vehicle_physics_cfg", None)
    if physics_cfg is None:
        physics_cfg = get_vehicle_physics(env_cfg.vehicle)
    
    if env_cfg.vehicle == "Shuttle_glider2":
        vehicle_cfg = ShuttleGliderBatchConfig2(cfg=physics_cfg, n_vehicles=n_envs)
        VehicleClass = ShuttleGliderBatch2
    elif env_cfg.vehicle == "Easyglider":
        vehicle_cfg = GliderBatchConfig(n_vehicles=n_envs, thrust_curve_cfg=physics_cfg)
        VehicleClass = GliderBatch
    else:
        vehicle_cfg = MultirotorBatchConfig(cfg=physics_cfg, n_vehicles=n_envs) # Shuttle
        VehicleClass = MultirotorBatch
    
    vehicle_cfg.backends = [backend]

    vehicle = VehicleClass(
        stage_prefix="/World/quadrotor", usd_file=ROBOTS[env_cfg.vehicle],
        vehicle_batch_id=1, n_vehicles=n_envs, spacing=2.5,
        config=vehicle_cfg,
    )

    # Grid neighbours are not isolated by PhysX; disable contacts as evaluation does
    vehicle.disable_collisions()

    env = EnvClass(env_cfg, backend, reset_manager=None)
    env._world = world

    # Initialize the prim views before the environment starts resetting vehicles.
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)
    if not args.headless:
        world.render()

    goal_cfg = GoalCfg(goal_pos_xy_range=env_cfg.goal_pos_xy_range, goal_pos_z_range=env_cfg.goal_pos_z_range)
    init_state_cfg = getattr(env_cfg, "init_state_cfg", None) if getattr(env_cfg, "randomize_init_state", False) else None

    env.reset_manager = ResetManager(vehicles=[backend._vehicle], device=device, goal_cfg=goal_cfg, init_state_cfg=init_state_cfg)
    env.setup()

    # Keep rendering separate from physics in both headless and graphical training.
    # SequentialTrainer calls the wrapper's render() once after each decision.
    env._render_enabled = False

    if args.smoke_test:
        # Check observations, physics cadence, render cadence and partial resets.
        obs, _ = env.reset()
        assert obs["policy"].shape == (n_envs, env_cfg.observation_space)
        physics_steps_before = getattr(backend._vehicle, "_state_update_count", 0)
        for _ in range(8):
            obs, reward, terminated, truncated, info = env.step(torch.zeros((n_envs, env_cfg.action_space), device=device))
            assert obs["policy"].shape == (n_envs, env_cfg.observation_space)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        assert backend._vehicle._state_update_count - physics_steps_before == 8 * env_cfg.decimation

        # Drawing one frame must leave the simulation clock and physics callbacks unchanged.
        time_before_render = world.current_time
        physics_steps_before_render = backend._vehicle._state_update_count
        env.render()
        assert world.current_time == time_before_render
        assert backend._vehicle._state_update_count == physics_steps_before_render

        before_pos = backend._vehicle.state.position.clone()
        before_vel = backend._vehicle.state.linear_velocity.clone()
        env._reset_idx(torch.tensor([0], device=device))
        if n_envs > 1:
            assert torch.equal(before_pos[1:], backend._vehicle.state.position[1:])
            assert torch.equal(before_vel[1:], backend._vehicle.state.linear_velocity[1:])
        print("SMOKE PASSED: 8 control steps, exact physics decimation, render without physics, observations, partial reset")
        backend._vehicle.close()
        timeline.stop()
        return

    # The algorithm module owns the optimization loop and checkpoint output.
    log_dir = args.log_dir or os.path.join(str(TASKS_DIR), *args.task.split("/"), "logs")
    train_fn(env=env, agent_cfg=agent_cfg, log_dir=log_dir, device=device, headless=args.headless, checkpoint=args.checkpoint)

    # Release the simulation resources after a completed training run.
    timeline.stop()
    backend._vehicle.close()
    world.clear_instance()

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("\n[TRAIN] Unhandled exception:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
