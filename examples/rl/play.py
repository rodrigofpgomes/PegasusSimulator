#!/usr/bin/env python
"""
| File: play.py
| Description: Generic RL inference launcher (skrl). Loads config and weights to run the model.
| License: BSD-3-Clause.
"""

import os
import sys
import csv
import copy
import argparse
import importlib
import importlib.util
from pathlib import Path

import torch
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent.parent
TASKS_DIR = RL_DIR / "tasks"
PEGASUS_EXT_DIR = PROJECT_ROOT / "extensions" / "pegasus.simulator"

for p in [str(RL_DIR), str(PROJECT_ROOT), str(PEGASUS_EXT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------
# Recorder utils
# ---------------------------------------------------------------------
def _to_np_ref(value, n_envs: int):
    if value is None:
        return np.full((n_envs, 3), np.nan, dtype=np.float32)

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] < 3:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

        arr = arr[:3][None, :]
        arr = np.repeat(arr, n_envs, axis=0)

    elif arr.ndim == 2:
        if arr.shape[1] < 3:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

        arr = arr[:, :3]

        if arr.shape[0] == 1:
            arr = np.repeat(arr, n_envs, axis=0)
        elif arr.shape[0] >= n_envs:
            arr = arr[:n_envs]
        else:
            return np.full((n_envs, 3), np.nan, dtype=np.float32)

    else:
        return np.full((n_envs, 3), np.nan, dtype=np.float32)

    return arr


def make_reference_provider(reset_manager, n_envs: int):
    """
    Reads reference from ResetManager.

    Position reference:
        reset_manager.goal_pos or reset_manager._goal_pos

    Velocity reference:
        zero velocity
    """

    def provider():
        if reset_manager is None:
            return {
                "position": np.full((n_envs, 3), np.nan, dtype=np.float32),
                "velocity": np.full((n_envs, 3), np.nan, dtype=np.float32),
            }

        ref_pos = _to_np_ref(reset_manager.goal_pos, n_envs)
        ref_vel = np.zeros((n_envs, 3), dtype=np.float32)

        return {
            "position": ref_pos,
            "velocity": ref_vel,
        }

    return provider


class EpisodeTrajectoryRecorder:
    """
    Records position, velocity, reference goal and applied body-frame force/torque
    of RL and optional LQR vehicles.

    Output:
      trajectories.csv:
        controller, env_id, episode_id, episode_step, global_step, t,
        x, y, z, vx, vy, vz, speed,
        goal_x, goal_y, goal_z,
        fx_body, fy_body, fz_body, body_force_norm,
        tx_body, ty_body, tz_body, body_torque_norm

      episodes.csv:
        env_id, episode_id, end_global_step, length_steps, duration_s, reason
    """

    def __init__(
        self,
        vehicles: dict,
        n_envs: int,
        step_dt: float,
        out_dir: str,
        record_every: int = 1,
        reset_manager=None,
        use_tensorboard: bool = False,
    ):
        self.vehicles = vehicles
        self.n_envs = n_envs
        self.step_dt = float(step_dt)
        self.record_every = max(1, int(record_every))
        self.reset_manager = reset_manager

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_step = np.zeros(n_envs, dtype=np.int64)
        self.completed_episodes = np.zeros(n_envs, dtype=np.int64)

        self.traj_file = open(self.out_dir / "trajectories.csv", "w", newline="")
        self.ep_file = open(self.out_dir / "episodes.csv", "w", newline="")

        self.traj_writer = csv.DictWriter(
            self.traj_file,
            fieldnames=[
                "controller",
                "env_id",
                "episode_id",
                "episode_step",
                "global_step",
                "t",

                "x", "y", "z",
                "vx", "vy", "vz",
                "speed",

                "goal_x", "goal_y", "goal_z",

                "fx_body", "fy_body", "fz_body",
                "body_force_norm",

                "tx_body", "ty_body", "tz_body",
                "body_torque_norm",
            ],
        )
        self.traj_writer.writeheader()

        self.ep_writer = csv.DictWriter(
            self.ep_file,
            fieldnames=[
                "env_id",
                "episode_id",
                "end_global_step",
                "length_steps",
                "duration_s",
                "reason",
            ],
        )
        self.ep_writer.writeheader()

        self.tb = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))

    def _get_goals_numpy(self):
        if self.reset_manager is None:
            return np.full((self.n_envs, 3), np.nan, dtype=np.float32)

        return self.reset_manager.goal_pos.detach().cpu().numpy()

    def _get_body_force_torque_numpy(self, vehicle):
        body_index = vehicle.body_index
        last_forces = vehicle._last_forces_local
        last_torques = vehicle._last_torques_local

        if last_forces is None:
            body_force = np.full((self.n_envs, 3), np.nan, dtype=np.float32)
        else:
            body_force = last_forces[:, body_index, :].detach().cpu().numpy()

        if last_torques is None:
            body_torque = np.full((self.n_envs, 3), np.nan, dtype=np.float32)
        else:
            body_torque = last_torques[:, body_index, :].detach().cpu().numpy()

        return body_force, body_torque

    def sample(self):
        """Record current state and last applied body-frame force/torque."""
        if self.global_step % self.record_every != 0:
            return

        goals = self._get_goals_numpy()

        for controller_name, vehicle in self.vehicles.items():
            if vehicle is None:
                continue

            state = vehicle.state

            pos = state.position.detach().cpu().numpy()
            vel = state.linear_velocity.detach().cpu().numpy()
            speed = np.linalg.norm(vel, axis=1)

            body_force, body_torque = self._get_body_force_torque_numpy(vehicle)

            body_force_norm = np.linalg.norm(body_force, axis=1)
            body_torque_norm = np.linalg.norm(body_torque, axis=1)

            for env_id in range(self.n_envs):
                row = {
                    "controller": controller_name,
                    "env_id": int(env_id),
                    "episode_id": int(self.episode_id[env_id]),
                    "episode_step": int(self.episode_step[env_id]),
                    "global_step": int(self.global_step),
                    "t": float(self.episode_step[env_id] * self.step_dt),

                    "x": float(pos[env_id, 0]),
                    "y": float(pos[env_id, 1]),
                    "z": float(pos[env_id, 2]),

                    "vx": float(vel[env_id, 0]),
                    "vy": float(vel[env_id, 1]),
                    "vz": float(vel[env_id, 2]),
                    "speed": float(speed[env_id]),

                    "goal_x": float(goals[env_id, 0]),
                    "goal_y": float(goals[env_id, 1]),
                    "goal_z": float(goals[env_id, 2]),

                    "fx_body": float(body_force[env_id, 0]),
                    "fy_body": float(body_force[env_id, 1]),
                    "fz_body": float(body_force[env_id, 2]),
                    "body_force_norm": float(body_force_norm[env_id]),

                    "tx_body": float(body_torque[env_id, 0]),
                    "ty_body": float(body_torque[env_id, 1]),
                    "tz_body": float(body_torque[env_id, 2]),
                    "body_torque_norm": float(body_torque_norm[env_id]),
                }

                self.traj_writer.writerow(row)

                if self.tb is not None:
                    prefix = f"{controller_name}/env_{env_id}"
                    self.tb.add_scalar(f"{prefix}/z", row["z"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/speed", row["speed"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fx_body", row["fx_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fy_body", row["fy_body"], self.global_step)
                    self.tb.add_scalar(f"{prefix}/fz_body", row["fz_body"], self.global_step)

    def advance(self, terminated, truncated):
        """
        Call after wrapped.step(...).

        terminated/truncated may be shape (n_envs, 1) or (n_envs,).
        """
        self.global_step += 1
        self.episode_step += 1

        terminated = terminated.detach().view(-1).cpu().numpy().astype(bool)
        truncated = truncated.detach().view(-1).cpu().numpy().astype(bool)
        done = np.logical_or(terminated, truncated)

        for env_id in np.where(done)[0]:
            self.completed_episodes[env_id] += 1

            if terminated[env_id]:
                reason = "terminated"
            elif truncated[env_id]:
                reason = "truncated"
            else:
                reason = "done"

            self.ep_writer.writerow({
                "env_id": int(env_id),
                "episode_id": int(self.episode_id[env_id]),
                "end_global_step": int(self.global_step),
                "length_steps": int(self.episode_step[env_id]),
                "duration_s": float(self.episode_step[env_id] * self.step_dt),
                "reason": reason,
            })

            self.episode_id[env_id] += 1
            self.episode_step[env_id] = 0

        self.traj_file.flush()
        self.ep_file.flush()

        if self.tb is not None:
            self.tb.flush()

    def completed(self, target_episodes_per_env: int) -> bool:
        if target_episodes_per_env <= 0:
            return False
        return np.all(self.completed_episodes >= target_episodes_per_env)

    def close(self):
        self.traj_file.close()
        self.ep_file.close()

        if self.tb is not None:
            self.tb.close()    
    

# ---------------------------------------------------------------------
# Isaac Sim imports
# ---------------------------------------------------------------------
from isaacsim import SimulationApp


def discover_tasks():
    """Recursively finds tasks (dirs containing a ``*_env.py``) and returns their
    paths relative to TASKS_DIR using ``/`` (e.g. ``double_integrator/01_isaac_lab``)."""
    if not TASKS_DIR.is_dir():
        return []

    tasks = []
    for root, dirs, files in os.walk(TASKS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith("_") and d != "agents"]
        if any(f.endswith("_env.py") for f in files):
            rel = os.path.relpath(root, TASKS_DIR)
            tasks.append(rel.replace(os.sep, "/"))
    return sorted(tasks)


def parse_args():
    tasks = discover_tasks() or ["quadcopter"]

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--task", required=True, choices=tasks)
    p.add_argument("--algo", default="ppo")
    p.add_argument("--preset", default="isaac_lab")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", default=False, action="store_true")
    p.add_argument("--compare_lqr", default=False, action="store_true")

    p.add_argument("--record", default=False, action="store_true")
    p.add_argument("--record_dir", default="play_records")
    p.add_argument("--record_every", type=int, default=1)
    p.add_argument("--num_episodes_per_env", type=int, default=0)

    return p.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})


# ---------------------------------------------------------------------
# Post-SimulationApp imports
# ---------------------------------------------------------------------
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager


def load_env(task):
    task_dir = TASKS_DIR.joinpath(*task.split("/"))

    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]
    if not env_files:
        raise FileNotFoundError(f"No *_env.py found in {task_dir}")

    env_file = env_files[0]
    module_name = env_file[:-3]

    # Numeric-prefixed nested packages (e.g. "double_integrator/01_isaac_lab")
    # are imported via importlib using the dotted string form.
    task_pkg = task.replace("/", ".")
    mod = importlib.import_module(f"tasks.{task_pkg}.{module_name}")

    base_name = module_name.replace("_env", "")
    cls_name = "".join(w.capitalize() for w in base_name.split("_")) + "Env"

    if not hasattr(mod, cls_name):
        raise AttributeError(f"Class '{cls_name}' not found in {module_name}")

    if not hasattr(mod, cls_name + "Cfg"):
        raise AttributeError(f"Config class '{cls_name}Cfg' not found in {module_name}")

    return getattr(mod, cls_name), getattr(mod, cls_name + "Cfg")()


def load_agent_cfg(task, algo, preset):
    task_pkg = task.replace("/", ".")
    mod = importlib.import_module(f"tasks.{task_pkg}.agents.{algo}_cfg")
    return getattr(mod, "PRESETS")[preset]


def load_algo_class(algo):
    algo_map = {
        "ppo": ("skrl.agents.torch.ppo", "PPO"),
        "sac": ("skrl.agents.torch.sac", "SAC"),
        "td3": ("skrl.agents.torch.td3", "TD3"),
        "ddpg": ("skrl.agents.torch.ddpg", "DDPG"),
    }

    if algo not in algo_map:
        raise ValueError(f"Unknown algo '{algo}'.")

    mod_name, cls_name = algo_map[algo]
    return getattr(importlib.import_module(mod_name), cls_name)


def load_lqr_backend_class():
    """
    Loads utils/lqr_controller_batch.py by absolute path.

    This avoids issues where Python imports a different package called `utils`.
    """
    path = RL_DIR / "utils" / "lqr_controller_batch.py"

    if not path.exists():
        raise FileNotFoundError(f"LQR backend not found: {path}")

    spec = importlib.util.spec_from_file_location("local_lqr_controller_batch", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "LQRBackend"):
        raise AttributeError(f"LQRBackend not found in {path}")

    return module.LQRBackend


def _prepare_cfg(cfg: dict, device: str) -> dict:
    cfg = copy.deepcopy(cfg)

    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device

    return cfg


def main():
    device = args.device
    n_envs = args.n_envs

    EnvClass, env_cfg = load_env(args.task)
    agent_cfg = load_agent_cfg(args.task, args.algo, args.preset)
    AgentClass = load_algo_class(args.algo)

    pg = PegasusInterface()
    pg.set_world_settings(
        physics_dt=env_cfg.sim_dt,
        rendering_dt=env_cfg.sim_dt * env_cfg.decimation,
        device=device,
    )

    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

    prim_utils.create_prim(
        "/World/Light/DomeLight",
        "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={
            "inputs:intensity": 5e3,
            "inputs:color": (1.0, 1.0, 1.0),
        },
    )

    if "cuda" in device:
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # -----------------------------------------------------------------
    # RL vehicle
    # -----------------------------------------------------------------
    rl_backend = RLBackend(n_vehicles=n_envs, action_mode=env_cfg.action_mode)

    physics_cfg = getattr(env_cfg, "vehicle_physics_cfg", None) or {}
    vehicle_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [rl_backend]

    rl_vehicle = MultirotorBatch(
        stage_prefix="/World/rl_quadrotor",
        usd_file=ROBOTS["Iris"],
        vehicle_batch_id=1,
        n_vehicles=n_envs,
        spacing=2.5,
        config=vehicle_cfg,
    )
    rl_vehicle.disable_collisions()

    env = EnvClass(env_cfg, rl_backend, reset_manager=None)
    env._world = world

    # -----------------------------------------------------------------
    # Optional LQR vehicle
    # -----------------------------------------------------------------
    lqr_vehicle = None
    lqr_backend = None
    has_lqr = args.compare_lqr and hasattr(env, "_compute_discounted_dlqr")

    if has_lqr:
        LQRBackend = load_lqr_backend_class()

        P, K = env._compute_discounted_dlqr()

        lqr_backend = LQRBackend(
            n_vehicles=n_envs,
            P=P,
            K=K,
            reset_manager=None,
            action_mode="direct_force",
        )

        lqr_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
        lqr_cfg.backends = [lqr_backend]

        lqr_vehicle = MultirotorBatch(
            stage_prefix="/World/lqr_quadrotor",
            usd_file=ROBOTS["Iris_White"],
            init_pos=rl_vehicle._init_pos,
            init_orientation=rl_vehicle._init_orientation,
            vehicle_batch_id=2,
            n_vehicles=n_envs,
            config=lqr_cfg,
        )

    # -----------------------------------------------------------------
    # Start simulation
    # -----------------------------------------------------------------
    world.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    print("RL has _root_prims:", hasattr(rl_vehicle, "_root_prims"), flush=True)

    if has_lqr:
        print("LQR has _root_prims:", hasattr(lqr_vehicle, "_root_prims"), flush=True)

        if not hasattr(lqr_vehicle, "_root_prims"):
            raise RuntimeError("LQR vehicle did not initialize. Do not add it to ResetManager.")

    if has_lqr:
        reset_manager = ResetManager(vehicles=[rl_vehicle, lqr_vehicle], device=device)
    else:
        reset_manager = ResetManager(vehicles=[rl_vehicle], device=device)

    env.reset_manager = reset_manager

    env.setup()
    env._render_enabled = not args.headless

    if has_lqr:
        lqr_backend.setup(reset_manager)

    from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper

    wrapped = PegasusSkrlWrapper(env)

    cfg = _prepare_cfg(agent_cfg["cfg"], device)
    models = agent_cfg["models"](
        wrapped.observation_space,
        wrapped.action_space,
        device,
    )

    cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

    agent = AgentClass(
        models=models,
        memory=None,
        cfg=cfg,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )

    print(f"\nLoading checkpoint: {args.checkpoint}", flush=True)
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")
    print("Loaded. Running inference...\n", flush=True)

    obs, _ = wrapped.reset()

    recorder = None

    if args.record:
        vehicles_to_record = {"rl": rl_vehicle}

        if has_lqr:
            vehicles_to_record["lqr"] = lqr_vehicle

        step_dt = getattr(env, "step_dt", env_cfg.sim_dt * env_cfg.decimation)

        recorder = EpisodeTrajectoryRecorder(
            vehicles=vehicles_to_record,
            n_envs=n_envs,
            step_dt=env.step_dt,
            out_dir=args.record_dir,
            record_every=args.record_every,
            reset_manager=reset_manager,
            use_tensorboard=False,
        )

    try:
        while simulation_app.is_running():
            if recorder is not None:
                recorder.sample()

            with torch.no_grad():
                # skrl's act() returns a STOCHASTIC sample from the policy
                # distribution (even after set_running_mode("eval")). For
                # deterministic evaluation we use the distribution mean from
                # outputs["mean_actions"] instead of the sampled action.
                _sampled, _, _outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = _outputs.get("mean_actions", _sampled)

            obs, _, terminated, truncated, _ = wrapped.step(actions)

            world.step(render=not args.headless)

            if recorder is not None:
                recorder.advance(terminated, truncated)

                if recorder.completed(args.num_episodes_per_env):
                    print(
                        f"Finished {args.num_episodes_per_env} episodes per env. Exiting.",
                        flush=True,
                    )
                    break

    finally:
        if recorder is not None:
            recorder.close()

        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()