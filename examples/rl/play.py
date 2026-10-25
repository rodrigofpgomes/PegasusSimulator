#!/usr/bin/env python
"""
| File: play.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Evaluate one skrl policy inside its own training task (task protocol).
|
| The policy runs through the same PegasusEnv used by train.py: the task draws
| its goals and references, applies its terminations and resets, and builds the
| observation. Use it for statistics over the training distribution (seeds,
| ablations); use play_multi.py to compare controllers on shared references.
|
| Timing: PegasusEnv.step applies each action for cfg.decimation physics steps of
| cfg.sim_dt, exactly as in training, so the policy runs at 1/(sim_dt*decimation).
| --run-metadata restores the training snapshot (including sim_dt and decimation).
| The graphical path draws one frame per decision; it never adds a physics step.
|
| Initial states: the task's own init_state_cfg randomization is off by default
| (nominal start); --train-init-state enables it as in train.py. Curriculum
| tasks (shuttle_glider) start at curriculum_start_level and update the
| curriculum unless --env-cfg test_mode=True; select the evaluated level with
| --env-cfg curriculum_start_level=N.
|
| Outputs: --metrics-dir writes steps.npz/episodes.csv/eval_config.json (offline
| metrics, used by scripts/shuttle); --record writes trajectories.csv/episodes.csv.
"""

# Standard library imports
import sys
import csv
import json
import argparse
import importlib
import importlib.util
from pathlib import Path

# Numerical utilities
import torch
import numpy as np


# =============================================================================
# Paths
# =============================================================================
RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent.parent
PEGASUS_EXT_DIR = PROJECT_ROOT / "extensions" / "pegasus.simulator"

for p in [str(RL_DIR), str(PROJECT_ROOT), str(PEGASUS_EXT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Optional trajectory recorder
# =============================================================================
class EpisodeTrajectoryRecorder:
    """
    Record position, velocity, goal and last applied body wrench for each vehicle.

    sample() reads the state before the next policy decision. advance() counts
    the decision just executed and writes a row when an environment terminates
    or reaches its time limit.

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
        """Open CSV files and initialize per-environment episode counters."""
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

        self.traj_file = open(self.out_dir / "trajectories.csv", "x", newline="")
        self.ep_file = open(self.out_dir / "episodes.csv", "x", newline="")

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
        """Read the current goal positions for the recorded environments."""
        if self.reset_manager is None:
            return np.full((self.n_envs, 3), np.nan, dtype=np.float32)

        return self.reset_manager.goal_pos.detach().cpu().numpy()

    def _get_body_force_torque_numpy(self, vehicle):
        """Read the last wrench applied at the vehicle body in its local frame."""
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
        """Advance one policy step and write the episodes that ended on it."""
        self.global_step += 1
        self.episode_step += 1

        terminated = terminated.detach().view(-1).cpu().numpy().astype(bool)
        truncated = truncated.detach().view(-1).cpu().numpy().astype(bool)
        done = np.logical_or(terminated, truncated)

        for env_id in np.where(done)[0]:
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

    def close(self):
        """Flush and close the optional trajectory and TensorBoard outputs."""
        self.traj_file.close()
        self.ep_file.close()

        if self.tb is not None:
            self.tb.close()    
    

class EvalMetricsRecorder:
    """
    Store the state needed for offline evaluation metrics.

    sample() runs before each policy decision and keeps position, inertial
    velocity, body angular rate, tilt and the task's goal for every environment.
    advance() labels finished episodes. The state after the final step of an
    episode is not stored because the environment resets inside step().

    Output (out_dir):
      steps.npz     arrays env, episode, step, t, pos, vel, w_b, goal, goal_vel, tilt_deg, quat (w, x, y, z),
                    action (policy output before the task's clipping, taken at the sampled state), plus the arrays
                    of the task's eval_extras() (name -> (n_envs, k) tensor) when it defines one
      episodes.csv  env_id, episode_id, length_steps, duration_s, reason
    """

    def __init__(self, env, backend, reset_manager, n_envs, step_dt, out_dir):
        """Allocate per-environment counters and chunk buffers."""
        self.env, self.backend, self.reset_manager = env, backend, reset_manager
        self.n_envs, self.step_dt = n_envs, float(step_dt)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode = np.zeros(n_envs, dtype=np.int64)
        self.step = np.zeros(n_envs, dtype=np.int64)
        self.chunks = []
        self.action_chunks = []
        self.extra_chunks = {}
        self.episodes = []

    def _goal(self, name, manager_attr):
        """Return the task's own goal buffer when it keeps one, otherwise the reset manager's."""
        buf = getattr(self.env, name, None)
        if buf is None:
            buf = getattr(self.reset_manager, manager_attr)
        return buf.detach().cpu().numpy()

    def sample(self):
        """Store the current state of every environment."""
        state = self.backend.get_state().detach().cpu().numpy()
        quat = state[:, 6:10]  # w, x, y, z
        tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0)))
        self.chunks.append(np.column_stack([
            np.arange(self.n_envs), self.episode, self.step, self.step * self.step_dt,
            state[:, 0:3], state[:, 3:6], state[:, 10:13],
            self._goal("_goal_pos", "goal_pos"), self._goal("_goal_vel", "goal_vel"), tilt, quat,
        ]).astype(np.float64))
        extras = getattr(self.env, "eval_extras", None)
        if extras is not None:
            for name, value in extras().items():
                self.extra_chunks.setdefault(name, []).append(value.detach().cpu().numpy().astype(np.float32))

    def record_action(self, actions):
        """Store the policy output decided at the state of the last sample()."""
        self.action_chunks.append(actions.detach().cpu().numpy().astype(np.float64))

    def advance(self, terminated, truncated):
        """Count the executed decision and label the episodes that ended on it."""
        terminated = terminated.detach().view(-1).cpu().numpy().astype(bool)
        truncated = truncated.detach().view(-1).cpu().numpy().astype(bool)
        self.step += 1
        for env_id in np.where(terminated | truncated)[0]:
            self.episodes.append((int(env_id), int(self.episode[env_id]), int(self.step[env_id]),
                                  float(self.step[env_id] * self.step_dt),
                                  "terminated" if terminated[env_id] else "truncated"))
            self.episode[env_id] += 1
            self.step[env_id] = 0

    def close(self):
        """Write the stored arrays and the episode table."""
        if self.chunks:
            data = np.concatenate(self.chunks, axis=0)
            np.savez_compressed(
                self.out_dir / "steps.npz",
                env=data[:, 0].astype(np.int64), episode=data[:, 1].astype(np.int64),
                step=data[:, 2].astype(np.int64), t=data[:, 3],
                pos=data[:, 4:7], vel=data[:, 7:10], w_b=data[:, 10:13],
                goal=data[:, 13:16], goal_vel=data[:, 16:19], tilt_deg=data[:, 19], quat=data[:, 20:24],
                **({"action": np.concatenate(self.action_chunks, axis=0)}
                   if len(self.action_chunks) == len(self.chunks) else {}),
                **{name: np.concatenate(chunks, axis=0) for name, chunks in self.extra_chunks.items()
                   if len(chunks) == len(self.chunks)},
            )
        with (self.out_dir / "episodes.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "episode_id", "length_steps", "duration_s", "reason"])
            writer.writerows(self.episodes)


# =============================================================================
# Task discovery and SimulationApp startup
# =============================================================================
from isaacsim import SimulationApp
from task_loader import (apply_env_overrides, control_timing, discover_tasks, load_agent_cfg, load_algo_class, resolve_preset,
                         load_env, prepare_agent_cfg)


def parse_args():
    """Parse the policy checkpoint, task and optional comparison settings."""
    tasks = discover_tasks()

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--task", required=True, choices=tasks)
    p.add_argument("--algo", default="ppo")
    p.add_argument("--preset", default=None, help="Agent preset; defaults to 'default' or the only preset.")
    p.add_argument("--checkpoint", default=None, help="Agent checkpoint (required with --controller policy).")
    p.add_argument("--controller", default="policy", choices=["policy", "lqr", "pretrain"],
                   help="policy: the trained agent; lqr: the task's discounted DLQR u = K obs on the policy vehicle "
                        "(needs a task with _compute_discounted_dlqr and an [ep, ev] observation); "
                        "pretrain: a RAPTOR pre-training actor (HDF5 --checkpoint) acting on the task observation "
                        "(26 inputs in the rl-tools layout; needs --pretrain-dynamics).")
    p.add_argument("--pretrain-dynamics", default=None,
                   help="rl-tools dynamics json of the pre-training run (rotor order of the actor), "
                        "required with --controller pretrain.")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", default=False, action="store_true")
    p.add_argument("--compare_lqr", default=False, action="store_true")

    p.add_argument("--record", default=False, action="store_true")
    p.add_argument("--record_dir", default=None,
                   help="Exact run directory; defaults to logs/evaluations/ground_truth/<task>/<timestamp>.")
    p.add_argument("--record_every", type=int, default=1)
    p.add_argument("--num_episodes_per_env", type=int, default=0)

    p.add_argument("--run-metadata", default=None,
                   help="run_metadata.json of the training run; its environment snapshot is applied to the task config "
                        "so the vehicle and task match training.")
    p.add_argument("--env-cfg", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a task config field after --run-metadata (repeatable).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed torch before the first goal draw, so different checkpoints see the same goals.")
    p.add_argument("--metrics-dir", default=None,
                   help="Write per-step state (steps.npz), episodes.csv and eval_config.json for offline metrics.")
    p.add_argument("--train-init-state", default=False, action="store_true",
                   help="Randomize initial states with the task's init_state_cfg, as train.py does "
                        "(default: nominal start).")

    p.add_argument("--goal-lemniscate", default=False, action="store_true",
                   help="Move the goal of tasks that keep _goal_pos/_goal_vel buffers along a figure eight around each "
                        "vehicle's spawn point (the reference of the direct_rotor ablation task, eval_reference=lemniscate); "
                        "use --env-cfg episode_length_s=20.")
    p.add_argument("--lem-amplitude", type=float, default=1.0, help="Half width of the figure eight [m].")
    p.add_argument("--lem-peak-acc", type=float, default=2.0, help="Peak reference acceleration [m/s^2].")
    p.add_argument("--lem-ramp", type=float, default=2.0, help="Time to ramp the reference speed up from rest [s].")
    p.add_argument("--lem-zero-vel", default=False, action="store_true",
                   help="Keep the goal velocity at zero (the value of the training), so the policy only sees the position.")

    args = p.parse_args()
    if args.controller in ("policy", "pretrain") and not args.checkpoint:
        p.error(f"--checkpoint is required with --controller {args.controller}")
    if args.controller == "pretrain" and not args.pretrain_dynamics:
        p.error("--pretrain-dynamics is required with --controller pretrain")
    return args


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})


# =============================================================================
# Imports that require an initialized SimulationApp
# =============================================================================
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.vehicles.shuttle_glider_batch2 import ShuttleGliderBatch2, ShuttleGliderBatchConfig2
from pegasus.simulator.logic.vehicles.glider_batch import GliderBatch, GliderBatchConfig
from pegasus.simulator.logic.vehicles.vehicle_physics import get_vehicle_physics
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager


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


def install_lemniscate_goal(env, reset_manager, amplitude: float, peak_acc: float, ramp_s: float, zero_vel: bool):
    """Drive the goal buffers of a task along the figure eight of the direct_rotor ablation task.

    The goal is refreshed before the task builds its observation, reward and termination (it follows the episode
    step), so the policy sees the reference of the current step. Only for tasks with _goal_pos / _goal_vel buffers
    that do not generate their own reference.
    """
    if getattr(env, "_trajectory", None) is not None or not hasattr(env, "_goal_pos") or not hasattr(env, "_goal_vel"):
        raise SystemExit("--goal-lemniscate needs a task with _goal_pos/_goal_vel buffers and no reference of its own; "
                         "for the direct_rotor ablation task use --env-cfg eval_reference=lemniscate")
    path = RL_DIR / "tasks/shuttle/direct_rotor/ablation/raptor_tracking/trajectory.py"
    spec = importlib.util.spec_from_file_location("lemniscate_trajectory", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = torch.arange(env.num_envs, device=env.device)
    trajectory = module.LemniscateTrajectory(env.num_envs, env.max_episode_length, env.step_dt, env.device,
                                             amplitude=amplitude, peak_acc=peak_acc, ramp_s=ramp_s)
    trajectory.reset(ids, reset_manager.init_pos.to(device=env.device, dtype=torch.float32))

    def sync():
        pos, vel, _ = trajectory.current(ids, env.episode_length_buf[ids])
        env._goal_pos[:] = pos
        env._goal_vel[:] = 0.0 if zero_vel else vel

    for name in ("_get_observations", "_get_rewards", "_get_dones"):
        original = getattr(env, name)
        setattr(env, name, (lambda f: lambda *a, **k: (sync(), f(*a, **k))[1])(original))
    sync()
    print(f"[goal] figure eight: A={amplitude} m, peak acceleration {peak_acc} m/s^2 (omega {trajectory.omega:.3f} rad/s), "
          f"ramp {ramp_s} s, goal velocity {'zero' if zero_vel else 'the reference'}", flush=True)


def load_pretrain_policy(checkpoint: str, dynamics_json: str, vehicle, device: str):
    """Deterministic RAPTOR pre-training actor acting on the task observation.

    The actor is the MLP of an rl-tools HDF5 checkpoint (26 -> 64 -> 64 -> 8, ReLU; the first four outputs are the
    mean, squashed by tanh). Its observation and action use the rotor order of the pre-training dynamics json;
    the vehicle's rotors are matched to it by position and spin direction, and the match must be unique.
    Returns a function obs (N, 26) -> action (N, 4) in the vehicle's rotor order.
    """
    import itertools
    import h5py

    dynamics = json.loads(Path(dynamics_json).read_text())["dynamics"]
    raptor_xy = torch.tensor(dynamics["rotor_positions"], dtype=torch.float32)[:, :2]
    raptor_dir = torch.tensor([d[2] for d in dynamics["rotor_torque_directions"]], dtype=torch.float32)
    vehicle_xy = vehicle._rotor_positions_body[0, :, :2].detach().cpu().float()
    vehicle_dir = vehicle._thrusters._rot_dir.detach().cpu().float().reshape(-1)

    # perm[i] = index of the vehicle rotor sitting at the position of actor rotor i
    matches = [perm for perm in itertools.permutations(range(4))
               if torch.allclose(vehicle_xy[list(perm)], raptor_xy, atol=0.02)
               and torch.equal(torch.sign(vehicle_dir[list(perm)]), torch.sign(raptor_dir))]
    if len(matches) != 1:
        raise RuntimeError(f"rotor order of {dynamics_json} does not match the vehicle uniquely: {matches}\n"
                           f"vehicle xy {vehicle_xy.tolist()} dir {vehicle_dir.tolist()}\n"
                           f"actor   xy {raptor_xy.tolist()} dir {raptor_dir.tolist()}")
    perm = torch.tensor(matches[0], dtype=torch.long, device=device)
    print(f"[pretrain] actor rotor i -> vehicle rotor {perm.tolist()}", flush=True)

    with h5py.File(checkpoint, "r") as f:
        layers = f["actor/layers/0"]
        W = [torch.tensor(layers[k]["weights/parameters"][()], dtype=torch.float32, device=device)
             for k in ("input_layer", "hidden_layer_0", "output_layer")]
        b = [torch.tensor(layers[k]["biases/parameters"][()], dtype=torch.float32, device=device).reshape(-1)
             for k in ("input_layer", "hidden_layer_0", "output_layer")]
    if W[0].shape[1] != 26 or W[2].shape[0] != 8:
        raise ValueError(f"unexpected actor shapes {[tuple(w.shape) for w in W]}")

    def act(obs: torch.Tensor) -> torch.Tensor:
        o = obs.clone()
        # last action and measured rotor speeds are in the vehicle order: put them in the actor order
        o[:, 18:22] = obs[:, 18:22][:, perm]
        o[:, 22:26] = obs[:, 22:26][:, perm]
        h = torch.relu(o @ W[0].T + b[0])
        h = torch.relu(h @ W[1].T + b[1])
        mean = torch.tanh((h @ W[2].T + b[2])[:, :4])
        out = torch.empty_like(mean)
        out[:, perm] = mean
        return out

    return act


# =============================================================================
# Evaluation lifecycle
# =============================================================================

def main():
    """Recreate the training vehicle, restore the agent and evaluate episodes."""
    device = args.device
    n_envs = args.n_envs

    if args.headless and args.num_episodes_per_env <= 0:
        raise ValueError("Headless evaluation requires a finite episode count")

    EnvClass, env_cfg = load_env(args.task)
    args.preset = resolve_preset(args.task, args.algo, args.preset)
    agent_cfg = load_agent_cfg(args.task, args.algo, args.preset)
    AgentClass = load_algo_class(args.algo)

    # Restore the training task parameters before the world and vehicle are built.
    override_log = apply_env_overrides(env_cfg, args.run_metadata, args.env_cfg)
    for key, change in override_log["changed"].items():
        print(f"[env-cfg] {key}: {change['default']!r} -> {change['applied']!r}", flush=True)
    if override_log["not_in_task_cfg"]:
        print(f"[env-cfg] snapshot fields absent from the task config: {override_log['not_in_task_cfg']}", flush=True)
    timing = control_timing(env_cfg)
    print(f"[timing] physics {timing['physics_hz']:.1f} Hz, policy {timing['control_hz']:.1f} Hz "
          f"(sim_dt={env_cfg.sim_dt}, decimation={env_cfg.decimation})", flush=True)

    # Match the task's physics step and draw interval used during training.
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
    # Create the policy vehicle with the task's physics override, if present.
    rl_backend = RLBackend(n_vehicles=n_envs, action_mode=env_cfg.action_mode)

    # Recreate the same physical vehicle and task overrides used in training.
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
        vehicle_cfg = MultirotorBatchConfig(cfg=physics_cfg, n_vehicles=n_envs)
        VehicleClass = MultirotorBatch

    vehicle_cfg.backends = [rl_backend]

    rl_vehicle = VehicleClass(
        stage_prefix="/World/rl_quadrotor",
        usd_file=ROBOTS[env_cfg.vehicle],
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
    has_lqr = args.compare_lqr and env_cfg.vehicle == "Iris" and hasattr(env, "_compute_discounted_dlqr")

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

        lqr_cfg = MultirotorBatchConfig(cfg=physics_cfg, n_vehicles=n_envs)
        lqr_cfg.backends = [lqr_backend]

        lqr_vehicle = MultirotorBatch(
            stage_prefix="/World/lqr_quadrotor",
            usd_file=ROBOTS["Iris"],
            init_pos=rl_vehicle._init_pos,
            init_orientation=rl_vehicle._init_orientation,
            vehicle_batch_id=2,
            n_vehicles=n_envs,
            config=lqr_cfg,
        )

    # -----------------------------------------------------------------
    # Start simulation
    # -----------------------------------------------------------------
    # Initialize prim views, then let the environment perform its own reset.
    world.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    if has_lqr:
        if not hasattr(lqr_vehicle, "_root_prims"):
            raise RuntimeError("LQR vehicle did not initialize. Do not add it to ResetManager.")

    # The task samples its own goals in setup()/reset(). Initial-state randomization
    # follows train.py only with --train-init-state; otherwise every episode starts nominal.
    init_state_cfg = None
    if args.train_init_state and getattr(env_cfg, "randomize_init_state", False):
        init_state_cfg = getattr(env_cfg, "init_state_cfg", None)
    vehicles = [rl_vehicle, lqr_vehicle] if has_lqr else [rl_vehicle]
    reset_manager = ResetManager(vehicles=vehicles, device=device, init_state_cfg=init_state_cfg)

    env.reset_manager = reset_manager

    if args.seed is not None:
        torch.manual_seed(args.seed)  # goals drawn in setup() and reset() are then identical across checkpoints
    env.setup()
    # The task owns all physics substeps. Draw once after each policy decision.
    env._render_enabled = False

    if has_lqr:
        lqr_backend.setup(reset_manager)

    from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper

    wrapped = PegasusSkrlWrapper(env)
    if args.goal_lemniscate:
        install_lemniscate_goal(env, reset_manager, args.lem_amplitude, args.lem_peak_acc, args.lem_ramp,
                                args.lem_zero_vel)

    agent = None
    lqr_gain = None
    pretrain_policy = None
    if args.controller == "pretrain":
        if wrapped.observation_space.shape[0] != 26:
            raise ValueError(f"the pre-training actor needs a 26-component observation, got {wrapped.observation_space.shape}")
        pretrain_policy = load_pretrain_policy(args.checkpoint, args.pretrain_dynamics, rl_vehicle, device)
    elif args.controller == "lqr":
        # The task's DLQR, with the discount of the agent preset, acting on the policy observation
        if not hasattr(env, "_compute_discounted_dlqr") or not hasattr(env_cfg, "max_acceleration"):
            raise ValueError(f"--controller lqr: task {args.task} has no _compute_discounted_dlqr/max_acceleration")
        gamma = float(agent_cfg["cfg"].get("discount_factor", 0.99))
        _, lqr_gain = env._compute_discounted_dlqr(gamma=gamma)
        print(f"\nLQR controller (gamma={gamma}, a_max={env_cfg.max_acceleration}): K row 0 = "
              f"{lqr_gain[0].tolist()}\n", flush=True)
    else:
        # Rebuild the model and preprocessors before restoring checkpoint state.
        cfg = prepare_agent_cfg(agent_cfg["cfg"], device)
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
        # skrl returns random warm-up actions while timestep < random_timesteps (SAC presets)
        agent._random_timesteps = 0
        print("Loaded. Running inference...\n", flush=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    obs, _ = wrapped.reset()

    metrics = None
    if args.metrics_dir:
        import subprocess
        try:
            commit = subprocess.run(["git", "-C", str(RL_DIR), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=10).stdout.strip()
        except Exception:
            commit = None
        if override_log["training_commit"] and commit and override_log["training_commit"] != commit:
            print(f"[WARN] evaluating commit {commit[:8]} but training used {override_log['training_commit'][:8]}", flush=True)
        metrics = EvalMetricsRecorder(env, rl_backend, reset_manager, n_envs, env.step_dt, args.metrics_dir)
        config_out = Path(args.metrics_dir) / "eval_config.json"
        config_out.write_text(json.dumps({
            "checkpoint": args.checkpoint, "controller": args.controller,
            "lqr_gain": lqr_gain.tolist() if lqr_gain is not None else None, "task": args.task, "algo": args.algo, "preset": args.preset,
            "n_envs": n_envs, "seed": args.seed, "num_episodes_per_env": args.num_episodes_per_env,
            "evaluation_commit": commit, "training_commit": override_log["training_commit"],
            "training_seed": override_log.get("training_seed"),
            "control_timing": timing,
            "train_init_state": init_state_cfg is not None,
            "env_cfg_overrides": override_log["changed"],
            "env_cfg_effective": {k: getattr(env_cfg, k) for k in dir(env_cfg)
                                  if not k.startswith("_") and isinstance(getattr(env_cfg, k), (int, float, str, bool, list, tuple))},
            "vehicle_physics": physics_cfg,
        }, indent=2, default=str))

    # CSV recording is optional; episode termination is tracked in either case.
    recorder = None

    if args.record:
        from evaluation_log import evaluation_directory, evaluation_identity

        controller = {"name": "rl", "type": args.algo, "task": args.task}
        run_dir = (Path(args.record_dir) if args.record_dir else
                   evaluation_directory(RL_DIR / "logs" / "evaluations", env_cfg.vehicle.lower(),
                                        [controller]))
        run_dir.mkdir(parents=True, exist_ok=True)
        args.record_dir = str(run_dir)
        mode, test_id = evaluation_identity(env_cfg.vehicle.lower(), [controller])
        metadata = {"source": "play", "evaluation_mode": mode,
                    "test_id": test_id, "checkpoint": args.checkpoint, "arguments": vars(args)}
        with (run_dir / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
        vehicles_to_record = {"rl": rl_vehicle}

        if has_lqr:
            vehicles_to_record["lqr"] = lqr_vehicle

        recorder = EpisodeTrajectoryRecorder(
            vehicles=vehicles_to_record,
            n_envs=n_envs,
            step_dt=env.step_dt,
            out_dir=str(run_dir),
            record_every=args.record_every,
            reset_manager=reset_manager,
            use_tensorboard=False,
        )

    try:
        completed_episodes = torch.zeros(n_envs, dtype=torch.long, device=device)

        while args.headless or simulation_app.is_running():
            if recorder is not None:
                recorder.sample()
            if metrics is not None:
                metrics.sample()

            with torch.no_grad():
                # skrl's act() returns a STOCHASTIC sample from the policy
                # distribution (even after set_running_mode("eval")). For
                # deterministic evaluation we use the distribution mean from
                # outputs["mean_actions"] instead of the sampled action.
                if pretrain_policy is not None:
                    actions = pretrain_policy(obs)
                elif lqr_gain is not None:
                    # Normalised action; the task clamps it to [-1, 1] (inactive when a_max exceeds the LQR peak)
                    actions = obs @ lqr_gain.T / env_cfg.max_acceleration
                else:
                    _sampled, _, _outputs = agent.act(obs, timestep=0, timesteps=0)
                    actions = _outputs.get("mean_actions", _sampled)
            if metrics is not None:
                metrics.record_action(actions)

            obs, _, terminated, truncated, _ = wrapped.step(actions)

            # PegasusEnv.step advances all physics substeps and resets done envs.
            completed_episodes += (terminated | truncated).reshape(-1).long()

            # Draw the resulting state once per policy decision in graphical play.
            if not args.headless:
                wrapped.render()

            if recorder is not None:
                recorder.advance(terminated, truncated)
            if metrics is not None:
                metrics.advance(terminated, truncated)

            if args.num_episodes_per_env > 0 and torch.all(completed_episodes >= args.num_episodes_per_env):
                print(f"Finished {args.num_episodes_per_env} episodes per env. Exiting.", flush=True)
                break

    finally:
        if recorder is not None:
            recorder.close()
        if metrics is not None:
            metrics.close()

        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
