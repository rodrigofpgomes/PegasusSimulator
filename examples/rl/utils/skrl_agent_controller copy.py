"""skrl_agent_controller.py

Pegasus batched backend that runs a trained skrl agent (PPO / PPO2 / SAC) in
inference, EXACTLY like examples/rl/play.py -- but with the observation and the
action built BY THE TASK ENVIRONMENT ITSELF.

Why this design
---------------
Instead of hard-coding one observation layout and one actuation path (which only
works for the task it was written for and breaks on a shape mismatch when the
checkpoint was trained on a different task), this backend reuses the task's own
``QuadcopterEnv``. It therefore:

    1. loads the task env class + config (like play.py:load_env);
    2. subclasses RLBackend so the env talks to it exactly as in training
       (get_state / set_forces_and_torques / _input_reference / goal markers);
    3. instantiates ``EnvClass(env_cfg, self, reset_manager)`` and calls setup();
    4. every step asks the env to build the observation (``_get_observations``)
       and to apply the action (``_pre_physics_step`` + ``_apply_action``).

Because the observation/action dimensions and the actuation mode come from
``env_cfg`` (``observation_space`` / ``action_space`` / ``action_mode``), the
same controller supports any task (e.g. raptor_pretrain 26-dim rotor_velocity,
double_integrator 12-dim direct_force, shuttle_glider2 34-dim rotor_velocity)
with no per-task code here.

The skrl agent is built and restored exactly like play.py, so any state
preprocessor baked into the checkpoint is applied identically. ``agent.act()``
returns a stochastic sample even in eval mode, so we take the deterministic
distribution mean from ``outputs["mean_actions"]``.

Reference (goal) handling
-------------------------
The reference is driven externally by play_multi (``reset_manager.goal_*``). The
env's own trajectory generator is disabled so it does not fight that reference.
Tasks that read ``reset_manager.goal_*`` directly follow it for free; tasks that
keep their own goal buffer are synced from ``reset_manager`` every step.
"""

__all__ = ["SkrlAgentBackend"]

import copy
import os
import importlib
from pathlib import Path

import torch
import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - skrl also supports classic gym
    import gym

from pegasus.simulator.logic.rl.rl_backend import RLBackend
from pegasus.simulator.logic.state_batch import StateBatch
import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

# examples/rl, so that "tasks.<...>" and "tasks.<...>.agents.<algo>_cfg" are
# importable exactly as in play.py (play_multi.py already inserts this dir).
RL_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = RL_DIR / "tasks"


def _load_env(task: str):
    """Mirror of play.py:load_env -> (EnvClass, env_cfg_instance)."""
    task_dir = TASKS_DIR.joinpath(*task.split("/"))
    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]
    if not env_files:
        raise FileNotFoundError(f"No *_env.py found in {task_dir}")
    module_name = env_files[0][:-3]
    task_pkg = task.replace("/", ".")
    mod = importlib.import_module(f"tasks.{task_pkg}.{module_name}")
    base_name = module_name.replace("_env", "")
    cls_name = "".join(w.capitalize() for w in base_name.split("_")) + "Env"
    if not hasattr(mod, cls_name):
        raise AttributeError(f"Class '{cls_name}' not found in {module_name}")
    if not hasattr(mod, cls_name + "Cfg"):
        raise AttributeError(f"Config class '{cls_name}Cfg' not found in {module_name}")
    return getattr(mod, cls_name), getattr(mod, cls_name + "Cfg")()


def _load_agent_cfg(task: str, algo: str, preset: str):
    """Mirror of play.py:load_agent_cfg."""
    task_pkg = task.replace("/", ".")
    mod = importlib.import_module(f"tasks.{task_pkg}.agents.{algo}_cfg")
    return getattr(mod, "PRESETS")[preset]


def _load_algo_class(algo: str):
    """Mirror of play.py:load_algo_class."""
    algo_map = {
        "ppo":  ("skrl.agents.torch.ppo", "PPO"),
        "ppo2": ("pegasus.simulator.logic.rl.algorithms.ppo2", "PPO2"),
        "sac":  ("skrl.agents.torch.sac", "SAC"),
        "td3":  ("skrl.agents.torch.td3", "TD3"),
        "ddpg": ("skrl.agents.torch.ddpg", "DDPG"),
    }
    if algo not in algo_map:
        raise ValueError(f"Unknown algo '{algo}'.")
    mod_name, cls_name = algo_map[algo]
    return getattr(importlib.import_module(mod_name), cls_name)


def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Mirror of play.py:_prepare_cfg."""
    cfg = copy.deepcopy(cfg)
    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device
    return cfg


class SkrlAgentBackend(RLBackend):
    """Batched Pegasus backend running a trained skrl agent (PPO/PPO2/SAC) in
    eval mode, with observation and action produced by the task env itself.

    Subclasses RLBackend so the instantiated task env interacts with it through
    the same interface used during training, and so the force/torque -> rotor
    conversion for ``rotor_velocity`` tasks is inherited for free.
    """

    def __init__(
        self,
        checkpoint_path: str,
        task: str,
        algo: str,
        preset: str = "isaac_lab",
        obs_dim: int = 26,
        act_dim: int = 4,
        n_vehicles: int = 1,
        reset_manager=None,
        action_mode: str = "rotor_velocity_direct",
        device: str = "cuda:0",
    ):
        # The observation/action dimensions and the actuation mode are taken
        # from env_cfg, NOT from obs_dim/act_dim/action_mode (kept only for
        # manifest/back-compatibility). This is what makes it task-agnostic.
        EnvClass, env_cfg = _load_env(task)

        # play_multi owns the reference (reset_manager goals); disable the env's
        # own trajectory generator so it does not fight the external reference.
        for attr, val in (("use_raptor_trajectory", False),
                          ("trajectory_generator", False),
                          ("test_mode", True)):
            if hasattr(env_cfg, attr):
                setattr(env_cfg, attr, val)

        resolved_action_mode = getattr(env_cfg, "action_mode", action_mode)
        super().__init__(n_vehicles=n_vehicles, action_mode=resolved_action_mode)

        self._checkpoint_path = checkpoint_path
        self._task = task
        self._algo = algo
        self._preset = preset
        self._EnvClass = EnvClass
        self._env_cfg = env_cfg
        self._obs_dim = int(getattr(env_cfg, "observation_space", obs_dim))
        self._act_dim = int(getattr(env_cfg, "action_space", act_dim))
        self.reset_manager = reset_manager
        self._ext_device = device
        self._env = None
        self._agent = None

        # Single shared reference marker (play_multi looks for create_goal_marker).
        self._ref_marker_view = None
        self._ref_marker_paths = []

        self._control_decimation = None
        self._control_counter = 0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def _is_rotor_mode(self) -> bool:
        return self._action_mode in ("rotor_velocity", "rotor_velocity_direct")

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------
    def initialize(self, vehicle):
        super().initialize(vehicle)

    def start(self):
        # Allocates _forces/_torques/_input_reference/_state_cache and pushes the
        # input mode onto the vehicle. Called during world.reset(), BEFORE
        # setup() provides the reset_manager, so env/agent are built later.
        super().start()
        self._prime_motors()

    def setup(self, reset_manager):
        # Called by play_multi after world.reset(): the vehicle is fully started
        # and the shared reset_manager is available, so we can now instantiate
        # the task env and build the agent.
        self.reset_manager = reset_manager
        self._build_env()
        self._build_agent()
        self._control_decimation = int(self._env.cfg.decimation)

    def stop(self):
        pass

    def reset(self):
        super().reset()
        # Restart the control-decimation phase so the policy recomputes an action
        # on the FIRST physics step of every episode (as it does at spawn).
        self._control_counter = 0
        # Reset the task's action-history buffers. Prefer the env's own hook so the
        # history is re-seeded to the post-reset rotor trim exactly as in training;
        # otherwise fall back to zeroing whatever buffers the task keeps.
        if self._env is not None:
            env_reset_hook = getattr(self._env, "reset_action_history", None)
            if callable(env_reset_hook):
                env_reset_hook()
            else:
                for attr in ("_last_action", "_prev_action", "_action_history_obs", "_actions"):
                    buf = getattr(self._env, attr, None)
                    if isinstance(buf, torch.Tensor):
                        buf.zero_()
        # Reproduce the fresh-spawn rotor condition: prime the command reference
        # (no free-fall) but leave the ACTUAL rotor velocity at 0, so the first
        # observation (rotor_speeds_norm) is not polluted by a spun-up state and
        # the policy resumes with the same first action as in episode 0.
        self._prime_motors(spin_up=False)
        # Re-arm the "first state received" latch so the policy computes an action
        # on the FIRST physics step of the new episode, independently of whether
        # reset_manager.reset_all() (which also re-arms it via set_state) runs
        # before or after this reset(). RLBackend.reset() clears this flag, which
        # otherwise made update() skip one control step (the primed motors were
        # applied for a step) and the episode diverged. The live physics state is
        # refreshed every physics step, so this never reads a stale state.
        if hasattr(self, "_received_first_state"):
            self._received_first_state = True

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, dt: float):
        if self._agent is None or self._env is None or not self._received_first_state:
            return

        # If the task keeps its own goal buffers, feed them the externally-driven
        # reference so observations are computed against play_multi's trajectory.
        # Tasks that read reset_manager.goal_* directly need nothing here.
        if self.reset_manager is not None:
            self._sync_env_goal("_goal_pos", "goal_pos")
            self._sync_env_goal("_goal_vel", "goal_vel")
            self._sync_env_goal("_goal_acc", "goal_acc")

        # Compute a new action only at the policy/control frequency.
        if self._control_counter % self._control_decimation == 0:
            obs = self._env._get_observations()["policy"]
            sampled_actions, _, outputs = self._agent.act(obs, timestep=0, timesteps=0)
            actions = outputs.get("mean_actions", sampled_actions)  # deterministic eval
            actions = torch.clamp(actions, -1.0, 1.0)
            self._env._pre_physics_step(actions)  # store the new action

        # Re-apply the stored action every physics step.
        self._env._apply_action()
        self._control_counter += 1

        # Inherited: for 'rotor_velocity' turns forces/torques into rotor speeds;
        # for 'rotor_velocity_direct'/direct-force this is a no-op.
        super().update(dt)

    def set_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ):
        super().set_state(env_ids, positions, attitudes, linear_velocity, angular_velocity)
        # Prime the command reference to avoid free-fall on reset transients, but
        # leave the actual rotor velocity at 0 (fresh-spawn condition) so a
        # deferred teleport cannot re-pollute the first observation of an episode.
        self._prime_motors(env_ids=env_ids, spin_up=False)

    # update_state / get_state / input_reference / set_forces_and_torques /
    # get_forces_and_torques are all inherited from RLBackend unchanged.

    # ------------------------------------------------------------------
    # Shared reference marker (single cube, used by play_multi)
    # ------------------------------------------------------------------
    def create_goal_marker(self, root_path: str = "/World/GoalMarker", size: float = 0.15, color: tuple = (1.0, 0.0, 0.0)):
        stage = self._vehicle._world.stage
        if not stage.GetPrimAtPath(root_path).IsValid():
            prim_utils.create_prim(root_path, "Xform")
        self._ref_marker_paths = []
        for i in range(self._n_vehicles):
            prim_path = f"{root_path}/goal_{i}"
            if not stage.GetPrimAtPath(prim_path).IsValid():
                prim_utils.create_prim(prim_path, "Cube", translation=[0.0, 0.0, -100.0], scale=[size, size, size])
            cube = UsdGeom.Cube(stage.GetPrimAtPath(prim_path))
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            self._ref_marker_paths.append(prim_path)
        self._ref_marker_view = XFormPrimView(
            prim_paths_expr=f"{root_path}/goal_*",
            name=f"skrl_ref_marker_view_{root_path.replace('/', '_')}",
        )

    def update_goal_marker(self, positions: torch.Tensor):
        if self._ref_marker_view is None:
            return
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        positions = positions.to(device=self._device, dtype=torch.float32)
        orientations = torch.zeros((positions.shape[0], 4), device=self._device, dtype=torch.float32)
        orientations[:, 0] = 1.0
        self._ref_marker_view.set_world_poses(positions=positions, orientations=orientations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_env_goal(self, env_attr: str, rm_attr: str):
        """Copy a reset_manager goal buffer into the env's own buffer if it keeps one."""
        env_buf = getattr(self._env, env_attr, None)
        rm_buf = getattr(self.reset_manager, rm_attr, None)
        if isinstance(env_buf, torch.Tensor) and isinstance(rm_buf, torch.Tensor):
            env_buf[:] = rm_buf.to(self._device, dtype=torch.float32)

    def _build_env(self):
        """Instantiate the task env on top of this backend and run its setup()."""
        self._env = self._EnvClass(self._env_cfg, self, self.reset_manager)
        self._env.setup()
        print(
            f"[SkrlAgentBackend] Env '{self._task}' ready "
            f"(obs={self._obs_dim}, act={self._act_dim}, action_mode='{self._action_mode}').",
            flush=True,
        )

    def _build_agent(self):
        """Build the skrl agent exactly like play.py and restore the checkpoint."""
        dev = str(self._device)
        agent_cfg = _load_agent_cfg(self._task, self._algo, self._preset)
        AgentClass = _load_algo_class(self._algo)

        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)

        cfg = _prepare_cfg(agent_cfg["cfg"], dev)
        if cfg.get("state_preprocessor") is not None and isinstance(cfg.get("state_preprocessor_kwargs"), dict):
            cfg["state_preprocessor_kwargs"]["size"] = observation_space

        models = agent_cfg["models"](observation_space, action_space, dev)
        self._agent = AgentClass(
            models=models,
            memory=None,
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=dev,
        )

        checkpoint = torch.load(self._checkpoint_path, map_location=dev, weights_only=False)
        is_teacher_checkpoint = isinstance(checkpoint, dict) and "actor_net_state_dict" in checkpoint
        if is_teacher_checkpoint:
            if self._algo != "ppo2":
                raise ValueError("Teacher-pretrained checkpoint currently expects the PPO2 policy architecture.")
            models["policy"].net.load_state_dict(checkpoint["actor_net_state_dict"], strict=True)
            print(f"[SkrlAgentBackend] Loaded teacher-pretrained actor into PPO2 policy: {self._checkpoint_path}", flush=True)
        else:
            self._agent.load(self._checkpoint_path)
            print(f"[SkrlAgentBackend] Loaded full {self._algo} checkpoint: {self._checkpoint_path}", flush=True)
        self._agent.set_running_mode("eval")

    def _prime_motors(self, env_ids: torch.Tensor | None = None, spin_up: bool = True):
        """Prime the rotor COMMAND to mid throttle so the drone does not drop
        before the first action. Keeps any non-rotor actuators neutral. No-op for
        direct-force tasks.

        spin_up: if True the ACTUAL rotor velocity is also set to mid throttle
        (instantaneous, used for reset transients / set_state). If False the
        actual velocity is set to 0 to reproduce the fresh-spawn condition (used
        on episode reset), so the first observation is not polluted."""
        if not self._is_rotor_mode or self._input_reference is None:
            return
        thrusters = getattr(self._vehicle, "_thrusters", None)
        if thrusters is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._n_vehicles, device=self._device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self._device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        min_w = thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32)
        max_w = thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32)
        omega_mid = (0.5 * (min_w + max_w)).unsqueeze(0).expand(env_ids.numel(), -1)
        num_rotors = omega_mid.shape[1]

        actuator_dim = self._input_reference.shape[1]
        if actuator_dim < num_rotors:
            raise RuntimeError(f"Actuator dimension ({actuator_dim}) cannot be smaller than number of rotors ({num_rotors}).")

        # First num_rotors entries are the rotor commands; keep the rest neutral.
        self._input_reference[env_ids] = 0.0
        self._input_reference[env_ids, :num_rotors] = omega_mid

        # Actual rotor state: mid throttle (spin_up) or fresh-spawn zero.
        rotor_target = omega_mid if spin_up else torch.zeros_like(omega_mid)
        rotor_velocity = getattr(thrusters, "_velocity", None)
        if isinstance(rotor_velocity, torch.Tensor):
            rotor_velocity[env_ids] = rotor_target
        rotor_input_reference = getattr(thrusters, "_input_reference", None)
        if isinstance(rotor_input_reference, torch.Tensor):
            rotor_input_reference[env_ids] = omega_mid

        # Composite propulsion models (e.g. ShuttleGlider2) propagate the
        # compatibility rotor state to their internal thrust models.
        sync_to_models = getattr(thrusters, "sync_to_models", None)
        if callable(sync_to_models):
            sync_to_models()
