"""
skrl_agent_controller.py
Pegasus batched backend that runs a trained skrl agent (PPO or SAC) for
inference, EXACTLY as examples/rl/play.py does -- but with the observation and
the action built BY THE TASK ENVIRONMENT ITSELF.

Why this design
---------------
The previous version hard-coded a single observation layout (the 26-dim
multirotor/raptor vector) and a single actuation path (rotor-velocity). That
only works for tasks trained with that exact protocol. A checkpoint trained on a
different task -- e.g. ``double_integrator/01_isaac_lab`` (12-dim observation,
direct thrust+torque actions) -- then fails at load time with a shape mismatch:

    size mismatch for net.0.weight: copying a param with shape [64, 12]
    from checkpoint, the shape in current model is [64, 26].

The robust fix is to stop re-implementing the observation/action here and
instead reuse the task's own ``QuadcopterEnv``. This backend therefore:

    1. loads the task env class + config (like play.py:load_env),
    2. subclasses RLBackend so the env can talk to it exactly like in training
       (get_state / set_forces_and_torques / _input_reference / create_goal_markers),
    3. instantiates ``EnvClass(env_cfg, self, reset_manager)`` and calls its
       ``setup()``,
    4. every step asks the env to build the observation
       (``env._get_observations()``) and to apply the action
       (``env._pre_physics_step()`` + ``env._apply_action()``).

Because the observation/action dimensions and the actuation mode now come from
``env_cfg`` (``observation_space`` / ``action_space`` / ``action_mode``), the
same controller transparently supports any task: raptor_pretrain
(26-dim, rotor_velocity_direct) and double_integrator (12-dim, direct_force)
alike, with no per-task code here.

The skrl agent is still built and restored exactly like play.py
(load_agent_cfg -> build models -> AgentClass -> agent.load -> eval mode), so
any state preprocessor baked into the checkpoint (e.g. PPO's
RunningStandardScaler) is applied identically. As in play.py, ``agent.act()``
returns a STOCHASTIC sample even in eval mode, so we take the deterministic
distribution mean from ``outputs["mean_actions"]``.

Reference (goal) handling
-------------------------
In the multi-vehicle player the reference is driven externally by play_multi
(``reset_manager._goal_pos`` / ``_goal_vel``, optionally a lemniscate). So:
  * any task that reads ``reset_manager.goal_*`` (e.g. raptor) follows it for
    free -- and we force ``use_raptor_trajectory = False`` so the env does NOT
    generate its own competing trajectory;
  * any task that keeps its own goal buffer (e.g. double_integrator's
    ``_goal_pos``) is synced from ``reset_manager.goal_pos`` every step.
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


# examples/rl (so that "tasks.<...>" and "tasks.<...>.agents.<algo>_cfg" are
# importable, exactly as in play.py; play_multi.py already inserts this
# directory into sys.path).
RL_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = RL_DIR / "tasks"


def _load_env(task: str):
    """Mirror of play.py:load_env -> (EnvClass, env_cfg_instance)."""
    task_dir = TASKS_DIR.joinpath(*task.split("/"))

    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]
    if not env_files:
        raise FileNotFoundError(f"No *_env.py found in {task_dir}")

    env_file = env_files[0]
    module_name = env_file[:-3]

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
    """Batched Pegasus backend running a trained skrl agent (PPO/SAC) in eval
    mode, with observation and action produced by the task env itself.

    It subclasses RLBackend so the instantiated task env interacts with it
    through the very same interface used during training
    (``get_state``/``set_forces_and_torques``/``_input_reference``/goal markers), and
    so the force/torque -> rotor conversion for ``rotor_velocity`` tasks is
    inherited for free.
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
        # Load the task env class + config up-front: the observation/action
        # dimensions and the actuation mode are taken from here, NOT from the
        # obs_dim/act_dim/action_mode arguments (kept only for manifest/back-
        # compatibility). This is what makes the controller task-agnostic.
        EnvClass, env_cfg = _load_env(task)

        # The reference trajectory in the multi-vehicle player is owned by
        # play_multi (reset_manager goals). Disable the env's own trajectory
        # generator so it does not fight the external reference.
        if hasattr(env_cfg, "use_raptor_trajectory"):
            env_cfg.use_raptor_trajectory = False

        if hasattr(env_cfg, "test_mode"):
            env_cfg.test_mode = True  # disable the env's own test-mode reference generator

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
        # setup() provides the reset_manager -- so the env/agent are built later.
        super().start()
        self._prime_motors()

    def setup(self, reset_manager):
        # Called by play_multi after world.reset(): the vehicle is fully started
        # (rotor geometry/thrusters allocated) and the shared reset_manager is
        # available, so we can now instantiate the task env and build the agent.
        self.reset_manager = reset_manager
        self._build_env()
        self._build_agent()

    def stop(self):
        pass

    def reset(self):
        super().reset()
        # Clear the env's action-history buffers (whatever the task keeps).
        if self._env is not None:
            for attr in ("_last_action", "_prev_action", "_action_history_obs", "_actions"):
                buf = getattr(self._env, attr, None)
                if isinstance(buf, torch.Tensor):
                    buf.zero_()
        self._prime_motors()

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, dt: float):
        if self._agent is None or self._env is None or not self._received_first_state:
            return

        # If the task keeps its own goal buffer (e.g. double_integrator's
        # _goal_pos), feed it the externally-driven reference so observations
        # are computed against play_multi's trajectory/goal. Tasks that read
        # reset_manager.goal_* directly (e.g. raptor) need nothing here.
        if self.reset_manager is not None:
            env_goal = getattr(self._env, "_goal_pos", None)
            if isinstance(env_goal, torch.Tensor):
                env_goal[:] = self.reset_manager.goal_pos.to(self._device, dtype=torch.float32)

        # Observation built by the env exactly as in training.
        obs = self._env._get_observations()["policy"]

        # Same call play.py uses. act() returns a STOCHASTIC sample even in eval
        # mode, so take the deterministic distribution mean. Any state
        # preprocessor restored by agent.load() is applied inside act().
        _sampled, _, _outputs = self._agent.act(obs, timestep=0, timesteps=0)
        actions = _outputs.get("mean_actions", _sampled)
        actions = torch.clamp(actions, -1.0, 1.0)

        # Action applied by the env exactly as in training: this sets either the
        # rotor reference (_input_reference + thrusters) or the body forces/torques
        # (set_forces_and_torques), depending on the task's action_mode.
        self._env._pre_physics_step(actions)
        self._env._apply_action()

        # Inherited conversion: for 'rotor_velocity' tasks turns the forces/
        # torques into rotor speeds; for 'rotor_velocity_direct' and direct-force
        # tasks this is a no-op (the env already set the actuation).
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
        # Keep the drone from free-falling on reset transients (rotor tasks).
        self._prime_motors(env_ids=env_ids)

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

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32
        )

        cfg = _prepare_cfg(agent_cfg["cfg"], dev)
        # play.py sets the state preprocessor size to the observation space.
        if isinstance(cfg.get("state_preprocessor_kwargs"), dict):
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

        self._agent.load(self._checkpoint_path)
        self._agent.set_running_mode("eval")
        print(
            f"[SkrlAgentBackend] Loaded {self._algo} ({self._task}/{self._preset}) "
            f"checkpoint: {self._checkpoint_path}",
            flush=True,
        )

    def _prime_motors(self, env_ids: torch.Tensor | None = None):
        """Set rotors to mid throttle so the drone does not drop before the first
        action. No-op for direct-force tasks (no rotor reference is used)."""
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
        self._input_reference[env_ids] = omega_mid
        thrusters._input_reference[env_ids] = omega_mid
        thrusters._velocity[env_ids] = omega_mid
