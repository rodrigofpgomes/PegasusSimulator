"""
| File: skrl_agent_controller.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Evaluate a skrl checkpoint through its original Pegasus task.
|
| The task defines the observation and action contracts. The first backend
| computes actions at task decimation, then applies the held command at each
| physics step, so the policy runs at 1/(sim_dt*decimation) as in
| PegasusEnv.step. run_metadata restores the training task configuration
| (action scaling, vehicle parameters, timing) before the task is built.
| play_multi owns the shared goal and reset schedule. In PX4
| estimator mode, only the state supplied to the task changes; direct-force
| actuation remains under the task. Body-rate mode requires its own policy.
| Ground-truth evaluation retains one state/action row per vehicle. PX4 modes
| currently require a single vehicle and cannot override selected batch rows.
"""

__all__ = ["SkrlAgentBackend"]

from pathlib import Path

import torch
import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - skrl also supports classic gym
    import gym

from pegasus.simulator.logic.rl.rl_backend import RLBackend
from task_loader import apply_env_overrides, control_timing, load_agent_cfg, load_algo_class, load_env, prepare_agent_cfg
import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, Gf

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
        preset: str | None = None,
        obs_dim: int = 26,
        act_dim: int = 4,
        n_vehicles: int = 1,
        reset_manager=None,
        action_mode: str = "rotor_velocity_direct",
        device: str = "cuda:0",
        px4: dict | None = None,
        run_metadata: str | None = None,
        env_cfg_overrides: dict | None = None,
    ):
        # The observation/action dimensions and the actuation mode are taken
        # from env_cfg, NOT from obs_dim/act_dim/action_mode (kept only for
        # manifest/back-compatibility). This is what makes it task-agnostic.
        EnvClass, env_cfg = load_env(task)

        # Training snapshot first: it sets the action scaling, vehicle
        # parameters and sim_dt/decimation that the checkpoint was trained with.
        self.override_log = apply_env_overrides(env_cfg, run_metadata)

        # play_multi owns the reference (reset_manager goals); disable the env's
        # own trajectory generator so it does not fight the external reference.
        # test_mode also freezes curricula and, in direct_rotor tasks, clamps the
        # position/velocity errors in the observation (clamp_observations_in_test).
        for attr, val in (("use_raptor_trajectory", False),
                          ("trajectory_generator", False),
                          ("curriculum_enabled", False),
                          ("test_mode", True)):
            if hasattr(env_cfg, attr):
                setattr(env_cfg, attr, val)

        # Explicit manifest overrides win over both.
        explicit = apply_env_overrides(env_cfg, values=env_cfg_overrides)
        self.override_log["changed"].update(explicit["changed"])
        self.timing = control_timing(env_cfg)

        self._px4_config = px4
        self._px4_backend = None
        self._px4_sensors = None
        self._px4_actions = None
        if px4 is not None and n_vehicles != 1:
            raise ValueError("PX4 SITL evaluation supports one vehicle at a time")
        if px4 is not None and px4.get("mode") == "body_rate":
            if env_cfg.vehicle != "Shuttle" or env_cfg.action_space != 4:
                raise ValueError("PX4 body-rate mode requires a four-action Shuttle policy")
            if px4.get("policy_action_contract") != "body_rate_thrust":
                raise ValueError("Declare policy_action_contract='body_rate_thrust' for a policy trained with these outputs")
            if getattr(env_cfg, "policy_action_contract", None) != "body_rate_thrust":
                raise ValueError("The task must declare its trained body_rate_thrust action contract")
            if len(px4.get("body_rate_scale_rad_s", ())) != 3 or sorted(px4.get("motor_order", ())) != [0, 1, 2, 3]:
                raise ValueError("PX4 body-rate mode requires three rate scales and an explicit four-motor order")
            resolved_action_mode = "rotor_velocity_direct"
        else:
            resolved_action_mode = getattr(env_cfg, "action_mode", action_mode)
        # play_multi owns the shared reference cube. Task setup may still call
        # create_goal_markers(), but this backend must not create a second cube.
        super().__init__(
            n_vehicles=n_vehicles,
            action_mode=resolved_action_mode,
            goal_markers_enabled=False,
        )

        # VehicleBatch owns sampling; the secondary backend owns PX4 transport.
        # No sensors or communication backend are allocated for ground truth.
        if px4 is not None:
            from pegasus.simulator.logic.backends.px4_mavlink_estimator_backend import PX4MavlinkEstimatorBackend
            from pegasus.simulator.logic.sensors.sensor_suite_batch import SensorSuiteBatch
            self._px4_backend = PX4MavlinkEstimatorBackend(px4)
            self._px4_sensors = SensorSuiteBatch(n_vehicles, px4)

        self._checkpoint_path = checkpoint_path
        self._task = task
        self._algo = algo
        self._preset = preset
        self._EnvClass = EnvClass
        self._env_cfg = env_cfg
        self._obs_dim = int(getattr(env_cfg, "observation_space", obs_dim))
        self._act_dim = int(getattr(env_cfg, "action_space", act_dim))
        self.reset_manager = reset_manager
        self._env = None
        self._agent = None

        # Single shared reference marker (play_multi looks for create_goal_marker).
        self._ref_marker_view = None

        self._control_decimation = None
        self._control_counter = 0

        # The evaluation log reads these actions after every physics step.
        self._policy_mean_raw = None
        self._policy_action_clipped = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def _is_rotor_mode(self) -> bool:
        return self._action_mode in ("rotor_velocity", "rotor_velocity_direct")

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Allocate backend buffers after the vehicle has joined the stage."""
        # Allocates _forces/_torques/_input_reference/_state_cache and pushes the
        # input mode onto the vehicle. Called during world.reset(), BEFORE
        # setup() provides the reset_manager, so env/agent are built later.
        super().start()
        self._prime_motors()

    def setup(self, reset_manager):
        """Build the task and restore the policy after the first world reset."""
        # Called by play_multi after world.reset(): the vehicle is fully started
        # and the shared reset_manager is available, so we can now instantiate
        # the task env and build the agent.
        self.reset_manager = reset_manager
        self._build_env()
        self._build_agent()
        self._control_decimation = int(self._env.cfg.decimation)

    def after_evaluation_reset(self):
        """Seed policy history after ResetManager establishes actual actuator state."""
        self._control_counter = 0
        self._policy_mean_raw = None
        self._policy_action_clipped = None
        if self._env is None:
            return
        thr = self._vehicle._thrusters
        low, high = thr.min_rotor_velocity, thr.max_rotor_velocity
        normalized = 2 * (thr.velocity - low) / (high - low).clamp_min(1e-8) - 1
        for name in ("_last_action", "_prev_action", "_prev_prev_action", "_action_history_obs", "_actions"):
            buf = getattr(self._env, name, None)
            if isinstance(buf, torch.Tensor):
                buf.zero_()
                if self._action_mode == "rotor_velocity_direct":
                    count = min(buf.shape[-1], normalized.shape[-1])
                    buf[:, :count] = normalized[:, :count]
        for name in ("_surface_deflections", "_prev_surface_deflections", "_control_surfaces", "_prev_surfaces"):
            buf = getattr(self._env, name, None)
            if isinstance(buf, torch.Tensor):
                buf.zero_()
        # Track-frame tasks (shuttle_glider) keep the last reference heading while the
        # reference speed is low; restart it from the reset yaw, as their training reset does.
        psi = getattr(self._env, "_psi_track", None)
        if isinstance(psi, torch.Tensor):
            w, x, y, z = self._vehicle.state.attitude.to(psi.device).unbind(-1)
            psi[:] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def cruise_start(self, speed: float) -> dict | None:
        """Return ResetManager overrides for a trimmed start at ``speed`` along world +x.

        Uses the task's own trim (AlignedTrimMixin._trim_from_speed, as in the
        shuttle_glider cruise starts): pitch up by the trim angle of attack,
        horizontal velocity, trim rotor speeds and neutral surfaces. The requested
        angle of attack is the robust envelope cap at that speed. Returns None if
        the task has no trim; a fixed-wing task should implement the same method.
        """
        trim = getattr(self._env, "_trim_from_speed", None)
        if trim is None:
            return None
        from pegasus.simulator.logic.transforms import euler_angles_to_matrix, matrix_to_quaternion

        n = self._n_vehicles
        V0 = torch.full((n,), float(speed), device=self._device)
        alpha, _surfaces, rotor_norm = trim(V0, torch.full_like(V0, float("inf")))
        angles = torch.stack((torch.zeros_like(alpha), -alpha, torch.zeros_like(alpha)), dim=1)
        velocity = torch.zeros((n, 3), device=self._device)
        velocity[:, 0] = V0
        return {"orientations": matrix_to_quaternion(euler_angles_to_matrix(angles, convention="ZYX")),
                "linear_velocity": velocity, "rotor_norm": rotor_norm,
                "angular_velocity": torch.zeros((n, 3), device=self._device),
                "alpha_trim_deg": float(torch.rad2deg(alpha[0]))}

    def stop(self):
        """Close the optional PX4 connection before the simulator stops."""
        if self._px4_backend is not None:
            self._px4_backend.close()
        if self._px4_config is not None:
            self._received_first_state = False

    def reset(self):
        """Restart policy history and motor references for a new episode."""
        if self._px4_backend is not None and self._px4_backend.ready:
            raise RuntimeError("PX4 SITL estimator cannot follow an episode teleport; restart SITL and evaluation")
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
        """Update PX4, compute a policy action when due, and hold it for physics.

        The shared goal is copied into tasks with private goal buffers before
        constructing the observation. The task then converts the selected
        action to a body wrench or rotor reference exactly as in training.
        """
        if self._agent is None or self._env is None or not self._received_first_state:
            return

        if self._px4_config is not None:
            if self._px4_backend is None:
                raise RuntimeError("PX4 SITL backend must be prepared before evaluation")
            self._px4_backend.step(dt)

        # If the task keeps its own goal buffers, feed them the externally-driven
        # reference so observations are computed against play_multi's trajectory.
        # Tasks that read reset_manager.goal_* directly need nothing here.
        if self.reset_manager is not None:
            self._sync_env_goal("_goal_pos", "goal_pos")
            self._sync_env_goal("_goal_vel", "goal_vel")
            self._sync_env_goal("_goal_acc", "goal_acc")

        # Compute a new action only at the policy/control frequency.
        if self._control_counter % self._control_decimation == 0:
            # Tasks with observed moving averages update them in _get_rewards during training, which this loop does
            # not call: advance them here, after the physics of the previous decision (none before the first one,
            # as after a reset in training)
            if self._control_counter > 0 and hasattr(self._env, "update_filters"):
                self._env.update_filters()
            obs = self._env._get_observations()["policy"]
            sampled_actions, _, outputs = self._agent.act(obs, timestep=0, timesteps=0)
            if self._algo in ("ppo", "ppo2") and "mean_actions" not in outputs:
                raise KeyError("PPO eval requires outputs['mean_actions']")
            mean_raw = outputs.get("mean_actions", sampled_actions)
            actions = torch.clamp(mean_raw, -1.0, 1.0)
            self._policy_mean_raw = mean_raw.detach().clone()
            self._policy_action_clipped = actions.detach().clone()
            self._env._pre_physics_step(actions)  # store the new action
            if self._px4_backend is not None:
                self._px4_backend.record(obs)

            if self._px4_backend is not None and self._px4_backend.mode == "body_rate":
                self._px4_actions = actions

        # Re-apply the stored action every physics step.
        if self._px4_backend is not None and self._px4_backend.mode == "body_rate":
            self._px4_backend.send_body_rate(self._px4_actions)
            self._px4_backend.request_offboard_and_arm()
            self._px4_backend._poll()
            self._input_reference = self._px4_backend.motor_velocities()
        else:
            self._env._apply_action()
        self._control_counter += 1

        # Inherited: for 'rotor_velocity' turns forces/torques into rotor speeds;
        # for 'rotor_velocity_direct'/direct-force this is a no-op.
        super().update(dt)

    def prepare_px4(self, dt: float, render, log_path: Path) -> None:
        """Connect PX4 and wait for a valid estimate before the first policy step."""
        if self._px4_config is None:
            return
        try:
            self._px4_backend.connect(log_path)
            self._px4_backend.warmup(dt, render)
            if self._px4_backend.mode == "body_rate":
                self._px4_backend.activate_body_rate(dt, render)
            self._received_first_state = True
        except Exception:
            self._px4_backend.close()
            raise

    def get_state(self) -> torch.Tensor:
        """Supply PX4 estimates to task observations when the backend is ready."""
        if self._px4_config is None:
            return super().get_state()
        if self._px4_backend is None or not self._px4_backend.ready:
            raise RuntimeError("PX4 estimated observations are unavailable")
        return self._px4_backend.current

    def update_state(self, state) -> None:
        """Cache physical state only when evaluation selects ground truth.

        In PX4 mode, get_state() reads the secondary estimator backend. Physical
        state remains available to that backend's diagnostic log through the
        vehicle, without entering the controller's observation cache.
        """
        if self._px4_config is None:
            super().update_state(state)

    def set_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ):
        """Set the reset state and prime rotor commands for rotor tasks."""
        if self._px4_config is None:
            super().set_state(env_ids, positions, attitudes, linear_velocity, angular_velocity)
        # Prime the command reference to avoid free-fall on reset transients, but
        # leave the actual rotor velocity at 0 (fresh-spawn condition) so a
        # deferred teleport cannot re-pollute the first observation of an episode.
        self._prime_motors(env_ids=env_ids, spin_up=False)

    # Actuator references and force/torque buffers retain the RLBackend contract.

    # ------------------------------------------------------------------
    # Shared reference marker (single cube, used by play_multi)
    # ------------------------------------------------------------------
    def create_goal_marker(self, root_path: str = "/World/GoalMarker", size: float = 0.15, color: tuple = (1.0, 0.0, 0.0)):
        """Create the shared play_multi reference marker for this batch."""
        stage = self._vehicle._world.stage
        if not stage.GetPrimAtPath(root_path).IsValid():
            prim_utils.create_prim(root_path, "Xform")
        for i in range(self._n_vehicles):
            prim_path = f"{root_path}/goal_{i}"
            if not stage.GetPrimAtPath(prim_path).IsValid():
                prim_utils.create_prim(prim_path, "Cube", translation=[0.0, 0.0, -100.0], scale=[size, size, size])
            cube = UsdGeom.Cube(stage.GetPrimAtPath(prim_path))
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        self._ref_marker_view = XFormPrimView(
            prim_paths_expr=f"{root_path}/goal_*",
            name=f"skrl_ref_marker_view_{root_path.replace('/', '_')}",
        )

    def update_goal_marker(self, positions: torch.Tensor):
        """Move the shared marker to the current external goal."""
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
        agent_cfg = load_agent_cfg(self._task, self._algo, self._preset)
        AgentClass = load_algo_class(self._algo)

        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)

        cfg = prepare_agent_cfg(agent_cfg["cfg"], dev)
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
        # skrl returns random warm-up actions while timestep < random_timesteps (SAC presets)
        self._agent._random_timesteps = 0

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
