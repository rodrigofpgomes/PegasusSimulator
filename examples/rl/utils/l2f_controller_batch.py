"""
| File: l2f_controller_batch.py
| License: BSD-3-Clause.
| Description: Pegasus batched backend for running a Learning-to-Fly policy.
|
| This backend adapts a Learning-to-Fly Crazyflie policy to Pegasus:
|   Pegasus StateBatch ENU/FLU state -> L2F policy input
|   L2F normalized motor action [-1, 1] -> rotor speed command [rad/s]
|
| Motor first-order delay is intentionally NOT applied here. It belongs in
| QuadraticThrustCurveBatch through the motor_time_constant parameter.
"""

__all__ = ["L2FBackend", "LQRBackend"]

from pathlib import Path
import itertools
import math
import sys

import torch
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state_batch import StateBatch

# l2f_controller_batch.py is often loaded with importlib.util.spec_from_file_location,
# so the utils directory is not guaranteed to be on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from l2f_policy_adapter import L2FPolicyAdapter


class L2FBackendCfg:
    """Constants required to adapt the L2F policy to Pegasus."""

    # Learning-to-Fly Crazyflie action limits. The actor output is normalized
    # in [-1, 1] and must be mapped back to this RPM interval before conversion
    # to Pegasus rotor angular velocity [rad/s].
    l2f_min_rpm: float = 0.0
    l2f_max_rpm: float = 21702.0

    # L2F initial_state() sets rpm to the middle of the action range and
    # initializes action_history to zero. Do the same in Pegasus instead of
    # starting motors at 0 RPM.
    l2f_initial_rpm: float = 0.5 * (l2f_min_rpm + l2f_max_rpm)

    # Kept unchanged for this debug version. If needed, tune this later.
    max_target_error_m: float = 0.30

    # Debug controls.
    debug_first_update: bool = False
    debug_start: bool = False
    debug_update_state: bool = False
    debug_action: bool = False
    debug_max_prints: int = 20

    # Rotor order used in learning_to_fly/parameters/dynamics/crazy_flie.h:
    #   0: x+, y-   1: x-, y-   2: x-, y+   3: x+, y+
    # in ENU/FLU body coordinates.
    l2f_rotor_xy = (
        (0.028, -0.028),
        (-0.028, -0.028),
        (-0.028, 0.028),
        (0.028, 0.028),
    )

    # Rotor yaw directions in L2F rotor order. These are copied into the Pegasus
    # thrust curve in Pegasus/USD rotor order at start(), after the physical
    # rotor positions have been discovered.
    l2f_rot_dir = (-1, 1, -1, 1)


class L2FBackend(Backend):
    """
    Batched Pegasus backend that runs a Learning-to-Fly actor and outputs rotor
    angular velocities in Pegasus rotor order.

    Input to the L2F model:
        position              ENU world position [m]
        orientation_wxyz       body-to-world quaternion [w, x, y, z]
        linear_velocity        ENU world velocity [m/s]
        angular_velocity       FLU body angular velocity [rad/s]
        rpm                    current motor speeds in L2F rotor order [RPM]
        target_position        ENU world target position [m]
        target_linear_velocity ENU world target velocity [m/s]

    Output adaptation:
        actor action [-1, 1]
          -> L2F RPM command [0, 21702]
          -> Pegasus motor angular velocity [rad/s]
          -> reordered from L2F rotor order to the USD/Pegasus rotor order.
    """

    def __init__(
        self,
        n_vehicles: int,
        reset_manager=None,
        action_mode: str = "rotor_velocity",
        checkpoint_path: str | None = None,
        # Kept only so older play.py files that still pass this argument do not
        # crash. The delay is not applied in this backend.
        emulate_motor_delay: bool | None = None,
        **_unused,
    ):
        super().__init__(config=None)

        self.cfg = L2FBackendCfg()

        self._n_vehicles = int(n_vehicles)
        self._action_mode = action_mode
        self.reset_manager = reset_manager

        self._checkpoint_path = checkpoint_path or (
            "/home/rodrigogomes/learning_to_fly/checkpoints/multirotor_td3/"
            "2026_05_03_00_30_01_d+o+a+r+h+c+f+w+e+_000/"
            "actor_000000000300000.h5"
        )

        self._parts_per_vehicle = None
        self._device = None
        self._vehicle = None

        self._forces = None
        self._torques = None
        self._input_ref = None
        self._state_cache = None
        self._current_rpm_l2f = None
        self._received_first_state = False

        self._l2f_to_pegasus = None
        self.policy = None

        # Debug/recording buffers.
        self._last_action_norm_l2f = None
        self._last_rpm_cmd_l2f = None
        self._last_current_rpm_l2f = None
        self._printed_first_update_debug = False

        # Repeated debug counters.
        self._debug_state_counter = 0
        self._debug_action_counter = 0

        self.counter = 0

    # ------------------------------------------------------------------
    # Backend Interface
    # ------------------------------------------------------------------

    def initialize(self, vehicle):
        """Stores the vehicle batch reference after spawning."""
        self._vehicle = vehicle

    def setup(self, reset_manager):
        """Sets up the backend with the shared ResetManager."""
        self.reset_manager = reset_manager

    def start(self):
        """Allocates buffers and loads the L2F policy when simulation starts."""
        self._n_vehicles = self.vehicle.n_vehicles
        self._parts_per_vehicle = self.vehicle.parts_per_vehicle
        self._device = self.vehicle.device

        self._forces = torch.zeros(
            (self._n_vehicles, self._parts_per_vehicle, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._torques = torch.zeros_like(self._forces)
        self._input_ref = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)
        self._current_rpm_l2f = torch.full(
            (self._n_vehicles, 4),
            self._initial_rpm(),
            dtype=torch.float32,
            device=self._device,
        )

        self._l2f_to_pegasus = self._infer_l2f_to_pegasus_rotor_order()
        self._sync_thruster_rotor_directions()

        self.policy = L2FPolicyAdapter(
            checkpoint_path=self._checkpoint_path,
            num_envs=self._n_vehicles,
            device=self._device,
        )

        self._last_action_norm_l2f = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)
        self._last_rpm_cmd_l2f = self._current_rpm_l2f.clone()
        self._last_current_rpm_l2f = self._current_rpm_l2f.clone()
        self._printed_first_update_debug = False
        self._debug_state_counter = 0
        self._debug_action_counter = 0

        self._received_first_state = False
        self.vehicle.set_input_mode(self._action_mode)
        self._prime_motors(self._initial_rpm())

        if self.cfg.debug_start:
            self._print_start_debug()

    def stop(self):
        pass

    def reset(self):
        """Clears command buffers and resets the policy history."""
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._state_cache.zero_()
            self._prime_motors(self._initial_rpm())

            if self._last_action_norm_l2f is not None:
                self._last_action_norm_l2f.zero_()
            if self._last_rpm_cmd_l2f is not None:
                self._last_rpm_cmd_l2f.fill_(self._initial_rpm())
            if self._last_current_rpm_l2f is not None:
                self._last_current_rpm_l2f.fill_(self._initial_rpm())
            self._printed_first_update_debug = False
            self._debug_state_counter = 0
            self._debug_action_counter = 0

        if self.policy is not None:
            self.policy.reset()

        self._received_first_state = False

    @torch.no_grad()
    def update(self, dt: float):
        """Runs the L2F policy and updates the Pegasus rotor-speed command."""
        if not self._received_first_state or self.policy is None:
            return

        if self._action_mode != "rotor_velocity":
            raise RuntimeError("L2FBackend expects action_mode='rotor_velocity'.")

        pos_w = self._state_cache[:, 0:3]
        vel_w = self._state_cache[:, 3:6]
        quat_wxyz = self._normalize_quaternion(self._state_cache[:, 6:10])
        ang_vel_b = self._state_cache[:, 10:13]

        current_rpm_l2f = self._read_current_rpm_l2f()
        goal_pos_w = self._read_target_position(default_like=pos_w)
        target_vel_w = torch.zeros_like(vel_w)

        #self._print_pre_act_debug(
        #    pos_w=pos_w,
        #    vel_w=vel_w,
        #    quat_wxyz=quat_wxyz,
        #    ang_vel_b=ang_vel_b,
        #    current_rpm_l2f=current_rpm_l2f,
        #    goal_pos_w=goal_pos_w,
        #    target_pos_w=goal_pos_w,
        #    dt=dt,
        #)

        debug = 0

        if debug == 1:
            print("[L2F OBS FINAL]")
            print("state pos used:", pos_w[0].detach().cpu().numpy())
            print("goal:", goal_pos_w[0].detach().cpu().numpy())
            print("goal - pos:", (goal_pos_w - pos_w)[0].detach().cpu().numpy())
            print("vel_w:", vel_w[0].detach().cpu().numpy())
            print("rpm:", current_rpm_l2f[0].detach().cpu().numpy())

        action_norm_l2f = self.policy.act(
            position=pos_w,
            orientation_wxyz=quat_wxyz,
            linear_velocity=vel_w,
            angular_velocity=ang_vel_b,
            rpm=current_rpm_l2f,
            target_position=goal_pos_w,
            target_linear_velocity=target_vel_w,
        )

        rpm_cmd_l2f = self._action_norm_to_rpm(action_norm_l2f)

        # No motor-delay emulation here. QuadraticThrustCurveBatch applies the
        # first-order motor dynamics through motor_time_constant.
        omega_cmd_pegasus = self._l2f_rpm_to_pegasus_rad_s(rpm_cmd_l2f)
        self._input_ref = omega_cmd_pegasus

        self._current_rpm_l2f = current_rpm_l2f
        self._last_current_rpm_l2f = current_rpm_l2f
        self._last_action_norm_l2f = action_norm_l2f
        self._last_rpm_cmd_l2f = rpm_cmd_l2f

        #self._print_post_act_debug(
        #    action_norm_l2f=action_norm_l2f,
        #    rpm_cmd_l2f=rpm_cmd_l2f,
        #    omega_cmd_pegasus=omega_cmd_pegasus,
        #)

        if self.counter % 50 == 0:
            print(
                "[L2F] update "
                f"pos={pos_w[0].detach().cpu().tolist()} "
                f"goal={goal_pos_w[0].detach().cpu().tolist()} "
                f"err={(pos_w[0] - goal_pos_w[0]).detach().cpu().tolist()} "
                f"rpm_in={current_rpm_l2f[0].detach().cpu().tolist()} "
                f"action={action_norm_l2f[0].detach().cpu().tolist()} "
                f"rpm_cmd={rpm_cmd_l2f[0].detach().cpu().tolist()} "
                f"omega_cmd_pegasus={omega_cmd_pegasus[0].detach().cpu().tolist()} "
                f"l2f_to_pegasus={self._l2f_to_pegasus.detach().cpu().tolist()}",
                flush=True,
            )

        self.counter += 1

        self._debug_action_counter += 1

    def update_state(self, state: StateBatch):
        """Caches the new physics state after a simulation step."""
        self._state_cache = torch.cat(
            [
                state.position,
                state.linear_velocity,
                state.attitude,
                state.angular_velocity,
            ],
            dim=-1,
        )
        self._received_first_state = True

        if self.cfg.debug_update_state and self._debug_state_counter < int(self.cfg.debug_max_prints):
            q = state.attitude[0]
            q_norm = torch.linalg.norm(q).item()

            print(
                "[L2F update_state] "
                f"i={self._debug_state_counter} "
                f"pos={state.position[0].detach().cpu().tolist()} "
                f"vel_w={state.linear_velocity[0].detach().cpu().tolist()} "
                f"quat_wxyz={state.attitude[0].detach().cpu().tolist()} "
                f"quat_norm={q_norm:.6f} "
                f"ang_vel_b={state.angular_velocity[0].detach().cpu().tolist()}",
                flush=True,
            )

        self._debug_state_counter += 1

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self) -> torch.Tensor:
        """Returns commanded rotor angular velocities in Pegasus order [rad/s]."""
        return self._input_ref

    # ------------------------------------------------------------------
    # VecEnv Interface
    # ------------------------------------------------------------------

    def set_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ):
        """Overrides cached state for selected environments, e.g. during resets."""
        if self._state_cache is None or env_ids.numel() == 0:
            return

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        lin_vel = linear_velocity if linear_velocity is not None else torch.zeros(
            (env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype
        )
        ang_vel = angular_velocity if angular_velocity is not None else torch.zeros(
            (env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype
        )

        self._state_cache[env_ids, 0:3] = positions.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 3:6] = lin_vel.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 6:10] = attitudes.to(self._device, dtype=torch.float32)
        self._state_cache[env_ids, 10:13] = ang_vel.to(self._device, dtype=torch.float32)

        self._prime_motors(self._initial_rpm(), env_ids=env_ids)
        if self._last_rpm_cmd_l2f is not None:
            self._last_rpm_cmd_l2f[env_ids] = self._initial_rpm()
        if self._last_current_rpm_l2f is not None:
            self._last_current_rpm_l2f[env_ids] = self._initial_rpm()
        if self._last_action_norm_l2f is not None:
            self._last_action_norm_l2f[env_ids] = 0.0

        # The pybind policy currently exposes only reset_history(batch_size), not
        # per-environment reset. Resetting the full history avoids feeding stale
        # action history to freshly reset environments.
        if self.policy is not None:
            self.policy.reset()

        self._received_first_state = True

    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forces, self._torques

    def get_state(self) -> torch.Tensor:
        return self._state_cache

    # ------------------------------------------------------------------
    # Optional debug accessors used by the recorder
    # ------------------------------------------------------------------

    @property
    def last_action_norm_l2f(self) -> torch.Tensor | None:
        return self._last_action_norm_l2f

    @property
    def last_rpm_cmd_l2f(self) -> torch.Tensor | None:
        return self._last_rpm_cmd_l2f

    @property
    def last_current_rpm_l2f(self) -> torch.Tensor | None:
        return self._last_current_rpm_l2f

    @property
    def l2f_to_pegasus(self) -> torch.Tensor | None:
        return self._l2f_to_pegasus

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _print_start_debug(self):
        rotor_positions = self.vehicle._rotor_positions_body
        thrusters = self.vehicle._thrusters

        print(
            "[L2F start] "
            f"device={self._device} "
            f"n_vehicles={self._n_vehicles} "
            f"parts_per_vehicle={self._parts_per_vehicle} "
            f"input_mode={self._action_mode} "
            f"initial_rpm={self._initial_rpm()} "
            f"l2f_to_pegasus={self._l2f_to_pegasus.detach().cpu().tolist()}",
            flush=True,
        )

        if rotor_positions is not None:
            print(
                "[L2F start] rotor_positions_body[0]="
                f"{rotor_positions[0].detach().cpu().tolist()}",
                flush=True,
            )

        if thrusters is not None:
            rot_dir = thrusters.rot_dir
            min_w = thrusters.min_rotor_velocity
            max_w = thrusters.max_rotor_velocity
            vel = thrusters.velocity

            def _maybe_list(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().tolist()
                return x

            if isinstance(vel, torch.Tensor) and vel.ndim >= 2:
                vel0 = vel[0]
            else:
                vel0 = vel

            print(
                "[L2F start] thrusters "
                f"rot_dir={_maybe_list(rot_dir)} "
                f"min_w={_maybe_list(min_w)} "
                f"max_w={_maybe_list(max_w)} "
                f"velocity0={_maybe_list(vel0)}",
                flush=True,
            )

    def _print_pre_act_debug(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        ang_vel_b: torch.Tensor,
        current_rpm_l2f: torch.Tensor,
        goal_pos_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        dt: float,
    ):
        if not self.cfg.debug_action or self._debug_action_counter >= int(self.cfg.debug_max_prints):
            return

        print(
            "[L2F pre-act] "
            f"i={self._debug_action_counter} "
            f"dt={float(dt):.6f} "
            f"pos={pos_w[0].detach().cpu().tolist()} "
            f"vel_w={vel_w[0].detach().cpu().tolist()} "
            f"quat_wxyz={quat_wxyz[0].detach().cpu().tolist()} "
            f"ang_vel_b={ang_vel_b[0].detach().cpu().tolist()} "
            f"rpm_in={current_rpm_l2f[0].detach().cpu().tolist()} "
            f"goal={goal_pos_w[0].detach().cpu().tolist()} "
            f"target={target_pos_w[0].detach().cpu().tolist()} "
            f"err_pos_minus_target={(pos_w[0] - target_pos_w[0]).detach().cpu().tolist()}",
            flush=True,
        )

    def _print_post_act_debug(
        self,
        action_norm_l2f: torch.Tensor,
        rpm_cmd_l2f: torch.Tensor,
        omega_cmd_pegasus: torch.Tensor,
    ):
        if not self.cfg.debug_action or self._debug_action_counter >= int(self.cfg.debug_max_prints):
            return

        print(
            "[L2F post-act] "
            f"i={self._debug_action_counter} "
            f"action={action_norm_l2f[0].detach().cpu().tolist()} "
            f"rpm_cmd_l2f={rpm_cmd_l2f[0].detach().cpu().tolist()} "
            f"omega_cmd_pegasus={omega_cmd_pegasus[0].detach().cpu().tolist()} "
            f"l2f_to_pegasus={self._l2f_to_pegasus.detach().cpu().tolist()}",
            flush=True,
        )

    def _hover_rpm(self) -> float:
        # L2F Crazyflie hover speed with m=0.027 kg, g=9.81 and kf=3.16e-10.
        return 14478.0

    def _initial_rpm(self) -> float:
        # L2F initial_state() uses the middle of the action range, not zero RPM.
        # action_history is then initialized to zero.
        return float(self.cfg.l2f_initial_rpm)

    def _prime_motors(self, rpm: float, env_ids: torch.Tensor | None = None):
        """Set Pegasus command and actual motor state to an L2F-consistent RPM."""
        if self._device is None or self._l2f_to_pegasus is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self._n_vehicles, dtype=torch.long, device=self._device)
        else:
            env_ids = env_ids.to(device=self._device, dtype=torch.long)

        rpm_l2f = torch.full(
            (env_ids.numel(), 4),
            float(rpm),
            dtype=torch.float32,
            device=self._device,
        )
        omega_pegasus = self._l2f_rpm_to_pegasus_rad_s(rpm_l2f)

        if self._input_ref is not None:
            self._input_ref[env_ids] = omega_pegasus
        if self._current_rpm_l2f is not None:
            self._current_rpm_l2f[env_ids] = rpm_l2f

        thrusters = self.vehicle._thrusters
        thrusters._input_reference[env_ids] = omega_pegasus
        thrusters._velocity[env_ids] = omega_pegasus

    def _normalize_quaternion(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(quat_wxyz, dim=1, keepdim=True).clamp_min(1e-6)
        return quat_wxyz / norm

    def _action_norm_to_rpm(self, action_norm_l2f: torch.Tensor) -> torch.Tensor:
        action_norm_l2f = torch.clamp(action_norm_l2f, -1.0, 1.0)
        half_range = 0.5 * (self.cfg.l2f_max_rpm - self.cfg.l2f_min_rpm)
        center = self.cfg.l2f_min_rpm + half_range
        return action_norm_l2f * half_range + center

    def _l2f_rpm_to_pegasus_rad_s(self, rpm_l2f: torch.Tensor) -> torch.Tensor:
        omega_l2f = rpm_l2f * (2.0 * math.pi / 60.0)

        # Reorder from L2F rotor order to the order expected by the Pegasus USD.
        omega_pegasus = torch.empty_like(omega_l2f)
        omega_pegasus[:, self._l2f_to_pegasus] = omega_l2f

        thrusters = self.vehicle._thrusters
        min_w = thrusters.min_rotor_velocity.to(device=self._device, dtype=torch.float32).unsqueeze(0)
        max_w = thrusters.max_rotor_velocity.to(device=self._device, dtype=torch.float32).unsqueeze(0)
        omega_pegasus = torch.clamp(omega_pegasus, min=min_w, max=max_w)

        return omega_pegasus

    def _read_current_rpm_l2f(self) -> torch.Tensor:
        """Reads Pegasus motor speed and returns it in L2F order as RPM."""
        omega_pegasus = self.vehicle._thrusters.velocity

        if omega_pegasus is None:
            # Fallback before the first thruster update.
            return self._current_rpm_l2f.clone()

        omega_pegasus = torch.as_tensor(omega_pegasus, dtype=torch.float32, device=self._device)
        if omega_pegasus.ndim == 1:
            omega_pegasus = omega_pegasus.unsqueeze(0).expand(self._n_vehicles, -1)

        # L2F treats RPM as a positive rotor speed magnitude.
        omega_pegasus = torch.abs(omega_pegasus)
        omega_l2f = omega_pegasus[:, self._l2f_to_pegasus]
        rpm_l2f = omega_l2f * (60.0 / (2.0 * math.pi))
        rpm_l2f = torch.clamp(rpm_l2f, self.cfg.l2f_min_rpm, self.cfg.l2f_max_rpm)

        self._current_rpm_l2f = rpm_l2f
        return rpm_l2f

    def _read_target_position(self, default_like: torch.Tensor) -> torch.Tensor:
        goal = None
        if self.reset_manager is not None:
            goal = self.reset_manager.goal_pos

        if goal is None:
            return torch.zeros_like(default_like)

        goal1 = torch.as_tensor(goal, dtype=torch.float32, device=self._device)

        goal = goal1 - self.reset_manager.main_vehicle._init_pos + self._vehicle._init_pos

        #print(f"difference between goal and goal1: {(goal - goal1)[0].detach().cpu().tolist()}")

        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(self._n_vehicles, -1)
        elif goal.shape[0] == 1 and self._n_vehicles > 1:
            goal = goal.expand(self._n_vehicles, -1)

        if goal.shape != default_like.shape:
            raise ValueError(
                f"target_position must have shape {tuple(default_like.shape)}, "
                f"got {tuple(goal.shape)}"
            )
        return goal

    def _local_target_position(self, pos_w: torch.Tensor, goal_pos_w: torch.Tensor) -> torch.Tensor:
        """Return a bounded local waypoint toward the real goal.

        The original L2F evaluation clamps tracking error to +/-0.6 m. Sending
        far Pegasus goals directly can pin the observation at the clamp and make
        the actor output saturated motor commands. This keeps the policy in a
        smaller local-tracking regime while still moving toward the final goal.
        """
        limit = float(self.cfg.max_target_error_m)
        if limit <= 0.0:
            return goal_pos_w
        delta = torch.clamp(goal_pos_w - pos_w, min=-limit, max=limit)
        return pos_w + delta

    def _infer_l2f_to_pegasus_rotor_order(self) -> torch.Tensor:
        """
        Returns a tensor p where p[i] is the Pegasus rotor index corresponding
        to L2F rotor i. In other words, omega_l2f = omega_pegasus[:, p].
        """
        identity = torch.arange(4, dtype=torch.long, device=self._device)
        rotor_positions = self.vehicle._rotor_positions_body

        if rotor_positions is None:
            return identity

        pegasus_xy = torch.as_tensor(rotor_positions[0, :, :2], dtype=torch.float32, device=self._device)
        if pegasus_xy.shape != (4, 2) or not torch.isfinite(pegasus_xy).all():
            return identity

        l2f_xy = torch.tensor(self.cfg.l2f_rotor_xy, dtype=torch.float32, device=self._device)

        # Compare signs/quadrants rather than absolute arm length, so this also
        # works if the USD geometry has the same X layout but a slightly different
        # arm length.
        pegasus_norm = pegasus_xy / pegasus_xy.abs().max().clamp_min(1e-6)
        l2f_norm = l2f_xy / l2f_xy.abs().max().clamp_min(1e-6)

        best_perm = tuple(range(4))
        best_cost = float("inf")
        for perm in itertools.permutations(range(4)):
            candidate = pegasus_norm[list(perm), :]
            cost = torch.sum((candidate - l2f_norm) ** 2).item()
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        return torch.tensor(best_perm, dtype=torch.long, device=self._device)

    def _sync_thruster_rotor_directions(self):
        """Writes L2F rotor yaw directions into the Pegasus thrust curve order."""
        thrusters = self.vehicle._thrusters

        l2f_dirs = torch.tensor(self.cfg.l2f_rot_dir, dtype=torch.int32, device=self._device)
        pegasus_dirs = torch.empty_like(l2f_dirs)
        pegasus_dirs[self._l2f_to_pegasus] = l2f_dirs
        thrusters._rot_dir = pegasus_dirs


