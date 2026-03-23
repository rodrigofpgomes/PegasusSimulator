"""
| File: rl_backend.py
| Description: Backend used to interface the multirotor simulator with
|              vectorized RL environments.
| License: BSD-3-Clause.
"""
__all__ = ["RLBackend"]

import torch
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.transforms import quaternion_to_matrix


class RLBackend(Backend):
    """
    Backend for RL training with vectorized multirotor environments.

    Callback order per physics step (VehicleBatch):

        PRE_PHYSICS_STEP  → MultirotorBatch.update(dt)
                                backend.update(dt)
                                applies forces via input_mode

        POST_PHYSICS_STEP → VehicleBatch.update_state(dt)
                                backend.update_state(s_{t+1})

    VecEnv flow:
        obs = get_state()               # read s_t
        a_t = policy(obs)
        env._apply_action()
            → backend.set_forces_and_torques()
        world.step()
            PRE:  backend.update()      # MultirotorBatch applies forces
            POST: backend.update_state(s_{t+1})
        obs = get_state()               # read s_{t+1}
    """

    def __init__(
        self,
        n_vehicles:  int,
        action_mode: str  = "direct_force",   # "direct_force" | "rotor_velocity"
        inner_loop:  bool = False,
    ):
        """
        Args:
            n_vehicles:  Number of parallel environments.
                         Required here so PegasusEnv.__init__ can read
                         backend.n_vehicles before start() is called.
            action_mode: "direct_force"   — env sends forces/torques directly.
                         "rotor_velocity" — backend converts to rotor speeds.
            inner_loop:  Only used in rotor_velocity mode.
                         If True, backend runs geometric attitude controller.
                         If False, policy provides thrust + torques directly.
        """
        super().__init__(config=None)

        # n_vehicles stored here for PegasusEnv to read before start()
        self._n_vehicles        = n_vehicles
        self._action_mode       = action_mode
        self._inner_loop        = inner_loop

        # Filled in start() once vehicle is initialised
        self._parts_per_vehicle = None
        self._device            = None

        self._forces      = None
        self._torques     = None
        self._input_ref   = None
        self._state_cache = None

        self._received_first_state = False

        # Attitude controller gains (rotor_velocity + inner_loop only)
        self._Kr = None
        self._Kw = None

    # ── properties (readable before start()) ─────────────────

    @property
    def n_vehicles(self) -> int:
        return self._n_vehicles

    @property
    def parts_per_vehicle(self):
        return self._parts_per_vehicle

    @property
    def device(self):
        return self._device

    # ── Backend interface ─────────────────────────────────────

    def initialize(self, vehicle):
        """Called by VehicleBatch.initialize() after spawn."""
        self._vehicle = vehicle

    def start(self):
        """
        Called when simulation starts (play), after MultirotorBatch.start()
        which calls initialize() where parts_per_vehicle is computed.
        """
        self._n_vehicles        = self.vehicle.n_vehicles
        self._parts_per_vehicle = self.vehicle.parts_per_vehicle
        self._device            = self.vehicle.device

        n, p, d = self._n_vehicles, self._parts_per_vehicle, self._device

        self._forces      = torch.zeros((n, p, 3), dtype=torch.float32, device=d)
        self._torques     = torch.zeros((n, p, 3), dtype=torch.float32, device=d)
        self._input_ref   = torch.zeros((n, 4),    dtype=torch.float32, device=d)
        self._state_cache = torch.zeros((n, 13),   dtype=torch.float32, device=d)

        self._received_first_state = False

        # Attitude controller gains
        self._Kr = torch.diag(torch.tensor([3.5, 3.5, 3.5], dtype=torch.float32, device=d))
        self._Kw = torch.diag(torch.tensor([0.5, 0.5, 0.5],  dtype=torch.float32, device=d))

        self.vehicle.set_input_mode(self._action_mode)

    def stop(self):
        """Called when simulation stops."""
        pass

    def reset(self):
        """Called on global reset — clears buffers."""
        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._input_ref.zero_()
            self._state_cache.zero_()
        self._received_first_state = False

    def update(self, dt: float):
        """
        Called by MultirotorBatch.update() in PRE_PHYSICS_STEP.

        direct_force:   MultirotorBatch reads get_forces_and_torques() directly.
        rotor_velocity: converts forces/torques to rotor angular velocities.
        """
        if not self._received_first_state:
            return

        if self._action_mode == "rotor_velocity":
            
            if self._inner_loop:
                
                # Get the desired forces
                F_des = self._forces[:, 0, :]

                # Get the current state
                q = self._state_cache[:, 6:10]
                w = self._state_cache[:, 10:13]
                R = quaternion_to_matrix(q)
                
                # Get the current axis Z_B (given by the last column of the rotation matrix)
                Z_B = R[:, :,2]

                # Compute the desired total thrust in Z_B direction (u_1)
                u_1 = torch.sum(F_des * Z_B, dim=1)

                # Compute the desired body-frame axis Z_b
                Z_b_des = F_des / torch.linalg.norm(F_des, dim=1, keepdim=True)

                # Desired yaw (fixed to zero)
                yaw_ref = torch.zeros((self._n_vehicles,), dtype=torch.float32, device=self._device)

                # Compute X_C_des 
                X_c_des = torch.stack((torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref),), dim=1)

                # Compute Y_b_des
                Z_b_cross_X_c = torch.cross(Z_b_des, X_c_des, dim=1)
                Y_b_des = Z_b_cross_X_c / torch.linalg.norm(Z_b_cross_X_c, dim=1, keepdim=True)

                # Compute X_b_des
                X_b_des = torch.cross(Y_b_des, Z_b_des, dim=1)

                # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
                R_des = torch.stack((X_b_des, Y_b_des, Z_b_des), dim=2)

                # Compute the rotation error
                e_R_mat = torch.matmul(R_des.transpose(1, 2), R) - torch.matmul(R.transpose(1, 2), R_des)
                e_R = 0.5 * self.vee_batch(e_R_mat)
            
                # desired angular velocity
                w_des = torch.zeros_like(w)
                
                # Compute the angular velocity error
                e_w = w - w_des

                # Compute the torques to apply on the rigid body
                tau = -(e_R @ self._Kr.T) - (e_w @ self._Kw.T)
            
            # RL directly provides thrust (u_1) and torques (tau)
            else:
                u_1 = self._forces[:, 0, 2]
                tau = self._torques[:, 0, :]

            # Use the allocation matrix provided by the Multirotor vehicle to convert the desired force and torque
            # to angular velocity [rad/s] references to give to each rotor
            self._input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau) # (n_envs, 4)

    def update_state(self, state: StateBatch):
        """
        Called by VehicleBatch.update_state() in POST_PHYSICS_STEP.
        Receives s_{t+1} — result of forces applied in PRE.
        """
        self._state_cache = torch.cat([
            state.position,           # [N, 3]
            state.linear_body_velocity,    # [N, 3]
            state.attitude,           # [N, 4]  wxyz
            state.angular_velocity,   # [N, 3]  body frame
        ], dim=-1)                    # → [N, 13]
        self._received_first_state = True

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self) -> torch.Tensor:
        """Used by MultirotorBatch in rotor_velocity mode."""
        return self._input_ref

    # ── VecEnv interface ──────────────────────────────────────

    def set_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """
        Called by env._apply_action() before world.step().
        Args:
            forces:  [N, n_bodies, 3]  Newtons
            torques: [N, n_bodies, 3]  Nm
        """
        self._forces  = forces
        self._torques = torques

    def set_state_for_envs(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_body_velocity: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        ):
        
        if self._state_cache is None or env_ids.numel() == 0:
            return

        if linear_body_velocity is None:
            linear_body_velocity = torch.zeros((env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype)
        if angular_velocity is None:
            angular_velocity = torch.zeros((env_ids.numel(), 3), device=self._device, dtype=self._state_cache.dtype)

        self._state_cache[env_ids, 0:3] = positions
        self._state_cache[env_ids, 3:6] = linear_body_velocity
        self._state_cache[env_ids, 6:10] = attitudes
        self._state_cache[env_ids, 10:13] = angular_velocity

        self._received_first_state = True

    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read by MultirotorBatch.update() in direct_force mode."""
        return self._forces, self._torques

    def get_state(self) -> torch.Tensor:
        """
        Returns s_{t+1} after world.step().
        Shape: [N, 13]  pos(3) + vel_world(3) + quat_wxyz(4) + omega_body(3)
        """
        return self._state_cache

    @staticmethod
    def vee_batch(S):
        return torch.stack((-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]), dim=1)