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
    Backend used for RL training with vectorized multirotor environments.

    Callback order per physics step (VehicleBatch):

        world.step()  (PRE_PHYSICS_STEP)
            └── update(dt)           
                MultirotorBatch calls backend.update(), then applies forces 
                according to the selected input_mode.

        world.step()  (POST_PHYSICS_STEP)
            └── update_state(dt)     
                VehicleBatch reads PhysX state and calls
                backend.update_state(s_{t+1}).

    VecEnv execution flow:

        obs = get_state()                  # read s_t from cache
        a_t = policy(obs)                  # shape [N, 4]
        env._pre_physics_step(a_t)         # store actions
        env._apply_action()
            → backend.set_forces_and_torques()   # env formats and writes commands
            → vehicle.set_forces_and_torques()   # backend passes commands to vehicle
        world.step()
            PRE:  update(dt)               # MultirotorBatch applies inputs
            POST: update_state(s_{t+1})    # cache updated to s_{t+1}
        obs = get_state()                  # read next state
    """

    def __init__(self, action_mode: str = "direct_force", inner_loop: bool = True):
        """
            Args:
            action_mode (str): Input mode used by the vehicle.
            inner_loop (bool): 
                If True, the backend computes attitude control (torques) from a desired force.
                If False, the RL policy is expected to directly provide thrust and torques.
        """

        super().__init__(config=None)

        self._action_mode = action_mode

        # buffers — inicializados em start() após conhecer o veículo
        self._forces = None
        self._torques = None
        self._state_cache = None
        self._input_ref = None

        self._n_vehicles = None
        self._parts_per_vehicle = None
        self._device = None

        self._received_first_state = False

        # Define the control gains matrix for the inner-loop (attitude) 
        self._Kr = None
        self._Kw = None

        # Define the dynamic parameters for the vehicle
        self._m = 1.50        # Mass in Kg
        self._g = 9.81       # The gravity acceleration ms^-2

        self._inner_loop = inner_loop


    def start(self):
        """
        Called when the simulation starts (play).

        At this point, self.vehicle is already initialized and parts_per_vehicle has been computed by MultirotorBatch.start().

        Initializes buffers and controller gains.
        """

        self._n_vehicles = self.vehicle.n_vehicles
        self._parts_per_vehicle = self.vehicle.parts_per_vehicle
        self._device = self.vehicle.device

        self._forces = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        self._torques = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
        
        self._state_cache = torch.zeros((self._n_vehicles, 13), dtype=torch.float32, device=self._device)

        self._input_ref = torch.zeros((self._n_vehicles, 4), dtype=torch.float32, device=self._device)

        self._received_first_state = False

        self._Kr = torch.diag(torch.tensor([3.5, 3.5, 3.5], dtype=torch.float32, device=self._device))
        self._Kw = torch.diag(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self._device))

        self.vehicle.set_input_mode(self._action_mode)


    def reset(self):
        """Chamado no reset global — limpa buffers."""

        if self._forces is not None:
            self._forces.zero_()
            self._torques.zero_()
            self._input_ref.zero_()
            self._state_cache.zero_()

        self._received_first_state = False



    def update(self, dt: float):
        """
        Called by MultirotorBatch.update() during PRE_PHYSICS_STEP,
        before forces are applied to the vehicle.

        Modes:

        - direct_force:
            Forces and torques are directly read from get_forces_and_torques()
            and applied to the vehicle.

        - rotor_velocity:
            Converts thrust and torques into rotor angular velocities using
            force_and_torques_to_velocities().

            Two operation modes are supported:

            1) inner_loop = True:
                The RL policy provides a desired force F_des in the world frame.
                The backend computes:
                    - total thrust (u_1)
                    - desired orientation (R_des)
                    - attitude control torques (tau)
                using a geometric controller.

            2) inner_loop = False:
                The RL policy directly provides:
                    - thrust (u_1)
                    - body torques (tau)
                The backend only performs allocation to rotor velocities.
        """

        if not self._received_first_state:
            return

        if self._action_mode == "rotor_velocity":
            
            if self._inner_loop = True:
                
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
        Updates the internal state cache with the latest simulator state.

        State layout:
            position          → indices 0:3
            linear velocity   → indices 3:6
            attitude (quat)   → indices 6:10
            angular velocity  → indices 10:13
        """

        self._state_cache = torch.cat([
                                state.position,
                                state.linear_velocity,
                                state.attitude,
                                state.angular_velocity,
                            ], dim=-1)

        self._received_first_state = True


    @staticmethod
    def vee_batch(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (torch.Tensor): A batch of matrices in so(3).
        """
        return torch.stack((-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]), dim=1)


    def set_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """
        Receives forces and torques from the environment.

        Args:
            forces:  shape (N, n_bodies, 3)
            torques: shape (N, n_bodies, 3)
        """

        self._forces  = forces
        self._torques = torques


    def get_forces_and_torques(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forces, self._torques


    def input_mode(self) -> str:
        return self._action_mode


    def input_reference(self) -> torch.Tensor:
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            torch.Tensor: shape (N, 4)
        """
        return self._input_ref
    
    
    def get_state(self) -> torch.Tensor:
        """
        Returns the cached state vector.

        Shape:
            (N, 13)
        """
        return self._state_cache




    """
    Properties
    """

    @property
    def device(self):
        return self._device

    @property
    def parts_per_vehicle(self):
        return self._parts_per_vehicle

    @property
    def n_vehicles(self):
        return self._n_vehicles