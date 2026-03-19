__all__ = ["RLBackend"]

import torch

from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state_batch import StateBatch


class RLBackend(Backend):

    def __init__(
        self,
        n_vehicles: int,
        device: str = "cuda",
        #max_rotor_vel: float = 1000.0,
        parts_per_vehicle: int = 5,
        action_mode: str = "direct_force",
    ):
        super().__init__()

        self.n_vehicles = n_vehicles
        self.device = device
        #self.max_rotor_vel = max_rotor_vel
        self.parts_per_vehicle = parts_per_vehicle
        self.action_mode = action_mode

        self._forces = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        self._torques = torch.zeros((self.n_vehicles, self.parts_per_vehicle, 3), dtype=torch.float32, device=self.device)

        self._state_cache = torch.zeros(n_vehicles, 13, device=device)

        self.received_first_state = False

    def update_state(self, state: StateBatch):

        self._state_cache = torch.cat([
                                state.position,
                                state.linear_velocity,
                                state.attitude,
                                state.angular_velocity,
                            ], dim=-1)

        self.received_first_state = True

    def update(self, dt: float):

        if not self.received_first_state:
            return

        if self.action_mode == "rotor_velocity":
            u_1 = self.forces[:, 0, 0]
            tau = self.torques[:, 0, :]

            # Use the allocation matrix provided by the Multirotor vehicle to convert the desired force and torque
            # to angular velocity [rad/s] references to give to each rotor
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau) # (n_envs, 4)

        if self.action_mode == "direct_force":
            self.input_actions = self._actions


    def input_mode(self) -> str:
        return self.action_mode

    def input_reference(self) -> torch.Tensor:
        return self.input_ref
    
    def input_actions(self) -> torch.Tensor:
        return self.input_actions
        

    def start(self):
        self._actions.zero_()
        self._state_cache.zero_()
        self.received_first_state = False

    def stop(self):
        pass

    def set_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor):
        """Recebe [N, n_bodies, 3] já formatado pelo env."""
        self._forces  = forces
        self._torques = torques

    def get_state(self) -> torch.Tensor:
        return self._state_cache