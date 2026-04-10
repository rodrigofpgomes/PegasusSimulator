"""
| File: multirotor_batch.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Adapted by: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the MultirotorBatch class, which serves as the base class for batched multirotor vehicles.
"""
import torch

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle_batch import VehicleBatch

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDragBatch
from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse
from pegasus.simulator.logic.thrusters import QuadraticThrustCurveBatch

# Default sensors can be enabled here if batch sensor support is required.
#from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Extension APIs
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class MultirotorBatchConfig:
    """
    Class used to configure a MultirotorBatch vehicle.
    """

    def __init__(self, n_vehicles=1):
        """
        Initialize the MultirotorBatch configuration.
        """
        # Define the same device that is running the simulation
        device = PegasusInterface()._world_settings["device"]

        # Stage prefix used when spawning the batched vehicles in the world
        self.stage_prefix = "quadrotor"

        # The USD file that describes the visual appearance of the vehicle, as well as properties such as mass and inertia
        self.usd_file = ""

        # Default thrust curve and drag model for the batched quadrotors
        self.thrust_curve = QuadraticThrustCurveBatch(n_vehicles=n_vehicles, device = device)
        self.drag = LinearDragBatch(n_vehicles=n_vehicles, drag_coefficients=[0.50, 0.30, 0.0])

        # Default onboard sensors for the batched quadrotors.
        # These are currently disabled until batch sensor support is enabled.
        #self.sensors = [Barometer(device=device), IMU(device=device), Magnetometer(device=device), GPS(device=device)]
        self.sensors = []

        # Default graphical sensors for the batched quadrotors
        self.graphical_sensors = []

        # Default OmniGraph graphs for the batched quadrotors
        self.graphs = []

        # Backends used to send commands to the batched vehicles.
        # This can be a PX4, ROS 2, or custom backend implementation, or an empty list if no backend is required.
        self.backends = []


class MultirotorBatch(VehicleBatch):
    """MultirotorBatch class - It defines a base interface for batched multirotor vehicles."""
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "quadrotor",
        usd_file: str = "",
        vehicle_batch_id: int = 0,
        n_vehicles: int = 1,
        # Spawning pose of the vehicle
        init_pos=None,
        init_orientation=None,
        spacing: float = 3.0,
        config=None,
    ):
        """Initialize the batched multirotor object.

        Args:
            stage_prefix (str): Base name used when spawning the batched vehicles in the simulator. Defaults to "quadrotor".
            usd_file (str): USD file describing the vehicle appearance and physical properties. Defaults to "".
            vehicle_batch_id (int): The id to be used for the vehicle batch. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qw, qx, qy, qz]. Defaults to [1.0, 0.0, 0.0, 0.0].
            config (MultirotorBatchConfig, optional): Defaults to MultirotorBatchConfig().
        """

        if config is None:
            config = MultirotorBatchConfig(n_vehicles=n_vehicles)

        # 1. Initialize the VehicleBatch base class
        super().__init__(stage_prefix, usd_file, n_vehicles, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends, spacing)

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag

        self._input_mode = None
        
    
    def _cache_rotor_positions_body(self):
        """
        Compute and cache rotor positions relative to the vehicle body frame.

        This only needs to be done once because the rotor locations are fixed
        with respect to the body.
        """

        pos, quat = self._vehicle_prims.get_world_poses()

        # Reshape from (n_vehicles * parts_per_vehicle, 3) to (n_vehicles, parts_per_vehicle, 3)
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self._device).reshape(self._n_vehicles, self._parts_per_vehicle, 3)
        quat = torch.as_tensor(quat, dtype=torch.float32, device=self._device).reshape(self._n_vehicles, self._parts_per_vehicle, 4)
        

        # Extract body and rotor poses
        body_pos = pos[:, self._body_index, :]       # (n_vehicles, 3)
        body_quat = quat[:, self._body_index, :]     # (n_vehicles, 4)
        rotor_pos = pos[:, self._body_index + 1:, :]     # (n_vehicles, num_rotors, 3)

        # Compute rotor positions relative to the body in the world frame
        relative_pos_world = rotor_pos - body_pos.unsqueeze(1)

        # Expand body quaternion so that one quaternion is associated with each rotor
        body_quat_expanded = body_quat.unsqueeze(1).expand(-1, self._thrusters._num_rotors, -1)

        # Rotate relative positions into the body frame
        self._rotor_positions_body = quat_rotate_inverse(body_quat_expanded.reshape(-1, 4), relative_pos_world.reshape(-1, 3)).view(self._n_vehicles, self._thrusters._num_rotors, 3)


    def _cache_allocation_matrix(self):
        """
        Build and cache the control allocation matrix and its pseudo-inverse.

        The matrix maps squared rotor angular velocities to total thrust and body
        torques. Since the vehicle geometry and rotor coefficients are fixed,
        this matrix can be computed once during initialization.
        """

        # Use the first vehicle as reference since all vehicles share the same geometry
        rotor_pos_body = self._rotor_positions_body[0]   # (num_rotors, 3)

        x = rotor_pos_body[:, 0]
        y = rotor_pos_body[:, 1]

        kf = self._thrusters._rotor_constant
        km = self._thrusters._rolling_moment_coefficient
        rot_dir = self._thrusters._rot_dir.to(torch.float32)

        self._allocation_matrix = torch.zeros((4, self._thrusters._num_rotors), dtype=torch.float32, device=self._device)

        # Total thrust contribution
        self._allocation_matrix[0, :] = kf

        # Roll torque contribution: tau_x = y * Fz
        self._allocation_matrix[1, :] = y * kf

        # Pitch torque contribution: tau_y = -x * Fz
        self._allocation_matrix[2, :] = -x * kf

        # Yaw torque contribution from rotor drag
        self._allocation_matrix[3, :] = km * rot_dir

        # Precompute pseudo-inverse for fast control allocation
        self._allocation_inv = torch.linalg.pinv(self._allocation_matrix)

    def start(self):
        """
        Precompute and cache quantities required for multirotor dynamics,
        such as rotor positions relative to the body frame and the control
        allocation matrix. This method is called when the simulation starts.
        """

        # 1. Initialize the vehicle primitives and internal state
        self.initialize()

        # 2. Cache the rotor positions relative to the vehicle body frame.
        #    These are used when computing the forces applied in the simulator.
        self._cache_rotor_positions_body()

        # 3. Cache the control allocation matrix that maps desired body-frame
        #    forces and torques to rotor angular velocities.
        self._cache_allocation_matrix()

    def stop(self):
        """
        No extra actions are required when the simulation stops.
        """
        pass

    def update(self, dt: float):
        """
        Compute and apply the batched forces and torques to the vehicles in simulation.

        Depending on the selected input mode, the commands are interpreted either as
        rotor angular velocities or as body-frame forces and torques. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if self._sim_running == False:
            return

        # Call the update methods in all backends
        for backend in self._backends:
            backend._vehicle = self
            backend.update(dt)

        # TODO: Add batched propeller visual updates for rotor animation.

        # Rotor angular velocity mode
        if self._input_mode == "rotor_velocity":

            if len(self._backends) != 0:
                desired_rotor_velocities = self._backends[0].input_reference()
            else:
                desired_rotor_velocities = torch.zeros((self._n_vehicles, self._thrusters._num_rotors), dtype=torch.float32, device=self._device)

            # Input the desired rotor velocities in the thruster model
            self._thrusters.set_input_reference(desired_rotor_velocities)

            forces = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)
            torques = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self._device)

            # Compute the rotor thrust forces and the desired rolling_moment
            forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

            # Apply the force in Z to each rotor in the rotor frame
            # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
            forces[:, 1:, 2] = forces_z
            torques[:, 0, 2] = rolling_moment

            # Compute the total linear drag force to apply to the vehicle's body frame
            # drag = self._drag.update(self._state, dt)

            # forces[:, 0, :] += drag
        
        # Direct body force and torque mode
        else:
            forces, torques = self._backends[0].get_forces_and_torques()

            # Optional: add linear drag on the vehicle body.
            # drag = self._drag.update(self._state, dt)
            # forces[:, 0, :] += drag


        # Apply the batched forces and torques in the simulator.
        self.apply_forces_and_torques_all_parts(forces, torques)


    def force_and_torques_to_velocities(self, force: torch.Tensor, torque: torch.Tensor):
        """
        Auxiliary method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (torch.Tensor): Desired total thrust along the body Z axis, shape (n_vehicles,)
            torque (torch.Tensor): A vector of the torque to be applied in the body frame of each vehicle [Nm], shape (n_vehicles, 3).

        Returns:
            torch.Tensor: Target rotor angular velocities, shape (n_vehicles, num_rotors).
        """

        # Build desired wrench vector [T, tau_x, tau_y, tau_z]
        wrench = torch.cat((force.unsqueeze(1), torque), dim=1)   # (n_vehicles, 4)

        # Solve for squared angular velocities
        squared_ang_vel = wrench @ self._allocation_inv.T         # (n_vehicles, num_rotors)

        # Clamp negative values caused by the pseudo-inverse
        squared_ang_vel = torch.clamp(squared_ang_vel, min=0.0)

        # Saturate while preserving the relative distribution between rotors
        max_thrust_vel_squared = torch.pow(self._thrusters.max_rotor_velocity[0], 2)
        max_val = torch.max(squared_ang_vel, dim=1, keepdim=True).values

        normalize = torch.clamp(max_val / max_thrust_vel_squared, min=1.0)
        squared_ang_vel = squared_ang_vel / normalize

        # Convert to angular velocity
        ang_vel = torch.sqrt(squared_ang_vel)

        return ang_vel
 
    
    def set_input_mode(self, input_mode: str):
        """
        Set the control input mode used by the batched multirotor.

        Args:
            input_mode (str): Control input mode. Expected values are "rotor_velocity" or "forces_torques".
        """
        self._input_mode = input_mode

