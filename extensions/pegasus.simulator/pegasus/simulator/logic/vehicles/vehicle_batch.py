"""
| File: vehicle_batch.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Adapted by: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the VehicleBatch class, adapted to support batched simulation of multiple vehicles.
"""

# Standard library imports
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

# Low level APIs
from pxr import UsdGeom, Gf, Usd, UsdPhysics

# High level Isaac sim APIs
from omni.usd import get_stage_next_free_path
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import RigidPrim, XFormPrim
from isaacsim.core.simulation_manager import SimulationManager, IsaacEvents
from isaacsim.core.cloner import GridCloner

# Extension APIs
from pegasus.simulator.logic.state_batch import StateBatch
from pegasus.simulator.logic.vehicle_manager import VehicleManager
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert

# Debug purposes
import carb


class VehicleBatch():
    """
    Base class for handling batched simulation of multiple vehicles.
    """
    def __init__(
        self,
        stage_prefix: str,
        vehicle_batch_id: int,
        usd_path: str,
        n_vehicles: int = 1,
        init_pos: torch.Tensor | None = None,
        init_orientation: torch.Tensor | None = None,
        sensors: Sequence[Any] | None = None,
        graphical_sensors: Sequence[Any] | None = None,
        graphs: Sequence[Any] | None = None,
        backends: Sequence[Any] | None = None,
        spacing: float = 3.0
    )-> None:
        """
        Initialize a batch of vehicles in the current Isaac Sim stage.

        Args:
            stage_prefix (str): Base path used when spawning the vehicle instances.
            usd_path (str): Path to the USD file that defines the vehicle model.
            n_vehicles (int): Number of vehicles to spawn in the batch.
            init_pos (torch.Tensor | None, optional): Initial positions of the vehicles in the inertial frame using the ENU convention. If None, vehicles are spawned using a grid layout.
            init_orientation (torch.Tensor | None, optional): Initial vehicle orientations as quaternions in the format [qw, qx, qy, qz].
            sensors (list, optional): List of physics-based sensors attached to the vehicles.
            graphical_sensors (list, optional): List of graphical sensors attached to the vehicles.
            graphs (list, optional): List of graphs associated with the vehicles.
            backends (list, optional): List of communication or control backends.
            spacing (float): Distance between vehicles when using automatic grid spawning.
        """        
        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")
        if spacing <= 0.0:
            raise ValueError("spacing must be greater than zero")

        # Define the same device that is running the simulation
        self._device = PegasusInterface()._world_settings["device"]

        # Get the current world in which the batched vehicles will be spawned
        self._world = PegasusInterface().world
        self._stage = self._world.stage
            
        # Save the base stage prefix for the batch and the USD file that defines the vehicle model
        base_stage_prefix = stage_prefix.rstrip("/")

        self._vehicle_batch_id = vehicle_batch_id
        self._batch_root = get_stage_next_free_path(self._stage, f"{base_stage_prefix}_batch_{vehicle_batch_id}", False)
        self._stage_prefix = f"{self._batch_root}/env"
        
        self._usd_file = usd_path
        self._n_vehicles = n_vehicles
        self._vehicle_name = self._stage_prefix.rpartition("/")[-1]

        # Spawn the batch of vehicles in the world stage
        self._spawn_batch(init_pos, init_orientation, spacing)

        self._parts_per_vehicle: int | None = None
        self._body_index: int | None = None

        # Create a batched view over all rigid prims belonging to the vehicles
        self._vehicle_expr = f"{self._stage_prefix}.*/.*"
        self._vehicle_prims = RigidPrim(prim_paths_expr=self._vehicle_expr, name=f"{self._stage_prefix}_prims")
        self._root_prims = RigidPrim(prim_paths_expr = f"{self._stage_prefix}.*/body", name = f"{self._vehicle_name}_roots")

        # Variable that stores the current batched state of the vehicles
        self._state = StateBatch(self.n_vehicles, self.device)

        self._last_forces_local: torch.Tensor | None = None
        self._last_torques_local: torch.Tensor | None = None

        # Register callback executed before each physics step. 
        # This method should be implemented in classes that inherit the vehicle object.
        self._cb_pre = SimulationManager.register_callback(self.update, event=IsaacEvents.PRE_PHYSICS_STEP, order=0)

        # Register the callback executed after each physics step to update the batched vehicle state
        self._cb_post = SimulationManager.register_callback(self.update_state, event=IsaacEvents.POST_PHYSICS_STEP, order=0)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        # Set variable that control if the batch is already closed
        self._closed = False

        # Add a callback to start/stop of the simulation once the play/stop button is hit
        self._world.add_timeline_callback(self._stage_prefix + "/start_stop_sim", self.sim_start_stop)

        # --------------------------------------------------------------------
        # -------------------- Add sensors to the vehicle --------------------
        # --------------------------------------------------------------------
        #self._sensors = tuple(sensors or ())
        
        #for sensor in self._sensors:
        #    sensor.initialize(
        #        self, 
        #        torch.tensor(PegasusInterface().latitude, dtype=torch.float32, device=self.device), 
        #        torch.tensor(PegasusInterface().longitude, dtype=torch.float32, device=self.device), 
        #        torch.tensor(PegasusInterface().altitude, dtype=torch.float32, device=self.device)
        #    )

        # Add callbacks to the physics engine to update each sensor at every timestep
        # and let the sensor decide depending on its internal update rate whether to generate new data
        #self._world.add_physics_callback(self._stage_prefix + "/Sensors", self.update_sensors)

        # --------------------------------------------------------------------
        # -------------------- Add the graphical sensors to the vehicle ------
        # --------------------------------------------------------------------
        #self._graphical_sensors = tuple(graphical_sensors or ())

        #for graphical_sensor in self._graphical_sensors:
        #    graphical_sensor.initialize(self)

        # Add callbacks to the rendering engine to update each graphical sensor at every timestep of the rendering engine
        #self._world.add_render_callback(self._stage_prefix + "/GraphicalSensors", self.update_graphical_sensors)


        # --------------------------------------------------------------------
        # -------------------- Add the graphs to the vehicle -----------------
        # --------------------------------------------------------------------
        #self._graphs = tuple(graphs or ())

        #for graph in self._graphs:
        #    graph.initialize(self)
        
        # --------------------------------------------------------------------
        # ---- Add (communication/control) backends to the vehicle -----------
        # --------------------------------------------------------------------
        self._backends = tuple(backends or ())


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the vehicle prim handles and allocate batched state tensors.
        """
        self._vehicle_prims.initialize()
        self._root_prims.initialize()

        # capture _init_pos and _init_orientation from the /body prim after world.reset() so it matches state.position (which also comes from _root_prims)
        init_pos, init_orientation = self._root_prims.get_world_poses()
        self._init_pos = torch.as_tensor(init_pos, dtype=torch.float32, device=self.device)
        self._init_orientation = torch.as_tensor(init_orientation, dtype=torch.float32, device=self.device)

        if self._vehicle_prims.count % self._n_vehicles != 0:
            raise RuntimeError(f"Rigid prim count ({self._vehicle_prims.count}) is not divisible by " f"n_vehicles ({self._n_vehicles})")

        self._parts_per_vehicle = self._vehicle_prims.count // self._n_vehicles

        self._last_forces_local = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self.device)
        self._last_torques_local = torch.zeros((self._n_vehicles, self._parts_per_vehicle, 3), dtype=torch.float32, device=self.device)

        print(f"Spawned {self._n_vehicles} vehicles with {self._parts_per_vehicle} parts each (total {self._vehicle_prims.count} prims)")

        self._body_index = next((index for index, path in enumerate(self._vehicle_prims.prim_paths[:self._parts_per_vehicle]) if path.endswith("/body")), None)

        if self._body_index is None:
            raise RuntimeError(f"No '/body' prim found in the first vehicle under {self._stage_prefix}")
        
        self._allocate_batch_state()

        #Initialize the backends
        for backend in self._backends:
            backend.initialize(self)


    def close(self) -> None:
        """
        Release callbacks and unregister this batch.
        """
        if self._closed:
            return

        SimulationManager.deregister_callback(self._cb_pre)
        SimulationManager.deregister_callback(self._cb_post)
        VehicleManager.get_vehicle_manager().remove_vehicle(self._stage_prefix)

        self._closed = True
    

    def __del__(self) -> None:
        """
        Method called when the vehicle batch object is destroyed.
        When this happens, the batch is also removed from the VehicleManager.
        """
        try:
            self.close()
        except Exception:
            pass


    def sim_start_stop(self, event) -> None:
        """
        Callback called whenever a timeline event occurs, such as starting or stopping the simulation.

        Args:
            event: A timeline event generated from Isaac Sim, such as starting or stoping the simulation.
        """

        # If the start/stop button was pressed, then call the start and stop methods accordingly
        if self._world.is_playing() and self._sim_running == False:
            # Initialize the sensors
            #for sensor in self._sensors:
            #    sensor.start()

            # Initialize the graphical sensors
            #for graphical_sensor in self._graphical_sensors:
            #    graphical_sensor.start()

            carb.log_info(f"Simulation started for vehicle batch {self.vehicle_name}.")

            # Invoke the start method of the vehicle (if it exists)
            self.start()

            # Intializes communication with all backends. This method is invoked automatically when the simulation starts
            for backend in self._backends:
                backend.start()

            self._sim_running = True

        if self._world.is_stopped() and self._sim_running == True:
            self._sim_running = False

            # Stop the sensors
            #for sensor in self._sensors:
            #    sensor.stop()

            # Stop the graphical sensors
            #for graphical_sensor in self._graphical_sensors:
            #    graphical_sensor.stop()

            # Signal all backends that the simulation has stopped. This method is invoked automatically when the simulation stops
            for backend in self._backends:
                backend.stop()

            self.stop()

    # ------------------------------------------------------------------
    # Spawn and state allocation
    # ------------------------------------------------------------------

    def _spawn_batch(
        self,
        init_pos: torch.Tensor | None,
        init_orientation: torch.Tensor | None,
        spacing: float,
    ) -> None:
        """
        This method spawns a batch of vehicles in the simulation stage.

        If explicit initial positions are provided, each vehicle is spawned individually at the specified position and orientation.
        Otherwise, a GridCloner is used to spawn the vehicles automatically in a grid formation with the given spacing.

        Args:
            spacing (float): Distance between vehicles when using the grid spawn mode.
        """

        # If explicit initial positions were provided, spawn each vehicle manually
        if init_pos is not None:

            for i in range(self._n_vehicles):
                
                # Create the prim path for this vehicle instance
                prim_path = f"{self._stage_prefix}_{i}"

                # Add the USD reference of the vehicle to the stage
                stage_utils.add_reference_to_stage(self._usd_file, prim_path=prim_path)

                # Get the prim and create XForm interfaces to manipulate its transform
                prim = self._stage.GetPrimAtPath(prim_path)
                xformable = UsdGeom.Xformable(prim)
                xform = UsdGeom.XformCommonAPI(xformable)

                # Set the initial position of the vehicle in world coordinates
                p = init_pos[i]
                xform.SetTranslate(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

                # If initial orientations were provided, apply them using a quaternion
                if init_orientation is not None:
                    q = init_orientation[i]   # [w, x, y, z]
                    quat = Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])).GetNormalized()

                    # Check if the prim already has an orient transform op
                    orient_attr = prim.GetAttribute("xformOp:orient")
                    
                    if orient_attr.IsValid():
                        orient_attr.Set(quat)
                    else:
                        orient_op = xformable.AddOrientOp()
                        orient_op.Set(quat)
        
        # If no explicit positions were given, spawn vehicles using GridCloner
        else:

            # Spawn the base vehicle that will be used as the cloning source
            prim_path = f"{self._stage_prefix}_0"
            stage_utils.add_reference_to_stage(self._usd_file, prim_path=prim_path)

            # Create a grid cloner that will distribute vehicles with the given spacing
            cloner = GridCloner(spacing=spacing)
            
            # Generate the prim paths for all vehicle instances
            target_paths = cloner.generate_paths(self._stage_prefix, self.n_vehicles)

            # Clone the base vehicle to the generated paths
            cloner.clone(source_prim_path=f"{self._stage_prefix}_0", prim_paths=target_paths, replicate_physics=True, copy_from_source=True, base_env_path=self._batch_root, root_path=f"{self._stage_prefix}_", enable_env_ids=True)

        # Create a view over the root prim of each vehicle
        vehicles = XFormPrim(prim_paths_expr=f"{self._stage_prefix}.*/")

        # Retrieve the world poses of the spawned vehicles
        set_init_pos, set_init_orientation = vehicles.get_world_poses()

        if init_pos is None:
            set_init_pos[:, 2] = 1.0

        vehicles.set_world_poses(set_init_pos, set_init_orientation)

        # Store them as tensors for later use
        self._init_pos = torch.as_tensor(set_init_pos, dtype=torch.float32, device=self.device).clone()
        self._init_orientation = torch.as_tensor(set_init_orientation, dtype=torch.float32, device=self.device).clone()

        #print(f"Initialized {self.n_vehicles} vehicles at positions: {self.init_pos} and orientations: {self.init_orientation}")


    def disable_collisions(self) -> None:
        """Disable collisions on every collider prim of this vehicle batch.

        Walks the batch's USD subtree and sets collisionEnabled=False on every
        prim that exposes the UsdPhysics CollisionAPI. Useful when several
        vehicle batches are spawned overlapping in space.
        """
        root = self._stage.GetPrimAtPath(self._batch_root)
        if not root or not root.IsValid():
            carb.log_warn(f"[VehicleBatch] disable_collisions: root '{self._batch_root}' not found.")
            return
        count = 0
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                attr = api.GetCollisionEnabledAttr()
                if not attr:
                    attr = api.CreateCollisionEnabledAttr()
                attr.Set(False)
                count += 1
        carb.log_warn(f"[VehicleBatch] Disabled collisions on {count} prim(s) under {self._batch_root}.")


    def _allocate_batch_state(self):
        """
        Allocate and initialize the batched state tensors for all vehicles.
        """
        n = self.n_vehicles
        zeros3 = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        zeros4 = torch.zeros((n, 4), dtype=torch.float32, device=self.device)

        self._state.position = zeros3.clone()
        self._state.attitude = zeros4.clone()
        self._state.attitude[:, 0] = 1.0
        self._state.linear_velocity = zeros3.clone()
        self._state.linear_body_velocity = zeros3.clone()
        self._state.angular_velocity = zeros3.clone()
        self._state.linear_acceleration = zeros3.clone()

    # ------------------------------------------------------------------
    # Physics and state
    # ------------------------------------------------------------------
    
    def apply_forces_and_torques_all_parts(self, forces: torch.Tensor, torques: torch.Tensor) -> None:
        """
        Method that apply forces and torques to all rigid parts matched by the batch view.

        Args:
            forces: Tensor of shape (n_vehicles, n_parts_per_vehicle, 3) or flattened shape (n_total_parts, 3).
            torques: Tensor of shape (n_vehicles, n_parts_per_vehicle, 3) or flattened shape (n_total_parts, 3).
        """

        forces = forces.reshape((self._vehicle_prims.count, 3))
        
        torques = torques.reshape((self._vehicle_prims.count, 3))

        #print("Applying forces:", forces)
        #print("Applying torques:", torques)

        if self._parts_per_vehicle is not None:
            self._last_forces_local = forces.reshape(self._n_vehicles, self._parts_per_vehicle, 3).detach().clone()
            self._last_torques_local = torques.reshape(self._n_vehicles, self._parts_per_vehicle, 3).detach().clone()

        # Apply the force to the rigidbody. The force should be expressed in the rigidbody frame (is_global=False)
        # Note that the mapping between tensor rows and prim paths follows self._vehicle_prims.prim_paths order
        self._vehicle_prims.apply_forces_and_torques_at_pos(forces, torques, is_global=False)


    def update_state(self, dt: float) -> None:
        """
        Callback called at every physics step to retrieve and update the current batched vehicle state.

        The state of each vehicle is defined with respect to its body prim.
        """
        
        if self._sim_running == False:
            return
                
        # Get the current position of the body in the inertial frame and its orientation relative to the inertial frame
        positions, orientations = self._root_prims.get_world_poses()

        # The linear velocity [x_dot, y_dot, z_dot] of the vehicle's body frame expressed in the inertial frame of reference
        linear_vel = self._root_prims.get_linear_velocities()

        # Get the angular velocity of the vehicle expressed in the body frame of reference
        angular_vel = self._root_prims.get_angular_velocities()

        # Get the linear acceleration of the body relative to the inertial frame, expressed in the inertial frame
        # Note: we must do this approximation, since the Isaac sim does not output the acceleration of the rigid body directly
        if dt > 0.0:
            linear_acceleration = (torch.as_tensor(linear_vel, dtype=torch.float32, device=self._device) - self._state.linear_velocity) / dt
        else:
            linear_acceleration = torch.zeros_like(torch.as_tensor(linear_vel, dtype=torch.float32, device=self._device))

        # Update the state — use .clone() so state tensors own their storage and
        # are never dangling views into PhysX buffers that get freed on stage reload.
        self._state.position = torch.as_tensor(positions, dtype=torch.float32, device=self._device).clone()
        self._state.attitude = torch.as_tensor(orientations, dtype=torch.float32, device=self._device).clone()

        # Express the velocity of the vehicle in the inertial frame X_dot = [x_dot, y_dot, z_dot]
        self._state.linear_velocity = torch.as_tensor(linear_vel, dtype=torch.float32, device=self._device).clone()

        # The linear velocity V =[u,v,w] of the vehicle's body frame expressed in the body frame of reference
        # Note that: x_dot = Rot * V
        self._state.linear_body_velocity = quaternion_apply(quaternion_invert(self._state.attitude), self._state.linear_velocity)

        # omega = [p,q,r], expressed in the body frame of reference
        self._state.angular_velocity = quaternion_apply(quaternion_invert(self._state.attitude), torch.as_tensor(angular_vel, dtype=torch.float32, device=self._device).clone())

        # The acceleration of the vehicle expressed in the inertial frame X_ddot = [x_ddot, y_ddot, z_ddot]
        self._state.linear_acceleration = linear_acceleration

        for backend in self._backends:
            backend.update_state(self._state)


    def set_state_batch(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        attitudes: torch.Tensor,
        linear_velocity: torch.Tensor | None = None, # expressed in the inertial frame
        angular_velocity: torch.Tensor | None = None,   # expressed in the body frame
    ) -> None:
        """
        Synchronize the cached vehicle state after an external reset.
        """

        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if linear_velocity is None:
            linear_velocity = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=torch.float32)

        if angular_velocity is None:
            angular_velocity = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=torch.float32)

        positions = positions.to(device=self.device, dtype=torch.float32)
        attitudes = attitudes.to(device=self.device, dtype=torch.float32)
        linear_velocity = linear_velocity.to(device=self.device, dtype=torch.float32)
        angular_velocity = angular_velocity.to(device=self.device, dtype=torch.float32)

        self._state.position[env_ids] = positions
        self._state.attitude[env_ids] = attitudes

        self._state.linear_velocity[env_ids] = linear_velocity

        self._state.linear_body_velocity[env_ids] = quaternion_apply(quaternion_invert(attitudes), linear_velocity)

        self._state.angular_velocity[env_ids] = angular_velocity

        self._state.linear_acceleration[env_ids] = 0.0

        for backend in self._backends:
            backend.set_state(
                env_ids=env_ids,
                positions=positions,
                attitudes=attitudes,
                linear_velocity=self._state.linear_velocity[env_ids],
                angular_velocity=self._state.angular_velocity[env_ids],
            )

    
    # ------------------------------------------------------------------
    # Optional legacy sensor forwarding
    # ------------------------------------------------------------------

    def update_sensors(self, dt: float):
        """Callback that is called at every physics steps and will call the sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_sensor method will also be called for
        every backend. For example, if new data is generated for an IMU and we have a PX4MavlinkBackend, then the update_sensor
        method will be called for that backend so that this data can latter be sent thorugh mavlink.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._sensors:
            sensor_data = sensor.update(self._state, dt)

            # If some data was updated and we have a mavlink backend or ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_sensor(sensor.sensor_type, sensor_data)


    def update_graphical_sensors(self, event):
        """Callback that is called at every rendering steps and will call the graphical_sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_graphical_sensor method will also be called for
        every backend. For example, if new data is generated for a monocular camera and we have a ROS2Backend, then the update_graphical_sensor
        method will be called for that backend so that this data can latter be sent through a ROS2 topic.

        Args:
            event (float): The timer event that contains the time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._graphical_sensors:
            sensor_data = sensor.update(self._state, event.payload['dt'])

            # If some data was updated and we have a ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_graphical_sensor(sensor.sensor_type, sensor_data)


    # ------------------------------------------------------------------
    # Helpers and API
    # ------------------------------------------------------------------

    def get_batch_layout_info(self) -> dict[str, Any]:
        """
        Return structural information about the current batch view.
        """
        return {
            "vehicle_prefix": self._stage_prefix,
            "n_vehicles": self._n_vehicles,
            "n_parts_per_vehicle": self._parts_per_vehicle,
            "n_total_prims": self._vehicle_prims.count,
            "prim_paths": list(self._vehicle_prims.prim_paths),
        }

    @property
    def state(self):
        """The batched state of the vehicles.

        Returns:
            StateBatch: The current state of all vehicles in the batch.
        """
        return self._state
    
    @property
    def vehicle_name(self) -> str:
        """Vehicle batch name.

        Returns:
            str: The last component of the vehicle batch prim path.
        """
        return self._stage_prefix.rpartition("/")[-1]

    @property
    def n_vehicles(self):
        """Number of vehicles in the batch."""
        return self._n_vehicles

    @property
    def parts_per_vehicle(self):
        """Number of parts per vehicle in the batch."""
        return self._parts_per_vehicle

    @property
    def device(self):
        """The device on which the state tensors are allocated."""
        return self._device

    @property
    def body_index(self):
        """The index of the body prim for each vehicle in the batch."""
        return self._body_index

    @abstractmethod
    def start(self) -> None:
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Method that computes and applies the forces to the vehicle insimulation. 
        This method must be implemented by a class that inherits this type and it's called periodically by the physics engine.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        pass