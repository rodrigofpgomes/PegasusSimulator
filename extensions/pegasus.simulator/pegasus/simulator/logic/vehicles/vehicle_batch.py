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
from pegasus.simulator.logic.transforms import quaternion_apply, quaternion_invert, quaternion_raw_multiply

# Debug purposes
import carb


class VehicleBatch():
    """
    Base class for vehicles that share one USD layout across a simulation batch.

    The root rigid prim of each vehicle is named ``body``. The rigid prim view
    must list the same parts in the same order for every vehicle because force
    and torque buffers use the shape (n_vehicles, parts_per_vehicle, 3).
    Subclasses implement the vehicle-specific physics in ``update``.
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
            vehicle_batch_id (int): Identifier used to name this batch in the stage.
            usd_path (str): Path to the USD file that defines the vehicle model.
            n_vehicles (int): Number of vehicles to spawn in the batch.
            init_pos (torch.Tensor | None, optional): ENU positions with shape (n_vehicles, 3). If None, vehicles are spawned on a grid.
            init_orientation (torch.Tensor | None, optional): Quaternions [qw, qx, qy, qz] with shape (n_vehicles, 4), used with explicit positions.
            sensors (Sequence | None, optional): Batched sensor suites consumed by backends through update_sensors().
            graphical_sensors (Sequence | None, optional): Reserved for future batch graphical sensor support.
            graphs (Sequence | None, optional): Reserved for future batch graph support.
            backends (Sequence | None, optional): Control or communication backends.
            spacing (float): Distance between vehicles when using automatic grid spawning.
        """        
        if n_vehicles <= 0:
            raise ValueError("n_vehicles must be greater than zero")
        if spacing <= 0.0:
            raise ValueError("spacing must be greater than zero")

        if graphical_sensors or graphs:
            raise NotImplementedError("VehicleBatch has no graphical sensor/graph routing yet")
        self._backends = tuple(backends or ())
        # RLBackend.update_sensor() is deliberately empty. Only the batched
        # update_sensors() consumer activates sampling and its physics callback.
        self._sensor_backends = tuple(backend for backend in self._backends
                                      if callable(getattr(backend, "update_sensors", None)))
        self._sensors = tuple(sensors or ()) if self._sensor_backends else ()
        self._graphical_sensors = ()
        self._graphs = ()

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

        # Create views over every rigid part and over the /body root of each vehicle
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

        # Sensors read the freshly updated physical state. Training backends
        # have no batch sensor consumer and therefore register no extra callback.
        self._cb_sensors = None
        if self._sensors:
            self._cb_sensors = SimulationManager.register_callback(self.update_sensors, event=IsaacEvents.POST_PHYSICS_STEP, order=1)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        # Set variable that control if the batch is already closed
        self._closed = False

        # Add a callback to start/stop of the simulation once the play/stop button is hit
        self._world.add_timeline_callback(self._stage_prefix + "/start_stop_sim", self.sim_start_stop)

        # initialize() binds sensors and backends after rigid prim views are ready


    # =============================================================================
    # Lifecycle
    # =============================================================================

    def initialize(self) -> None:
        """
        Initialize rigid prim views and cache the common batch layout.

        The first vehicle defines part indices for the entire batch. Verify
        every other vehicle before reshaping prim data into per-vehicle tensors.
        """
        self._vehicle_prims.initialize()
        self._root_prims.initialize()

        # Read the /body pose after world.reset(); update_state() uses the same root view
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
        
        # Reject a different part order before reshaping can mix vehicle channels
        paths = self._vehicle_prims.prim_paths
        roots = self._root_prims.prim_paths
        layout = [p.rsplit("/", 1)[-1] for p in paths[:self._parts_per_vehicle]]
        for i, root in enumerate(roots):
            group = paths[i * self._parts_per_vehicle:(i + 1) * self._parts_per_vehicle]
            if [p.rsplit("/", 1)[-1] for p in group] != layout or any(p.rsplit("/", 1)[0] != root.rsplit("/", 1)[0] for p in group):
                raise RuntimeError("RigidPrim ordering does not match the batched body/part contract")
        # Cache each part pose in the /body frame for resets and trim calculations
        positions, attitudes = self._vehicle_prims.get_world_poses()
        positions = torch.as_tensor(positions, device=self.device).reshape(self.n_vehicles, self.parts_per_vehicle, 3)
        attitudes = torch.as_tensor(attitudes, device=self.device).reshape(self.n_vehicles, self.parts_per_vehicle, 4)
        inverse = quaternion_invert(self._init_orientation)[:, None, :].expand(-1, self.parts_per_vehicle, -1)
        self._reset_part_positions_body = quaternion_apply(inverse, positions - self._init_pos[:, None, :]).clone()
        self._reset_part_attitudes_body = quaternion_raw_multiply(inverse, attitudes).clone()
        coms, _ = self._vehicle_prims.get_coms()
        self._reset_part_coms = torch.as_tensor(coms, device=self.device).reshape(self.n_vehicles, self.parts_per_vehicle, 3).clone()
        self._allocate_batch_state()

        # Initialize sensor origins before communication backends bind the suite.
        origin = PegasusInterface()
        for suite in self._sensors:
            suite.initialize(self, origin.latitude, origin.longitude, origin.altitude)

        # Bind backends only after the vehicle layout and state have been initialized
        for backend in self._backends:
            backend.initialize(self)


    def close(self) -> None:
        """
        Release physics and timeline callbacks and unregister this batch.

        Repeated calls are safe because the destructor may also call close().
        """
        if self._closed:
            return

        SimulationManager.deregister_callback(self._cb_pre)
        SimulationManager.deregister_callback(self._cb_post)
        if self._cb_sensors is not None:
            SimulationManager.deregister_callback(self._cb_sensors)
        self._world.remove_timeline_callback(self._stage_prefix + "/start_stop_sim")
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
        Start the vehicle before its backends and stop the backends first.

        Args:
            event: Timeline event delivered by the simulation world.
        """

        # If the start/stop button was pressed, then call the start and stop methods accordingly
        if self._world.is_playing() and self._sim_running == False:
            carb.log_info(f"Simulation started for vehicle batch {self.vehicle_name}.")

            # Invoke the start method of the vehicle (if it exists)
            self.start()

            # Start backends after the vehicle has resolved its prims and buffers
            for backend in self._backends:
                backend.start()

            self._sim_running = True

        if self._world.is_stopped() and self._sim_running == True:
            self._sim_running = False

            # Signal all backends that the simulation has stopped. This method is invoked automatically when the simulation stops
            for backend in self._backends:
                backend.stop()

            self.stop()

    # =============================================================================
    # Spawn and state allocation
    # =============================================================================

    def _spawn_batch(
        self,
        init_pos: torch.Tensor | None,
        init_orientation: torch.Tensor | None,
        spacing: float,
    ) -> None:
        """
        Spawn vehicle USD instances in the simulation stage.

        Explicit positions create individual instances. Without positions,
        GridCloner creates a grid and the root prims are placed at z = 1 m.

        Args:
            init_pos (torch.Tensor | None): ENU spawn positions, one per vehicle.
            init_orientation (torch.Tensor | None): Spawn quaternions [w, x, y, z], used only with explicit positions.
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

    def disable_collisions(self) -> None:
        """Disable collisions on every collider prim of this vehicle batch.

        Traverses the USD subtree and sets collisionEnabled to false on each collider.
        This can be used when comparison batches overlap in the stage.
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
        Allocate the position, attitude, velocity and acceleration state buffers.
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

    # =============================================================================
    # Physics and state
    # =============================================================================
    
    def apply_forces_and_torques_all_parts(self, forces: torch.Tensor, torques: torch.Tensor) -> None:
        """
        Apply local-frame forces and torques to every rigid part in the batch.

        The flattened rows must follow ``_vehicle_prims.prim_paths``. PhysX
        receives each wrench in the local frame of its corresponding prim.

        Args:
            forces (torch.Tensor): Shape (n_vehicles, parts_per_vehicle, 3) or (n_total_parts, 3), in N.
            torques (torch.Tensor): Same layout as forces, in Nm.
        """

        forces = forces.reshape((self._vehicle_prims.count, 3))
        
        torques = torques.reshape((self._vehicle_prims.count, 3))

        # Retain the last applied local wrench for diagnostics
        if self._parts_per_vehicle is not None:
            self._last_forces_local.copy_(forces.reshape(self._n_vehicles, self._parts_per_vehicle, 3).detach())
            self._last_torques_local.copy_(torques.reshape(self._n_vehicles, self._parts_per_vehicle, 3).detach())

        if getattr(self, "record_wrench_geometry", False):
            # Evaluation only: cache geometry at application time, before integration.
            positions, attitudes = self._vehicle_prims.get_world_poses()
            coms, _ = self._vehicle_prims.get_coms()
            self._wrench_geometry = tuple(x.detach().clone() for x in
                                          (positions, attitudes, coms, self._vehicle_prims.get_masses()))

        # Apply each wrench in the local frame of its rigid prim
        self._vehicle_prims.apply_forces_and_torques_at_pos(forces, torques, is_global=False)


    def update_state(self, dt: float) -> None:
        """
        Read the /body pose and velocity after each physics step.

        World-frame position and linear velocity come from ``_root_prims``.
        The position is the /body prim origin, but PhysX reports the linear
        velocity of the /body centre of mass (verified: a pure torque leaves it
        zero while the origin moves). The two coincide only when the prim
        origin is at its CoM. The cached body velocity and angular velocity are
        expressed in the body frame; acceleration is a finite difference in the
        world frame.

        Args:
            dt (float): Time since the previous physics step, in seconds.
        """
        
        if self._sim_running == False:
            return
        self._last_physics_dt = float(dt)
        self._state_update_count = getattr(self, "_state_update_count", 0) + 1
                
        # Get the current position of the body in the inertial frame and its orientation relative to the inertial frame
        positions, orientations = self._root_prims.get_world_poses()

        # The linear velocity [x_dot, y_dot, z_dot] of the vehicle's body frame expressed in the inertial frame of reference
        linear_vel = self._root_prims.get_linear_velocities()

        # Read angular velocity from PhysX in the world frame
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
        Synchronize cached state and backends after an external PhysX reset.

        Args:
            env_ids (torch.Tensor): Indices of vehicles being reset.
            positions (torch.Tensor): /body positions in the world frame.
            attitudes (torch.Tensor): /body quaternions [w, x, y, z].
            linear_velocity (torch.Tensor | None): World-frame velocities in m/s.
            angular_velocity (torch.Tensor | None): Body-frame rates in rad/s.
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
            if hasattr(backend, "set_state"):
                backend.set_state(
                    env_ids=env_ids,
                    positions=positions,
                    attitudes=attitudes,
                    linear_velocity=self._state.linear_velocity[env_ids],
                    angular_velocity=self._state.angular_velocity[env_ids],
                )

            else:
                backend.update_state(self._state)

    
    # =============================================================================
    # Post-physics sensor forwarding
    # =============================================================================

    def update_sensors(self, dt: float):
        """Sample the updated physical state and forward per-vehicle readings.

        The callback runs after update_state(), at POST_PHYSICS_STEP order 1.
        Estimator warmup can call it explicitly while physics is stopped.

        Args:
            dt (float): Time since the previous physics step, in seconds.
        """

        if not self._sim_running:
            return
        for suite in self._sensors:
            samples = suite.update(self._state, dt)
            for backend in self._sensor_backends:
                backend.update_sensors(samples, dt)


    def update_graphical_sensors(self, event):
        """Forward rendered sensor data if batch graphical sensors are enabled.

        The constructor currently rejects graphical sensors, and no render
        callback is registered for this method.

        Args:
            event: Render event containing the elapsed time in ``payload['dt']``.
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._graphical_sensors:
            sensor_data = sensor.update(self._state, event.payload['dt'])

            # If some data was updated and we have a ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_graphical_sensor(sensor.sensor_type, sensor_data)


    # =============================================================================
    # Helpers and API
    # =============================================================================

    def get_batch_layout_info(self) -> dict[str, Any]:
        """
        Return the batch size, part count and rigid prim paths for diagnostics.
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
        Initialize the vehicle-specific physics after the prim views are ready.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Release vehicle-specific state when the simulation stops.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Compute and apply the vehicle's forces before each physics step.

        Args:
            dt (float): Time since the previous physics step, in seconds.
        """
        pass
