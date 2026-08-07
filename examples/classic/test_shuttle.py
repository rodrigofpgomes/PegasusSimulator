#!/usr/bin/env python
"""
| File: python_control_backend.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| Adapted by: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This file serves as an example on how to use the control backends API to create a custom batch controller
for multiple vehicles from scratch and use it to perform a simulation, without using PX4 nor ROS.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Used for adding extra lights to the environment
import isaacsim.core.utils.prims as prim_utils

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Import the custom python control backend
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/utils')
from nonlinear_controller_batch import NonlinearControllerBatch

# Auxiliary scipy and numpy modules
import numpy as np
import torch
from pegasus.simulator.logic.transforms import euler_angles_to_matrix, matrix_to_quaternion


# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path

import random
from isaacsim.util.debug_draw import _debug_draw

from pxr import UsdPhysics

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self, n_envs=8, spacing=3.0, device="cpu"):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.

        world_settings = dict(self.pg._world_settings) 
        world_settings["device"] = device         

        self.pg._world = World(**world_settings)

        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Add a custom light with a high-definition HDR surround environment of an exhibition hall,
        # instead of the typical ground plane
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            position=np.array([1.0, 1.0, 1.0]),
            attributes={
                "inputs:intensity": 5e3,
                "inputs:color": (1.0, 1.0, 1.0),
                "inputs:texture:file": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr"
                # Alternative sky: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr
            }
        )

        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())


        cfg_shuttle = {
            "num_rotors": 4,
            "rotor_constant": [1.709716e-05, 1.709716e-05, 1.709716e-05, 1.709716e-05],
            "rolling_moment_coefficient": [1e-06, 1e-06, 1e-06, 1e-06],
            "rot_dir": [-1, -1, 1, 1],
            "min_rotor_velocity": [0, 0, 0, 0],                             # rad/s
            "max_rotor_velocity": [1400, 1400, 1400, 1400],                 # rad/s
            "motor_time_constant": [0.008, 0.008, 0.008, 0.008],            # s
        }

        # Create the vehicle 1
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorBatchConfig(cfg=cfg_shuttle, n_vehicles=1)


        # A single trajectory is shared across all vehicles; multiple trajectories are assigned per vehicle.
        config_multirotor.backends = [NonlinearControllerBatch(
            trajectory_files=[self.curr_dir + "/trajectories/pitch_relay_90_deg_2.csv"],
            results_files=[self.curr_dir + "/results/batch_statistics.npz"],
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0],
            n_vehicles=n_envs,
            device=self.pg._world_settings["device"]
        )]

        # Spawn the multirotor batch
        MultirotorBatch1 = MultirotorBatch(
            stage_prefix="/World/quadrotor",
            usd_file=ROBOTS["Shuttle"],
            vehicle_batch_id=1,
            n_vehicles=n_envs,
            config=config_multirotor,
        )
            
        self.world.reset()


    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Run the batch pipeline with a single vehicle for validation
    pg_app = PegasusApp(n_envs=1, spacing=3.0, device="cuda")

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()