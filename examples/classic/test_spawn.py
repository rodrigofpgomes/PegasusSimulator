#!/usr/bin/env python3
"""
Minimal Pegasus spawn test:
- no backend
- no controller
- no reset manager
- no RL env
- only tests whether 1 or 2 MultirotorBatch objects spawn correctly
"""

import argparse
import carb
from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vehicles", type=int, default=4)
    parser.add_argument("--second_batch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


args = parse_args()

simulation_app = SimulationApp({"headless": args.headless})


# -----------------------------
# Isaac / Pegasus imports
# -----------------------------
import omni.timeline
from omni.isaac.core.world import World

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import (
    MultirotorBatch,
    MultirotorBatchConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import numpy as np
from pxr import UsdPhysics


def make_positions(n, y, z=8.0, spacing=2.5):
    return [[i * spacing, y, z] for i in range(n)]


def print_prim_info(stage, path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        print(f"{path:45s} | exists=False")
        return False

    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    type_name = prim.GetTypeName()

    print(
        f"{path:45s} | exists=True  "
        f"type={type_name:12s} rigid_body={has_rb}"
    )

    return has_rb


def diagnose_batch(vehicle, prefix, n_vehicles):
    stage = stage_utils.get_current_stage()

    print("\n" + "=" * 80)
    print(f"Diagnosing batch: {prefix}")
    print("=" * 80)

    print(f"vehicle has _root_prims: {hasattr(vehicle, '_root_prims')}")
    print(f"vehicle has _prim_paths : {hasattr(vehicle, '_prim_paths')}")
    print(f"vehicle has _stage_prefix: {hasattr(vehicle, '_stage_prefix')}")

    found_body_rbs = 0

    for i in range(n_vehicles):
        print(f"\nVehicle {i}")
        body_path = f"{prefix}_{i}/body"

        if print_prim_info(stage, body_path):
            found_body_rbs += 1

        for rotor_id in range(4):
            rotor_path = f"{prefix}_{i}/rotor{rotor_id}"
            print_prim_info(stage, rotor_path)

    print("\nSummary:")
    print(f"Expected body rigid bodies: {n_vehicles}")
    print(f"Found body rigid bodies   : {found_body_rbs}")

    print("\nAll prims under prefix:")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prefix):
            print("  ", path, "| type:", prim.GetTypeName())


class PegasusSpawnTest:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()

        world_settings = dict(self.pg._world_settings)
        world_settings["device"] = "cuda"

        self.pg._world = World(**world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            position=np.array([1.0, 1.0, 1.0]),
            attributes={
                "inputs:intensity": 5e3,
                "inputs:color": (1.0, 1.0, 1.0),
            },
        )

        n = args.n_vehicles

        # -----------------------------
        # Batch 1: no backend
        # -----------------------------
        cfg1 = MultirotorBatchConfig(n_vehicles=n)
        cfg1.backends = []

        self.batch1 = MultirotorBatch(
            stage_prefix="/World/rl",
            usd_file=ROBOTS["Iris"],
            vehicle_batch_id=1,
            n_vehicles=n,
            spacing=2.5,
            #init_pos=make_positions(n, y=-2.0),
            config=cfg1,
        )

        # -----------------------------
        # Optional Batch 2: no backend
        # -----------------------------
        self.batch2 = None

        if args.second_batch:
            cfg2 = MultirotorBatchConfig(n_vehicles=n)
            cfg2.backends = []

            self.batch2 = MultirotorBatch(
                stage_prefix="/World/lqr",
                usd_file=ROBOTS["Iris_White"],
                vehicle_batch_id=2,
                n_vehicles=n,
                spacing=2.5,
                #init_pos=make_positions(n, y=+2.0),
                config=cfg2,
            )

        #self.pg.set_viewport_camera([8.0, -8.0, 6.0], [3.0, 0.0, 6.0])

        print("\nCalling world.reset()...")
        self.world.reset()

        print("Starting timeline...")
        self.timeline.play()

        print("Stepping world once...")
        self.world.step(render=not args.headless)

        diagnose_batch(self.batch1, "/World/a/quadrotor_a", n)

        if self.batch2 is not None:
            diagnose_batch(self.batch2, "/World/b/quadrotor_b", n)

    def run(self):
        while simulation_app.is_running():
            self.world.step(render=not args.headless)

        carb.log_warn("PegasusSpawnTest closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusSpawnTest()

    app.run()


if __name__ == "__main__":
    main()