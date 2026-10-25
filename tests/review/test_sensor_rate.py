"""
| File: test_sensor_rate.py
| Description: Check that scalar sensor sampling follows its requested period.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


class SensorRateContract(unittest.TestCase):
    def test_ten_hertz_sensor_updates_every_ten_physics_steps(self):
        """Float32 addition must not turn a 0.1 s period into 0.11 s."""
        source = Path(__file__).resolve().parents[2] / (
            "extensions/pegasus.simulator/pegasus/simulator/logic/sensors/sensor.py"
        )
        state_module = ModuleType("pegasus.simulator.logic.state")
        state_module.State = type("State", (), {})
        spec = importlib.util.spec_from_file_location("scalar_sensor_rate", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {state_module.__name__: state_module}):
            spec.loader.exec_module(module)

        class ProbeSensor(module.Sensor):
            def __init__(self):
                super().__init__("Probe", torch.tensor(10.0))

            @module.Sensor.update_at_rate
            def update(self, state, dt):
                return float(dt)

        sensor = ProbeSensor()
        samples = [step for step in range(21)
                   if sensor.update(None, torch.tensor(0.01)) is not None]

        self.assertEqual(samples, [0, 10, 20])


if __name__ == "__main__":
    unittest.main()
