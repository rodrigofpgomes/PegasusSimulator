"""CPU check of PPO2 configuration preparation without skrl imports."""

import ast
import copy
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/algorithms/ppo2.py"


class AlgorithmConfigContract(unittest.TestCase):
    def test_ppo2_prepares_a_copy_for_selected_device(self):
        tree = ast.parse(SOURCE.read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_cfg")
        namespace = {"copy": copy}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)

        original = {"rollouts": 32, "state_preprocessor_kwargs": {"size": 8}}
        prepared = namespace["_prepare_cfg"](original, "cuda:0")

        self.assertEqual(prepared["rollouts"], 32)
        self.assertEqual(prepared["state_preprocessor_kwargs"]["device"], "cuda:0")
        self.assertNotIn("device", original["state_preprocessor_kwargs"])


if __name__ == "__main__":
    unittest.main()
