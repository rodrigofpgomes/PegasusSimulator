"""
| File: test_evaluation_layout.py
| Description: Evaluation run identity and shared log directory contracts.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path


RL_DIR = Path(__file__).resolve().parents[2] / "examples" / "rl"
sys.path.insert(0, str(RL_DIR))

from evaluation_log import EvaluationLog, evaluation_directory, evaluation_identity


class EvaluationLayoutContract(unittest.TestCase):

    def test_task_and_state_mode_identify_run(self):
        controllers = [{"name": "shuttle_ppo2", "type": "ppo2",
                        "task": "shuttle/thrust_moment/isaac_lab",
                        "px4": {"mode": "estimator"}}]
        self.assertEqual(evaluation_identity("shuttle", controllers),
                         ("px4_estimator", "shuttle_thrust_moment_isaac_lab"))

        with tempfile.TemporaryDirectory() as temporary:
            first = evaluation_directory(temporary, "shuttle", controllers)
            second = evaluation_directory(temporary, "shuttle", controllers)
            self.assertEqual(first.parent, Path(temporary) / "px4_estimator"
                             / "shuttle_thrust_moment_isaac_lab")
            self.assertNotEqual(first, second)

    def test_audit_uses_callers_run_directory_and_records_test(self):
        controllers = [{"name": "shuttle_ppo2", "type": "ppo2",
                        "task": "shuttle/thrust_moment/isaac_lab"}]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "dashboard_run"
            run_dir.mkdir()
            (run_dir / "vehicles.json").write_text(json.dumps(controllers))

            audit = EvaluationLog(run_dir, {}, controllers, {}, "shuttle")
            audit.stream.close()

            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["evaluation_mode"], "ground_truth")
            self.assertEqual(metadata["test_id"], "shuttle_thrust_moment_isaac_lab")
            self.assertTrue((run_dir / "raw.jsonl").exists())
            self.assertTrue((run_dir / "vehicles.json").exists())
            self.assertFalse(any(run_dir.parent.glob("legacy_*")))


if __name__ == "__main__":
    unittest.main()
