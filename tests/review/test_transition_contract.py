"""CPU check of terminal observations and timeout bootstrap under autoreset."""

import ast
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/algorithms/transition_contract.py"


class ParentPPO:
    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        return next_states, self._rewards_shaper(rewards, timestep, timesteps)


class ValueModel:
    def act(self, inputs, role):
        return torch.full((inputs["states"].shape[0], 1), 3.0), None, None


class TransitionContract(unittest.TestCase):
    def test_terminal_rows_and_timeout_value(self):
        tree = ast.parse(SOURCE.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        namespace = {"torch": torch, "PPO": ParentPPO}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)

        agent = namespace["PegasusPPO"]()
        agent.value = ValueModel()
        agent._state_preprocessor = lambda value: value
        agent._value_preprocessor = lambda value, inverse: value
        agent._discount_factor = 0.9
        agent._time_limit_bootstrap = True
        shaper = lambda rewards, timestep, timesteps: rewards * 2
        agent._rewards_shaper = shaper

        next_states = torch.tensor([[1.0], [2.0], [3.0]])
        final = torch.tensor([[0.0], [20.0], [30.0]])
        terminated = torch.tensor([[False], [True], [False]])
        truncated = torch.tensor([[False], [False], [True]])
        infos = {"final_observation": final, "final_observation_mask": torch.tensor([False, True, True])}

        physical, reward = agent.record_transition(
            next_states, None, torch.ones((3, 1)), next_states,
            terminated, truncated, infos, 0, 1,
        )

        self.assertEqual(physical[:, 0].tolist(), [1.0, 20.0, 30.0])
        torch.testing.assert_close(reward[:, 0], torch.tensor([2.0, 2.0, 4.7]))
        self.assertIs(agent._rewards_shaper, shaper)
        self.assertTrue(agent._time_limit_bootstrap)
        self.assertEqual(next_states[:, 0].tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
