"""CPU check of terminal observations and timeout bootstrap under autoreset."""

import ast
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[2] / "extensions/pegasus.simulator/pegasus/simulator/logic/rl/algorithms/transition_contract.py"


class ParentPPO:
    def __init__(self):
        self.write_interval = 1
        self._cumulative_rewards = None
        self._cumulative_timesteps = None
        self.ended_lengths = []

    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        # Reproduce skrl 1.4.3's episode tracking, including its 2D index.
        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

        self._cumulative_rewards.add_(rewards)
        self._cumulative_timesteps.add_(1)
        finished = (terminated + truncated).nonzero(as_tuple=False)

        if finished.numel():
            self.ended_lengths.extend(self._cumulative_timesteps[finished][:, 0].reshape(-1).tolist())
            self._cumulative_rewards[finished] = 0
            self._cumulative_timesteps[finished] = 0

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

    def test_other_vehicle_timeout_preserves_first_vehicle_episode_length(self):
        """An offset timeout must not turn vehicle 0's 499-step episode into one step."""
        tree = ast.parse(SOURCE.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        namespace = {"torch": torch, "PPO": ParentPPO}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)

        agent = namespace["PegasusPPO"]()
        agent._time_limit_bootstrap = False
        agent._rewards_shaper = None
        observations = torch.zeros((2, 1))
        rewards = torch.ones((2, 1))
        upstream = ParentPPO()
        upstream._rewards_shaper = lambda reward, timestep, timesteps: reward

        for step in range(1, 500):
            truncated = torch.tensor([[step == 499], [step == 498]])
            infos = {"final_observation": observations} if truncated.any() else {}
            upstream.record_transition(
                observations, None, rewards, observations,
                torch.zeros_like(truncated), truncated, infos, step, 500,
            )
            agent.record_transition(
                observations, None, rewards, observations,
                torch.zeros_like(truncated), truncated, infos, step, 500,
            )

        self.assertEqual(upstream.ended_lengths, [498, 1])
        self.assertEqual(agent.ended_lengths, [498, 499])
        self.assertEqual(agent._cumulative_timesteps[:, 0].tolist(), [0, 1])
        self.assertEqual(agent._cumulative_rewards[:, 0].tolist(), [0.0, 1.0])

    def test_true_one_step_termination_is_still_recorded(self):
        tree = ast.parse(SOURCE.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        namespace = {"torch": torch, "PPO": ParentPPO}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)

        agent = namespace["PegasusPPO"]()
        agent._time_limit_bootstrap = False
        agent._rewards_shaper = None
        observations = torch.zeros((2, 1))
        terminated = torch.tensor([[True], [False]])
        agent.record_transition(
            observations, None, torch.ones((2, 1)), observations,
            terminated, torch.zeros_like(terminated),
            {"final_observation": observations}, 1, 500,
        )

        self.assertEqual(agent.ended_lengths, [1])
        self.assertEqual(agent._cumulative_timesteps[:, 0].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
