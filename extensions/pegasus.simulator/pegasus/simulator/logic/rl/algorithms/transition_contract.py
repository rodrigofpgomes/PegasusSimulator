"""
| File: transition_contract.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Terminal observations and timeout bootstrapping for same-step autoreset.
"""
import torch
from skrl.agents.torch.ppo import PPO


def terminal_next_states(next_states, terminated, truncated, infos):
    """Replace autoreset rows with terminal states for replay or bootstrapping.

    The environment returns the new episode's observation after reset. The
    final observation in infos belongs to the completed transition instead.
    """
    done = (terminated | truncated).view(-1)
    if not done.any():
        return next_states
    if not infos or "final_observation" not in infos:
        raise ValueError("Pegasus autoreset requires final_observation for completed transitions")
    final = infos["final_observation"].to(device=next_states.device, dtype=next_states.dtype)
    if final.shape != next_states.shape:
        raise ValueError("final_observation shape differs from next_states")
    mask = infos.get("final_observation_mask")
    if mask is not None and not mask.view(-1).to(done.device)[done].all():
        raise ValueError("Missing terminal observation for a completed transition")
    physical = next_states.clone()
    physical[done] = final[done]
    return physical


class PegasusPPO(PPO):
    """SKRL PPO that bootstraps timeouts from the physical terminal state."""
    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        """Correct next states and timeout values before storing the transition."""
        physical = terminal_next_states(next_states, terminated, truncated, infos)
        bootstrap, shaper = self._time_limit_bootstrap, self._rewards_shaper
        correction = torch.zeros_like(rewards)
        if bootstrap and (truncated & ~terminated).any():
            with torch.no_grad():
                values, _, _ = self.value.act({"states": self._state_preprocessor(physical)}, role="value")
                values = self._value_preprocessor(values, inverse=True)
            correction = self._discount_factor * values * (truncated & ~terminated)
        # Apply the correction after reward shaping; avoid skrl's own timeout term
        self._time_limit_bootstrap = False
        self._rewards_shaper = lambda r, t, ts: (shaper(r, t, ts) if shaper else r) + correction
        try:
            return super().record_transition(states, actions, rewards, physical,
                terminated, truncated, infos, timestep, timesteps)
        finally:
            self._time_limit_bootstrap, self._rewards_shaper = bootstrap, shaper
