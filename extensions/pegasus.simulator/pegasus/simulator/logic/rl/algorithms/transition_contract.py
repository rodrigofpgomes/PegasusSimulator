"""
| File: transition_contract.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Preserve the physical terminal observation when an environment
| returns the first observation of the next episode in the same step.
|
| PPO and PPO2 use PegasusPPO to bootstrap time-limit transitions from that
| terminal observation. SAC uses terminal_next_states for replay. The helper
| leaves non-terminal rows unchanged and never bootstraps true terminations.
"""

import torch
from skrl.agents.torch.ppo import PPO


# =============================================================================
# Terminal observation contract
# =============================================================================

def terminal_next_states(next_states, terminated, truncated, infos):
    """Return next states belonging to the transitions just completed.

    PegasusEnv resets completed rows before returning from step(). Consequently,
    next_states contains the next episode's initial observation on those rows.
    infos["final_observation"] contains the observation before reset. Replay
    and value bootstrapping must use the latter for terminated and truncated
    rows; the returned tensor is a copy only when a row has completed.
    """
    # Keep the ordinary transition path free of an extra tensor copy.
    done = (terminated | truncated).view(-1)
    if not done.any():
        return next_states

    # A completed transition has no valid next state without its final observation.
    if not infos or "final_observation" not in infos:
        raise ValueError("Pegasus autoreset requires final_observation for completed transitions")
    final = infos["final_observation"].to(device=next_states.device, dtype=next_states.dtype)
    if final.shape != next_states.shape:
        raise ValueError("final_observation shape differs from next_states")

    # The optional mask distinguishes a stored terminal row from an empty slot.
    mask = infos.get("final_observation_mask")
    if mask is not None and not mask.view(-1).to(done.device)[done].all():
        raise ValueError("Missing terminal observation for a completed transition")

    physical = next_states.clone()
    physical[done] = final[done]
    return physical


# =============================================================================
# PPO time-limit bootstrap
# =============================================================================

class PegasusPPO(PPO):
    """SKRL PPO with terminal-state bootstrapping under same-step autoreset."""

    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        """Store physical next states and add a value target at time limits.

        Reward shaping runs before the timeout correction. The parent PPO
        bootstrap is disabled for this call because it would read the reset
        observation and could add the correction twice.
        """
        # Replace reset observations only on the rows that ended this step.
        physical = terminal_next_states(next_states, terminated, truncated, infos)

        # Save skrl's settings so the temporary reward wrapper affects one call.
        bootstrap, shaper = self._time_limit_bootstrap, self._rewards_shaper
        correction = torch.zeros_like(rewards)

        # Truncation is a time limit; termination is a physical episode end.
        if bootstrap and (truncated & ~terminated).any():
            with torch.no_grad():
                values, _, _ = self.value.act({"states": self._state_preprocessor(physical)}, role="value")
                values = self._value_preprocessor(values, inverse=True)
            correction = self._discount_factor * values * (truncated & ~terminated)

        # Apply the correction after reward shaping and suppress skrl's timeout term.
        self._time_limit_bootstrap = False
        self._rewards_shaper = lambda r, t, ts: (shaper(r, t, ts) if shaper else r) + correction
        try:
            return super().record_transition(states, actions, rewards, physical,
                terminated, truncated, infos, timestep, timesteps)
        finally:
            # Keep the agent's configuration intact for the following transition.
            self._time_limit_bootstrap, self._rewards_shaper = bootstrap, shaper
