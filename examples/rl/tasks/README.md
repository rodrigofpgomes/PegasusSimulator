# RL task families

New training tasks are grouped by vehicle: `iris_crazyflie/`, `shuttle/`,
`easyglider/` and `shuttle_glider/`. Each experiment directory should contain
one `*_env.py`, its `agents/` presets and a README recording the observation,
action, physics, reward, reset and reference contracts.

The existing `shuttle_glider2/` and quadcopter task IDs remain available while
their checkpoint and manifest references are audited. Historical candidates
in `/home/rodrigogomes/PegasusSimulatorRL/examples/rl/tasks` are catalogued
in `docs/RL_EXAMPLES_AUDIT.md` before being moved into these families.
