# RL task families

New training tasks are grouped by vehicle: `iris_crazyflie/`, `shuttle/`,
`easyglider/` and `shuttle_glider/`. Each experiment directory should contain
one `*_env.py`, its `agents/` presets and a README recording the observation,
action, physics, reward, reset and reference contracts.

`shuttle_glider/no_surfaces/` is the first migrated task. The older task folders
remain available while their useful methods are selected for fresh training.
Their checkpoints are not part of the new experiment series. Historical
candidates in `/home/rodrigogomes/PegasusSimulatorRL/examples/rl/tasks` are
catalogued in `docs/RL_EXAMPLES_AUDIT.md`.
