# RL task families

New training tasks are grouped by vehicle: `iris_crazyflie/`, `shuttle/`,
`easyglider/` and `shuttle_glider/`. Each experiment directory should contain
one `*_env.py`, its `agents/` presets and a README recording the observation,
action, physics, reward, reset and reference contracts.

The Shuttle tasks are split by action level in `double_integrator/` and
`direct_rotor/`. ShuttleGlider tasks are split by surface actuation in
`no_surfaces/` and `surfaces/`. The original Iris/Crazyflie double-integrator
study is preserved separately. Vehicle READMEs list the selected ablations.
Historical checkpoints are not part of the new experiment series. Unselected
source variants remain in `/home/rodrigogomes/PegasusSimulatorRL/examples/rl/tasks`
and the former branch backups remain in Git history. The selection rationale
is recorded in `docs/RL_EXAMPLES_AUDIT.md`.
