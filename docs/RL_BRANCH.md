# RL branch

This branch continues the `dev_rl` history with the current batched vehicle and
reinforcement learning code. The first commits import the working RL implementation
and its ShuttleGlider final task variants. Later cleanup should be committed in
small, focused changes.

## ShuttleGlider control contract

`ShuttleGliderBatch2` receives eight values per vehicle:

```text
[omega_0, omega_1, omega_2, omega_3, Omega_p, delta_e, delta_a, delta_r]
```

The four shuttle rotors and the puller use separate thrust models. The combined
five-rotor interface supplies velocity limits and rotor state to the RL backend and
reset logic. It synchronizes reset references and velocities into both physical
models and exposes their current references, velocities, and forces to RL code.
Longitudinal trim also reads the five thrust coefficients and rotor
positions in the body frame.

The batched thrust models use `rotor_constant`, `reaction_moment_coefficient`,
`rot_dir`, `motor_time_constant_up`, and `motor_time_constant_down`. The seven
`vehicle_final` task configurations were migrated to these keys. For variants that
used `moment_constant` and `thrust_constant`, the reaction coefficient retains the
previous relation `moment_constant * thrust_constant`.
The `play_multi.py` vehicle profiles use `reaction_moment_coefficient` as well;
the batched shuttle thrust curve rejects `rolling_moment_coefficient` to avoid silently
loading its default coefficient.

## Validation and scope

- Python syntax was checked for the changed vehicle modules and seven task files.
- All seven final task configurations instantiated and updated both batched thrust
  models in a CPU check.
- The four CPU checks in `tests/review/test_torch_contracts.py` passed, including
  a reset and motor-update check for the combined rotor interface.
- An Isaac Sim or SITL run has not yet been performed for this branch.
- Training logs, checkpoints, recovered snapshots, and backup copies were excluded
  from the initial import. Historical tasks with five-action contracts still need a
  separate migration decision.
