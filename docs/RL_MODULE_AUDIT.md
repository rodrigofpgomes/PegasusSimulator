# RL module usage audit

This is a static usage review of `extensions/pegasus.simulator/pegasus/simulator/logic/rl`.
It identifies modules that can be removed from the working tree without losing
their implementation from Git history. It does not establish Isaac Sim or training
correctness.

| Module | Current use | Decision |
| --- | --- | --- |
| `__init__.py` | Exports the environment, backend and reset classes used by `train.py`, `play.py` and `play_multi.py`. | Keep. |
| `base_env.py` | Base class for the quadcopter and ShuttleGlider RL tasks. | Keep. |
| `rl_backend.py` | Backend created by the training and playback launchers. | Keep. |
| `reset_manager.py` | Active reset implementation used by launchers, tasks and `GliderBatch.reset_in_flight()`. | Keep. |
| `aligned_trim.py` | Used by the ShuttleGlider `vehicle_final` tasks and a CPU contract check. | Keep. |
| `skrl_pegasus_wrapper.py` | Used by all three training algorithms and playback. | Keep. |
| `training_export.py` | Used by PPO, PPO2 and SAC training. | Keep. |
| `algorithms/ppo.py` | Loaded by `train.py` for `--algo ppo`; task presets exist. | Keep. |
| `algorithms/ppo2.py` | Loaded by `train.py` for `--algo ppo2` and by playback; task presets exist. | Keep. |
| `algorithms/sac.py` | Loaded by `train.py` for `--algo sac`; task presets exist. | Keep. |
| `algorithms/transition_contract.py` | Supplies terminal observations and timeout bootstrapping to PPO, PPO2 and SAC. It has no `train()` function. | Keep as a helper, exclude from algorithm discovery. |
| `algorithms/__init__.py` | Package entry point for algorithm imports. | Keep; review its eager imports and incomplete `__all__` during the algorithm pass. |
| `reset_manager copy.py` | No imports or calls; superseded by `reset_manager.py`. | Remove from the working tree. |

## Logic retained for future use

The removed `reset_manager copy.py` initialized a single vehicle batch, reset
selected root rigid prims to their spawn poses with zero linear and angular
velocities, provided `reset_all()`, and exposed `init_pos`. The active
`reset_manager.py` contains those operations and also supports multiple vehicle
batches, state overrides, rotor initialization and goals.

The exact removed source remains available in Git history:

```bash
git show 64a14a7:"extensions/pegasus.simulator/pegasus/simulator/logic/rl/reset_manager copy.py"
```

## Reset lifecycle review

`PegasusEnv.step()` advances physics, computes reward and termination, captures
the terminal observation, and then asks the task to reset completed rows. The
observation returned after that reset starts the next episode. Tasks call
`ResetManager.reset_envs()`, which writes the root pose and velocity to PhysX,
updates the vehicle state cache, and forwards the new state to its backends.

`RLBackend.set_state()` now clears forces, torques, rotor commands and external
disturbances for only the reset environments. `RLBackend.reset()` clears those
buffers for a full reset. This prevents the previous episode's commands from
being reused after a teleport while leaving other vehicles' commands intact.
The playback agent can then prime its own rotor reference in its `set_state()`
override.

The three batched vehicles now accept the task mode `direct_force` directly.
`rotor_velocity_direct` passes commanded speeds to the thruster models, while
`direct_force` applies the requested per-part loads without motor dynamics.

For `MultirotorBatch`, `rotor_velocity` converts body thrust and moments into
rotor-speed commands. With `inner_loop=True`, the backend also computes an
attitude correction from the requested body-frame force and current attitude;
the requested body moment is retained as a feedforward term. The resulting
rotor speeds pass through the multirotor's thruster dynamics. This mode is
restricted to `MultirotorBatch`. Its thrust projection is limited to positive
values because the rotors cannot reverse their thrust direction. The current
task presets still use `direct_force` or `rotor_velocity_direct`; the inner
loop is available for future multirotor tasks.

The separate playback LQR controller also retains attitude-control logic.
Checks protecting reset orientation and batch layout remain because invalid
values would be passed directly to PhysX.

## Next review points

- Check how `skrl_pegasus_wrapper.py` and `algorithms/transition_contract.py`
  handle terminal observations when a step resets an environment.
- Review PPO, PPO2 and SAC against their task presets and the installed skrl
  version before simplifying their training code.
