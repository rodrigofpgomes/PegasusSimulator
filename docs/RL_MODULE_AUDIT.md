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

## Next review points

- Check the reset lifecycle across `base_env.py`, `reset_manager.py` and
  `rl_backend.py`, including partial airborne resets.
- Check how `skrl_pegasus_wrapper.py` and `algorithms/transition_contract.py`
  handle terminal observations when a step resets an environment.
- Review PPO, PPO2 and SAC against their task presets and the installed skrl
  version before simplifying their training code.
