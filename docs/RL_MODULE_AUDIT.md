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
| `algorithms/__init__.py` | Package entry point for algorithm imports. | Keep; load selected algorithms by name and export PPO, PPO2 and SAC. |
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

`aligned_trim.py` remains active in the ShuttleGlider tasks. Its unused
`_trim_rotor_norm()` compatibility method, unused cached coefficients and
`PHYSICS_REVISION` constant were removed. The previous implementation is
available in commit `aa2061c`.

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

`ResetManager.reset_envs()` initializes rotor velocities. `PegasusEnv` derives
normalized rotor values from those velocities and the corresponding vehicle's
rotor limits when a task needs them, including when seeding action history.
No normalized value is cached on a thruster. The reset manager does not call
`sync_to_models()`: `ShuttleGliderBatch2.update()` does so before either
physical propulsion model advances. The method remains necessary there.

The reset manager no longer clears vehicle force, torque or surface buffers.
Each vehicle starts its next physics update by zeroing its applied-force
buffers; `VehicleBatch` replaces its last-applied wrench at the end of that
update. Rotor-mode gliders replace surface deflections from the first backend.
Tasks build reset observations from state, goals and their own action history,
not these vehicle buffers. `RLBackend.set_state()` still clears backend commands
and external disturbances for reset rows, preventing previous-episode loads
from reaching the next physics step.

`PegasusEnv` now builds the policy and optional critic spaces directly. Four
unused copies of config dimensions and episode duration were removed; the
configuration remains available as `env.cfg`. Its reset callback hook stays
because task reset methods call it, allowing external users to attach reset
bookkeeping without changing task implementations. No current launcher
registers such a callback.
The physics step now uses the `World` attached by the training or playback
launcher directly, so a missing world fails at the point of use.

The three batched vehicles now accept the task mode `direct_force` directly.
`rotor_velocity_direct` passes commanded speeds to the thruster models, while
`direct_force` applies the requested per-part loads without motor dynamics.

For `MultirotorBatch`, `rotor_velocity` converts body thrust and moments into
rotor-speed commands. With `inner_loop=True`, the backend also computes an
attitude correction from the requested body-frame force and current attitude.
It computes the moment from attitude and angular-rate errors alone. The resulting
rotor speeds pass through the multirotor's thruster dynamics. This mode is
restricted to `MultirotorBatch`. Its thrust projection is limited to positive
values because the rotors cannot reverse their thrust direction. The current
task presets still use `direct_force` or `rotor_velocity_direct`; the inner
loop is available for future multirotor tasks.

The restored inner loop follows the original geometric construction: desired
body Z from force, fixed yaw reference, orthonormal desired attitude, vee-map
attitude error, and proportional attitude/rate moments. It is **not exactly the
same calculation** as the original `553538e` version. The task supplies force
in body axes, so the current loop rotates it to world axes before constructing
the target; it also uses the resolved body index, clips negative collective
thrust, and handles zero force and parallel reference axes. The original loop
used part index zero and treated its force as already world-aligned. Both loops
compute the moment from attitude and rate errors without a requested moment.

The separate playback LQR controller also retains attitude-control logic.
Checks protecting reset orientation and batch layout remain because invalid
values would be passed directly to PhysX.

## Algorithm and wrapper review

`PegasusSkrlWrapper` converts the task's policy/action spaces to skrl tensors.
It flattens both ordinary observations and `final_observation` with the same
policy space, and exposes a separate critic state only when the task provides
one. Its first reset is cached because PPO2 initializes the task before the
trainer requests its own reset. Rendering and closing remain delegated so a
future launcher can choose how to manage the environment lifecycle.

`ppo.py` trains standard skrl PPO through `PegasusPPO`, which corrects timeout
bootstrapping. `ppo2.py` keeps its RSL-RL-style minibatch and adaptive KL update
as a separate algorithm. `sac.py` keeps its custom update for delayed actor
steps, entropy learning, optional timeout bootstrapping and terminal-state
replay, even when the configured delay is one. `transition_contract.py` supplies
the physical final observation to both PPO variants and SAC. The algorithms
write configuration metadata and use `training_export.py` to export scalar
histories after training, including interrupted runs.

The commented-out PPO2 episode-length randomization and its unused variables
were removed. The active training and playback scripts currently select these
algorithms and construct the backend; their setup can evolve independently of
the task, transition and wrapper contracts described here. CPU contract tests
do not execute skrl or Isaac Sim training.

## Terminal transition review

`PegasusEnv.step()` captures `final_observation` before resetting completed
vehicles and returns the new episode's observation for those rows. The wrapper
flattens both observations with the same policy space. `terminal_next_states()`
then replaces completed rows with their terminal observations for learning;
the observation returned by the wrapper remains the next episode's state for
interaction. PPO uses the terminal state to compute its timeout value, and SAC
stores the terminal state in replay memory. When supplied, the completion mask
confirms that every completed row has a terminal observation.

`PegasusSkrlWrapper.reset()` initializes the environment once because each
task already resets completed rows during `step()`. Its later calls return the
current observation. The unused `skrl.config` import was removed. `skrl` is
not installed in this workspace, so the installed library's trainer behavior
and the PPO timeout override still need an Isaac Sim training run.

## Algorithm loading review

`train.py` discovers algorithm modules by a top-level `train()` function and
imports only the selected module. The package initializer no longer imports
PPO and SAC eagerly; its export list now includes PPO2. This lets package
discovery avoid loading the other training modules. The unused `skrl.config`
import was also removed from SAC. The custom PPO2 and SAC update loops depend
on skrl internals, so they remain unchanged until they can be checked against
the version bundled with the training environment.

## Backend interface and goal markers

`RLBackend.update_sensor()` and `update_graphical_sensor()` remain because the
abstract `Backend` interface requires them. Goal marker creation and updates
are used by active tasks. `hide_goal_markers()` is retained for future task
control; the unused stored marker paths remain removed.

## Trim and training export review

The ShuttleGlider `vehicle_final` tasks call `AlignedTrimMixin` to compute
physically feasible starting speed, elevator and rotor references. Its
feasibility checks protect the force and pitch equilibrium, so they remain.
`training_export.py` is called by all three training algorithms to record run
metadata and export TensorBoard scalars. Its export failure handling preserves
the original training exception, so the module remains active.

## Next review points

- Review PPO, PPO2 and SAC against their task presets and the installed skrl
  version before simplifying their training code.
