# RL examples: launcher and experiment audit

This audit compares `examples/rl` on `rl-cleanup` with the experimental copy in
`/home/rodrigogomes/PegasusSimulatorRL/examples/rl`. It is a source review; it
does not establish Isaac Sim training performance or scientific conclusions.

## Order of work

1. Fix the shared vehicle configuration and the train/play contract. Otherwise
   the same checkpoint can be evaluated with different motor dynamics or USD.
2. Select one representative task for each experimental question. Record its
   observation, action, reward, reset distribution, trajectory and physics in
   its README before retraining it.
3. Reorganize selected tasks by vehicle. Their old checkpoints are not inputs
   to the new experiments, so old task IDs need no compatibility aliases.
   Keep historical variants in the source copy for comparison of methods.
4. Consolidate playback, controller adapters and dashboard around the final
   recorded data schema after the task contracts are stable.

The shared physical presets now live in
`pegasus.simulator.logic.vehicles.vehicle_physics`. `get_vehicle_physics()`
returns a deep copy. Selected tasks for retraining should use this preset for
motor parameters. A task-specific override is appropriate only for an explicit
physics comparison and should be named in the task README. In `direct_force`
mode, the vehicle applies the requested body wrench without advancing the
thruster model, so motor coefficients do not affect that actuation path. The
vehicle USD, mass, inertia and body layout still affect the dynamics. Training
and single vehicle playback select the batch class and asset from
`env_cfg.vehicle`.
Multi-vehicle playback uses the same preset and replaces it with a task's
explicit physics when evaluating a checkpoint. The Crazyflie USDA from the
experimental copy is now registered in `ROBOTS` for its baseline task.
The seven ShuttleGlider surface tasks now use the shared motor preset for new
training. `reward_offload_custom_aero` retains its explicit aerodynamic
coefficient override. The earlier results used different motor settings and
must not be compared as if they used this new configuration.

## Script dependency map

| Component | Current role | Review finding |
| --- | --- | --- |
| `train.py` | Discovers tasks and algorithm presets, starts Isaac Sim, creates the RL backend and vehicle. | Recursive discovery now sees the named vehicle/action families; backups were removed from the active task tree. |
| `play.py` | Restores one skrl policy and can compare with LQR and record CSV. | Previously instantiated Iris even for another task; vehicle selection and physics now follow the task. LQR remains an Iris-only comparison. |
| `play_multi.py` | Manifest-driven comparison of controllers in one world. | Uses the task's physics for skrl checkpoints and the shared baseline for external controllers. It also owns trajectories, recorder and reset logic, so extracting those parts needs a separate pass. |
| `utils/skrl_agent_controller.py` | Builds the task environment for inference and restores PPO/PPO2/SAC. | Duplicates task import and preset loading from the launchers. Keep the task environment as the observation/action authority. |
| `utils/foundation_policy_controller.py` and `foundation_policy_pre_controller.py` | Adapt external RAPTOR policies to Pegasus rotor commands. | Keep separate from skrl checkpoint loading; verify rotor order and motor priming against the selected Iris/Crazyflie physics. |
| `utils/lqr_controller_batch.py` and `non_linear_controller.py` | Classical comparison controllers. | Keep where a comparison actually uses them; review their assumptions after task selection. |
| `dashboard.py` | Starts/monitors playback and plots its CSV outputs. | The recorder already writes rotor commands/speeds, surface angles and aerodynamic wrench, but this dashboard does not plot them. The experimental `dashboard_final.py` has matching plots; port only the useful views and keep one dashboard entry point. |

## Task inventory and structure

The experimental copy contains 145 `*_env.py` files: 102 under
`shuttle_glider2`, 19 under `raptor_pretrain_folder`, 13 under
`shuttle_testes`, 9 under `double_integrator`, and 2 under `shuttle_glider`.
The branch now has 36 active task environments: 9 Iris/Crazyflie, 19 Shuttle
and 8 ShuttleGlider. EasyGlider has no task yet. The experimental copy remains
available for reviewing unselected variants.

| Family | Current experiments | Open comparison |
| --- | --- | --- |
| `iris_crazyflie/` | The nine original double-integrator tasks, including both Isaac Lab baselines and Iris reward ablations. | Retrain if useful as baseline. Learned Lyapunov requires a new value checkpoint. |
| `shuttle/double_integrator/` | Six reward stages and three additional ablations from the former `quadcopter*` tasks. | Verify the new 3.5 kg acceleration-to-force conversion in Isaac Sim. |
| `shuttle/direct_rotor/` | Ten selected RAPTOR-style, observation, reward, termination and trajectory ablations. | Train with the common Shuttle motor preset and compare under matched goals. |
| `easyglider/` | No task. | Implement four-action fixed-wing control and a suitable curriculum. |
| `shuttle_glider/no_surfaces/` | Five-action RAPTOR-style tracking. | Establish the no-surface result. |
| `shuttle_glider/surfaces/curriculum/` | Seven eight-action reward/curriculum variants. | Add a matched no-curriculum surface-control task. |

The table identifies candidates, not an equivalence between similarly named
variants. The double-integrator study uses a three-dimensional acceleration
action for stages 2–7; that differs from four-rotor Shuttle control. The
Shuttle copy changes the vehicle and the acceleration-to-force mass to match
its 3.5 kg USDA. Old
RAPTOR tasks also use `rolling_moment_coefficient` and `motor_time_constant`,
which the cleaned thrusters reject. Selected rotor-control tasks use the new
shared preset instead of translating each legacy motor dictionary.

## Experiment comparison contract

For each selected task, record the USD and physical preset/override, physics
and policy frequencies, observation order and units, action mapping, reward
formula, termination, initial-state distribution, goal generator and curriculum
levels. Give new runs an experiment ID in checkpoint metadata and evaluation
CSVs so the result is tied to a method and physical configuration.
Use identical evaluation trajectories, seeds and vehicle parameters when
comparing reward variants. For the ShuttleGlider, include rotor and surface
commands, airspeed, angle of attack, aerodynamic forces and episode outcome in
the recorder/dashboard contract before interpreting learned flight regimes.
The current `play_multi.py` recorder already exports rotor, surface and
aerodynamic wrench columns; the active dashboard needs corresponding plots.
