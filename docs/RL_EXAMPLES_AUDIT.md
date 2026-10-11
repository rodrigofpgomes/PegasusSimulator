# RL examples: launcher and experiment audit

This audit compares `examples/rl` on `rl-cleanup` with the experimental copy in
`/home/rodrigogomes/PegasusSimulatorRL/examples/rl`. It is a source review; it
does not establish Isaac Sim training performance or scientific conclusions.

## Order of work

1. Fix the shared vehicle configuration and the train/play contract. Otherwise
   the same checkpoint can be evaluated with different motor dynamics or USD.
2. Select one representative task for each experimental question. Record its
   observation, action, reward, reset distribution, trajectory and physics in
   its README before changing its path or training it again.
3. Reorganize selected tasks by vehicle. Update task IDs in scripts, manifests
   and checkpoint metadata together. Keep historical variants in the source
   copy and Git history until the selected task passes an Isaac Sim smoke test.
4. Consolidate playback, controller adapters and dashboard around the final
   recorded data schema after the task contracts are stable.

The shared physical presets now live in
`pegasus.simulator.logic.vehicles.vehicle_physics`. `get_vehicle_physics()`
returns a deep copy. Task-specific `vehicle_physics_cfg` takes precedence, so
an intentional experimental physics change is retained. Training and single
vehicle playback select the batch class and asset from `env_cfg.vehicle`.
Multi-vehicle playback uses the same preset and replaces it with a task's
explicit physics when evaluating a checkpoint. The active branch has no
Crazyflie USD asset; the Crazyflie physics preset is available, but a
Crazyflie evaluation cannot run until the asset is added and registered.
The seven ShuttleGlider tasks already on this branch contain distinct physical
overrides, including motor time constants and aerodynamic coefficients. Their
overrides remain in place; replacing them with the common baseline would change
the experiments and invalidate comparisons with their existing checkpoints.

## Script dependency map

| Component | Current role | Review finding |
| --- | --- | --- |
| `train.py` | Discovers tasks and algorithm presets, starts Isaac Sim, creates the RL backend and vehicle. | Recursive discovery includes experimental/backup folders; a selected task needs a stable ID. |
| `play.py` | Restores one skrl policy and can compare with LQR and record CSV. | Previously instantiated Iris even for another task; vehicle selection and physics now follow the task. LQR remains an Iris-only comparison. |
| `play_multi.py` | Manifest-driven comparison of controllers in one world. | Uses the task's physics for skrl checkpoints and the shared baseline for external controllers. It also owns trajectories, recorder and reset logic, so extracting those parts needs a separate pass. |
| `utils/skrl_agent_controller.py` | Builds the task environment for inference and restores PPO/PPO2/SAC. | Duplicates task import and preset loading from the launchers. Keep the task environment as the observation/action authority. |
| `utils/foundation_policy_controller.py` and `foundation_policy_pre_controller.py` | Adapt external RAPTOR policies to Pegasus rotor commands. | Keep separate from skrl checkpoint loading; verify rotor order and motor priming against the selected Iris/Crazyflie physics. |
| `utils/lqr_controller_batch.py` and `non_linear_controller.py` | Classical comparison controllers. | Keep where a comparison actually uses them; review their assumptions after task selection. |
| `dashboard.py` | Starts/monitors playback and plots its CSV outputs. | The recorder already writes rotor commands/speeds, surface angles and aerodynamic wrench, but this dashboard does not plot them. The experimental `dashboard_final.py` has matching plots; port only the useful views and keep one dashboard entry point. |

## Task inventory and proposed structure

The experimental copy contains 145 `*_env.py` files: 102 under
`shuttle_glider2`, 19 under `raptor_pretrain_folder`, 13 under
`shuttle_testes`, 9 under `double_integrator`, and 2 under `shuttle_glider`.
The branch currently has 24 task environments, including 7 ShuttleGlider
`vehicle_final` variants. These counts include historical and backup variants;
they are not 145 independent experiments ready to train.

| Future family | Questions to retain | Candidate source |
| --- | --- | --- |
| `iris_crazyflie/` | Initial Isaac Lab and external RAPTOR baselines. | `double_integrator/01_isaac_lab_{iris,crazyflie}`; Iris RAPTOR variants. Crazyflie requires an asset. |
| `shuttle/` | Isaac Lab starting point, quadratic/exponential/death/Lyapunov rewards, then RAPTOR-style pretraining with static and Langevin references. | `double_integrator/02_quadratic` through `07_curriculum_ppo2`, `shuttle_testes`, and `raptor_pretrain_folder`. The `double_integrator` variants must be checked for their declared vehicle before inclusion. |
| `easyglider/` | Future fixed-wing training and a curriculum adapted to aerodynamic flight. | Empty until a task with four actions is implemented and checked. |
| `shuttle_glider/` | Without curriculum, then curriculum; first without surface control, then with all surfaces; compare flight regimes, aerodynamic loads and actuation. | `shuttle_glider/sg_f2_rep*` and `shuttle_glider2/vehicle_final/*`, especially `task5_v1_tracking`, `task5_all7_6`, `c_task5_all5` and the physical reward variants. |

The table identifies candidates, not an equivalence between similarly named
variants. For example, the double-integrator study says stages 2–7 use a
three-dimensional acceleration action; that differs from four-rotor Shuttle
control. Likewise, old RAPTOR tasks use `rolling_moment_coefficient` and
`motor_time_constant`, which the cleaned thrusters reject. Their task physics
must be translated explicitly before they can be run in this branch.

## Experiment comparison contract

For each selected task, record the USD and physical preset/override, physics
and policy frequencies, observation order and units, action mapping, reward
formula, termination, initial-state distribution, goal generator and curriculum
levels. Retain an experiment ID in checkpoint metadata and evaluation CSVs so
that changing a Python package path does not change the meaning of a result.
Use identical evaluation trajectories, seeds and vehicle parameters when
comparing reward variants. For the ShuttleGlider, include rotor and surface
commands, airspeed, angle of attack, aerodynamic forces and episode outcome in
the recorder/dashboard contract before interpreting learned flight regimes.
The current `play_multi.py` recorder already exports rotor, surface and
aerodynamic wrench columns; the active dashboard needs corresponding plots.
