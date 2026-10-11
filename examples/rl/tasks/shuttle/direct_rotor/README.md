# Shuttle: direct rotor-speed control

These tasks command four rotor speeds through `rotor_velocity_direct`. Every
task uses the same Shuttle motor preset, so the folder names describe the
training-method change instead of a motor-parameter change. The implementations
are adapted from selected `shuttle_testes` and `raptor_pretrain_folder` variants
for new Shuttle training. Variants originally written for Iris are identified
below; their results must be rerun on Shuttle.

| Task ID suffix | Source | Main difference |
| --- | --- | --- |
| `baseline/raptor_tracking` | `shuttle_pretrain` | RAPTOR-style tracking baseline. |
| `observation/body_reference_v3` | `shuttle_v3_obs` | Revised tracking observation. |
| `termination/wide_bounds` | `shuttle_v3_obs_05` | Wider position and velocity termination bounds. |
| `reward/orientation_weight_01` | `shuttle_w01` | Orientation weight 0.1. |
| `reward/orientation_weight_03` | `shuttle_w03` | Orientation weight 0.3. |
| `reward/velocity_tracking` | Iris `raptor_pretrain4_5_w_v` | Adds a velocity cost to the RAPTOR-style reward. |
| `reward/angular_rate` | Iris `raptor_pretrain6_2` | Adds an angular-rate cost. |
| `termination/death_penalty_200` | Iris `raptor_pretrain_mod` | Terminal penalty 200. |
| `trajectory/duration_2p5` | `v3_obs` | Reference segment lasts 2.5 s. |
| `trajectory/duration_5p0` | `v3_obs2` | Reference segment lasts 5.0 s. |

The source code for the 19 Iris RAPTOR variants and 13 Shuttle test variants
remains in `/home/rodrigogomes/PegasusSimulatorRL/examples/rl/tasks`. This
folder contains the variants selected for the next comparison, not every
intermediate test from that history.

The RAPTOR-derived and most `shuttle_testes` variants provide a SAC preset.
Use `--algo sac` for those tasks. The two trajectory-duration variants also
provide PPO and PPO2 presets.
