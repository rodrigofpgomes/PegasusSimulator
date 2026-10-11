# Shuttle: direct body-force control

All tasks use `vehicle="Shuttle"`, `action_mode="direct_force"`, and a 3.5 kg
vehicle mass. The actuator command bypasses the rotor model. The reward stages
are adapted from the older `quadcopter*` experiments for new Shuttle training.

| Task ID suffix | Source | Main comparison |
| --- | --- | --- |
| `01_isaac_lab` | `quadcopter` | Isaac Lab-style starting reward and four actions. |
| `02_quadratic` | `quadcopter3` | Negative quadratic position and velocity cost. |
| `03_exponential` | `quadcopter3_2` | Exponential reward from the tracking cost. |
| `04_constant_death` | `quadcopter3_3` | Alive reward with death penalty. |
| `05_lyapunov_reward` | `quadcopter5` | Analytic Lyapunov change in the reward. |
| `06_lyapunov_learned` | `quadcopter7` | Learned value/Lyapunov term; requires a newly trained value checkpoint specified by `lyapunov_checkpoint_path`. |
| `ablations/reward_curriculum` | `quadcopter3_4` | Changing reward weights during training. |
| `ablations/reward_curriculum_death` | `quadcopter3_7` | Curriculum with explicit terminal penalty. |
| `ablations/lyapunov_rate_scaling` | `quadcopter6` | Different analytic Lyapunov scaling and additional position/velocity terms. |

The `quadcopter8` environment duplicated `quadcopter7`. Its code and the old
backup directories remain available in Git history; they are not discovered
as active training tasks. The learned Lyapunov task has no bundled checkpoint.
