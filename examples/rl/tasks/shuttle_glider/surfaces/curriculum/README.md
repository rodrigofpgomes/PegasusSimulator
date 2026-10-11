# Surface-control curriculum ablations

| Task ID suffix | Source variant | Main comparison |
| --- | --- | --- |
| `reward_tracking` | `task5_v1_tracking` | Tracking reward. |
| `reward_physical` | `task5_v2_physical` | Adds physical flight terms. |
| `reward_offload` | `task5_v3_offload` | Encourages aerodynamic offload. |
| `reward_offload_custom_aero` | `task5_v3_offload2` | Offload reward with explicit aerodynamic coefficients. |
| `surface_shaping` | `task5_all7_6` | Surface-use shaping. |
| `cost_shaping_v4` | `c_task5_all4` | Earlier cost-shaping variant. |
| `cost_shaping_v5` | `c_task5_all5` | Later cost-shaping variant. |

All seven tasks use the shared ShuttleGlider motor preset and retain their
original observation, action, reward and curriculum implementation. The
individual task READMEs contain the full observation and reward contracts.
