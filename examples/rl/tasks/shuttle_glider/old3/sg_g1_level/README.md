# sg_g1_level -- remove the cancelled excess thrust (WARM START from sg_f2_level)

## What changed relative to the parent
**One line.** `w_level: 0.25 -> 0.5`. Everything else is `sg_f2_level` exactly
(`w_use = 1.0`, `w_align = 0.0`, `hr_ref_max = 1.5`).

## The problem it targets
f2 delivers **2.51 N** of signed longitudinal thrust where the test only demands **1.03 N**.
There is no fuselage drag in the simulation, so the excess **must** be trimmed out by the
quad pitching back. Consistent with that, `tilt_RMSE` rose from 6.93 deg (f0) to 8.07 deg (f2).

Part of that force is genuinely useful. Part may be the original pathology in new clothes --
a standing thrust bias, now merely *aligned* with travel instead of sideways. This must not
be reported as fully resolved until it is separated.

## The mechanism
Net bonus is `f5_frac * (w_use * align - w_level)`, so the puller pays for itself only when

```
align > w_level / w_use
```

| | threshold | i.e. within |
|---|---|---|
| f2 (`w_level` 0.25) | 0.25 | ~75 deg of travel |
| **g1 (`w_level` 0.5)** | **0.50** | **~60 deg of travel** |

The rotor becomes twice as expensive to hold on, so the optimum shifts towards firing it
less and only when better aligned -- pushing the delivered force towards the ~1.03 N the
task actually needs.

The direction is already supported: f1 (`w_level = 0`) delivered 4.29 N at 74 % efficiency,
while f2 (`w_level = 0.25`) delivered less total thrust (3.09 N) at **higher** efficiency
(81 %) and much better flight quality (`att_ff` 0.165 vs 0.760). g1 extends that trend by
one step.

## Expected result
`T5_mean` falls from 3.09 N towards ~1.5-2.0 N, `signed_T5 %` rises above 81 %,
`tilt_RMSE` falls back from 8.07 deg towards f0's 6.93 deg, and test RMSE holds or improves.

## Pre-registered reading criterion
PASS requires **both**:
- `signed_T5` stays **>= 1.0 N** (the criterion floor -- do not overshoot into switching the
  rotor off entirely)
- `tilt_RMSE` falls **below 7.5 deg** while test RMSE does not degrade by more than 0.05 m

## Failure mode to watch for
If `w_level = 0.5` makes `signed_T5` collapse towards zero, the term has gone from pricing
the rotor to banning it, and 0.35-0.4 is the range to explore instead. The `f5_frac` by
alignment band table is the diagnostic: the > 0.5 band should stay high (~0.8), only the
lower bands should fall.

## Shared context (all four tasks)

Observation is **34D**, identical to family F:
`[pos_error(3), vel_error(3), R_flat(9), ang_b(3), action_history(5), rotor_speeds_norm(5), acc_ref_b(3), vel_ref_b(3)]`

Reward:
```
cost    = clamp(w_pos*|e_p| + w_vel*|e_v| + w_d_action*|da[:4]| + w_d_action_5*|da[4]|,  max=cost_clip)
shaping = w_align * 0.5*(1 - align)*gate  +  w_lateral * |v_lateral_b|*gate
bonus   = w_use * f5_frac*align*gate*feasible  -  w_level * f5_frac*gate
reward  = constant - cost - shaping + bonus
```
`shaping` and `bonus` are applied **outside** the clip. Putting them inside saturates
`cost_clip` and zeroes the position gradient -- that is the mechanism that broke the v4
heading environments.

Reference generator: `trajectory.py` (H8 Frenet family, `sigma_z = 0.05`). Each task
carries its own copy; edit the copy inside the task folder you are running.

### Install and launch
```bash
cp -r sg_f0_cont sg_g0_break sg_g1_level sg_f2_rep \
      examples/rl/tasks/shuttle_glider/

bash examples/rl/tasks/shuttle_glider/launch_parallel_g.sh
```

### Prerequisite, not yet applied
`examples/rl/train.py` line 182 is `if env_cfg.vehicle == "Shuttle_glider":` and must be
`.startswith("Shuttle_glider")`. Without it any vehicle variant falls through to the
`else` branch and builds a `MultirotorBatch` with no rotor 5. Verify with:
```bash
grep -n 'env_cfg.vehicle' examples/rl/train.py
```

### Reading the results
Use `signed_T5 %` (= `T5_along / T5_mean`), **not** `heading_RMSE` and **not**
`corr_thrust5_accdem`. Both of the latter were retracted: `heading_RMSE` ignores that the
policy only fires the rotor while aligned (f2 reads 95 deg mean but 36 deg thrust-weighted),
and `corr_thrust5_accdem` correlates thrust against the *acceleration* demand while the
policy actually modulates against *velocity* (it reads ~0 while `corr(T5, v_forward)` is +0.92).
