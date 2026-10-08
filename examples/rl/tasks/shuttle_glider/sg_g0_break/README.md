# sg_g0_break -- break the gradient dead zone (WARM START from sg_f2_level)

## What changed relative to the parent
**One line.** `w_align: 0.0 -> 0.3`. Everything else is `sg_f2_level` exactly
(`w_use = 1.0`, `w_level = 0.25`, `hr_ref_max = 1.5`).

## The problem it targets
The f2 bonus is `w_use * f5_frac * align`, so

```
d(bonus)/d(yaw)  is proportional to  f5_frac
```

Measured `f5_frac` by alignment band (f2, episode 0):

| alignment | < -0.5 | -0.5..0 | 0..0.5 | > 0.5 |
|---|---|---|---|---|
| `f5_frac` | **0.009** | 0.025 | 0.237 | 0.804 |
| % of time | 24.6 % | 23.5 % | 11.2 % | 40.9 % |

When badly misaligned the policy correctly switches the rotor off -- and **all yaw gradient
vanishes with it**. Aligning becomes free and inconsequential: a local optimum. Escaping
would require simultaneously turning the rotor on (paying `w_level`) and rotating, with no
immediate benefit from either step alone.

This is the mechanism behind the observed "it tries to turn towards the trajectory but
cannot, and flies backwards". Time-window decomposition (f2, mean of 3 episodes):

| | 0-2 s | 2-5 s | 5-10 s | 10-20 s |
|---|---|---|---|---|
| alignment | +0.82 | -0.30 | +0.06 | +0.12 |
| % backwards | 0 % | **80 %** | 57 % | 53 % |
| `f5_frac` | 0.83 | 0.07 | 0.37 | 0.36 |

The first 2 s are the *best* window: the vehicle spawns aligned with world +x, which is the
lemniscate's initial heading. The collapse is at 2-5 s, where the curve turns.

## Why w_align works now and failed in family E
Identical term, `align_cost = 0.5*(1 - align)*gate`, already present in the code and
subtracted outside the clip. Two differences:

1. **It is now observable.** The family E observation carried only `vel_error = v - goal_vel`,
   from which `goal_vel` is *not* recoverable without `v`, which is also absent. E was
   graded on a quantity it could not see. `vel_ref_b` (family F) fixed exactly this.
2. **The weight is 0.3, not 1.5.** E used 1.5 and destroyed flight quality (RMSE 0.709).

Crucially, `align_cost` is **independent of `f5_frac`**, so the yaw gradient no longer
switches itself off.

## Expected result
Mean alignment rises from 0.109 towards 0.4-0.6, backwards flight in the 10-20 s window
falls from 53 %, and `signed_T5` holds at or above f2's 2.51 N (better alignment should make
the rotor *more* useful, not less).

## Pre-registered reading criterion
PASS requires **all three**:
- mean alignment (not thrust-weighted) rises 0.109 -> **> 0.4**
- % backwards in the 10-20 s window falls 53 % -> **< 25 %**
- test RMSE degrades by **no more than 0.05 m** from f2's 0.288

## Falsifiable prediction
If alignment does not move, the correct reading is that the local optimum was **inherited
from the f2 parent** -- which holds a converged strategy of switching the rotor off when
misaligned, precisely what `w_align` must overturn. The follow-up is then `g0` **cold**.

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
