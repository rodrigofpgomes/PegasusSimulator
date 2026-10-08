# sg_f0_cont -- budget control (RESUME from sg_f0_obs)

## What changed relative to the parent
**Nothing.** Byte-identical configuration to `sg_f0_obs`
(`w_use = 0.0`, `w_level = 0.0`, `w_align = 0.0`, `hr_ref_max = None`).

This is not a new experiment. It is the **missing control**.

## Why it exists
`sg_f1_use` and `sg_f2_level` were warm-started from the best `sg_f0_obs` checkpoint, so
they accumulated **2 M steps against f0's 1 M**. Worse, f0 spent its first 100 k steps
learning not to crash (reward -95, `died` 0.598, episode length 254) while f1/f2 inherited
that for free and spent their whole 1 M on refinement.

So the most interesting claim about family F -- "adding the rotor-5 objective improved
position tracking" (0.288 / 0.348 against 0.532) -- **is confounded with training budget
and cannot currently be interpreted.**

f0 is also the only one of the four that had **not** plateaued: end-slope of total reward
**+28.9 +/- 8.2** per 100 k steps (3.5 sigma), last four deciles 183 -> 170 -> 192 -> 228,
`died` still falling 0.208 -> 0.159, episode length still rising 423 -> 441. The other three
read -4.7 +/- 9.5, +0.2 +/- 9.6 and +8.2 +/- 8.2 -- all flat.

## Expected result
Test RMSE improves from 0.532 towards ~0.35-0.45 with `signed_T5` staying near **0.01 N**.

## How to read it
- **If f0_cont reaches ~0.30 m**: the f1/f2 tracking gain was **budget**, not the bonus.
  The rotor-5 result stands on its own (0.01 -> 2.51 N), but stop claiming the bonus
  improves tracking.
- **If f0_cont stays near 0.50 m**: the bonus genuinely improved tracking, which is the
  more interesting outcome and is worth stating explicitly in the thesis.

Either way `signed_T5` must stay near zero. If it rises without any `w_use`, something is
wrong with the measurement, not with the policy.

## Caveat
Rising training reward is **not** the same as falling test RMSE. f0 already flew worse in
the test than its training reward suggested. This task measures the test, not the curve.

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
