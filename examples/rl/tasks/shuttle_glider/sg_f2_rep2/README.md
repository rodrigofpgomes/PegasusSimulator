# sg_f2_rep -- seed replicate of sg_f2_level (WARM START from sg_f0_obs, seed 11)

## What changed relative to the parent
**Nothing in the configuration.** Byte-identical to `sg_f2_level`. The only difference is
the launch flag: `--seed 11` instead of `--seed 10`.

Warm-start from the same `sg_f0_obs` best checkpoint, to reproduce the f2 lineage exactly.

## Why it exists
**Every number in this investigation comes from a single seed.** Evaluation is deterministic
(RMSE is identical to three decimal places across the three test episodes), so the three
episodes carry no statistical information whatsoever -- n = 1, not n = 3.

Without this replicate there is no way to know whether a difference between 0.288 and, say,
0.31 means anything at all. It costs one slot and converts every other comparison in the
family into an interpretable result.

It also probes a second question: none of the four F runs converged (`died` 0.157-0.218,
mean episode length 408-439 of 500 -- roughly 15 % still dying at 1 M steps). Two runs of
the same configuration bound how much of that residual instability is seed-dependent.

## Expected result
Test RMSE within roughly +/- 0.05 m of 0.288 and `signed_T5` within +/- 0.5 N of 2.51 N.

## How to read it
- **Spread below ~0.05 m**: the family F ordering (f2 < f1 < f0 ~ f3) is real, and the g0/g1
  criteria above can be applied as written.
- **Spread above ~0.10 m**: single-seed differences of that size are noise. The f0-vs-e0
  regression (0.532 vs 0.267) becomes unattributable, the g0/g1 RMSE tolerances must be
  widened accordingly, and **any headline claim needs >= 3 seeds** before it goes in the
  thesis.

The 0.08 -> 2.51 N jump in signed thrust is two orders of magnitude with an identified
mechanism, so it is very unlikely to be seed noise. The **tracking** numbers are the ones
at risk.

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
