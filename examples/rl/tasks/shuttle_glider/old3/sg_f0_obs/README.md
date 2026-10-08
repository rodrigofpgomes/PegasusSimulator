# sg_f0_obs — the observability fix, alone

## What this is

`sg_e0_base` with **two changes and no new reward term**:

1. `observation_space` 31 → **34**: adds `vel_ref_b`, the reference velocity in
   the body frame.
2. `sigma_z` 0.20 → **0.05** in the H8 reference.

`w_use = 0.0`, `w_level = 0.0`, `hr_ref_max = None`, and
`w_align = w_lateral = 0.0` as in `sg_e0_base`. This variant exists to price
those two changes on their own, before any rotor-5 incentive is switched on.

## Why `vel_ref_b` — the defect that made family E fail

The family-E observation was

```
[pos_error(3), vel_error(3), R_flat(9), ang_b(3), action_history(5),
 rotor_speeds_norm(5), acc_ref_b(3)]
```

`vel_error = v - goal_vel`. Neither `v` nor `goal_vel` appears anywhere else, so
**`goal_vel` is not recoverable from the observation**. The heading reward was

```
align = fwd_hat · normalise(goal_vel_xy)
```

i.e. a function of a quantity the policy could not see. The only partial
information available was `acc_ref_b`, and for a smooth path the acceleration is
roughly perpendicular to the velocity — it locates the direction of travel only
to within ±90°, with a sign ambiguity.

The measured signature matches that diagnosis exactly. `w_align = 1.5` raised
yaw *activity* without producing any yaw *tracking*:

| variant | total yaw path / 20 s | net yaw | achieved `hr_rms` | demanded `hr_rms` | `corr(hr, hr_ref)` |
|---|---|---|---|---|---|
| `sg_e0_base` (`w_align` 0) | 158° | 26° | 0.19 | 1.12 | +0.07 |
| `sg_e1_align` (`w_align` 1.5) | 331° | 40° | 0.39 | 1.12 | **−0.06** |
| `sg_e2_align_lat` | 384° | 19° | 0.42 | 1.12 | **−0.10** |

The gradient existed; the signal did not.

`vel_ref_b` closes that gap, and it does so in a directly usable form: `cos` and
`sin` of the heading error are `vel_ref_b[0]` and `vel_ref_b[1]` divided by the
xy norm. There is no rotation left for the network to learn. It also supplies
the reference speed as feedforward — which is the channel of the along-track
error that dominated `sg_e1_align` (0.565 m along-track against 0.411 m
cross-track).

## Why `sigma_z` 0.05

The vertical error was **0.131 m of the 0.267 m** total RMSE in `sg_e0_base`.
It comes from the vertical noise in the training reference, while the evaluation
condition holds z constant at 1.5 m. That accuracy was being spent tracking a
disturbance the test never applies.

## Expected result

RMSE should drop **below** the 0.267 m of `sg_e0_base` and move toward the
0.176 m that `shuttle_glider_v3_obs` achieves in the same test. Most of the gain
should appear in the vertical component.

Rotor 5 should stay near-idle, or hold the same lazy constant bias as in family
E — there is still nothing rewarding its use here. If `Metrics/thrust5_frac`
settles high again while `Metrics/thrust5_signed_N` stays near zero, that
confirms the level-cost diagnosis that `sg_f2_level` acts on.

If this variant does **not** beat `sg_e0_base`, stop: the extra three
observation dimensions are costing more than they buy, and `sg_f1_use` /
`sg_f2_level` must not be interpreted.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_f0_obs \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Note: the observation is 34D, so family-E checkpoints (31D) **cannot** be
warm-started into this family. Cold start is required for `sg_f0_obs`; the other
F variants can then warm-start from it.
