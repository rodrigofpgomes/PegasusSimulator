# sg_e1_align — heading incentive

## What this is

`sg_e0_base` **plus one change**: a reward term that asks the nose to point
along the direction of travel.

```
w_align = 1.5    (sg_e0_base: 0.0)
```

Everything else is identical to `sg_e0_base` — same 31D observation, same H8
reference, same rotor-5 cap of 1000, same `w_d_action_5 = 0.1`. Read that
README first for the shared changes.

## Why this term exists

Measured on the family-D runs (complete 20 s lemniscate episodes): heading
alignment between the nose and the direction of travel was **−0.068 to +0.058**
— i.e. indistinguishable from random. `|heading error| < 20°` happened 9.0–14.7 %
of the time, against an 11 % random rate. Total yaw travel over 20 s was
**4–17°**: the airframe is not actively pointing the wrong way, yaw is simply a
**free degree of freedom** that nothing in the reward ever constrained.

Rotor 5 pushes along body **+X** with thrust ≥ 0. If the nose sits ~90° off the
direction of travel, its thrust projects onto the demand with a factor of ~0,
and no amount of capacity or training time can make it useful. Making the
reference trackable (family D) was necessary but not sufficient:

> **Possibility is not incentive.**

## The term

```python
fwd_hat   = normalise(R[:, :2, 0])            # nose, world XY
ref_hat   = normalise(goal_vel[:, :2])        # demanded direction of travel
gate      = (‖goal_vel_xy‖ > 0.15)            # moving reference only
align     = clamp(fwd_hat · ref_hat, -1, 1)
align_cost = 0.5 * (1 - align) * gate         # 0 aligned .. 1 backwards
```

Three design details that matter:

1. **The gate threshold is 0.15 m/s, the same value `play_multi.py` uses for
   `ref_speed_xy`.** Reward and reported metric therefore agree, and static
   episodes (where the demanded direction is undefined) are excluded.
2. **The cost is bounded in [0, 1]**, so `w_align = 1.5` is directly comparable
   with `w_pos = 1.0` on a 1 m position error.
3. **It is subtracted OUTSIDE the cost clip:**

   ```python
   cost    = clamp(w_pos*pos + w_vel*vel + w_d_action*d4 + w_d_action_5*d5, max=cost_clip)
   shaping = w_align*align_cost + w_lateral*lateral_cost
   reward  = constant - cost - shaping
   ```

   Inside the clip the new term would saturate `cost_clip = 3.0` and zero the
   position gradient. **That is exactly the mechanism that broke the v4 heading
   environments** — the weight was never the problem.

## Expected result

Read in this order:

1. `Metrics/heading_align` must rise clearly above 0. This is the term's own
   objective; if it does not move, the term is too weak or the gate is wrong,
   and nothing downstream is interpretable.
2. `Metrics/thrust5_N` and `Metrics/thrust5_frac` should then rise, because
   pushing along +X finally helps.
3. `Metrics/corr_thrust5_accdem` should follow. Note the **structural ceiling**:
   for any bounded periodic reference, longitudinal demand is positive only
   ~50 % of the time and rotor-5 thrust cannot be negative, so the maximum
   plausible correlation is ≈0.5–0.7, not 1.0.
4. `Episode_Reward/pos` and the test RMSE will probably get slightly **worse**
   than `sg_e0_base`. Some degradation is the honest price of the constraint;
   the question is how much. Compare against e0 in the same test.

## Run

Cold start:

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_e1_align \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Warm start from the best `sg_e0_base` checkpoint is preferable once e0 has
converged — the observation is 31D in both, so the network transfers directly.
