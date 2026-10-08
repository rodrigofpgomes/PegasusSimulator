# sg_f2_level — level cost + feasibility gate

## What this is

`sg_f1_use` **plus two changes**:

```
w_level    = 0.25   (sg_f1_use: 0.0)
hr_ref_max = 1.5    (sg_f1_use: None)
```

Read `sg_f0_obs/README.md` and `sg_f1_use/README.md` first.

> This is the one variant in the family that changes two things at once. They
> are bundled deliberately because each is a guard on the same failure mode of
> `sg_f1_use` — a bonus that can be collected without genuine alignment. If f2
> beats f1 and you need to know which guard did the work, run the two
> separately afterwards.

## 1. Cost on the rotor-5 level

```python
level_cost = f5_frac * gate
bonus      = w_use * use_signed - w_level * level_cost
```

`w_d_action_5` penalises **changes** to the puller command. Nothing has ever
penalised its **level**. Family E found the obvious optimum:

| | `w5` p50 | of cap | `T5_mean` | CV | sideslip | tilt | signed `T5_along` |
|---|---|---|---|---|---|---|---|
| `sg_e0_base` | **713** | **71 %** | 4.33 N | **0.27** | 86.5° | 7.86° | 0.08 N |

A CV of 0.27 is a near-constant bias. 4.33 N pointing 86° away from the
direction of travel is a **self-inflicted disturbance** that has to be trimmed
out with a permanent 7.9° of tilt — and it is part of why `sg_e0_base` measured
0.267 m against the 0.176 m of `shuttle_glider_v3_obs`. "Using the puller" and
"flying badly" were coupled through a term that was never in the reward.

The pair of weights combines into a threshold:

```
bonus = f5_frac * (w_use * align - w_level)
```

So the puller pays for itself only when `align > w_level / w_use = 0.25`, i.e.
within about 75° of the direction of travel. Below that it costs. Holding it
high in a random direction is no longer free.

## 2. Feasibility gate on the demanded heading rate

```python
# for a planar curve, hr = (v x a)_z / |v|^2, from goal_vel and goal_acc
hr_ref   = (gv_x*ga_y - gv_y*ga_x) / max(gv_x^2 + gv_y^2, 1e-4)
feasible = (abs(hr_ref) < hr_ref_max)
```

The bonus is zeroed while the reference demands a heading rate the airframe
cannot track. Yaw authority is **5.92 rad/s^2**; a T = 6 s lemniscate exceeds it
22 % of the time and a T = 4 s one 60 % of the time. Without the gate, the
policy is offered a reward it cannot collect during the sharp reversals at the
centre of the figure-8, and the pressure it feels there is pure noise.

`hr_ref_max = 1.5 rad/s` sits just above the p50 of the T = 10 test (0.82) and
well below its peak (1.99), so most of the evaluation condition stays inside the
gate.

## Expected result

The informative comparison is **f2 versus f1**:

- If f1 already produces `Metrics/thrust5_signed_N` near 1.0 N with a sane
  `thrust5_frac`, f2 changes little — prefer f1 as the simpler formulation.
- If f1 saturates `thrust5_frac` with a stagnant signed thrust, f2 is the
  variant that breaks the constant-bias strategy, and it should show the best
  RMSE **and** the best signed thrust of the three.
- `Metrics/feasible_frac` reports how much of each episode is inside the gate.
  If it is very low, `hr_ref_max` is too tight and the bonus is rarely on.

If f2 collapses, drop `w_level` to 0.1 first. Raising `hr_ref_max` is the second
knob; changing `w_use` is the last.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_f2_level \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Warm start from the best `sg_f1_use` checkpoint is preferable — the observation
is 34D in both.
