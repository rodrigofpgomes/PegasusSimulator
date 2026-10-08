# sg_f1_use — signed useful-thrust bonus

## What this is

`sg_f0_obs` **plus one change**:

```
w_use = 1.0    (sg_f0_obs: 0.0)
```

Read `sg_f0_obs/README.md` first for the observability fix that both variants
share.

## The term

```python
f5_frac    = (w5 / w5_max)**2                 # = f5 / f5_max, in [0, 1]
align      = clamp(fwd_hat . ref_hat, -1, 1)  # cos(heading error)
gate       = (norm(goal_vel_xy) > 0.15)       # moving reference only
use_signed = f5_frac * align * gate           # in [-1, 1]

bonus  = w_use * use_signed
reward = constant - cost - shaping + bonus
```

The bonus is **added outside the cost clip**, for the same reason the shaping is
subtracted outside it: inside, it would saturate `cost_clip = 3.0` and zero the
position gradient. That is the mechanism that broke the v4 heading environments.

## Why a signed bonus instead of family E's heading penalty

Three reasons, all measured.

**1. It rewards the quantity that actually matters.** Family E penalised the
geometry (`0.5(1 - align)`) and hoped rotor-5 usage would follow. It did not.
This term pays directly for `f5 * cos(heading error)`, which is literally what
the evaluation reports as `thrust_5_along_ref` — and what the test needs to
reach ≈**1.03 N**.

**2. It pays for partial alignment and for modulation, and never demands a
nose-lock.** That matters because a nose-lock is physically impossible at speed.
Demanded yaw acceleration for the Gerono lemniscate, against the 5.92 rad/s^2
available:

| T | max heading rate | `hacc` p90 | `hacc` max | % of time infeasible |
|---|---|---|---|---|
| 10 s | 1.99 | 2.40 | 2.68 | **0 %** |
| 8 s | 2.49 | 3.75 | 4.18 | **0 %** |
| 6 s | 3.32 | 6.65 | 7.42 | **22 %** |
| 4 s | 4.98 | 14.93 | 16.67 | **60 %** |

Below T ≈ 7 s, asking for nose-on-velocity is asking for the impossible, and a
geometric penalty charges for it anyway. A signed bonus simply earns less.

**3. The signed form cannot be gamed by a constant bias.** Family E's headline
number came from the evaluation harness's
`thrust_5_useful = max(T5 * align, 0)`. With `align ~ 0` and `T5` large, that
rectifier returns a positive mean **by construction** — only the positive half
of the noise survives. `sg_e0_base` scored 1.45 N of "useful" thrust while its
signed value was **0.08 N**. Never optimise the rectified version.

| variant | signed `T5_along` | `T5_mean` | `alignment` | RMSE |
|---|---|---|---|---|
| `sg_e0_base` | 0.08 N | 4.33 N | −0.014 | 0.267 |
| `sg_e1_align` | **0.29 N** | 2.11 N | 0.063 | 0.708 |
| `sg_e2_align_lat` | 0.03 N | 0.72 N | 0.143 | 0.560 |

Against the 1.03 N the test demands, none of them is close.

## The encouraging part of family E

`sg_e0_base` reached `corr(T5, a_long_body) = +0.662`: it *did* modulate the
puller against the longitudinal component of the demand, in its own arbitrary
heading frame. The mechanism was already being learned — what was missing was a
frame worth modulating in. That is what `vel_ref_b` supplies and what this bonus
pays for.

## Expected result

Read in this order:

1. `Metrics/thrust5_signed_N` must rise clearly above zero. It is the term's own
   objective. Target ≈**1.0 N**.
2. `Metrics/heading_align` should rise as a *consequence*, not as a target — the
   policy aligns because alignment is what makes the bonus payable.
3. `Metrics/corr_thrust5_accdem` should follow. Structural ceiling ≈0.5–0.7:
   longitudinal demand is positive only ≈50 % of the time for a bounded
   periodic reference, and rotor-5 thrust cannot be negative.
4. RMSE will probably rise somewhat above `sg_f0_obs`. That is the honest price
   of the constraint. Compare against `sg_f0_obs`, not against family E.

Failure mode to watch: the bonus is maximised by holding `f5_frac` high whenever
`align > 0`, which can reproduce family E's constant-bias behaviour in a
half-rectified form. If `Metrics/thrust5_frac` saturates while
`Metrics/thrust5_signed_N` stagnates, that is exactly what `sg_f2_level`
addresses. If instead the position error collapses, halve `w_use` to 0.5 before
changing anything else.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_f1_use \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Warm start from the best `sg_f0_obs` checkpoint is preferable once f0 has
converged — the observation is 34D in both. Family E learned its shaping term
from step zero and never acquired the base flying skill; the curriculum is meant
to avoid repeating that.
