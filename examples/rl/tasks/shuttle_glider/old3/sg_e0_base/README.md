# sg_e0_base — control variant (no heading incentive)

## What this is

The **denominator** of the E family. It carries every change of the family
*except* the heading and sideslip rewards, so that any improvement observed in
`sg_e1_align` / `sg_e2_align_lat` can be attributed to those two terms and not
to the rebase, the new reference or the rotor-5 cap.

Base: `shuttle_glider_v3_obs` (**low limits**), not `_9`.

## Changes relative to `shuttle_glider_v3_obs`

| # | Change | Value | Why |
|---|---|---|---|
| 1 | `observation_space` | 28 → **31** | adds `acc_ref_b` (3), the reference acceleration in the body frame. This is the only signal the puller can act on, and it is exactly what the 28D ablation removed. All three E variants share the dimension so the warm-start chain e0 → e1 → e2 works. |
| 2 | `max_rotor_velocity[4]` | 3500 → **1000** | `play_multi.py` hard-codes 1000 for evaluation. Training at 3500 created a train/test mismatch and put the useful operating point at ~4 % of the command range, where `dF/da` is ~10× smaller. |
| 3 | `w_d_action_5` | new, **0.1** | `w_d_action` now covers rotors 1–4 only. With a single `‖d_action‖` over all five dimensions, modulating the puller is punished exactly as much as modulating lift — the behaviour we want was actively discouraged. |
| 4 | `max_lin_vel_per_axis` | 2.0 → **3.0** | the H8 reference itself commands up to 1.40 m/s. At 2.0 there is almost no margin before spurious velocity terminations. |
| 5 | `trajectory.py` | Langevin → **H8 Frenet** | see below. |
| 6 | `trajectory_mixture_langevin_prob` | 0.5 → **0.65** | with the H8 generator this flag means "probability of a *moving* reference"; 35 % of episodes stay static, which is one of the two evaluation conditions. |
| 7 | logging | `Metrics/*` masked | `corr_thrust5_accdem` and `heading_align` are averaged over **moving episodes only**. A static episode has zero demand variance and contributes a forced zero, which halves the reported number. |
| 8 | `w_align`, `w_lateral` | **0.0 / 0.0** | this variant only. |

## The H8 reference

The Cartesian Langevin process fails a kinematic requirement, not a preference:
its demand direction rotates at a median of 17 rad/s (p90 66 rad/s), which
would need ≈47 rad/s² of yaw authority against the **5.92 rad/s²** available.
No reward shaping can repair that.

H8 builds the path in a **Frenet frame** instead: speed and heading rate are two
slow Ornstein–Uhlenbeck processes integrated into a unicycle path, so direction
persistence is a construction property. Three-way mixture: static / frenet /
randomised lemniscate.

It is also **envelope-matched to the test**. Measured on the evaluation
condition (lemniscate A = 1.5 m, T = 10 s, 20 s):

| Quantity | Test demands | H8 provides |
|---|---|---|
| position extent (p95) | 1.50 m | 2.12 m (**2.0×** the area) |
| speed | 0.62 – 1.33 m/s | 0.15 – 1.40 m/s |
| heading rate (p90) | 1.88 rad/s | trackable, `a_need` 3.26 < 5.92 |
| usable rotor-5 force | 1.03 N | **3.76 N** (44 % of cap) |

The previous Frenet reference (family D) covered **9.7×** the test area. Support
coverage was already 100 % at 2×; the excess bought no generalisation and cost
precision where the metric is actually measured.

## Expected result

The policy should fly at least as well as `shuttle_glider_v3_obs` does in the
test (RMSE ≈ 0.18 m on the lemniscate A = 1.5, T = 10), and rotor 5 should stay
**idle** — `Metrics/thrust5_frac` low, `Metrics/heading_align` ≈ 0,
`Metrics/corr_thrust5_accdem` ≈ 0. That is the expected outcome and it is the
point: it is the baseline against which the alignment variants are read.

If this variant does *not* fly well, the problem is the rebase/reference and
not the heading reward, and the alignment variants must not be interpreted.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_e0_base \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

## Prerequisite (blocking)

`examples/rl/train.py` line 182 reads

```python
if env_cfg.vehicle == "Shuttle_glider":
```

It must be

```python
if env_cfg.vehicle.startswith("Shuttle_glider"):
```

Otherwise any vehicle variant falls through to the `else` branch and silently
builds a plain `MultirotorBatch`, which voids the run.
