# sg_f3_body — one single frame for every vector

## What this is

`sg_f0_obs` with **one change, in the observation only**:

```python
# sg_f0_obs
obs = [pos_error,   vel_error,   R_flat, ang_b, action_history, rotor_speeds_norm, acc_ref_b, vel_ref_b]
#      ^^ world     ^^ world             ^^ body                                   ^^ body    ^^ body

# sg_f3_body
obs = [pos_error_b, vel_error_b, R_flat, ang_b, action_history, rotor_speeds_norm, acc_ref_b, vel_ref_b]
#      ^^ body      ^^ body              ^^ body                                   ^^ body    ^^ body
```

The reward and the termination bounds are untouched and keep their world-frame
per-axis meaning, so `max_pos_error_per_axis` still means what it always meant.
`observation_space` stays at 34.

## Why this variant exists

Because the family-E and `sg_f0_obs` observation mixes frames: `pos_error` and
`vel_error` are in the world frame while `ang_b`, `acc_ref_b` and `vel_ref_b`
are in the body frame. That is inherited from the RAPTOR-style formulation this
environment descends from.

**It is not a correctness bug.** `R_flat` is in the observation, so the
body-frame quantities are recoverable from the world-frame ones — no information
is lost, and the state remains Markovian. This is the decisive difference from
the defect that actually broke family E, where `goal_vel` was genuinely absent
and therefore genuinely unrecoverable.

But mixing frames has two real costs:

1. **It asks the network to learn a bilinear map.** Recovering a body-frame
   error means computing `R^T . e`, a product of observation components. The
   SAC actor is `Linear(obs, 64) - ReLU - Linear(64, 64) - ReLU`; small MLPs
   represent multiplicative interactions poorly. Supplying the product directly
   costs nothing.
2. **It breaks yaw symmetry.** In the world frame, the same physical situation
   seen at a different yaw angle is a *different* observation, so the task has
   to be learned once per heading. In the body frame it is learned once and
   transfers to every heading. This matters far more for a heading-dependent
   objective than it did for the pure hover task the environment came from — and
   poor yaw generalisation is precisely the symptom being chased.

## The honest counter-argument

`shuttle_glider_v3_obs` uses the mixed-frame observation and reaches **0.176 m**
RMSE in this exact test — the best number measured so far in this whole
investigation. So the mixing is demonstrably not fatal to position tracking, and
it should **not** be presented as the cause of the RMSE gap. This variant is a
hypothesis to be measured, not a diagnosis.

## Expected result

Compare against `sg_f0_obs` only — both have `w_use = w_level = 0`, so this is a
pure representation ablation on flight quality.

- If RMSE improves, rebase `sg_f1_use` and `sg_f2_level` on this variant before
  reading their rotor-5 numbers.
- If it is indifferent, keep `sg_f0_obs` as the base: it is closer to the
  `v3_obs` formulation that produced 0.176 m, and being closer to a known-good
  configuration is worth something.
- If it is worse, that is informative too: it means `R_flat` combined with
  world-frame errors is doing useful work, and the yaw-invariance argument below
  should be dropped rather than pursued.

Yaw behaviour is the secondary thing to watch. `Metrics/heading_align` has no
reward term behind it in this variant, so it should stay near zero; but total
yaw activity and `corr(hr, hr_ref)` are worth logging, because if the body-frame
representation alone makes yaw motion *coherent* rather than merely more
frequent, that is strong support for the symmetry argument.

## The follow-up this variant is a step toward

If the body frame wins, the principled endpoint is a **fully yaw-invariant**
observation: replace `R_flat` (9 dims) with just the gravity direction in the
body frame, `R[:, :, 2]` (3 dims). Every remaining vector is then body-frame and
absolute yaw disappears from the observation entirely. The task is
yaw-equivariant — following a reference and pointing at it are both defined by
*relative* angles — so nothing needed is lost, and the policy can no longer
overfit to particular headings. That is a larger change with real risk of
breaking flight, so it is deliberately not bundled here.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_f3_body \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Cold start required (the observation semantics differ from every earlier
checkpoint, even at the same 34 dimensions).
