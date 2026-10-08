# sg_e2_align_lat — heading incentive + sideslip penalty

## What this is

`sg_e1_align` **plus one change**: a penalty on lateral body velocity.

```
w_align   = 1.5     (same as sg_e1_align)
w_lateral = 0.25    (sg_e1_align: 0.0)
```

Read the `sg_e0_base` README for the shared family changes and the
`sg_e1_align` README for the heading term.

## Why this term exists

`w_align` constrains the **nose direction**. It does not constrain the vehicle
to actually *travel* along the nose. A quadrotor can hold the nose on the
reference heading while crabbing sideways, because horizontal force comes from
tilting and tilting is free — there is no fuselage drag in the simulation, so
sideways flight costs nothing.

That matters because it is a way to satisfy `w_align` without ever needing
rotor 5. Measured kinematic sideslip on the family-D runs was **73–92°**, with
forward body velocity of only 0.02–0.27 m/s against a total speed of ~0.9 m/s:
the airframe was flying almost entirely sideways.

```python
vel_b        = Rᵀ · vel_world
lateral_cost = |vel_b[:, 1]| * gate          # same moving-reference gate
```

The two terms together define "fly like an aircraft": nose on the velocity
vector *and* velocity along the nose. Only then is a forward-only thruster
physically useful.

## Why the weight is 0.25 and not 1.5

`lateral_cost` is in m/s and is **unbounded**, unlike `align_cost ∈ [0, 1]`. At
the reference's top speed of 1.40 m/s a full 90° crab produces a cost of 1.40,
so `w_lateral = 0.25` contributes at most ≈0.35 — deliberately smaller than the
heading term. It is a tie-breaker, not a primary objective.

Like `align_cost`, it is subtracted **outside** the cost clip.

## Expected result

The informative comparison is **e2 versus e1**, not e2 versus e0:

- If e1 already produces low sideslip, e2 changes little and the extra term is
  unnecessary — prefer e1 as the simpler formulation.
- If e1 shows high alignment but sideslip stays large and `thrust5_N` stays
  low, e2 is the variant that closes the loop, and it should show the largest
  `Metrics/corr_thrust5_accdem` of the three.
- If e2 collapses (position error grows, episodes terminate early), the total
  shaping budget is too large relative to `cost_clip = 3.0`. Lower `w_lateral`
  to 0.1 before touching `w_align`.

`Metrics/heading_align` should be **at least as high** as in e1. If it is
lower, the two terms are fighting and the weights need rebalancing.

## Run

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_e2_align_lat \
    --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

Warm start from the best `sg_e1_align` checkpoint is preferable — the
observation is 31D in both.

## One change at a time

e0 → e1 → e2 differ by exactly one weight each. Do not adjust anything else
while the comparison is running, or the family loses its ability to attribute
the result.
