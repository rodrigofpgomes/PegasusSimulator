# sg_d4_com — remove the moment — puller at centre-of-mass height

A **family D** variant of the `Shuttle_glider` task, derived from `shuttle_glider_v3_obs_9`. Goal of the family: find out whether the policy can learn to use the 5th rotor (the glider's puller) usefully, instead of leaving it at a constant bias that the four quadrotor rotors then cancel out.

## What this variant does

Test whether what kills rotor 5 is purely geometric. The puller sits 0.3475 m below the centre of mass, so every newton of thrust costs 0.3475 N.m of pitch that has to be trimmed out continuously, consuming attitude headroom and therefore **worsening** `pos_cost`. This env uses an alternative USD with the puller raised to CoM height (z = -0.245 -> z = +0.10253066), where thrust becomes pure force with no moment.

## What changes relative to the parent (`shuttle_glider_v3_obs_9`)

- everything `sg_d3_cap` changes
- `vehicle`: `Shuttle_glider` -> **`Shuttle_glider_com`**

Everything else is inherited unchanged: `w_pos = 1.0`, `w_vel = 0.3`, `w_d_action = 1.0`, `constant = 1.5`, `termination_penalty = 200.0`, `cost_clip = 3.0`, envelope `max_pos_error_per_axis = 3.0` / `max_lin_vel_per_axis = 8.0` / `max_ang_vel_per_axis = 35.0`, `trajectory_mixture_langevin_prob = 0.5`, `langevin_omega = 2.0`, `episode_length_s = 5.0`, `sim_dt = 0.01`.

## Expected result

This is a diagnostic, not a solution: the real vehicle has the puller where it has it. If **only** d4 shows rotor-5 usage, the conclusion is that the force-moment coupling is the dominant obstacle and that no observation or reward formulation will fix it — it will have to be solved with pitch authority, or by accepting a different trim attitude. If d3 and d4 read the same, geometry was not the problem.

### Prerequisites specific to this variant

1. `shuttle_glider_com.usda` in `extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Shuttle/`, with the puller raised in **two** places: the `xformOp:translate` of the `/vehicle/rotor_puller` prim **and** the `physics:localPos0` of `/vehicle/rotor_puller_joint`. Editing only the first leaves the PhysX anchor at -0.245 and the test is silently invalid.
2. A `"Shuttle_glider_com"` entry in the `ROBOTS` dict in `extensions/pegasus.simulator/pegasus/simulator/params.py`.
3. **`examples/rl/train.py`, line ~182:** change `if env_cfg.vehicle == "Shuttle_glider":` to `if env_cfg.vehicle.startswith("Shuttle_glider"):`. Without this the env falls into the `else` branch, instantiates `MultirotorBatch` instead of `ShuttleGliderBatch`, and rotor 5 does not even exist.


## Effective configuration

| field | value |
|---|---|
| `observation_space` | `31` |
| `action_space` | `5` |
| `vehicle` | `Shuttle_glider_com` |
| `max_rotor_velocity` | `[1400, 1400, 1400, 1400, 1000]` |
| `w_d_action_5` | `0.10` |
| `w_tilt` | `2.0` |
| `trajectory_mixture_langevin_prob` | `0.5` |
| `langevin_sigma` | `6.0` |

### Rotor-5 capacity at this cap

- maximum force: `8.549e-06 * 1000^2` = **8.5 N** (18% of the 48.8 N weight)
- force at the neutral action (`a4 = 0`, i.e. half the cap): **2.1 N** (4% of weight)
- corresponding pitch moment, with a 0.3475 m arm: **0.74 N.m** (total quadrotor authority is ~15.5 N.m)

## Observation (31D)

`[pos_error(3), vel_error(3), R_flat(9), ang_b(3), action_history(5), rotor_speeds_norm(5), acc_ref_b(3)]`

`acc_ref_b` is the reference acceleration projected into the body frame:

```python
acc_ref_b = torch.bmm(R.transpose(1, 2), self.reset_manager.goal_acc.unsqueeze(-1)).squeeze(-1)
```

## Reward

```python
pos_cost      = ||pos_error||
vel_cost      = ||vel_error||
d_cost        = w_d_action * ||d_action[:, :4]|| \
              + w_d_action_5 * |d_action[:, 4]|    # split
tilt_cost     = acos(R[2, 2].clamp(-1 + 1e-6, 1 - 1e-6))   # radians
cost          = w_pos*pos_cost + w_vel*vel_cost + d_cost + w_tilt*tilt_cost
cost          = cost.clamp(max=cost_clip)
reward        = constant - cost
reward[died]  = -termination_penalty
```

With `w_tilt = 2.0` and `acos`, a 7.4 degree tilt (0.129 rad) costs **0.26**, against a typical `pos_cost` of ~0.5.

## Reference demand, and why sigma = 6.0

`trajectory.py` integrates `acc = -gamma*v - omega^2*(x - centre) + sigma*noise`, storing that same `acc` into `acc_traj`. That stored value is the exact discrete acceleration of the reference (verified to 1e-14), so the white-noise term is not an artefact to be filtered out — it is what generates the trajectory, and passing it to the policy provides a genuine one-step preview of where the reference is going.

What matters for rotor 5 is not the RMS of that acceleration but its **sustained** part. Rotor 5 is slow and every newton costs 0.3475 N.m of pitch to trim continuously, so demand that reverses every few control steps is served better by the four fast rotors. Only ~22-24% of the raw acceleration RMS survives a 0.2 s moving average.

Measured by simulating the generator's exact integrator, at `sigma = 6.0`, `gamma = 1.0`, `omega = 2.0`, `dt = 0.01`:

- `sigma_v` = 0.424 m/s (stationary velocity spread of the reference)
- **sustained** horizontal demand = 2.078 m/s^2 -> **10.34 N (21.2% of weight)**
- trim tilt required if the quadrotor alone supplies it: **12.0 degrees**

For comparison, the evaluation lemniscate `A = 1.5 m, T = 10 s` (Gerono: `x = A sin(wt)`, `y = (A/2) sin(2wt)`, so the Y axis accelerates at twice the frequency) demands a sustained 0.952 m/s^2 = 4.74 N (9.7% of weight), and is ~100% sustained because it is a 0.1 Hz sinusoid with no fast component.

So `sigma = 6.0` places the sustained training demand at **2.18x** the demand of the condition the policy is evaluated on. That is a deliberate margin rather than a match: the previous `sigma = 9.0` overshot by ~3.3x and cost real tracking accuracy (the measured sweep gives RMSE 0.214 at sigma 2, 0.249 at sigma 4, 0.389 at sigma 9), while a matched sigma of ~2.7 would leave the policy with no headroom outside the training region — which was one of the observed failure modes.

The margin also has a specific virtue for the capped variants. At this sigma the sustained demand is 10.34 N, against the 8.55 N that rotor 5 can deliver at `cap = 1000`. Rotor 5 can therefore supply about 83% of the sustained demand: enough to matter, but not so much that holding it at full throttle would suffice — which would simply reproduce the constant bias this whole family exists to eliminate.

`trajectory_mixture_langevin_prob` stays at **0.5**, unchanged from the parent and from the entire measured sigma sweep: 50% of episodes use a Langevin reference and the rest hold a static reference at spawn. The static episodes are kept deliberately — static goals are one of the two evaluation conditions, so dropping them from the training distribution would degrade exactly one of the things being measured, and would break comparability with every run already completed.

## How to train

```bash
isaac_run examples/rl/train.py --task shuttle_glider/sg_d4_com \
  --algo sac --n_envs 1 --seed 10 --headless --device cuda:0
```

`--algo sac` is mandatory (`train.py` defaults to `ppo`) and so is `--n_envs 1` (the default is 4096, but the `isaac_lab` preset was designed for a single env). Logs are written to `logs/` inside this folder.

The preset in `agents/sac_cfg.py` needs no edit: `Policy` and `Critic` build their layers from `observation_space` (`nn.Linear(self.num_observations, 64)`), so the 31D observation is picked up automatically.

## How to read the result

The deciding metric is **`Metrics/corr_thrust5_accdem`** — the within-episode Pearson correlation between rotor-5 thrust and the longitudinal component of the demanded acceleration:

| reading | interpretation |
|---|---|
| `|corr| < 0.1` | rotor 5 is still a constant bias — the variant failed |
| `0.1 - 0.4` | marginal modulation, probably noise |
| `> 0.4` | rotor 5 became an actuator — the variant worked |

`Metrics/thrust5_frac` on its own is **not** a valid read-out: a large constant bias produces a large fraction with no usefulness whatsoever. That is precisely the behaviour being eliminated.

Also read these, to rule out that any apparent gain is just degraded flight:

- `Episode_Reward/tilt` and `Episode_Reward/d_action_5`
- `Metrics/die_pos`, `Metrics/die_vel`, `Metrics/clip_hit_frac`
- `Metrics/episode_length` (if it drops, the policy is dying more often)

## Place in the family

| task | obs | `w_d_action_5` | `w_tilt` | rotor-5 cap | vehicle |
|---|---|---|---|---|---|
| `sg_d0_ctrl` | 28 | `None` | `0.0` | 3500 | `Shuttle_glider` |
| `sg_d1_obs` | 31 | `0.10` | `0.0` | 3500 | `Shuttle_glider` |
| `sg_d2_tilt` | 31 | `0.10` | `2.0` | 3500 | `Shuttle_glider` |
| `sg_d3_cap` | 31 | `0.10` | `2.0` | 1000 | `Shuttle_glider` |
| `sg_d4_com` **<-- this one** | 31 | `0.10` | `2.0` | 1000 | `Shuttle_glider_com` |
| `sg_d5_hard` | 31 | `0.10` | `8.0` | 1000 | `Shuttle_glider` |

Recommended priority if the GPU cannot host all six in parallel: `sg_d3_cap` -> `sg_d4_com` -> `sg_d5_hard` -> `sg_d2_tilt` -> `sg_d1_obs` -> `sg_d0_ctrl`.

`sg_d0_ctrl` is the control and `sg_d3_cap` is the main bet. Running d3 without d0 leaves a positive result ambiguous.
