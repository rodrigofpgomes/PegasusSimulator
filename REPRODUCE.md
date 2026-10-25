# Reproducing the results

This file lists, for every experiment of the Shuttle and of the Shuttle-Glider, the commands that train and evaluate
the policies, the scripts that
turn the evaluation logs into the tables and figures, and reference values to check the outcome. Trained weights and
raw logs are not part of the repository: every result is regenerated from scratch.

All commands are run from the repository root. Training and evaluation logs are written under
`examples/rl/tasks/<task>/logs/` (ignored by git); the scripts skip finished steps (`.done` files), so they can be
restarted.

## 1. Environment

| Component | Version used |
| --- | --- |
| OS / GPU | Ubuntu 24.04, NVIDIA GeForce RTX 5090 |
| Isaac Sim | 5.1.0 (Python 3.11, torch 2.7.0+cu128) |
| skrl / rsl-rl-lib | 1.4.3 / 5.0.1 (installed in the Isaac Sim Python) |
| PX4-Autopilot | v1.14.3 (only for the PX4 state-estimation flights) |
| rl-tools (RAPTOR) | `rl-tools/raptor` at `e43ae4bc`, built with OpenBLAS (only for the RAPTOR pre-training reference) |
| MATLAB | R2020b or later (figures; `results/shuttle/rotor_velocities` also has a matplotlib version) |

1. Install Isaac Sim 5.1.0 and the Pegasus Simulator extension as in the
   [Pegasus documentation](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html)
   (`ISAACSIM_PYTHON -m pip install --editable pegasus.simulator` from `extensions/`).
2. Install the RL libraries in the Isaac Sim Python:
   `~/isaacsim_5.1.0/python.sh -m pip install skrl==1.4.3 rsl-rl-lib==5.0.1`.
3. For the export scripts, the diagnostics and the offline tests, use a Python 3 environment with
   `numpy scipy torch matplotlib usd-core` (`pip install usd-core` provides the USD bindings that read the vehicle
   assets without Isaac Sim).
4. Environment variables read by the scripts (defaults in brackets):
   - `ISAACSIM_PYTHON` [`~/isaacsim_5.1.0/python.sh`]: Isaac Sim interpreter;
   - `RAPTOR_DIR` [`~/raptor`]: clone of `rl-tools/raptor` (RAPTOR pre-training reference only);
   - `px4_dir` in `extensions/pegasus.simulator/config/configs.yaml` [`~/PX4-Autopilot`].
5. Offline check of the installation: `python -m pytest tests/review` (no Isaac Sim needed).

Results are reproducible statistically, not bit for bit: the GPU physics (PhysX) and the parallel training are not
deterministic. The evaluations use a fixed seed (1000), so all policies of a section see the same episodes.

## 2. Shuttle - thrust and moment control

```bash
examples/rl/scripts/shuttle/thrust_moment/run_isaac_lab_study.sh            # k_T = 1.9, 8 configurations x 10 seeds
KT=3.9 examples/rl/scripts/shuttle/thrust_moment/run_isaac_lab_study.sh     # k_T = 3.9 (sensitivity test)
python3 results/shuttle/isaac_lab_study/export_results.py                   # -> results/shuttle/isaac_lab_study/data
# MATLAB, in results/shuttle/isaac_lab_study:  plot_results
```

Each run trains PPO2 for 200 iterations (4096 vehicles, about 80 s on one GPU) and evaluates the final checkpoint on
200 goals. Check: with c_M = 0.12, c_D = 0.8 and w_psi = 1.01 the final error is about 0.018 m, the RMSE 0.449 m and
t20 1.37 s (mean over ten seeds).

## 3. Shuttle - acceleration control

```bash
examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh              # a_max = 7 m/s^2, rewards Q, A, D, E, AD and the LQR
A_MAX=1.0 examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh    # a_max = 1 m/s^2 (LQR saturated)
EXPERIMENTS=alive_death200 examples/rl/scripts/shuttle/outer_loop/run_outer_loop_study.sh   # r^AD with d = -200
python3 results/shuttle/outer_loop_amax7/export_results.py                  # needs torch (equilibrium of each policy)
python3 results/shuttle/outer_loop_amax1/export_results.py
# MATLAB: plot_results in each folder
```

The study script also writes `outer_loop_by_run.csv` (`summarize_outer_loop_eval.py`), read by the export. Check:
the quadratic reward terminates every episode (survival 0), the constant, death-penalty and exponential rewards
survive every episode with a discounted return about 10 % below the LQR (G_LQR = -24.57).

## 4. Shuttle - direct rotor velocity control

```bash
S=examples/rl/scripts/shuttle/direct_rotor
$S/run_direct_rotor_baseline.sh                       # SAC (RAPTOR setup, 1e6 transitions, seed 1) and PPO2 (400 it.)
$S/run_ppo2_initial_600.sh                            # PPO2, initial configuration, 600 iterations, seeds 1-3
PRESET=tanh $S/run_ppo2_variant.sh                    # PPO2, tanh-bounded mean, seed 1
$S/run_ppo2_tanh_seeds23.sh                           # ... seeds 2 and 3
PRESET=tanh STD_MIN=0.2 $S/run_ppo2_variant.sh        # PPO2, tanh + sigma_min = 0.2 (retained configuration)
PRESET=default STD_MIN=0.2 PPO_SEEDS=1 $S/run_ppo2_variant.sh   # sigma_min only
$S/run_direct_rotor_ablation.sh                       # reward and observation variants (seed 3)
$S/run_direct_rotor_ablation_extra.sh                 # control runs of the R5 and R6 variants with seeds 1 and 2
$S/eval_direct_rotor_yaw.sh                           # heading of the ablation policies
$S/eval_direct_rotor_lemniscate.sh                    # lemniscate, peak acceleration 0.5-10 m/s^2
$S/eval_force_moment_lemniscate.sh                    # the thrust-and-moment policies on the lemniscate
results/shuttle/rotor_velocities/make_results.sh      # checkpoint evaluations, export, figures (matplotlib)
```

RAPTOR pre-training reference: build `rl-tools` with OpenBLAS (`-DRL_TOOLS_BACKEND_ENABLE_OPENBLAS=ON`) and train one
teacher with the Shuttle dynamics of this repository, then evaluate it with the same protocol:

```bash
cd "$RAPTOR_DIR/rl-tools"
OPENBLAS_NUM_THREADS=1 RL_TOOLS_EXTRACK_EXPERIMENT=shuttle_pretrain \
  ./build_openblas/src/foundation_policy/foundation_policy_pre_training <repo>/examples/rl/configs/raptor/shuttle.json
cd <repo> && examples/rl/scripts/shuttle/direct_rotor/eval_raptor_pretrain.sh
```

Check: PPO2 with tanh and sigma_min reaches a final error of about 5.2 cm (three seeds) with no command at its limits;
on the lemniscate at 2 m/s^2 the reference acceleration in the observation lowers the RMSE from about 15.8 to 5.2 cm.

## 5. Shuttle-Glider

Policies: `base` (B), `wls_ema` (F, filtered differential) and `wls_ema_m02` (M, retained), seed 3, 5000 iterations.

```bash
G=examples/rl/scripts/shuttle_glider
VARIANTS="base wls_ema wls_ema_m02" $G/run_tracking.sh                     # training (about 35 min per policy)
# reward variants of the variant table (one seed each):
VARIANTS="power01 power02 dact025 rate0 surfq conflict surfq_conflict conflict_alloc alloc wls wls_q wls_ema_m" $G/run_tracking.sh

# regimes of the regime table and of the steady-flight figure (200 vehicles, t >= 3 s)
OUT_ROOT=examples/rl/tasks/shuttle_glider/tracking/logs/regime_figures VARIANTS="base wls_ema wls_ema_m02" \
REGIMES="hover mixed accel_v5 accel_v9 accel_v13 decel_v5_0 decel_v9_0 decel_v13_9 \
  cruise_v1 cruise_v2 cruise_v3 cruise_v4 cruise_v5 cruise_v6 cruise_v7 cruise_v8 cruise_v9 cruise_v10 cruise_v11 \
  cruise_v12 cruise_v13 cruise_v14 turn_v8_r25 turn_v10_r40 turn_v12_r30 turn_v14_r40 climb_v0_z1 climb_v10_z1 climb_v10_z-1" \
  $G/eval_tracking.sh
results/shuttle_glider/tracking_regimes/eval_regimes.sh                    # deceleration and helix episodes

# missions at 12 and 14 m/s (true state)
TRAIN_VARIANTS=" " EVAL_VARIANTS=" " MISSION_VARIANTS="base wls_ema wls_ema_m02" $G/run_round.sh

# surface ablation of the base policy (surface commands replaced by the trim or by zero)
$G/run_surface_ablation.sh
```

Mission with the PX4 state estimate. Install the airframes first: copy
`examples/rl/px4_vehicles/10050_none_shuttle` and `10051_none_shuttle_glider` to
`PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/`, add them to the `CMakeLists.txt` of that folder and run
`make px4_sitl_default` in PX4-Autopilot. Then, for `CONFIG` in `shuttle_glider_true_state`,
`px4_shuttle_glider_estimator` and `px4_shuttle_glider_estimator_gps_nodrift`, with `CASE` = `mission`, `mission_px4`
and `mission_px4_gps_nodrift`:

```bash
"$ISAACSIM_PYTHON" examples/rl/play_multi.py --config examples/rl/configs/$CONFIG.json \
  --trajectory mission --traj_speed 12 --traj_radius 50 --traj_a_max 3 --mission_decel_amax 0.7 \
  --episode_duration 101 --traj_z 10 --spawn-altitude 10 --mission_climb 10 --mission_climb_rate 1.0 \
  --n_envs 1 --headless --record --failure-distance 1000 \
  --record_dir examples/rl/tasks/shuttle_glider/tracking/logs/variants/wls_ema_m02/$CASE/mission_v12alt_s03
```

(For the appendix flights of the base policy, use copies of the three manifests with `wls_ema_m02` replaced by
`base` and record to `variants/base/...`.)

Tables, figures and analyses:

```bash
python3 examples/rl/diagnostics/shuttle_glider_trim.py --limits-table      # minimum-power trim
python3 examples/rl/scripts/shuttle_glider/analysis/surface_authority.py  # control authority of surfaces and rotors
python3 examples/rl/scripts/shuttle_glider/analysis/wing_only_trim.py     # level flight on the wing alone
python3 results/shuttle_glider/tracking_regimes/export_results.py          # -> data/ (regimes, episodes, missions)
# MATLAB, in results/shuttle_glider/tracking_regimes:  plot_results
```

The variant table (missions and Delta Omega) is in `examples/rl/tasks/shuttle_glider/tracking/logs/variants/
missions_summary.csv`, written by `run_round.sh`. Check: the three policies survive every regime with an RMSE below
2.4 cm; the minimum-power trim falls from 663.6 W in hover to 45.8 W at 14 m/s; with the PX4 estimate the retained
policy completes the mission, tracking the estimated position within about 12 cm.

Mission video (Lake World): run the `video_mission_v12_px4` flight of
`examples/rl/scripts/shuttle_glider/manifests/tracking_wls_ema_m02_px4_video.json`, then
`python3 examples/rl/scripts/shuttle_glider/video/compose_mission_video.py <record_dir>`.

## 6. EasyGlider

```bash
"$ISAACSIM_PYTHON" examples/rl/diagnostics/easyglider_equilibrium.py      # longitudinal trim of the EasyGlider
```
