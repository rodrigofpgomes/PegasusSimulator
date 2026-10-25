# Tasks

Each folder with a `quadcopter_env.py` and an `agents/` folder is a task. Its id is the path relative to this folder,
e.g. `shuttle/thrust_moment/isaac_lab`, and is passed to `train.py`, `play.py` and the manifests of `play_multi.py`
(`--task <id>`). The environment fields (`QuadcopterEnvCfg`) can be overridden from the command line with
`--env-cfg KEY=VALUE`; the agent presets are defined in `agents/<algo>_cfg.py`.

| Task | Vehicle | Action |
| --- | --- | --- |
| `shuttle/thrust_moment/isaac_lab` | Shuttle | collective thrust and body moments, applied directly to the body |
| `shuttle/outer_loop/double_integrator` | Shuttle | inertial acceleration with an ideal inner loop |
| `shuttle/direct_rotor/baseline/raptor_tracking` | Shuttle | rotor speeds through the motor dynamics |
| `shuttle/direct_rotor/ablation/raptor_tracking` | Shuttle | as above, with the reward and observation variants and the lemniscate |
| `shuttle_glider/tracking` | Shuttle-Glider | lift-rotor and puller speeds and the elevator, aileron and rudder deflections |

## Shuttle

- **`thrust_moment/isaac_lab`**: the quadcopter task of Isaac Lab ported to the Shuttle. Static goals within
  ±2 m horizontally and 0.5–1.5 m in altitude, 10 s episodes, termination outside 0.1–2.0 m of altitude. Reward
  `[15 (1 - tanh(|e_p| / c_D)) - 0.05 |v|^2 - 0.01 |w|^2 + yaw_rate_reward_scale * w_z^2] dt`. Parameters studied:
  `moment_scale` (c_M), `distance_reward_denominator` (c_D), `yaw_rate_reward_scale` and `thrust_to_weight` (k_T).
- **`outer_loop/double_integrator`**: the policy commands `u = max_acceleration * a`; the stage cost is
  `(q_pos |e_p|^2 + q_vel |e_v|^2 + r_u |u|^2) dt` and the reward is `-c` or `exp(-c)` (`reward_shape`) plus
  `alive_bonus` and `death_penalty` on termination. The discounted LQR of the same cost is the reference
  (`_compute_discounted_dlqr`).
- **`direct_rotor/baseline/raptor_tracking`**: the RAPTOR pre-training task (rl-tools): 26 observations, reward
  `1.5 - |e_p| - 0.1 * 2 acos(1 - |q_z|) - |a_k - a_(k-1)|`, -100 on termination (position error > 1 m, velocity
  > 2 m/s or angular rate > 35 rad/s on any axis), 50 % fixed point and 50 % Langevin references.
- **`direct_rotor/ablation/raptor_tracking`**: the same task with optional velocity, yaw-rate and angular-rate costs,
  a geometric attitude cost, the reference acceleration in the observation, and the lemniscate reference used to
  test the generalisation to moving references.

## Shuttle-Glider

- **`tracking`**: one policy for the whole envelope, from hover to 14 m/s, without curriculum. Each 10 s episode
  follows a reference built from a mixture of hover, acceleration, deceleration, cruise, turns and climbs, and
  starts at the minimum-power trim of its first sample (`logic/rl/shuttle_glider_trim.py`) with a random
  perturbation. The reward combines tracking, heading, angular rate, shaft power, surface deflection and command
  variation costs; the allocation terms of the retained policy (`w_rotor_diff`, `rotor_diff_tau`, `w_moment_ema`,
  `moment_ema_tau`, `w_surface_l1`) are exponential moving averages with a 2 s time constant, appended to the
  observation. The policies of the results are the variants `base`, `wls_ema` and `wls_ema_m02` of
  `scripts/shuttle_glider/run_tracking.sh`. The evaluation regimes are selected with `eval_episode`
  (`hover`, `accel`, `decel`, `cruise`, `turn`, `climb`) and `eval_speed`, `eval_speed_end`, `eval_turn_radius`,
  `eval_vz`.
