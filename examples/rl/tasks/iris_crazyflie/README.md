# Iris and Crazyflie baseline tasks

[`double_integrator/`](double_integrator/README.md) preserves the original
direct-force study from the experimental copy. It contains nine tasks:

- `01_isaac_lab_iris` and `01_isaac_lab_crazyflie`: Isaac Lab-style baseline on each airframe.
- `02_quadratic`, `03_exp`, `04_constant_death` and `04_constant_death2`: reward progression on Iris.
- `05_lyapunov_reward` and `06_lyapunov_learned`: analytic and learned Lyapunov terms on Iris.
- `07_curriculum_ppo2`: Iris reward curriculum.

The Crazyflie USDA is included and registered under `ROBOTS`. The learned
Lyapunov task requires a separately trained Iris value checkpoint through
`lyapunov_checkpoint_path`; no old training weights are bundled. Other copied
tasks retain their original reward/action implementation, with explicit Iris
and `direct_force` declarations where those were previously inherited.
