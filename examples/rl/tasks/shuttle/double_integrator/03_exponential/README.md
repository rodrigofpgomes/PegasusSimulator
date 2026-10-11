# Exponential tracking reward

Body-force command with the tracking cost transformed into an exponential reward.

The task uses the Shuttle asset with `direct_force` and a 3.5 kg mass. Its
reward, reset, reference and observation settings are defined in [quadcopter_env.py](quadcopter_env.py).
Available training presets: PPO, SAC. This experiment is selected for new
training; historical checkpoints are not associated with it.
