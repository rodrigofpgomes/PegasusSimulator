# Shuttle RL experiments

All active tasks here instantiate the Shuttle USDA. The two action levels are
kept separate:

| Group | Policy action | Experiments |
| --- | --- | --- |
| [`double_integrator/`](double_integrator/README.md) | Body forces and moments through `direct_force`. | Isaac Lab starting point; quadratic, exponential, death and Lyapunov rewards; reward curriculum and rate scaling ablations. |
| [`direct_rotor/`](direct_rotor/README.md) | Four rotor speeds through `rotor_velocity_direct`. | RAPTOR-style baseline; observation, reward, trajectory and termination ablations. |

The double-integrator tasks use a total Shuttle mass of 3.5 kg when converting
desired acceleration to force. Direct-rotor tasks use the Shuttle motor preset
from `pegasus.simulator.logic.vehicles.vehicle_physics`. These are new training
experiments; historical checkpoint results are not carried over.
