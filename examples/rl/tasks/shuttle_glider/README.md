# ShuttleGlider experiments

The action groups separate the effect of aerodynamic surface control:

| Group | Policy action | Experiments |
| --- | --- | --- |
| [`no_surfaces/`](no_surfaces/README.md) | Five rotor speeds; neutral elevator, aileron and rudder. | RAPTOR-style tracking without curriculum. |
| [`surfaces/`](surfaces/README.md) | Five rotor speeds and three surface deflections. | Reward and curriculum ablations across flight regimes. |

Both groups use the same ShuttleGlider motor preset. A named aerodynamic
override remains in `reward_offload_custom_aero`; compare it separately from
experiments using the default aerodynamic coefficients.
