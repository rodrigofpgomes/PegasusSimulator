# ShuttleGlider experiments

`no_surfaces/` is the first selected task. It uses five rotor actions, sends
zero surface deflections and has no curriculum. It is adapted from
`sg_f2_rep3` for a fresh training run with the shared ShuttleGlider physics.

The next comparison should enable surface control without curriculum, then
introduce curriculum while keeping the vehicle physics fixed. Record rotor
and puller effort, surface angles, airspeed and aerodynamic loads across the
flight regimes.
