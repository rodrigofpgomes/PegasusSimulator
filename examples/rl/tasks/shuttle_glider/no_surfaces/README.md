# ShuttleGlider without active surfaces

[`raptor_tracking/`](raptor_tracking/README.md) is adapted from the
`sg_f2_rep3` experiment. It uses five rotor-speed actions and sends zeros for
elevator, aileron and rudder. It has no curriculum and uses the shared motor
preset for a new training run. The historical `sg_f2_rep2` implementation was
identical except for its header, so it is not duplicated here.
