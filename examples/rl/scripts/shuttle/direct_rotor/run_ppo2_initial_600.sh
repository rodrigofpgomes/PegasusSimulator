#!/usr/bin/env bash
# Shuttle, direct rotor velocity (shuttle/direct_rotor/baseline/raptor_tracking): initial PPO2 configuration, 600 iterations.
#
# Trains the "default" preset (RSL-RL setup: unbounded Gaussian mean, std not bounded) from scratch for 600 iterations
# with seeds 1, 2 and 3, so that it has the same length as the tanh and tanh + std variants (run_ppo2_variant.sh).
# The agent that run_direct_rotor_baseline.sh trained for 400 iterations is not reused: it is a different run length.
# Output: logs/ppo2_default_600 (train, eval, summary.csv). Finished steps (.done) are skipped on restart.
#
# Usage (inside tmux):
#   tmux new -s ppo2_initial 'examples/rl/scripts/shuttle/direct_rotor/run_ppo2_initial_600.sh 2>&1 | tee ppo2_initial_600.out'
#   examples/rl/scripts/shuttle/direct_rotor/run_ppo2_initial_600.sh --dry-run
# Environment (forwarded): PPO_SEEDS (default 1 2 3), CKPT_ITERS, N_ENVS, N_EVAL_ENVS, EVAL_SEED, ROOT.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRESET=default ITERATIONS=600
unset STD_MIN
exec "$HERE/run_ppo2_variant.sh" "$@"
