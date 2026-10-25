#!/usr/bin/env bash
# Shuttle, direct rotor velocity (shuttle/direct_rotor/baseline/raptor_tracking): PPO2 "tanh" preset, seeds 2 and 3.
#
# Completes the tanh variant (policy mean = tanh(net), std not bounded, 600 iterations): seed 1 is already trained in
# logs/ppo2_tanh_600 and is skipped, so only seeds 2 and 3 are trained. All three seeds are evaluated and listed in
# logs/ppo2_tanh_600/summary.csv together with the reference runs.
#
# Usage (inside tmux):
#   tmux new -s ppo2_tanh 'examples/rl/scripts/shuttle/direct_rotor/run_ppo2_tanh_seeds23.sh 2>&1 | tee ppo2_tanh_s23.out'
#   examples/rl/scripts/shuttle/direct_rotor/run_ppo2_tanh_seeds23.sh --dry-run
# Environment (forwarded): CKPT_ITERS, N_ENVS, N_EVAL_ENVS, EVAL_SEED, ROOT.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRESET=tanh ITERATIONS=600 PPO_SEEDS="1 2 3"
unset STD_MIN
exec "$HERE/run_ppo2_variant.sh" "$@"
