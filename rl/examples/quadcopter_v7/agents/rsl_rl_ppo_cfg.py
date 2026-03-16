# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .custom_actor_critic import AxisDecoupledActorCritic

import builtins

builtins.AxisDecoupledActorCritic = AxisDecoupledActorCritic


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "quadcopter_direct_s7"

    policy = RslRlPpoActorCriticCfg(
        class_name="AxisDecoupledActorCritic",
        actor_hidden_dims=[1],   #ignored                     
        critic_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )