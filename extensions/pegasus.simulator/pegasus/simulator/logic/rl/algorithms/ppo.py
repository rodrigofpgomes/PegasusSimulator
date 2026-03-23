"""
| File: ppo.py
| Description: PPO training using rsl_rl OnPolicyRunner (rsl_rl 2.x).
| License: BSD-3-Clause.

rsl_rl 2.x learn() reads these keys directly from cfg root:
    cfg["num_steps_per_env"]   ← RolloutStorage
    cfg["save_interval"]       ← checkpoint saving
    cfg["check_for_nan"]       ← optional, defaults True
"""
import os
import datetime


def train(env, agent_cfg, log_dir: str, device: str):
    try:
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.models import MLPModel
        from rsl_rl.modules.distribution import GaussianDistribution
    except ImportError:
        raise ImportError("rsl_rl not installed. Install with: pip install rsl-rl")

    from .wrappers.rsl_rl_wrapper import RslRlVecEnvWrapper

    wrapped_env = RslRlVecEnvWrapper(
        env,
        clip_obs     = agent_cfg.clip_obs,
        clip_actions = agent_cfg.clip_actions,
    )

    train_cfg_dict = {
        # ── top-level: read directly by construct_algorithm and learn() ──
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval":     agent_cfg.save_interval,

        # ── observation routing ───────────────────────────────────────────
        "obs_groups": {
            "actor":  ["policy"],
            "critic": ["policy"],
        },

        # ── actor: stochastic MLPModel ────────────────────────────────────
        "actor": {
            "class_name":  MLPModel,
            "hidden_dims": agent_cfg.actor_hidden_dims,
            "activation":  agent_cfg.activation,
            "distribution_cfg": {
                "class_name": GaussianDistribution,   # pass class — not string
                "init_std":   agent_cfg.init_noise_std,
            },
        },

        # ── critic: deterministic MLPModel ────────────────────────────────
        "critic": {
            "class_name":  MLPModel,
            "hidden_dims": agent_cfg.critic_hidden_dims,
            "activation":  agent_cfg.activation,
        },

        # ── algorithm ─────────────────────────────────────────────────────
        "algorithm": {
            "class_name":             "PPO",
            "clip_param":             agent_cfg.clip_param,
            "desired_kl":             agent_cfg.desired_kl,
            "entropy_coef":           agent_cfg.entropy_coef,
            "gamma":                  agent_cfg.gamma,
            "lam":                    agent_cfg.lam,
            "learning_rate":          agent_cfg.learning_rate,
            "max_grad_norm":          agent_cfg.max_grad_norm,
            "num_learning_epochs":    agent_cfg.num_learning_epochs,
            "num_mini_batches":       agent_cfg.num_mini_batches,
            "schedule":               agent_cfg.schedule,
            "use_clipped_value_loss": agent_cfg.use_clipped_value_loss,
            "value_loss_coef":        agent_cfg.value_loss_coef,
            "rnd_cfg":                None,   # required by Logger — disables RND
        },

        # ── runner ────────────────────────────────────────────────────────
        "runner": {
            "algorithm_class_name":    "PPO",
            "num_steps_per_env":       agent_cfg.num_steps_per_env,
            "max_iterations":          agent_cfg.max_iterations,
            "save_interval":           agent_cfg.save_interval,
            "empirical_normalization": False,
        },

        # ── multi-GPU: required key, empty = disabled ─────────────────────
        "multi_gpu": {},
    }

    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name    = f"{agent_cfg.experiment_name}_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    _save_config_log(run_log_dir, "ppo", agent_cfg)

    runner = OnPolicyRunner(
        env       = wrapped_env,
        train_cfg = train_cfg_dict,
        log_dir   = run_log_dir,
        device    = device,
    )

    runner.learn(
        num_learning_iterations = agent_cfg.max_iterations,
        init_at_random_ep_len   = True,
    )

    print(f"\nTraining complete. Logs saved to: {run_log_dir}")


def _save_config_log(log_dir: str, algo: str, cfg):
    import dataclasses
    path = os.path.join(log_dir, "config.txt")
    with open(path, "w") as f:
        f.write(f"Algorithm: {algo.upper()}\n")
        f.write("=" * 40 + "\n")
        if dataclasses.is_dataclass(cfg):
            for field in dataclasses.fields(cfg):
                f.write(f"{field.name}: {getattr(cfg, field.name)}\n")
        else:
            f.write(str(cfg))