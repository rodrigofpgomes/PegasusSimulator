"""
| File: ppo.py
| Description: PPO training via rsl_rl 2.x OnPolicyRunner.
|
| actor_class — fully configurable via agent_cfg.actor_class
| seed        — reproducibility via agent_cfg.seed
"""
import os
import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from .wrappers.rsl_rl_wrapper import RslRlVecEnvWrapper


def train(env, agent_cfg, log_dir: str, device: str):

    # ── seed ─────────────────────────────────────────────────
    _set_seed(getattr(agent_cfg, "seed", None))

    # ── actor class ──────────────────────────────────────────
    actor_class  = getattr(agent_cfg, "actor_class",  None) or MLPModel
    actor_kwargs = getattr(agent_cfg, "actor_kwargs", None) or {}

    actor_cfg = {
        "class_name":  actor_class,
        "hidden_dims": agent_cfg.actor_hidden_dims,
        "activation":  agent_cfg.activation,
        "distribution_cfg": {
            "class_name": GaussianDistribution,
            "init_std":   agent_cfg.init_noise_std,
        },
        **actor_kwargs,
    }

    # ── rsl_rl config ─────────────────────────────────────────
    train_cfg = {
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval":     agent_cfg.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor":  actor_cfg,
        "critic": {
            "class_name":  MLPModel,
            "hidden_dims": agent_cfg.critic_hidden_dims,
            "activation":  agent_cfg.activation,
        },
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
            "rnd_cfg":                None,
        },
        "runner": {
            "algorithm_class_name":    "PPO",
            "num_steps_per_env":       agent_cfg.num_steps_per_env,
            "max_iterations":          agent_cfg.max_iterations,
            "save_interval":           agent_cfg.save_interval,
            "empirical_normalization": False,
        },
        "multi_gpu": {},
    }

    # ── log dir ───────────────────────────────────────────────
    # log_dir is already <tasks_dir>/<task>/ — append logs/<timestamp> only
    ts          = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(log_dir, "logs", ts)
    os.makedirs(run_log_dir, exist_ok=True)
    _save_config_log(run_log_dir, agent_cfg)
    print(f"\n[PPO] Logs: {run_log_dir}\n")

    # ── train ─────────────────────────────────────────────────
    wrapped = RslRlVecEnvWrapper(
        env,
        clip_obs     = agent_cfg.clip_obs,
        clip_actions = agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env       = wrapped,
        train_cfg = train_cfg,
        log_dir   = run_log_dir,
        device    = device,
    )

    runner.learn(
        num_learning_iterations = agent_cfg.max_iterations,
        init_at_random_ep_len   = True,
    )

    print(f"[PPO] Done. Logs: {run_log_dir}")


def _set_seed(seed):
    if seed is None:
        return
    import torch, random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[PPO] Seed: {seed}")


def _save_config_log(log_dir, cfg):
    import dataclasses
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write("Algorithm: PPO\n" + "="*40 + "\n")
        if dataclasses.is_dataclass(cfg):
            for fld in dataclasses.fields(cfg):
                val = getattr(cfg, fld.name)
                if isinstance(val, type):
                    val = val.__name__
                f.write(f"{fld.name}: {val}\n")
        else:
            f.write(str(cfg))