"""
| File: task_loader.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Task, agent preset and algorithm lookup shared by training and evaluation.
|
| Tasks are directories under tasks/ that contain a *_env.py file. Their ids use
| "/" separators (e.g. "shuttle/thrust_moment/isaac_lab").
|
| apply_env_overrides restores the task configuration of a training run from its
| run_metadata.json, so play.py and play_multi.py evaluate the policy with the
| same action scaling, vehicle parameters and control timing used in training.
"""

import ast
import copy
import importlib
import json
import os
import sys
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
TASKS_DIR = RL_DIR / "tasks"

# The task packages are imported as "tasks.<dotted task id>"
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

# skrl agent classes that can restore a checkpoint for evaluation
_AGENT_CLASSES = {
    "ppo": ("skrl.agents.torch.ppo", "PPO"),
    "ppo2": ("pegasus.simulator.logic.rl.algorithms.ppo2", "PPO2"),
    "sac": ("skrl.agents.torch.sac", "SAC"),
}


def discover_tasks() -> list[str]:
    """Return every task id under TASKS_DIR, skipping private and agents folders."""
    tasks = []
    for root, dirs, files in os.walk(TASKS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith("_") and d != "agents"]
        if any(f.endswith("_env.py") for f in files):
            tasks.append(os.path.relpath(root, TASKS_DIR).replace(os.sep, "/"))
    return sorted(tasks)


def load_env(task: str):
    """Return (EnvClass, env_cfg instance) for a task id.

    The module ``<name>_env.py`` must define ``<Name>Env`` and ``<Name>EnvCfg``.
    """
    task_dir = TASKS_DIR.joinpath(*task.split("/"))
    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]
    if not env_files:
        raise FileNotFoundError(f"No *_env.py found in {task_dir}")

    # Numeric-prefixed packages are valid only through import_module
    module_name = env_files[0][:-3]
    module = importlib.import_module(f"tasks.{task.replace('/', '.')}.{module_name}")
    class_name = "".join(w.capitalize() for w in module_name.replace("_env", "").split("_")) + "Env"
    return getattr(module, class_name), getattr(module, class_name + "Cfg")()


def resolve_preset(task: str, algo: str, preset: str | None = None) -> str:
    """Return the preset name to load from tasks/<task>/agents/<algo>_cfg.py.

    Without an explicit name, 'default' is used, or the only preset when the
    file defines exactly one.
    """
    module = importlib.import_module(f"tasks.{task.replace('/', '.')}.agents.{algo}_cfg")
    available = sorted(module.PRESETS)
    if preset is None:
        if "default" in module.PRESETS:
            return "default"
        if len(available) == 1:
            return available[0]
        raise KeyError(f"No preset given for {task} / {algo}; choose one of {available}")
    if preset not in module.PRESETS:
        raise KeyError(f"Preset '{preset}' not found for {task} / {algo}; available: {available}")
    return preset


def load_agent_cfg(task: str, algo: str, preset: str | None = None) -> dict:
    """Return the selected preset from tasks/<task>/agents/<algo>_cfg.py."""
    module = importlib.import_module(f"tasks.{task.replace('/', '.')}.agents.{algo}_cfg")
    return module.PRESETS[resolve_preset(task, algo, preset)]


def load_algo_class(algo: str):
    """Return the skrl agent class used to restore a checkpoint of ``algo``."""
    module_name, class_name = _AGENT_CLASSES[algo]
    return getattr(importlib.import_module(module_name), class_name)


def prepare_agent_cfg(cfg: dict, device: str) -> dict:
    """Copy an agent config and place its preprocessors on ``device``."""
    cfg = copy.deepcopy(cfg)
    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device
    return cfg


def apply_env_overrides(env_cfg, metadata_path=None, items=(), values=None) -> dict:
    """Apply a training run's environment snapshot and explicit overrides to ``env_cfg``.

    The order is: snapshot in ``metadata_path`` (run_metadata.json written by
    train.py), then ``values`` (a mapping, used by play_multi manifests), then
    ``items`` ("KEY=VALUE" strings, used by --env-cfg). Nested config objects
    accept mappings. Returns a dict describing every changed field and every
    snapshot field the task config does not have, so the evaluation can be audited.
    """

    def assign(target, key, value, prefix, log):
        """Set one field, recursing into nested config objects given as mappings."""
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dict__") and not isinstance(current, dict):
            for sub_key, sub_value in value.items():
                if hasattr(current, sub_key):
                    assign(current, sub_key, sub_value, f"{prefix}{key}.", log)
            return
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        if current != value:
            log["changed"][prefix + key] = {"default": current, "applied": value}
            setattr(target, key, value)

    log = {"changed": {}, "not_in_task_cfg": [], "training_commit": None, "training_seed": None,
           "training_timing": None}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text())
        environment = metadata.get("environment", {})
        log["training_commit"] = metadata.get("git_commit")
        log["training_seed"] = metadata.get("seed")
        if "sim_dt" in environment and "decimation" in environment:
            log["training_timing"] = {"sim_dt": environment["sim_dt"], "decimation": environment["decimation"]}
        for key, value in environment.items():
            if hasattr(env_cfg, key):
                assign(env_cfg, key, value, "", log)
            else:
                log["not_in_task_cfg"].append(key)

    for key, value in (values or {}).items():
        if not hasattr(env_cfg, key):
            raise SystemExit(f"env_cfg '{key}' is not a field of {type(env_cfg).__name__}")
        assign(env_cfg, key, value, "", log)

    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not hasattr(env_cfg, key):
            raise SystemExit(f"--env-cfg '{item}': expected KEY=VALUE with KEY a field of {type(env_cfg).__name__}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        assign(env_cfg, key, value, "", log)
    return log


def control_timing(env_cfg) -> dict:
    """Return the physics and policy rates [Hz] defined by a task configuration."""
    return {"sim_dt": env_cfg.sim_dt, "decimation": env_cfg.decimation,
            "physics_hz": 1.0 / env_cfg.sim_dt, "control_hz": 1.0 / (env_cfg.sim_dt * env_cfg.decimation)}
