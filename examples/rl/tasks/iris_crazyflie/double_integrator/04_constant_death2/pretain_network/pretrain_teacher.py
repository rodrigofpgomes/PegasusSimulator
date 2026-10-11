#!/usr/bin/env python
"""
| File: pretrain_teacher.py
| Description: Teacher-student behavior-cloning pretraining for the 6D -> 3D
|              quadcopter acceleration policy.

The dataset is intentionally a mixture of four regions:

1. reset:
   Reproduces the actual observation distribution immediately after the current
   environment reset: randomized goal, fixed spawn state, zero velocity.

2. near_goal:
   Dense Gaussian sampling around the equilibrium.

3. operational:
   Samples the mostly-unsaturated region so the MLP learns the actual linear
   slope of the controller rather than only +/-1 saturation.

4. recovery:
   Broad position/velocity errors representative of off-nominal states.

The actor mean network is trained. PPO's std_parameter is left untouched.
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from gymnasium.spaces import Box
except ImportError:
    from gym.spaces import Box

from nonlinear_controller import NonlinearControllerTeacher
from ppo2_cfg import Policy, TEACHER_PRETRAIN_CFG


REGION_NAMES = {
    0: "reset",
    1: "near_goal",
    2: "operational",
    3: "recovery",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain PPO actor from nonlinear-controller teacher")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--spawn-z", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def uniform_columns(n: int, ranges, device: torch.device) -> torch.Tensor:
    lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=device)
    hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=device)

    return lo + (hi - lo) * torch.rand((n, len(ranges)), device=device)


def allocate_counts(n: int, sampling_cfg: dict) -> dict[str, int]:
    fractions = {
        "reset": float(sampling_cfg["reset_fraction"]),
        "near_goal": float(sampling_cfg["near_goal_fraction"]),
        "operational": float(sampling_cfg["operational_fraction"]),
        "recovery": float(sampling_cfg["recovery_fraction"]),
    }

    total = sum(fractions.values())

    if abs(total - 1.0) > 1.0e-6:
        raise ValueError(f"Sampling fractions must sum to 1.0, got {total}")

    counts = {
        name: int(round(n * fraction))
        for name, fraction in fractions.items()
    }

    # Force the exact requested batch size despite integer rounding.
    counts["recovery"] += n - sum(counts.values())

    return counts


def sample_reset_region(n: int, cfg: dict, device: torch.device):
    """
    Match the current environment reset exactly at the observation level.

    Current reset behavior:
        vehicle position = spawn position
        vehicle velocity = 0
        goal x/y = spawn x/y + U(goal_pos_xy_range)
        goal z   = U(goal_pos_z_range)
        goal velocity = 0

    Therefore:
        ep_x/y = U(goal_pos_xy_range)
        ep_z   = U(goal_pos_z_range) - spawn_z
        ev     = 0
    """
    if n == 0:
        return torch.empty((0, 6), device=device)

    xy_lo, xy_hi = cfg["goal_pos_xy_range"]
    z_lo, z_hi = cfg["goal_pos_z_range"]
    spawn_z = float(cfg["spawn_z"])

    ep = torch.empty((n, 3), dtype=torch.float32, device=device)
    ep[:, 0].uniform_(xy_lo, xy_hi)
    ep[:, 1].uniform_(xy_lo, xy_hi)
    ep[:, 2].uniform_(z_lo, z_hi)
    ep[:, 2] -= spawn_z

    ev = torch.zeros((n, 3), dtype=torch.float32, device=device)

    return torch.cat((ep, ev), dim=1)


def sample_near_goal_region(n: int, cfg: dict, device: torch.device):
    if n == 0:
        return torch.empty((0, 6), device=device)

    sampling = cfg["sampling"]

    ep = float(sampling["near_goal_ep_std"]) * torch.randn((n, 3), device=device)
    ev = float(sampling["near_goal_ev_std"]) * torch.randn((n, 3), device=device)

    return torch.cat((ep, ev), dim=1)


def sample_operational_region(n: int, cfg: dict, device: torch.device):
    if n == 0:
        return torch.empty((0, 6), device=device)

    sampling = cfg["sampling"]

    ep = uniform_columns(n, sampling["operational_ep_range"], device)
    ev = uniform_columns(n, sampling["operational_ev_range"], device)

    return torch.cat((ep, ev), dim=1)


def sample_recovery_region(n: int, cfg: dict, device: torch.device):
    if n == 0:
        return torch.empty((0, 6), device=device)

    sampling = cfg["sampling"]

    ep = uniform_columns(n, sampling["recovery_ep_range"], device)
    ev = uniform_columns(n, sampling["recovery_ev_range"], device)

    return torch.cat((ep, ev), dim=1)


def generate_observation_batch(n: int, cfg: dict, device: torch.device):
    counts = allocate_counts(n, cfg["sampling"])

    parts = [
        sample_reset_region(counts["reset"], cfg, device),
        sample_near_goal_region(counts["near_goal"], cfg, device),
        sample_operational_region(counts["operational"], cfg, device),
        sample_recovery_region(counts["recovery"], cfg, device),
    ]

    labels = [
        torch.full((counts["reset"],), 0, dtype=torch.long, device=device),
        torch.full((counts["near_goal"],), 1, dtype=torch.long, device=device),
        torch.full((counts["operational"],), 2, dtype=torch.long, device=device),
        torch.full((counts["recovery"],), 3, dtype=torch.long, device=device),
    ]

    obs = torch.cat(parts, dim=0)
    region = torch.cat(labels, dim=0)

    perm = torch.randperm(obs.shape[0], device=device)

    return obs[perm], region[perm]


def generate_dataset(cfg: dict, teacher: NonlinearControllerTeacher, device: torch.device):
    target_n = int(cfg["num_samples"])
    generation_bs = int(cfg["generation_batch_size"])

    obs_chunks = []
    action_chunks = []
    raw_chunks = []
    saturation_chunks = []
    region_chunks = []

    generated = 0

    print(f"Generating {target_n:,} teacher samples on {device}...")

    while generated < target_n:
        n = min(generation_bs, target_n - generated)

        obs, region = generate_observation_batch(n, cfg, device)

        with torch.no_grad():
            actions, raw_acceleration, saturated = teacher.compute_from_observation(obs)

        obs_chunks.append(obs.cpu())
        action_chunks.append(actions.cpu())
        raw_chunks.append(raw_acceleration.cpu())
        saturation_chunks.append(saturated.cpu())
        region_chunks.append(region.cpu())

        generated += n

    obs = torch.cat(obs_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    raw_acceleration = torch.cat(raw_chunks, dim=0)
    saturated = torch.cat(saturation_chunks, dim=0)
    region = torch.cat(region_chunks, dim=0)

    perm = torch.randperm(obs.shape[0])

    return (
        obs[perm],
        actions[perm],
        raw_acceleration[perm],
        saturated[perm],
        region[perm],
    )


def print_reset_definition(cfg: dict):
    xy_lo, xy_hi = cfg["goal_pos_xy_range"]
    z_lo, z_hi = cfg["goal_pos_z_range"]
    spawn_z = float(cfg["spawn_z"])

    print("\nReset distribution represented by the dataset:")
    print(f"  ep_x: [{xy_lo:.3f}, {xy_hi:.3f}] m")
    print(f"  ep_y: [{xy_lo:.3f}, {xy_hi:.3f}] m")
    print(f"  ep_z: [{z_lo - spawn_z:.3f}, {z_hi - spawn_z:.3f}] m")
    print("  ev_x = ev_y = ev_z = 0 m/s")
    print(f"  assumed Iris spawn_z = {spawn_z:.3f} m")


def print_dataset_statistics(
    actions: torch.Tensor,
    raw_acceleration: torch.Tensor,
    saturated: torch.Tensor,
    region: torch.Tensor,
):
    labels = ["x", "y", "z"]

    print("\nUnsaturated teacher acceleration [m/s^2]:")
    for i, name in enumerate(labels):
        x = raw_acceleration[:, i]
        q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
        print(
            f"  a_{name}: min={x.min(): .3f} max={x.max(): .3f} "
            f"p01={q[0]: .3f} p05={q[1]: .3f} p50={q[2]: .3f} "
            f"p95={q[3]: .3f} p99={q[4]: .3f}"
        )

    print("\nExecutable teacher action [-1, 1] = acceleration [m/s^2]:")
    for i, name in enumerate(labels):
        x = actions[:, i]
        q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
        print(
            f"  u_{name}: min={x.min(): .3f} max={x.max(): .3f} "
            f"p01={q[0]: .3f} p05={q[1]: .3f} p50={q[2]: .3f} "
            f"p95={q[3]: .3f} p99={q[4]: .3f}"
        )

    print("\nSaturation by sampling region:")
    for region_id, region_name in REGION_NAMES.items():
        mask = region == region_id

        if not mask.any():
            continue

        fraction = saturated[mask].float().mean().item()
        count = int(mask.sum().item())

        print(
            f"  {region_name:>11s}: "
            f"{100.0 * fraction:6.2f}% saturated "
            f"({count:,} samples)"
        )

    overall = saturated.float().mean().item()

    print(f"  {'overall':>11s}: {100.0 * overall:6.2f}% saturated")
    print(
        "\nNote: high saturation in the reset/recovery regions is expected with "
        "max_acceleration=1 m/s^2 and goal offsets up to 2 m. The near-goal and "
        "operational regions are included specifically to teach the unsaturated "
        "slope of the nonlinear controller."
    )


def action_balance_weights(actions: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Approximate inverse-density reweighting over each action dimension.

    Saturated reset samples naturally produce many +/-1 targets. Reweighting
    prevents those boundary bins from overwhelming the interior controller law.
    """
    if not cfg["balance_action_bins"]:
        return torch.ones(actions.shape[0], dtype=torch.double)

    bins = int(cfg["num_action_bins"])
    max_weight = float(cfg["max_sample_weight"])
    lo = -float(cfg["max_acceleration"])
    hi = float(cfg["max_acceleration"])

    weights = torch.zeros(actions.shape[0], dtype=torch.float64)

    for d in range(actions.shape[1]):
        x = actions[:, d]
        idx = torch.clamp(((x - lo) / (hi - lo) * bins).long(), 0, bins - 1)

        counts = torch.bincount(idx, minlength=bins).double().clamp_min(1.0)
        inv = 1.0 / counts[idx]
        inv /= inv.mean()

        weights += inv

    weights /= actions.shape[1]
    weights = torch.clamp(weights, max=max_weight)
    weights /= weights.mean()

    return weights


def make_policy(cfg: dict, device: torch.device) -> Policy:
    obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(cfg["num_observations"]),),
        dtype=np.float32,
    )

    max_acc = float(cfg["max_acceleration"])

    action_space = Box(
        low=-max_acc,
        high=max_acc,
        shape=(int(cfg["num_actions"]),),
        dtype=np.float32,
    )

    return Policy(obs_space, action_space, device).to(device)


@torch.no_grad()
def evaluate(
    policy: Policy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    batch_size: int,
):
    policy.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_elements = 0

    loader = DataLoader(
        TensorDataset(obs, actions),
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_obs, batch_actions in loader:
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)

        prediction = policy.net(batch_obs)

        total_sq += torch.sum((prediction - batch_actions) ** 2).item()
        total_abs += torch.sum(torch.abs(prediction - batch_actions)).item()
        total_elements += batch_actions.numel()

    return total_sq / total_elements, total_abs / total_elements


@torch.no_grad()
def evaluate_by_region(
    policy: Policy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    region: torch.Tensor,
    device: torch.device,
    batch_size: int,
):
    """Compute MSE, RMSE and MAE separately for each validation region."""
    policy.eval()

    metrics = {}

    for region_id, region_name in REGION_NAMES.items():
        mask = region == region_id

        if not mask.any():
            metrics[region_name] = {
                "count": 0,
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            }
            continue

        region_obs = obs[mask]
        region_actions = actions[mask]

        mse, mae = evaluate(
            policy=policy,
            obs=region_obs,
            actions=region_actions,
            device=device,
            batch_size=batch_size,
        )

        metrics[region_name] = {
            "count": int(mask.sum().item()),
            "mse": mse,
            "rmse": mse ** 0.5,
            "mae": mae,
        }

    return metrics


def print_validation_metrics_by_region(metrics: dict):
    print("\\nValidation metrics by sampling region:")
    print(
        f"{'region':>12s} | {'samples':>8s} | "
        f"{'MSE':>12s} | {'RMSE':>12s} | {'MAE':>12s}"
    )
    print("-" * 68)

    for region_name in ("reset", "near_goal", "operational", "recovery"):
        values = metrics[region_name]

        print(
            f"{region_name:>12s} | "
            f"{values['count']:8d} | "
            f"{values['mse']:12.6e} | "
            f"{values['rmse']:12.6e} | "
            f"{values['mae']:12.6e}"
        )


def train(
    policy: Policy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    region: torch.Tensor,
    cfg: dict,
    device: torch.device,
):
    n = obs.shape[0]
    n_val = max(1, int(round(n * float(cfg["validation_fraction"]))))
    n_train = n - n_val

    train_obs = obs[:n_train]
    train_actions = actions[:n_train]
    val_obs = obs[n_train:]
    val_actions = actions[n_train:]
    val_region = region[n_train:]

    sample_weights = action_balance_weights(train_actions, cfg)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_train,
        replacement=True,
    )

    train_loader = DataLoader(
        TensorDataset(train_obs, train_actions),
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        drop_last=False,
    )

    # Optimize only the deterministic mean network.
    # PPO's learnable sigma remains at its original initialization.
    optimizer = torch.optim.Adam(
        policy.net.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_actor_state = deepcopy(policy.net.state_dict())

    print("\nStarting behavior-cloning training...")
    print(f"  train samples:      {n_train:,}")
    print(f"  validation samples: {n_val:,}")
    print(f"  batch size:         {cfg['batch_size']}")
    print(f"  epochs:             {cfg['epochs']}")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        policy.train()

        running_loss = 0.0
        n_batches = 0

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            prediction = policy.net(batch_obs)
            loss = criterion(prediction, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        val_mse, val_mae = evaluate(
            policy,
            val_obs,
            val_actions,
            device=device,
            batch_size=int(cfg["batch_size"]),
        )

        train_mse = running_loss / max(n_batches, 1)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_actor_state = deepcopy(policy.net.state_dict())

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | "
            f"train MSE={train_mse:.6e} | "
            f"val MSE={val_mse:.6e} | "
            f"val MAE={val_mae:.6e}"
        )

    policy.net.load_state_dict(best_actor_state)

    regional_metrics = evaluate_by_region(
        policy=policy,
        obs=val_obs,
        actions=val_actions,
        region=val_region,
        device=device,
        batch_size=int(cfg["batch_size"]),
    )

    print_validation_metrics_by_region(regional_metrics)

    return best_val_mse, regional_metrics


def save_checkpoint(
    policy: Policy,
    cfg: dict,
    best_val_mse: float,
    regional_metrics: dict,
    output_path: str,
):
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "actor_net_state_dict": policy.net.state_dict(),
        "std_parameter": policy.std_parameter.detach().cpu(),
        "pretrain_cfg": cfg,
        "best_validation_mse": best_val_mse,
        "validation_metrics_by_region": regional_metrics,
    }

    torch.save(checkpoint, output_path)

    print(f"\nSaved pretrained policy to: {output_path}")


def main():
    args = parse_args()
    cfg = deepcopy(TEACHER_PRETRAIN_CFG)

    if args.samples is not None:
        cfg["num_samples"] = args.samples
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.spawn_z is not None:
        cfg["spawn_z"] = args.spawn_z
    if args.output is not None:
        cfg["checkpoint_path"] = args.output

    device = torch.device(args.device)

    set_seed(int(cfg["seed"]))
    print_reset_definition(cfg)

    teacher = NonlinearControllerTeacher(
        mass=cfg["mass"],
        Kp=cfg["Kp"],
        Kd=cfg["Kd"],
        max_acceleration=cfg["max_acceleration"],
        device=device,
    )

    obs, actions, raw_acceleration, saturated, region = generate_dataset(
        cfg,
        teacher,
        device,
    )

    print_dataset_statistics(
        actions=actions,
        raw_acceleration=raw_acceleration,
        saturated=saturated,
        region=region,
    )

    if args.dataset is not None:
        torch.save(
            {
                "observations": obs,
                "actions": actions,
                "teacher_raw_acceleration": raw_acceleration,
                "saturated": saturated,
                "region": region,
                "region_names": REGION_NAMES,
                "cfg": cfg,
            },
            args.dataset,
        )

        print(f"\nSaved dataset to: {args.dataset}")

    policy = make_policy(cfg, device)

    best_val_mse, regional_metrics = train(
        policy=policy,
        obs=obs,
        actions=actions,
        region=region,
        cfg=cfg,
        device=device,
    )

    save_checkpoint(
        policy=policy,
        cfg=cfg,
        best_val_mse=best_val_mse,
        regional_metrics=regional_metrics,
        output_path=cfg["checkpoint_path"],
    )


if __name__ == "__main__":
    main()
