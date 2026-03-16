# Copyright (c) 2022–2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from typing import Any, Tuple
from rsl_rl.modules.actor_critic import ActorCritic


def _to_int_dim(x: Any) -> int:
    """Converte tipos variados (int, dict, list, tensor, etc.) para int."""
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if hasattr(x, "shape"):  # tensor ou torch.Size
        s = tuple(x.shape)
        return int(s[-1]) if len(s) > 0 else 0
    if isinstance(x, (list, tuple)):
        for elem in reversed(x):
            d = _to_int_dim(elem)
            if d > 0:
                return d
        return 0
    if isinstance(x, dict):
        for v in x.values():
            d = _to_int_dim(v)
            if d > 0:
                return d
        return 0
    if isinstance(x, str):
        return 0
    try:
        return int(x)
    except Exception:
        return 0


def infer_feat_dim(x: Any) -> int:
    """
    Tenta obter a dimensão de features (último eixo) a partir de TensorDicts, dicts ou tensores.
    """
    # TensorDict com campo 'policy'
    try:
        if hasattr(x, "get") and callable(getattr(x, "get")):
            v = x.get("policy", None)
            if v is not None and hasattr(v, "shape") and len(v.shape) >= 2:
                return int(v.shape[-1])
    except Exception:
        pass

    # Mapping genérico
    try:
        if hasattr(x, "values"):
            for v in x.values():
                if hasattr(v, "shape") and len(v.shape) >= 2:
                    return int(v.shape[-1])
    except Exception:
        pass

    # Tensor
    if hasattr(x, "shape"):
        s = tuple(x.shape)
        return int(s[-1]) if len(s) > 0 else 0

    return _to_int_dim(x)


def infer_act_dim(x: Any, default: int = 3) -> int:
    """
    Tenta extrair dimensão de ação, ou retorna default (3) se não conseguir.
    """
    d = _to_int_dim(x)
    if d > 0:
        return d
    try:
        if hasattr(x, "get"):
            for k in ("action_dim", "actions", "act", "policy"):
                v = x.get(k, None)
                dv = _to_int_dim(v)
                if dv > 0:
                    return dv
    except Exception:
        pass
    return default


# ============================================================
# Actor: Desacoplado por eixo com saída tanh
# ============================================================

class AxisDecoupledPDActor(nn.Module):
    """
    Actor desacoplado por eixo.
    Cada saída (ax, ay, az) depende apenas do respetivo par de observações (ep?, ev?).
    A saída final está em [-1, 1] via tanh.
    """

    def __init__(self, obs_dim, act_dim, act_func = True, bias_z=False):
        super().__init__()

        obs_dim = infer_feat_dim(obs_dim)
        act_dim = infer_act_dim(act_dim, default=3)

        if obs_dim < 6:
            raise ValueError(f"obs_dim deve ser >= 6 (obtido {obs_dim}).")
        if act_dim != 3:
            print(f"[AxisDecoupledTanhActor] Aviso: act_dim={act_dim} (esperado 3). Forçando para 3.")
            act_dim = 3

        self.Kpx = nn.Parameter(0.05 * torch.randn(()))
        self.theta_dx = nn.Parameter(torch.randn(())) 

        self.Kpy = nn.Parameter(0.05 * torch.randn(()))
        self.theta_dy = nn.Parameter(0.05 * torch.randn(()))

        self.Kpz = nn.Parameter(0.05 * torch.randn(()))
        self.theta_dz = nn.Parameter(0.05 * torch.randn(()))
        self.bias_z = nn.Parameter(torch.zeros(())) if bias_z else None

        # [epx, epy, epz, evx, evy, evz]
        self.idx_x: Tuple[int, int] = (0, 3)
        self.idx_y: Tuple[int, int] = (1, 4)
        self.idx_z: Tuple[int, int] = (2, 5)

        self.in_features = 6  # ONNX

        self.act_func = act_func


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if bool(obs.dim() != 2 or obs.size(1) < 6):
            raise RuntimeError("obs deve ter shape [B, >=6]")

        x_pair = obs[:, [self.idx_x[0], self.idx_x[1]]]
        y_pair = obs[:, [self.idx_y[0], self.idx_y[1]]]
        z_pair = obs[:, [self.idx_z[0], self.idx_z[1]]]

        epx = x_pair[:, 0]
        evx = x_pair[:, 1]
        Kdx = self.theta_dx * self.theta_dx
        raw_x = self.Kpx * epx + Kdx * evx

        epy = y_pair[:, 0]
        evy = y_pair[:, 1]
        Kdy = self.theta_dy * self.theta_dy
        raw_y = self.Kpy * epy + Kdy * evy

        # extract EPZ and EVZ
        epz = z_pair[:, 0]
        evz = z_pair[:, 1]
        Kdz = self.theta_dz * self.theta_dz   # square to force positivity
        raw_z = self.Kpz * epz + Kdz * evz

        if self.act_func:
            out_x = torch.tanh(raw_x).unsqueeze(-1)
            out_y = torch.tanh(raw_y).unsqueeze(-1)
            out_z = torch.tanh(raw_z).unsqueeze(-1)
        else:
            out_x = raw_x.unsqueeze(-1)
            out_y = raw_y.unsqueeze(-1)
            out_z = raw_z.unsqueeze(-1)
    
        return torch.cat((out_x, out_y, out_z), dim=1)


class AxisDecoupledPDActorLinear(nn.Module):
    """
    Actor desacoplado por eixo.
    Cada saída (ax, ay, az) depende apenas do respetivo par de observações (ep?, ev?).
    A saída final está em [-1, 1] via tanh.
    """

    def __init__(self, obs_dim, act_dim, act_func = True, bias_z=False):
        super().__init__()

        obs_dim = infer_feat_dim(obs_dim)
        act_dim = infer_act_dim(act_dim, default=3)

        if obs_dim < 6:
            raise ValueError(f"obs_dim deve ser >= 6 (obtido {obs_dim}).")
        if act_dim != 3:
            print(f"[AxisDecoupledTanhActor] Aviso: act_dim={act_dim} (esperado 3). Forçando para 3.")
            act_dim = 3

        self.fcx = nn.Linear(2, 1, bias=False)
        self.fcy = nn.Linear(2, 1, bias=False)
        self.fcz = nn.Linear(2, 1, bias=bias_z)

        nn.init.uniform_(self.fcx.weight, -0.1, 0.1)
        nn.init.uniform_(self.fcy.weight, -0.1, 0.1)
        nn.init.uniform_(self.fcz.weight, -0.1, 0.1)

        # [epx, epy, epz, evx, evy, evz]
        self.idx_x: Tuple[int, int] = (0, 3)
        self.idx_y: Tuple[int, int] = (1, 4)
        self.idx_z: Tuple[int, int] = (2, 5)

        self.in_features = 6  # ONNX

        self.act_func = act_func


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if bool(obs.dim() != 2 or obs.size(1) < 6):
            raise RuntimeError("obs deve ter shape [B, >=6]")

        x_pair = obs[:, [self.idx_x[0], self.idx_x[1]]]
        y_pair = obs[:, [self.idx_y[0], self.idx_y[1]]]
        z_pair = obs[:, [self.idx_z[0], self.idx_z[1]]]
        

        if self.act_func:
            out_x = torch.tanh(self.fcx(x_pair))
            out_y = torch.tanh(self.fcy(y_pair))
            out_z = torch.tanh(self.fcz(z_pair))
        else:
            out_x = self.fcx(x_pair)
            out_y = self.fcy(y_pair)
            out_z = self.fcz(z_pair)
    
        return torch.cat((out_x, out_y, out_z), dim=1)

# ============================================================
# Critic: xᵀ P x
# ============================================================

class QuadraticFormCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        obs_dim = infer_feat_dim(obs_dim)
        if obs_dim != 6:
            raise ValueError(f"obs_dim deve ser = 6 (obtido {obs_dim}).")

        self.Px = nn.Parameter(torch.randn(2, 2))
        self.Py = nn.Parameter(torch.randn(2, 2))
        self.Pz = nn.Parameter(torch.randn(2, 2))

        self.idx_x = (0, 3)
        self.idx_y = (1, 4)
        self.idx_z = (2, 5)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.size(1) != 6:
            raise RuntimeError(f"obs deve ter shape [B, {self.P.size(0)}]")

        x = obs[:, [self.idx_x[0], self.idx_x[1]]]  # (B, 2)
        y = obs[:, [self.idx_y[0], self.idx_y[1]]]  # (B, 2)
        z = obs[:, [self.idx_z[0], self.idx_z[1]]]  # (B, 2)

        # Simetrizar P
        Px = 0.5 * (self.Px + self.Px.T)
        Py = 0.5 * (self.Py + self.Py.T)
        Pz = 0.5 * (self.Pz + self.Pz.T)

        xPx = torch.einsum('bi,ij,bj->b', x, Px, x)
        yPy = torch.einsum('bi,ij,bj->b', y, Py, y)
        zPz = torch.einsum('bi,ij,bj->b', z, Pz, z)

        # xᵀ P x por amostra
        V = xPx + yPy + zPz

        return -V.unsqueeze(1)  # (B, 1)

class LQRCritic(nn.Module):
    """
    Critic que aprende o kernel S da função-Q do LQR:
        Q(x,u) = 0.5 * [x,u]^T S [x,u]
    """
    def __init__(self, obs_dim=6, act_dim=3):
        super().__init__()

        obs_dim = infer_feat_dim(obs_dim)
        act_dim = infer_act_dim(act_dim)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        S = torch.randn(obs_dim + act_dim, obs_dim + act_dim)
        self.S_param = nn.Parameter(0.5*(S + S.T))

    def forward(self, obs):
        B = obs.size(0)

        S = 0.5 * (self.S_param + self.S_param.T)

        S_xx = S[:self.obs_dim, :self.obs_dim]
        S_xu = S[:self.obs_dim, self.obs_dim:]
        S_ux = S[self.obs_dim:, :self.obs_dim]
        S_uu = S[self.obs_dim:, self.obs_dim:]

        K = torch.linalg.solve(S_uu, S_ux)   # (3x6)

        # matriz P equivalente
        P = S_xx - S_xu @ K  # (6x6)

        # V(x) = -1/2 x^T P x
        V = torch.einsum('bi,ij,bj->b', obs, P, obs)

        return -V.unsqueeze(1)
