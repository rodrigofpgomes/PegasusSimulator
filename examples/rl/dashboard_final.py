#!/usr/bin/env python
"""
dashboard_final.py - Unified multi-vehicle evaluation dashboard (final).

Launches play_multi.py and lets you enable any subset of controllers
(RAPTOR Iris, RAPTOR Crazyflie, pre-train Iris, SAC Iris) in one shared world,
then records and inspects their trajectories live. Features:
  • Controller selection (RAPTOR / Crazyflie toggles + pre-train and SAC checkpoints)
  • Trajectory selector: static goal OR lemniscate (figure-8)
  • Lemniscate parameter controls (amplitude, period, altitude)
  • XY and 3D trajectory plots: actual vs desired (controllers discovered from the CSV)
  • Position-vs-time, velocity, rotor RPM and body-force plots
  • RMSE / mean / max error metrics per controller in the header

Records are written to ./play_records_dashboard_final/ next to this script.

Run with:
    python dashboard_final.py   →  http://127.0.0.1:8055

Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
"""

import math
import os
import json
import shutil
import signal
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update


# -------------------------------------------------------------------------
# Paths & globals
# -------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
RUNS_DIR = APP_DIR / "play_records_dashboard_final"

PROCESS = None
LOG_HANDLE = None
CURRENT_RUN_DIR = None

# Known controllers and their colors. The plots discover whichever controllers
# are actually present in the CSV, so any subset renders correctly; unknown
# names fall back to a neutral gray (see _ctrl_color).
CONTROLLERS = ["raptor_iris", "raptor_cf", "pretrain_iris", "sac_iris"]
CONTROLLER_COLORS = {
    "raptor_iris":   "#1f77b4",   # blue   (RAPTOR foundation policy, Iris)
    "raptor_cf":     "#ff7f0e",   # orange (RAPTOR foundation policy, Crazyflie)
    "pretrain_iris": "#2ca02c",   # green  (RAPTOR pre-training, Iris)
    "sac_iris":      "#9467bd",   # purple (SAC-trained, Iris)
}

# Palette for controllers not in CONTROLLER_COLORS (the manifest can define
# arbitrary names). Colors are assigned deterministically per name.
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Default manifest shown in the JSON editor. Edit/add/remove entries freely;
# play_multi.py consumes this via --config.
DEFAULT_VEHICLES_JSON = json.dumps({
    "vehicles": [
        {"name": "raptor",   "type": "raptor",   "checkpoint": None},
        {"name": "sac",      "type": "sac",      "checkpoint": "/path/to/sac/best_agent.pt"},
        {"name": "pretrain", "type": "pretrain", "checkpoint": "/path/to/pretrain.hdf5"},
        {"name": "sac_skrl", "type": "sac_skrl", "checkpoint": "/path/to/sac/best_agent.pt",
         "task": "raptor_pretrain", "preset": "isaac_lab"},
        {"name": "ppo",      "type": "ppo",      "checkpoint": "/path/to/ppo/best_agent.pt",
         "task": "raptor_pretrain", "preset": "isaac_lab"},
    ]
}, indent=2)


# -------------------------------------------------------------------------
# Isaac / env helpers (same as dashboard3)
# -------------------------------------------------------------------------
def default_isaac_python() -> str:
    env_value = os.environ.get("ISAACSIM_PYTHON")
    if env_value and Path(env_value).exists():
        return env_value
    isaacsim_path = os.environ.get("ISAACSIM_PATH")
    if isaacsim_path:
        candidate = Path(isaacsim_path) / "python.sh"
        if candidate.exists():
            return str(candidate)
    for candidate in [
        Path.home() / "isaacsim_5.1.0" / "python.sh",
        Path("/home/rodrigogomes/isaacsim_5.1.0/python.sh"),
    ]:
        if candidate.exists():
            return str(candidate)
    return "python.sh"


def infer_isaacsim_path(isaac_python: str) -> Path | None:
    p = Path(isaac_python).expanduser()
    if p.name == "python.sh":
        return p.resolve().parent
    env_value = os.environ.get("ISAACSIM_PATH")
    if env_value:
        return Path(env_value)
    return None


def ubuntu_version() -> str:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return "unknown"
    for line in os_release.read_text(errors="replace").splitlines():
        if line.startswith("VERSION_ID="):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def clean_ld_library_path(value: str) -> str:
    blocked = ("/opt/ros/humble", "/opt/ros/jazzy", "/opt/ros/iron")
    parts = [p for p in value.split(":") if p and not any(p.startswith(b) for b in blocked)]
    return ":".join(parts)


def build_isaac_env(isaac_python: str) -> dict:
    env = os.environ.copy()
    for key in ["ROS_VERSION", "ROS_PYTHON_VERSION", "ROS_DISTRO", "AMENT_PREFIX_PATH",
                "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"]:
        env.pop(key, None)
    env["LD_LIBRARY_PATH"] = clean_ld_library_path(env.get("LD_LIBRARY_PATH", ""))
    ros_distro = "jazzy" if ubuntu_version() == "24.04" else "humble"
    env["ROS_DISTRO"] = ros_distro
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    isaacsim_path = infer_isaacsim_path(isaac_python)
    if isaacsim_path is not None:
        bridge_lib = isaacsim_path / "exts" / "isaacsim.ros2.bridge" / ros_distro / "lib"
        if bridge_lib.exists():
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{current}:{bridge_lib}" if current else str(bridge_lib)
    return env


# -------------------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------------------
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_log_tail(path: Path, n_lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception as exc:
        return f"Could not read log: {exc}"


# -------------------------------------------------------------------------
# Lemniscate reference curve (Python-only, for plotting)
# -------------------------------------------------------------------------
def lemniscate_curve(cx: float, cy: float, cz: float, A: float, period: float, n: int = 400):
    """Returns (xs, ys, zs) arrays for one full period of the lemniscate."""
    w = 2.0 * math.pi / period
    t = np.linspace(0.0, 2.0 * math.pi / w, n)
    xs = cx + A * np.sin(w * t)
    ys = cy + 0.5 * A * np.sin(2.0 * w * t)
    zs = np.full_like(xs, cz)
    return xs, ys, zs


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame, skip: int = 0) -> dict:
    if skip > 0:
        df = df.iloc[skip:]
    if df.empty:
        return dict(rmse=float("nan"), mean_err=float("nan"),
                    max_err=float("nan"), max_speed=float("nan"),
                    att_ff_mean=float("nan"), att_pd_mean=float("nan"),
                    att_ff_rmse=float("nan"), att_pd_rmse=float("nan"),
                    att_ff_max=float("nan"), att_pd_max=float("nan"),
                    att_ff_angle_rmse=float("nan"), att_ff_angle_max=float("nan"))

    # Position RMSE
    sq = (df["x"] - df["goal_x"]) ** 2 + (df["y"] - df["goal_y"]) ** 2 + (df["z"] - df["goal_z"]) ** 2

    # Attitude RMSEs
    att_ff_sq = df["att_err_ff"] ** 2 if "att_err_ff" in df.columns else None
    att_pd_sq = df["att_err_pd"] ** 2 if "att_err_pd" in df.columns else None

    att_ff_angle = (
        np.degrees(np.arccos(np.clip(1.0 - df["att_err_ff"] / 2.0, -1.0, 1.0)))
        if "att_err_ff" in df.columns
        else None
    )

    att_pd_angle = (
        np.degrees(np.arccos(np.clip(1.0 - df["att_err_pd"] / 2.0, -1.0, 1.0)))
        if "att_err_pd" in df.columns
        else None
    )
    
    metrics = dict(
        rmse=math.sqrt(sq.mean()),
        mean_err=df["pos_error"].mean(),
        max_err=df["pos_error"].max(),
        max_speed=df["speed"].max(),
        att_ff=df["att_err_ff"].mean()
            if "att_err_ff" in df.columns
            else float("nan"),
        att_pd=df["att_err_pd"].mean()
            if "att_err_pd" in df.columns
            else float("nan"),
        att_ff_rmse=math.sqrt(att_ff_sq.mean())
            if att_ff_sq is not None
            else float("nan"),
        att_pd_rmse=math.sqrt(att_pd_sq.mean())
            if att_pd_sq is not None
            else float("nan"),
        att_ff_max=df["att_err_ff"].max()
            if "att_err_ff" in df.columns
            else float("nan"),
        att_pd_max=df["att_err_pd"].max()
            if "att_err_pd" in df.columns
            else float("nan"),
        att_ff_angle_rmse=math.sqrt(
            (att_ff_angle**2).mean()
        ) if att_ff_angle is not None else float("nan"),
        att_pd_angle_rmse=math.sqrt(
            (att_pd_angle**2).mean()
        ) if att_pd_angle is not None else float("nan"),
    )

    glider_required = {
        "heading_valid",
        "heading_error_deg",
        "heading_alignment",
        "tilt_angle_deg",
        "thrust_5",
        "thrust_5_along_ref",
        "thrust_5_useful",
        "delta_v_slipstream_est",
    }
    
    metrics["has_glider_metrics"] = glider_required.issubset(df.columns)
    
    if metrics["has_glider_metrics"]:
        valid = pd.to_numeric(df["heading_valid"], errors="coerce").fillna(0.0) > 0.5

        dg = df.loc[valid]

        heading_error = pd.to_numeric(
            dg["heading_error_deg"],
            errors="coerce",
        ).to_numpy(dtype=float)

        alignment = pd.to_numeric(
            dg["heading_alignment"],
            errors="coerce",
        ).to_numpy(dtype=float)

        thrust_5 = pd.to_numeric(
            dg["thrust_5"],
            errors="coerce",
        ).to_numpy(dtype=float)

        thrust_along = pd.to_numeric(
            dg["thrust_5_along_ref"],
            errors="coerce",
        ).to_numpy(dtype=float)

        thrust_useful = pd.to_numeric(
            dg["thrust_5_useful"],
            errors="coerce",
        ).to_numpy(dtype=float)

        tilt = pd.to_numeric(
            df["tilt_angle_deg"],
            errors="coerce",
        ).to_numpy(dtype=float)

        delta_v = pd.to_numeric(
            df["delta_v_slipstream_est"],
            errors="coerce",
        ).to_numpy(dtype=float)

        heading_error = heading_error[
            np.isfinite(heading_error)
        ]

        alignment = alignment[
            np.isfinite(alignment)
        ]

        tilt = tilt[np.isfinite(tilt)]
        delta_v = delta_v[np.isfinite(delta_v)]

        finite_thrust = (
            np.isfinite(thrust_5)
            & np.isfinite(thrust_along)
            & np.isfinite(thrust_useful)
        )

        thrust_5 = thrust_5[finite_thrust]
        thrust_along = thrust_along[finite_thrust]
        thrust_useful = thrust_useful[finite_thrust]

        total_thrust = np.sum(thrust_5)

        metrics.update(
            heading_rmse_deg= math.sqrt(np.mean(heading_error**2)) if heading_error.size else float("nan"),
            heading_mae_deg= np.mean(np.abs(heading_error)) if heading_error.size else float("nan"),
            heading_alignment_mean=np.mean(alignment) if alignment.size else float("nan"),
            tilt_rmse_deg= math.sqrt(np.mean(tilt**2)) if tilt.size else float("nan"),
            thrust_5_mean= np.mean(thrust_5) if thrust_5.size else float("nan"),
            thrust_5_along_mean=np.mean(thrust_along) if thrust_along.size else float("nan"),
            useful_thrust_fraction=np.sum(thrust_useful) / (total_thrust + 1e-9) if thrust_5.size else float("nan"),
            signed_thrust_fraction= np.sum(thrust_along) / (total_thrust + 1e-9) if thrust_5.size else float("nan"),
            slipstream_delta_v_mean= np.mean(delta_v) if delta_v.size else float("nan"),
        )

    return metrics


def metrics_text(dff: pd.DataFrame) -> str:
    rows = []
    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl]
        m = compute_metrics(d, skip=10)
        
        text = (
            f"{ctrl}: "
            f"RMSE={m['rmse']:.3f} m  "
            f"mean={m['mean_err']:.3f} m  "
            f"max={m['max_err']:.3f} m  "
            f"max_speed={m['max_speed']:.2f} m/s  "
            f"att_ff_RMSE={m['att_ff_rmse']:.3f}  "
            f"att_ff_angle_RMSE="
            f"{m['att_ff_angle_rmse']:.2f} deg"
        )
        
        if m.get("has_glider_metrics", False):
            text += (
                f"  heading_RMSE="
                f"{m['heading_rmse_deg']:.2f} deg"
                f"  alignment="
                f"{m['heading_alignment_mean']:.3f}"
                f"  tilt_RMSE="
                f"{m['tilt_rmse_deg']:.2f} deg"
                f"  T5_mean="
                f"{m['thrust_5_mean']:.2f} N"
                f"  T5_along="
                f"{m['thrust_5_along_mean']:.2f} N"
                f"  useful_T5="
                f"{100.0 * m['useful_thrust_fraction']:.1f}%"
                f"  signed_T5="
                f"{100.0 * m['signed_thrust_fraction']:.1f}%"
                f"  slipstream_dV="
                f"{m['slipstream_delta_v_mean']:.2f} m/s"
            )
            
        rows.append(text)

    return "  |  ".join(rows) if rows else ""


# -------------------------------------------------------------------------
# Plot builders
# -------------------------------------------------------------------------
def empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", uirevision="keep")
    return fig


def _ctrl_color(name: str) -> str:
    if name in CONTROLLER_COLORS:
        return CONTROLLER_COLORS[name]
    # Stable per-name color from the palette for manifest-defined controllers.
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


# XY trajectory
def make_xy_fig(dff: pd.DataFrame, A: float, period: float, traj_type: str) -> go.Figure:
    fig = go.Figure()
    if dff.empty:
        return empty_fig("XY Trajectory")

    # Desired reference curve (env 0, episode reference)
    if traj_type == "lemniscate" and {"goal_x", "goal_y"}.issubset(dff.columns):
        ref_row = dff[dff["controller"] == sorted(dff["controller"].unique())[0]].sort_values("t")
        if not ref_row.empty:
            # At t=0 the lemniscate is at its centre (sin(0)=0), so iloc[0] gives cx/cy exactly.
            cx = float(ref_row["goal_x"].iloc[0])
            cy = float(ref_row["goal_y"].iloc[0])
            cz = float(ref_row["goal_z"].iloc[0]) if "goal_z" in ref_row.columns else 1.5
            xs, ys, _ = lemniscate_curve(cx, cy, cz, A, period)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="desired", line=dict(color="#888", dash="dash", width=2)))
    
    elif traj_type == "static":
        # Mark the static goal
        ref_row = dff[dff["controller"] == sorted(dff["controller"].unique())[0]]
        if not ref_row.empty and {"goal_x", "goal_y"}.issubset(ref_row.columns):
            fig.add_trace(go.Scatter(
                x=[float(ref_row["goal_x"].iloc[0])],
                y=[float(ref_row["goal_y"].iloc[0])],
                mode="markers", name="goal",
                marker=dict(color="red", size=12, symbol="x"),
            ))

    # Actual trajectories per controller
    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="lines", name=ctrl,
            line=dict(color=_ctrl_color(ctrl), width=2),
        ))

    fig.update_layout(
        title="XY Trajectory - actual vs desired",
        xaxis_title="x [m]", yaxis_title="y [m]",
        yaxis_scaleanchor="x",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        uirevision="keep",
    )
    return fig


# 3D trajector
def make_3d_fig(dff: pd.DataFrame, A: float, period: float, traj_type: str) -> go.Figure:
    fig = go.Figure()
    if dff.empty:
        return empty_fig("3D Trajectory")

    # Desired reference
    if traj_type == "lemniscate" and {"goal_x", "goal_y", "goal_z"}.issubset(dff.columns):
        ref_row = dff[dff["controller"] == sorted(dff["controller"].unique())[0]].sort_values("t")
        if not ref_row.empty:
            cx = float(ref_row["goal_x"].iloc[0])
            cy = float(ref_row["goal_y"].iloc[0])
            cz = float(ref_row["goal_z"].iloc[0])
            xs, ys, zs = lemniscate_curve(cx, cy, cz, A, period)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name="desired",
                line=dict(color="#aaa", dash="dash", width=3),
            ))
    elif traj_type == "static":
        ref_row = dff[dff["controller"] == sorted(dff["controller"].unique())[0]]
        if not ref_row.empty and {"goal_x", "goal_y", "goal_z"}.issubset(ref_row.columns):
            fig.add_trace(go.Scatter3d(
                x=[float(ref_row["goal_x"].iloc[0])],
                y=[float(ref_row["goal_y"].iloc[0])],
                z=[float(ref_row["goal_z"].iloc[0])],
                mode="markers", name="goal",
                marker=dict(color="red", size=6, symbol="diamond"),
            ))

    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"], mode="lines", name=ctrl,
            line=dict(color=_ctrl_color(ctrl), width=4),
        ))

    fig.update_layout(
        title="3D Trajectory",
        template="plotly_white",
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        legend=dict(orientation="h", y=-0.05),
        uirevision="keep",
    )
    return fig


# Single-axis time series
def make_ts_fig(dff: pd.DataFrame, y_col: str, title: str, y_title: str,
                ref_col: str | None = None) -> go.Figure:
    fig = go.Figure()
    if dff.empty or y_col not in dff.columns:
        return empty_fig(title)

    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        fig.add_trace(go.Scatter(
            x=d["t"], y=d[y_col], mode="lines", name=ctrl,
            line=dict(color=_ctrl_color(ctrl)),
        ))

    if ref_col and ref_col in dff.columns:
        ref = dff[dff["controller"] == sorted(dff["controller"].unique())[0]].sort_values("t")
        fig.add_trace(go.Scatter(
            x=ref["t"], y=ref[ref_col], mode="lines", name="reference",
            line=dict(color="#888", dash="dash"),
        ))

    fig.update_layout(title=title, xaxis_title="t [s]", yaxis_title=y_title,
                      template="plotly_white", uirevision="keep")
    return fig


# --- Position error -------------------------------------------------------
def make_pos_error_fig(dff: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if dff.empty:
        return empty_fig("Position error")

    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        if "pos_error" in d.columns:
            fig.add_trace(go.Scatter(
                x=d["t"], y=d["pos_error"], mode="lines", name=f"{ctrl} |e|",
                line=dict(color=_ctrl_color(ctrl)),
            ))
        if "z_error" in d.columns:
            fig.add_trace(go.Scatter(
                x=d["t"], y=d["z_error"], mode="lines", name=f"{ctrl} ez",
                line=dict(color=_ctrl_color(ctrl), dash="dot"),
            ))

    fig.update_layout(title="Position error", xaxis_title="t [s]",
                      yaxis_title="error [m]", template="plotly_white", uirevision="keep")
    return fig


ROTOR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f"]

# --- Rotor RPM -----------------------------------------------------------
def make_rotor_fig(dff: pd.DataFrame, prefix: str, title: str) -> go.Figure:
    fig = go.Figure()
    if dff.empty:
        return empty_fig(title)

    n_rotors = 0
    while f"{prefix}_{n_rotors}" in dff.columns:
        n_rotors += 1
    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        for i in range(n_rotors):
            col = f"{prefix}_{i}"
            if col not in d.columns:
                continue
            fig.add_trace(go.Scatter(
                x=d["t"], y=d[col], mode="lines", name=f"{ctrl} r{i}",
                line=dict(color=ROTOR_COLORS[i % len(ROTOR_COLORS)], dash="dot"),
            ))
    fig.update_layout(title=title, xaxis_title="t [s]", yaxis_title="RPM",
                      template="plotly_white", uirevision="keep")
    return fig


# --- Attitude error ------------------------------------------------------
def make_att_error_fig(dff: pd.DataFrame) -> go.Figure:
    """Attitude error trace(I - Rd^T R) per controller.

    Rd is built from the desired acceleration with the yaw fixed to 0 (see
    play_multi.py). Two curves per controller: '(pd)' uses the full outer-loop
    desired acceleration (PD feedback + feed-forward), '(ff)' uses the
    feed-forward reference only.
    """
    fig = go.Figure()
    if dff.empty:
        return empty_fig("Attitude error")

    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")
        #if "att_err_pd" in d.columns:
        #    fig.add_trace(go.Scatter(
        #        x=d["t"], y=d["att_err_pd"], mode="lines", name=f"{ctrl} (pd)",
        #        line=dict(color=_ctrl_color(ctrl)),
        #    ))
        if "att_err_ff" in d.columns:
            fig.add_trace(go.Scatter(
                x=d["t"], y=d["att_err_ff"], mode="lines", name=f"{ctrl} (ff)",
                line=dict(color=_ctrl_color(ctrl), dash="dot"),
            ))

    fig.update_layout(title="Attitude error  \u03a8 = trace(I - Rd\u1d40 R)",
                      xaxis_title="t [s]", yaxis_title="\u03a8 [-]",
                      template="plotly_white", uirevision="keep")
    return fig


# -------------------------------------------------------------------------
# All figures bundle
# -----------------------------------------s--------------------------------
_N_FIGS = 19


def make_multi_ts_fig(dff: pd.DataFrame, series, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()

    if dff.empty:
        return empty_fig(title)

    added = False
    dashes = ["solid", "dash", "dot", "dashdot"]

    for ctrl in sorted(dff["controller"].unique()):
        d = dff[dff["controller"] == ctrl].sort_values("t")

        for index, (column, label) in enumerate(series):
            if column not in d.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=d["t"],
                    y=d[column],
                    mode="lines",
                    name=f"{ctrl} {label}",
                    line=dict(
                        color=_ctrl_color(ctrl),
                        dash=dashes[
                            index % len(dashes)
                        ],
                    ),
                )
            )

            added = True

    if not added:
        return empty_fig(title)

    fig.update_layout(
        title=title,
        xaxis_title="t [s]",
        yaxis_title=y_title,
        template="plotly_white",
        uirevision="keep",
    )

    return fig



def make_figures(df: pd.DataFrame, env_id: int, episode_id: int,
                 traj_type: str, A: float, period: float):
    empty_all = tuple(empty_fig() for _ in range(_N_FIGS))
    if df.empty:
        return empty_all

    dff = df[(df["env_id"] == env_id) & (df["episode_id"] == episode_id)].copy()
    if dff.empty:
        return empty_all

    label = f"env {env_id} | ep {episode_id}"
    
    return (
        make_xy_fig(dff, A, period, traj_type),
        make_3d_fig(dff, A, period, traj_type),
        make_pos_error_fig(dff),
        make_ts_fig(dff, "x",     f"Position X | {label}", "x [m]",  ref_col="goal_x"),
        make_ts_fig(dff, "y",     f"Position Y | {label}", "y [m]",  ref_col="goal_y"),
        make_ts_fig(dff, "z",     f"Position Z | {label}", "z [m]",  ref_col="goal_z"),
        make_ts_fig(dff, "speed", f"Speed | {label}",      "m/s"),
        make_ts_fig(dff, "vx",    f"Velocity X | {label}", "m/s",    ref_col="ref_vx"),
        make_ts_fig(dff, "vy",    f"Velocity Y | {label}", "m/s",    ref_col="ref_vy"),
        make_ts_fig(dff, "vz",    f"Velocity Z | {label}", "m/s",    ref_col="ref_vz"),
        make_rotor_fig(dff, "cmd_rpm",    f"Commanded RPM | {label}"),
        make_rotor_fig(dff, "actual_rpm", f"Actual RPM | {label}"),
        make_ts_fig(dff, "body_force_norm",  f"Body force | {label}", "|F| [N]"),
        make_att_error_fig(dff),
        
        make_ts_fig(dff, "heading_error_abs_deg", f"Absolute heading error | {label}", "heading error [deg]"),
        make_ts_fig(dff, "heading_alignment", f"Heading alignment | {label}", "cos(e_heading) [-]"),
        make_multi_ts_fig(dff, [("thrust_5", "total"),("thrust_5_along_ref", "along trajectory"), ("thrust_5_useful", "useful")], f"Rotor 5 thrust | {label}", "thrust [N]"),
        make_ts_fig(dff, "tilt_angle_deg", f"Tilt angle | {label}", "tilt [deg]"),
        make_multi_ts_fig(dff, [("v_forward_body", "forward"), ("v_lateral_body", "lateral")], f"Body-frame velocity | {label}", "velocity [m/s]")
    )


# -------------------------------------------------------------------------
# Dash layout
# -------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Multi-Vehicle Dashboard - Final"

_graph_ids = [
    "traj-xy", "traj-3d", "pos-error",
    "pos-x", "pos-y", "pos-z", "speed",
    "vel-x", "vel-y", "vel-z",
    "cmd-rpm", "actual-rpm", "force-norm",
    "att-error",
    "heading-error", "heading-alignment",
    "thrust-5", "tilt-angle",
    "body-velocity",
]

_LABEL_STYLE  = {"fontWeight": "bold", "marginBottom": "2px"}
_INPUT_STYLE  = {"width": "100%"}
_SECTION_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(4, 1fr)",
    "gap": "12px",
    "marginBottom": "8px",
}

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "margin": "24px", "maxWidth": "1440px"},
    children=[
        html.H2("Multi-Vehicle Evaluation - Final Dashboard"),
        html.P("Select controllers, launch play_multi.py, record trajectories and inspect them live."),

        dcc.Store(id="run-dir-store"),
        dcc.Store(id="traj-params-store", data={"type": "none", "A": 1.5, "period": 8.0, "z": 1.5}),

        # ---- Launcher controls ------------------------------------------------
        html.Details(open=True, children=[
            html.Summary(html.B("Simulation settings"), style={"cursor": "pointer"}),
            html.Br(),

            # Row 1: Isaac + device + headless + n_envs + episode_duration
            html.Div(style=_SECTION_STYLE, children=[
                html.Div(style={"gridColumn": "span 2"}, children=[
                    html.Label("Isaac Python", style=_LABEL_STYLE),
                    dcc.Input(id="isaac-python", type="text", value=default_isaac_python(),
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Device", style=_LABEL_STYLE),
                    dcc.Input(id="device", type="text", value="cuda:0", style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Headless", style=_LABEL_STYLE),
                    dcc.Checklist(id="headless", options=[{"label": "enable", "value": "yes"}],
                                  value=["yes"]),
                ]),
            ]),

            # Row 2: envs / duration / episodes / record_every
            html.Div(style=_SECTION_STYLE, children=[
                html.Div([
                    html.Label("N envs", style=_LABEL_STYLE),
                    dcc.Input(id="n-envs", type="number", value=1, min=1, step=1,
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Episode duration [s]", style=_LABEL_STYLE),
                    dcc.Input(id="episode-duration", type="number", value=20.0, min=1.0, step=1.0,
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Episodes per env", style=_LABEL_STYLE),
                    dcc.Input(id="num-episodes", type="number", value=3, min=1, step=1,
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Record every N steps", style=_LABEL_STYLE),
                    dcc.Input(id="record-every", type="number", value=1, min=1, step=1,
                              style=_INPUT_STYLE),
                ]),
            ]),

            # Row 3: Trajectory selector + params
            html.Div(style={"marginBottom": "8px"}, children=[
                html.Label("Trajectory type", style=_LABEL_STYLE),
                dcc.RadioItems(
                    id="traj-type",
                    options=[
                        {"label": "Static goal", "value": "none"},
                        {"label": "Lemniscate (figure-8)", "value": "lemniscate"},
                    ],
                    value="lemniscate",
                    inline=True,
                    style={"marginBottom": "8px"},
                ),
            ]),

            html.Div(id="lemniscate-params-div", style=_SECTION_STYLE, children=[
                html.Div([
                    html.Label("Amplitude A [m]", style=_LABEL_STYLE),
                    dcc.Input(id="traj-amplitude", type="number", value=1.5, min=0.1, step=0.1,
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Period T [s]", style=_LABEL_STYLE),
                    dcc.Input(id="traj-period", type="number", value=8.0, min=1.0, step=0.5,
                              style=_INPUT_STYLE),
                ]),
                html.Div([
                    html.Label("Altitude z [m]", style=_LABEL_STYLE),
                    dcc.Input(id="traj-z", type="number", value=1.5, min=0.1, step=0.1,
                              style=_INPUT_STYLE),
                ]),
            ]),

            html.Div(id="static-params-div", style=_SECTION_STYLE, children=[
                html.Div([
                    html.Label("Goal XY range [m]  (e.g. -2  2)", style=_LABEL_STYLE),
                    html.Div(style={"display": "flex", "gap": "4px"}, children=[
                        dcc.Input(id="goal-xy-low",  type="number", placeholder="low",  style={"width": "50%"}),
                        dcc.Input(id="goal-xy-high", type="number", placeholder="high", style={"width": "50%"}),
                    ]),
                ]),
                html.Div([
                    html.Label("Goal Z range [m]  (e.g. 0.5  1.5)", style=_LABEL_STYLE),
                    html.Div(style={"display": "flex", "gap": "4px"}, children=[
                        dcc.Input(id="goal-z-low",  type="number", placeholder="low",  style={"width": "50%"}),
                        dcc.Input(id="goal-z-high", type="number", placeholder="high", style={"width": "50%"}),
                    ]),
                ]),
            ]),

            # ---- Vehicle model + controllers ------------------------------------
            html.Label("Vehicle model (applies to ALL controllers)", style=_LABEL_STYLE),
            html.Div(style=_SECTION_STYLE, children=[
                html.Div(style={"gridColumn": "span 2"}, children=[
                    dcc.Dropdown(id="vehicle-model",
                                 options=[{"label": "Iris", "value": "iris"},
                                          {"label": "Crazyflie", "value": "crazyflie"},
                                          {"label": "Shuttle", "value": "shuttle"},
                                          {"label": "Shuttle_glider", "value": "shuttle_glider"}],
                                 value="iris", clearable=False),
                ]),
            ]),

            # Controllers — declared as a JSON manifest. play_multi.py reads it via
            # --config; every controller is instantiated on the vehicle model above,
            # so all spawned vehicles are identical.
            html.Div(style=_SECTION_STYLE, children=[
                html.Div(style={"gridColumn": "span 4"}, children=[
                    html.Label("Controllers manifest (JSON) - one entry per controller",
                               style=_LABEL_STYLE),
                    dcc.Textarea(id="vehicles-json", value=DEFAULT_VEHICLES_JSON,
                                 style={"width": "100%", "height": "200px",
                                        "fontFamily": "monospace", "fontSize": "12px"}),
                    html.Div("Each entry: name (unique), type "
                             "(raptor|pretrain|sac|ppo|sac_skrl|nonlinear), checkpoint (path or null). "
                             "sac/pretrain/ppo/sac_skrl need a checkpoint; raptor and nonlinear do not. "
                             "ppo and sac_skrl run the full skrl agent like play.py and also "
                             "need 'task' (+ optional 'preset'). All controllers use the "
                             "vehicle model above.",
                             style={"fontSize": "11px", "color": "#777", "marginTop": "4px"}),
                ]),
            ]),

            html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px"}, children=[
                html.Button("▶ Start run", id="start-button", n_clicks=0,
                            style={"background": "#1f77b4", "color": "white",
                                   "border": "none", "padding": "8px 18px", "cursor": "pointer"}),
                html.Button("■ Stop run",  id="stop-button",  n_clicks=0,
                            style={"background": "#d62728", "color": "white",
                                   "border": "none", "padding": "8px 18px", "cursor": "pointer"}),
            ]),
            html.Div(id="status", style={"marginTop": "10px", "fontWeight": "bold",
                                          "color": "#555"}),
        ]),

        html.Hr(),

        # ---- Viewer controls --------------------------------------------------
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "140px 140px minmax(0, 1fr)", "gap": "12px", "width": "100%", "maxWidth": "100%"},
            children=[
                html.Div([html.Label("Env"),     dcc.Dropdown(id="env-dropdown")]),
                html.Div([html.Label("Episode"), dcc.Dropdown(id="episode-dropdown")]),
                html.Div([
                    html.Label("Metrics", style=_LABEL_STYLE),
                    html.Div(id="metrics-text",
                             style={"fontSize": "12px", "color": "#333",
                                    "paddingTop": "6px", "fontFamily": "monospace", 
                                    "whiteSpace": "pre-wrap", "overflowWrap": "anywhere"})
                ]),
            ],
        ),

        dcc.Interval(id="refresh", interval=1500, n_intervals=0),

        # ---- Trajectory plots ------------------------------------------------
        html.H3("Trajectory - actual vs desired"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                 children=[
                     dcc.Graph(id="traj-xy", style={"height": "520px"}),
                     dcc.Graph(id="traj-3d", style={"height": "520px"}),
                 ]),

        # ---- Position & error -----------------------------------------------
        html.H3("Position"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                 children=[
                     dcc.Graph(id="pos-x"), dcc.Graph(id="pos-y"),
                     dcc.Graph(id="pos-z"), dcc.Graph(id="pos-error"),
                 ]),

        # ---- Velocity -------------------------------------------------------
        html.H3("Velocity"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                 children=[
                     dcc.Graph(id="speed"),
                     dcc.Graph(id="vel-x"),
                     dcc.Graph(id="vel-y"),
                     dcc.Graph(id="vel-z"),
                 ]),

        # ---- Rotors --------------------------------------------------------
        html.H3("Rotor RPM"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                 children=[
                     dcc.Graph(id="cmd-rpm"), dcc.Graph(id="actual-rpm"),
                 ]),

        # ---- Forces --------------------------------------------------------
        html.H3("Body forces"),
        dcc.Graph(id="force-norm"),

        # ---- Attitude error ------------------------------------------------
        html.H3("Attitude error  \u03a8 = trace(I - Rd\u1d40 R)"),
        dcc.Graph(id="att-error"),
        
        # --- Shuttle glider metrics ------------------------------------------
        html.H3("Shuttle glider - heading and rotor utility"),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "8px",
            },
            children=[
                dcc.Graph(id="heading-error"),
                dcc.Graph(id="heading-alignment"),
                dcc.Graph(id="thrust-5"),
                dcc.Graph(id="tilt-angle"),
                dcc.Graph(id="body-velocity"),
            ],
        ),

        # ---- Log -----------------------------------------------------------
        html.H3("Run log"),
        html.Pre(id="log-output", style={
            "background": "#111", "color": "#eee",
            "padding": "12px", "height": "320px",
            "overflowY": "scroll", "fontSize": "12px",
        }),
    ],
)


# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------

# Show/hide param panels depending on mode
@app.callback(
    Output("lemniscate-params-div", "style"),
    Output("static-params-div",     "style"),
    Input("traj-type", "value"),
)
def toggle_params(traj_type):
    active   = dict(_SECTION_STYLE)
    inactive = dict(_SECTION_STYLE, opacity="0.3", pointerEvents="none")
    if traj_type == "lemniscate":
        return active, inactive
    else:
        return inactive, active


# Start / stop simulation
@app.callback(
    Output("run-dir-store",  "data"),
    Output("traj-params-store", "data"),
    Output("status",         "children"),
    Input("start-button",    "n_clicks"),
    Input("stop-button",     "n_clicks"),
    State("isaac-python",    "value"),
    State("n-envs",          "value"),
    State("episode-duration","value"),
    State("num-episodes",    "value"),
    State("record-every",    "value"),
    State("device",          "value"),
    State("headless",        "value"),
    State("traj-type",       "value"),
    State("traj-amplitude",  "value"),
    State("traj-period",     "value"),
    State("traj-z",          "value"),
    State("goal-xy-low",     "value"),
    State("goal-xy-high",    "value"),
    State("goal-z-low",      "value"),
    State("goal-z-high",     "value"),
    State("vehicles-json",   "value"),
    State("vehicle-model",   "value"),
    prevent_initial_call=True,
)
def control_run(
    _start, _stop,
    isaac_python, n_envs, episode_duration, num_episodes, record_every, device, headless,
    traj_type, traj_amplitude, traj_period, traj_z,
    goal_xy_low, goal_xy_high, goal_z_low, goal_z_high,
    vehicles_json, vehicle_model,
):
    global PROCESS, LOG_HANDLE, CURRENT_RUN_DIR

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

    traj_params = {
        "type":   traj_type   or "none",
        "A":      float(traj_amplitude or 1.5),
        "period": float(traj_period    or 8.0),
        "z":      float(traj_z         or 1.5),
    }

    if trigger == "stop-button":
        if PROCESS is not None and PROCESS.poll() is None:
            try:
                os.killpg(os.getpgid(PROCESS.pid), signal.SIGTERM)
                PROCESS.wait(timeout=8)
            except Exception:
                try:
                    PROCESS.kill()
                except Exception:
                    pass
        if LOG_HANDLE is not None:
            LOG_HANDLE.close()
            LOG_HANDLE = None
        return str(CURRENT_RUN_DIR) if CURRENT_RUN_DIR else None, traj_params, "Stopped."

    if PROCESS is not None and PROCESS.poll() is None:
        return str(CURRENT_RUN_DIR), traj_params, "A run is already active."

    if not isaac_python:
        return no_update, traj_params, "Missing Isaac Python executable."

    isaac_python_path = Path(str(isaac_python)).expanduser()
    if not isaac_python_path.exists():
        return no_update, traj_params, f"Isaac Python not found: {isaac_python_path}"

    # Parse + validate the vehicles manifest (JSON editor).
    try:
        manifest = json.loads(vehicles_json or "")
    except Exception as e:
        return no_update, traj_params, f"Invalid vehicles JSON: {e}"
    vehicles = manifest.get("vehicles") if isinstance(manifest, dict) else manifest
    if not vehicles:
        return no_update, traj_params, "Vehicles manifest has no 'vehicles' entries."
    _known_types = {"raptor", "pretrain", "sac", "ppo", "sac_skrl", "nonlinear"}
    _seen = set()
    for i, entry in enumerate(vehicles):
        nm = entry.get("name")
        if not nm:
            return no_update, traj_params, f"Vehicle #{i} is missing 'name'."
        if nm in _seen:
            return no_update, traj_params, f"Duplicate controller name '{nm}'."
        _seen.add(nm)
        if entry.get("type") not in _known_types:
            return no_update, traj_params, f"Controller '{nm}': unknown type '{entry.get('type')}'."
        if entry.get("type") in ("sac", "pretrain", "ppo", "sac_skrl") and not entry.get("checkpoint"):
            return no_update, traj_params, f"Controller '{nm}' ({entry.get('type')}) needs a 'checkpoint'."
        if entry.get("type") in ("ppo", "sac_skrl") and not entry.get("task"):
            return no_update, traj_params, f"Controller '{nm}' ({entry.get('type')}) needs a 'task' (skrl agent config, like play.py)."

    # Single fixed temporary folder: reused and wiped clean on every run so only
    # the latest run's CSV/logs are ever present (no timestamped history).
    CURRENT_RUN_DIR = RUNS_DIR
    if CURRENT_RUN_DIR.exists():
        shutil.rmtree(CURRENT_RUN_DIR, ignore_errors=True)
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the manifest (incl. the global vehicle model) and point
    # play_multi.py at it via --config. All controllers use this vehicle model.
    vehicle_model = (vehicle_model or "iris")
    config_path = CURRENT_RUN_DIR / "vehicles.json"
    with open(config_path, "w") as f:
        json.dump({"vehicle": vehicle_model, "vehicles": vehicles}, f, indent=2)

    cmd = [
        str(isaac_python_path),
        str(APP_DIR / "play_multi.py"),
        "--n_envs",               str(int(n_envs or 1)),
        "--episode_duration",     str(float(episode_duration or 20.0)),
        "--device",               str(device or "cuda:0"),
        "--record",
        "--record_dir",           str(CURRENT_RUN_DIR),
        "--record_every",         str(int(record_every or 1)),
        "--num_episodes_per_env", str(int(num_episodes or 3)),
        "--trajectory",           traj_params["type"],
    ]

    cmd += ["--config", str(config_path), "--vehicle", vehicle_model]

    if "yes" in (headless or []):
        cmd.append("--headless")

    if traj_params["type"] == "lemniscate":
        cmd += [
            "--traj_amplitude", str(traj_params["A"]),
            "--traj_period",    str(traj_params["period"]),
            "--traj_z",         str(traj_params["z"]),
        ]
    else:
        if goal_xy_low is not None and goal_xy_high is not None:
            cmd += ["--goal_xy_range", str(float(goal_xy_low)), str(float(goal_xy_high))]
        if goal_z_low is not None and goal_z_high is not None:
            cmd += ["--goal_z_range", str(float(goal_z_low)), str(float(goal_z_high))]

    log_path = CURRENT_RUN_DIR / "run.log"
    LOG_HANDLE = open(log_path, "w", buffering=1)
    LOG_HANDLE.write(f"[DashboardFinal] cwd: {PROJECT_ROOT}\n")
    LOG_HANDLE.write(f"[DashboardFinal] cmd: {' '.join(cmd)}\n")
    LOG_HANDLE.flush()

    PROCESS = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=LOG_HANDLE, stderr=subprocess.STDOUT,
        text=True, env=build_isaac_env(str(isaac_python_path)),
        start_new_session=True,
    )

    return str(CURRENT_RUN_DIR), traj_params, f"Started - {CURRENT_RUN_DIR.name}"


# Update dropdowns + log
@app.callback(
    Output("env-dropdown",     "options"),
    Output("env-dropdown",     "value"),
    Output("episode-dropdown", "options"),
    Output("episode-dropdown", "value"),
    Output("log-output",       "children"),
    Input("refresh",           "n_intervals"),
    Input("run-dir-store",     "data"),
    State("env-dropdown",      "value"),
    State("episode-dropdown",  "value"),
)
def update_options(_, run_dir, current_env, current_ep):
    if not run_dir:
        return [], None, [], None, ""
    run_dir = Path(run_dir)
    log_text = read_log_tail(run_dir / "run.log")
    df = read_csv_safe(run_dir / "trajectories.csv")
    if df.empty:
        return [], None, [], None, log_text

    envs = sorted(int(v) for v in df["env_id"].dropna().unique())
    env_value = current_env if current_env in envs else envs[0]

    episodes = sorted(int(v) for v in df[df["env_id"] == env_value]["episode_id"].dropna().unique())
    if not episodes:
        return [{"label": f"env {e}", "value": e} for e in envs], env_value, [], None, log_text

    ep_value = current_ep if current_ep in episodes else episodes[0]
    return (
        [{"label": f"env {e}", "value": e} for e in envs], env_value,
        [{"label": f"ep {e}",  "value": e} for e in episodes], ep_value,
        log_text,
    )


# Update all graphs + metrics text
@app.callback(
    *[Output(gid, "figure") for gid in _graph_ids],
    Output("metrics-text", "children"),
    Input("refresh",           "n_intervals"),
    Input("run-dir-store",     "data"),
    Input("env-dropdown",      "value"),
    Input("episode-dropdown",  "value"),
    State("traj-params-store", "data"),
)
def update_graphs(_, run_dir, env_id, episode_id, traj_params):
    empty_all = tuple(empty_fig() for _ in _graph_ids) + ("",)
    if not run_dir or env_id is None or episode_id is None:
        return empty_all

    params = traj_params or {"type": "none", "A": 1.5, "period": 8.0, "z": 1.5}
    traj_type = params.get("type", "none")
    A      = float(params.get("A",      1.5))
    period = float(params.get("period", 8.0))

    run_dir = Path(run_dir)
    df = read_csv_safe(run_dir / "trajectories.csv")
    if df.empty:
        return empty_all

    dff = df[(df["env_id"] == int(env_id)) & (df["episode_id"] == int(episode_id))]
    metrics_str = metrics_text(dff)

    figs = make_figures(df, int(env_id), int(episode_id), traj_type, A, period)
    return figs + (metrics_str,)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, host="127.0.0.1", port=8055)
