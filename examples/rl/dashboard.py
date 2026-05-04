#!/usr/bin/env python

import os
import sys
import signal
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
RUNS_DIR = APP_DIR / "play_records_web"

PROCESS = None
LOG_HANDLE = None
CURRENT_RUN_DIR = None


# ---------------------------------------------------------------------
# Isaac helpers
# ---------------------------------------------------------------------
def default_isaac_python() -> str:
    env_value = os.environ.get("ISAACSIM_PYTHON")
    if env_value and Path(env_value).exists():
        return env_value

    isaacsim_path = os.environ.get("ISAACSIM_PATH")
    if isaacsim_path:
        candidate = Path(isaacsim_path) / "python.sh"
        if candidate.exists():
            return str(candidate)

    candidates = [
        Path.home() / "isaacsim_5.1.0" / "python.sh",
        Path("/home/rodrigogomes/isaacsim_5.1.0/python.sh"),
    ]

    for candidate in candidates:
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
    if not value:
        return ""

    blocked = (
        "/opt/ros/humble",
        "/opt/ros/jazzy",
        "/opt/ros/iron",
    )

    parts = []
    for part in value.split(":"):
        if not part:
            continue

        if any(part.startswith(prefix) for prefix in blocked):
            continue

        parts.append(part)

    return ":".join(parts)


def build_isaac_env(isaac_python: str) -> dict:
    """
    Build a clean environment for Isaac Sim.

    Avoids depending on the shell function isaac_run.
    Avoids sourcing ROS setup files with Isaac Python, which can cause:
        AssertionError: SRE module mismatch
    """
    env = os.environ.copy()

    for key in [
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "ROS_DISTRO",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
    ]:
        env.pop(key, None)

    env["LD_LIBRARY_PATH"] = clean_ld_library_path(env.get("LD_LIBRARY_PATH", ""))

    version = ubuntu_version()

    if version == "24.04":
        ros_distro = "jazzy"
    else:
        ros_distro = "humble"

    env["ROS_DISTRO"] = ros_distro
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

    isaacsim_path = infer_isaacsim_path(isaac_python)

    if isaacsim_path is not None:
        bridge_lib = isaacsim_path / "exts" / "isaacsim.ros2.bridge" / ros_distro / "lib"

        if bridge_lib.exists():
            current = env.get("LD_LIBRARY_PATH", "")
            if current:
                env["LD_LIBRARY_PATH"] = f"{current}:{bridge_lib}"
            else:
                env["LD_LIBRARY_PATH"] = str(bridge_lib)

    return env


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(title=title)
    return fig


def _has_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].dropna().empty


def _reference_df(dff: pd.DataFrame) -> pd.DataFrame:
    """
    Reference is duplicated for RL/LQR rows, so keep only one row per time.
    """
    if dff.empty:
        return dff

    return (
        dff.sort_values("t")
        .drop_duplicates(subset=["env_id", "episode_id", "t"])
    )


def _add_reference_trace(fig, ref_df, x_col, y_col, row, name):
    if ref_df.empty:
        return

    if not _has_column(ref_df, y_col):
        return

    fig.add_trace(
        go.Scatter(
            x=ref_df[x_col],
            y=ref_df[y_col],
            mode="lines",
            name=name,
            line=dict(dash="dash"),
        ),
        row=row,
        col=1,
    )


def make_single_axis_fig(
    dff: pd.DataFrame,
    y_col: str,
    title: str,
    y_title: str,
    ref_col: str | None = None,
):
    fig = go.Figure()

    if dff.empty:
        fig.update_layout(title=title)
        return fig

    controllers = sorted(dff["controller"].unique())

    for controller in controllers:
        d = dff[dff["controller"] == controller]

        if y_col in d.columns:
            fig.add_trace(go.Scatter(
                x=d["t"],
                y=d[y_col],
                mode="lines",
                name=controller,
            ))

    # Reference line, only once
    if ref_col is not None and ref_col in dff.columns:
        ref = dff[dff["controller"] == controllers[0]] if controllers else dff

        fig.add_trace(go.Scatter(
            x=ref["t"],
            y=ref[ref_col],
            mode="lines",
            name="reference",
            line=dict(dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="t [s]",
        yaxis_title=y_title,
    )

    return fig


def make_figures(df: pd.DataFrame, env_id: int, episode_id: int):
    if df.empty:
        return tuple([
            empty_fig("Position X"),
            empty_fig("Position Y"),
            empty_fig("Position Z"),
            empty_fig("Velocity X"),
            empty_fig("Velocity Y"),
            empty_fig("Velocity Z"),
            empty_fig("Body force X"),
            empty_fig("Body force Y"),
            empty_fig("Body force Z"),
            empty_fig("3D trajectory"),
        ])

    dff = df[
        (df["env_id"] == env_id)
        & (df["episode_id"] == episode_id)
    ]

    if dff.empty:
        return tuple([
            empty_fig("Position X"),
            empty_fig("Position Y"),
            empty_fig("Position Z"),
            empty_fig("Velocity X"),
            empty_fig("Velocity Y"),
            empty_fig("Velocity Z"),
            empty_fig("Body force X"),
            empty_fig("Body force Y"),
            empty_fig("Body force Z"),
            empty_fig("3D trajectory"),
        ])

    fig_px = make_single_axis_fig(
        dff,
        y_col="x",
        ref_col="goal_x",
        title=f"Position X | env {env_id} | episode {episode_id}",
        y_title="x [m]",
    )

    fig_py = make_single_axis_fig(
        dff,
        y_col="y",
        ref_col="goal_y",
        title=f"Position Y | env {env_id} | episode {episode_id}",
        y_title="y [m]",
    )

    fig_pz = make_single_axis_fig(
        dff,
        y_col="z",
        ref_col="goal_z",
        title=f"Position Z | env {env_id} | episode {episode_id}",
        y_title="z [m]",
    )

    fig_vx = make_single_axis_fig(
        dff,
        y_col="vx",
        title=f"Velocity X | env {env_id} | episode {episode_id}",
        y_title="vx [m/s]",
    )

    fig_vy = make_single_axis_fig(
        dff,
        y_col="vy",
        title=f"Velocity Y | env {env_id} | episode {episode_id}",
        y_title="vy [m/s]",
    )

    fig_vz = make_single_axis_fig(
        dff,
        y_col="vz",
        title=f"Velocity Z | env {env_id} | episode {episode_id}",
        y_title="vz [m/s]",
    )

    fig_fx = make_single_axis_fig(
        dff,
        y_col="fx_body",
        title=f"Applied body force X | env {env_id} | episode {episode_id}",
        y_title="Fx body [N]",
    )

    fig_fy = make_single_axis_fig(
        dff,
        y_col="fy_body",
        title=f"Applied body force Y | env {env_id} | episode {episode_id}",
        y_title="Fy body [N]",
    )

    fig_fz = make_single_axis_fig(
        dff,
        y_col="fz_body",
        title=f"Applied body force Z | env {env_id} | episode {episode_id}",
        y_title="Fz body [N]",
    )

    fig_3d = go.Figure()

    controllers = sorted(dff["controller"].unique())

    for controller in controllers:
        d = dff[dff["controller"] == controller]

        fig_3d.add_trace(go.Scatter3d(
            x=d["x"],
            y=d["y"],
            z=d["z"],
            mode="lines",
            name=controller,
        ))

    # Add reference point / trajectory
    if {"goal_x", "goal_y", "goal_z"}.issubset(dff.columns):
        ref = dff[dff["controller"] == controllers[0]] if controllers else dff

        fig_3d.add_trace(go.Scatter3d(
            x=ref["goal_x"],
            y=ref["goal_y"],
            z=ref["goal_z"],
            mode="markers",
            name="reference",
            marker=dict(size=4),
        ))

    fig_3d.update_layout(
        title=f"3D trajectory | env {env_id} | episode {episode_id}",
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
        ),
    )

    return (
        fig_px,
        fig_py,
        fig_pz,
        fig_vx,
        fig_vy,
        fig_vz,
        fig_fx,
        fig_fy,
        fig_fz,
        fig_3d,
    )




# ---------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------
app = Dash(__name__)
app.title = "Pegasus RL vs LQR Dashboard"

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "margin": "24px"},
    children=[
        html.H2("Pegasus RL / LQR trajectory dashboard"),

        dcc.Store(id="run-dir-store"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "12px",
                "maxWidth": "1300px",
            },
            children=[
                html.Div(style={"gridColumn": "span 2"}, children=[
                    html.Label("Isaac Python executable"),
                    dcc.Input(
                        id="isaac-python",
                        type="text",
                        value=default_isaac_python(),
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Task"),
                    dcc.Input(
                        id="task",
                        type="text",
                        value="quadcopter3",
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Algorithm"),
                    dcc.Input(
                        id="algo",
                        type="text",
                        value="ppo",
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Preset"),
                    dcc.Input(
                        id="preset",
                        type="text",
                        value="isaac_lab",
                        style={"width": "100%"},
                    ),
                ]),
                html.Div(style={"gridColumn": "span 3"}, children=[
                    html.Label("Checkpoint"),
                    dcc.Input(
                        id="checkpoint",
                        type="text",
                        placeholder="/path/to/checkpoint.pt",
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("N envs"),
                    dcc.Input(
                        id="n-envs",
                        type="number",
                        value=4,
                        min=1,
                        step=1,
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Trajetórias por env"),
                    dcc.Input(
                        id="num-episodes",
                        type="number",
                        value=3,
                        min=1,
                        step=1,
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Record every N steps"),
                    dcc.Input(
                        id="record-every",
                        type="number",
                        value=1,
                        min=1,
                        step=1,
                        style={"width": "100%"},
                    ),
                ]),
                html.Div([
                    html.Label("Compare LQR"),
                    dcc.Checklist(
                        id="compare-lqr",
                        options=[{"label": "enable", "value": "yes"}],
                        value=["yes"],
                    ),
                ]),
                html.Div([
                    html.Label("Headless"),
                    dcc.Checklist(
                        id="headless",
                        options=[{"label": "enable", "value": "yes"}],
                        value=[],
                    ),
                ]),
            ],
        ),

        html.Br(),

        html.Button("Start run", id="start-button", n_clicks=0),
        html.Button(
            "Stop run",
            id="stop-button",
            n_clicks=0,
            style={"marginLeft": "8px"},
        ),

        html.Div(id="status", style={"marginTop": "12px", "fontWeight": "bold"}),

        html.Hr(),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "12px",
                "maxWidth": "700px",
            },
            children=[
                html.Div([
                    html.Label("Env"),
                    dcc.Dropdown(id="env-dropdown"),
                ]),
                html.Div([
                    html.Label("Trajetória / episódio"),
                    dcc.Dropdown(id="episode-dropdown"),
                ]),
            ],
        ),

        dcc.Interval(id="refresh", interval=1000, n_intervals=0),

        html.H3("Position"),
        dcc.Graph(id="position-x-graph"),
        dcc.Graph(id="position-y-graph"),
        dcc.Graph(id="position-z-graph"),

        html.H3("Velocity"),
        dcc.Graph(id="velocity-x-graph"),
        dcc.Graph(id="velocity-y-graph"),
        dcc.Graph(id="velocity-z-graph"),

        html.H3("Applied body force"),
        dcc.Graph(id="body-force-x-graph"),
        dcc.Graph(id="body-force-y-graph"),
        dcc.Graph(id="body-force-z-graph"),

        html.H3("3D trajectory"),
        dcc.Graph(id="trajectory-graph"),

        html.H3("Run log"),
        html.Pre(
            id="log-output",
            style={
                "background": "#111",
                "color": "#eee",
                "padding": "12px",
                "height": "360px",
                "overflowY": "scroll",
                "fontSize": "12px",
            },
        ),
    ],
)


@app.callback(
    Output("run-dir-store", "data"),
    Output("status", "children"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("isaac-python", "value"),
    State("task", "value"),
    State("algo", "value"),
    State("preset", "value"),
    State("checkpoint", "value"),
    State("n-envs", "value"),
    State("num-episodes", "value"),
    State("record-every", "value"),
    State("compare-lqr", "value"),
    State("headless", "value"),
    prevent_initial_call=True,
)
def control_run(
    start_clicks,
    stop_clicks,
    isaac_python,
    task,
    algo,
    preset,
    checkpoint,
    n_envs,
    num_episodes,
    record_every,
    compare_lqr,
    headless,
):
    global PROCESS, LOG_HANDLE, CURRENT_RUN_DIR

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

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

        return (
            str(CURRENT_RUN_DIR) if CURRENT_RUN_DIR is not None else None,
            "Stopped.",
        )

    if PROCESS is not None and PROCESS.poll() is None:
        return str(CURRENT_RUN_DIR), "A run is already active."

    if not checkpoint:
        return no_update, "Missing checkpoint path."

    if not isaac_python:
        return no_update, "Missing Isaac Python executable."

    isaac_python_path = Path(str(isaac_python)).expanduser()

    if not isaac_python_path.exists():
        return no_update, f"Isaac Python executable not found: {isaac_python_path}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_RUN_DIR = RUNS_DIR / f"{task}_{timestamp}"
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(isaac_python_path),
        str(APP_DIR / "play.py"),
        "--task", str(task),
        "--algo", str(algo),
        "--preset", str(preset),
        "--checkpoint", str(checkpoint),
        "--n_envs", str(int(n_envs)),
        "--record",
        "--record_dir", str(CURRENT_RUN_DIR),
        "--record_every", str(int(record_every)),
        "--num_episodes_per_env", str(int(num_episodes)),
    ]

    if "yes" in compare_lqr:
        cmd.append("--compare_lqr")

    if "yes" in headless:
        cmd.append("--headless")

    log_path = CURRENT_RUN_DIR / "run.log"
    LOG_HANDLE = open(log_path, "w", buffering=1)

    LOG_HANDLE.write(f"[Dashboard] cwd: {PROJECT_ROOT}\n")
    LOG_HANDLE.write(f"[Dashboard] cmd: {' '.join(cmd)}\n")
    LOG_HANDLE.write(f"[Dashboard] record_dir: {CURRENT_RUN_DIR}\n")
    LOG_HANDLE.flush()

    env = build_isaac_env(str(isaac_python_path))

    PROCESS = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )

    return str(CURRENT_RUN_DIR), f"Started run in {CURRENT_RUN_DIR}"


@app.callback(
    Output("env-dropdown", "options"),
    Output("env-dropdown", "value"),
    Output("episode-dropdown", "options"),
    Output("episode-dropdown", "value"),
    Output("log-output", "children"),
    Input("refresh", "n_intervals"),
    Input("run-dir-store", "data"),
    State("env-dropdown", "value"),
    State("episode-dropdown", "value"),
)
def update_options(_, run_dir, current_env, current_episode):
    if not run_dir:
        return [], None, [], None, ""

    run_dir = Path(run_dir)

    log_text = read_log_tail(run_dir / "run.log")
    df = read_csv_safe(run_dir / "trajectories.csv")

    if df.empty:
        return [], None, [], None, log_text

    envs = sorted(int(v) for v in df["env_id"].dropna().unique())

    if current_env in envs:
        env_value = current_env
    else:
        env_value = envs[0]

    df_env = df[df["env_id"] == env_value]
    episodes = sorted(int(v) for v in df_env["episode_id"].dropna().unique())

    if not episodes:
        return (
            [{"label": f"env {e}", "value": e} for e in envs],
            env_value,
            [],
            None,
            log_text,
        )

    episode_value = current_episode if current_episode in episodes else episodes[0]

    env_options = [{"label": f"env {e}", "value": e} for e in envs]
    episode_options = [{"label": f"episode {e}", "value": e} for e in episodes]

    return env_options, env_value, episode_options, episode_value, log_text


@app.callback(
    Output("position-x-graph", "figure"),
    Output("position-y-graph", "figure"),
    Output("position-z-graph", "figure"),
    Output("velocity-x-graph", "figure"),
    Output("velocity-y-graph", "figure"),
    Output("velocity-z-graph", "figure"),
    Output("body-force-x-graph", "figure"),
    Output("body-force-y-graph", "figure"),
    Output("body-force-z-graph", "figure"),
    Output("trajectory-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("run-dir-store", "data"),
    Input("env-dropdown", "value"),
    Input("episode-dropdown", "value"),
)
def update_graphs(_, run_dir, env_id, episode_id):
    if not run_dir or env_id is None or episode_id is None:
        return tuple([
            empty_fig("Position X"),
            empty_fig("Position Y"),
            empty_fig("Position Z"),
            empty_fig("Velocity X"),
            empty_fig("Velocity Y"),
            empty_fig("Velocity Z"),
            empty_fig("Body force X"),
            empty_fig("Body force Y"),
            empty_fig("Body force Z"),
            empty_fig("3D trajectory"),
        ])

    df = read_csv_safe(Path(run_dir) / "trajectories.csv")
    return make_figures(df, int(env_id), int(episode_id))


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, host="127.0.0.1", port=8050)