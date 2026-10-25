"""
| File: evaluation_log.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Raw evaluation records with explicit timing, failures and numerical metrics.
|
| Written by play_multi.py. It imports only NumPy, so the records and manifests can be
| read and validated without Isaac Sim.
|
| Run directory written by play_multi.py:
|   metadata.json  schema, arguments, controllers, physics and control timing
|   raw.jsonl      one record per controller/vehicle/physics step
|   events.jsonl   one record per finished episode (timeout or failure)
|   metrics.json   vector RMSE, failures and actuation summaries per controller/vehicle
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import numpy as np


# =============================================================================
# Controller manifest
# =============================================================================
# Vehicle key -> asset/physics key in pegasus.simulator.params.ROBOTS.
VEHICLES = {
    "iris": "Iris",
    "crazyflie": "Crazyflie",
    "shuttle": "Shuttle",
    "shuttle_glider2": "Shuttle_glider2",
    "easyglider": "Easyglider",
}
# Vehicles with aerodynamic surfaces, and their puller rotor index.
AERO_PULLER_INDEX = {"shuttle_glider2": 4, "easyglider": 0}
# Fixed-wing vehicles cannot hover: they start in flight and follow moving references.
FIXED_WING = {"easyglider"}

SKRL_TYPES = ("ppo", "ppo2", "sac_skrl")
CONTROLLER_TYPES = ("lqr", "raptor", "pretrain", *SKRL_TYPES, "nonlinear", "px4_position")


def validate_manifest(manifest):
    """Validate a controller manifest and return (entries, vehicle or None).

    Accepts {"vehicle": key, "vehicles": [...]} or a bare list. Each entry
    needs a unique name and a type in CONTROLLER_TYPES; skrl entries also need
    a checkpoint and a task, and may give run_metadata (training snapshot),
    env_cfg (field overrides) and preset. lqr entries need gains (see
    play_multi.resolve_lqr). Raises ValueError with the reason.
    """
    if isinstance(manifest, dict):
        entries, vehicle = manifest.get("vehicles"), manifest.get("vehicle")
    else:
        entries, vehicle = manifest, None
    if not entries:
        raise ValueError("manifest has no 'vehicles' entries")
    if vehicle is not None and vehicle not in VEHICLES:
        raise ValueError(f"manifest vehicle '{vehicle}' must be one of {sorted(VEHICLES)}")
    seen = set()
    for i, entry in enumerate(entries):
        name, kind = entry.get("name"), entry.get("type")
        if not name:
            raise ValueError(f"vehicle #{i} is missing 'name'")
        if name in seen:
            raise ValueError(f"duplicate controller name '{name}'")
        seen.add(name)
        if kind not in CONTROLLER_TYPES:
            raise ValueError(f"controller '{name}' has unknown type '{kind}'; known: {sorted(CONTROLLER_TYPES)}")
        if kind in (*SKRL_TYPES, "pretrain") and not entry.get("checkpoint"):
            raise ValueError(f"controller '{name}' ({kind}) needs a 'checkpoint'")
        if kind in SKRL_TYPES and not entry.get("task"):
            raise ValueError(f"controller '{name}' ({kind}) needs a 'task'")
        if kind == "lqr" and not (entry.get("gains") or "K" in entry):
            raise ValueError(f"controller '{name}' (lqr) needs 'gains': K, kp/kd, q_pos/q_vel/r or task")
        if entry.get("run_metadata") and not Path(entry["run_metadata"]).is_file():
            raise ValueError(f"controller '{name}': run_metadata '{entry['run_metadata']}' not found")
    if vehicle in FIXED_WING and any(entry["type"] not in SKRL_TYPES for entry in entries):
        raise ValueError(f"{vehicle} is evaluated with skrl policies only")
    return entries, vehicle


def evaluation_identity(vehicle, controllers):
    """Name the state source and tested task(s) without relying on a log path."""
    px4 = next((entry.get("px4") for entry in controllers if entry.get("px4")), None)
    mode = f"px4_{px4.get('mode', 'estimator')}" if px4 else "ground_truth"
    if px4 and not all(entry.get("px4") for entry in controllers):
        mode = f"mixed_ground_truth_{mode}"   # some controllers observe the true state, others the PX4 estimate
    if len(controllers) == 1:
        entry = controllers[0]
        test = entry.get("task") or f"{vehicle}_{entry['type']}"
    else:
        test = f"{vehicle}_comparison_" + "_".join(entry["name"] for entry in controllers)
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", test).strip("_")
    # A folder name is limited to 255 bytes; long comparisons keep a prefix and a hash of the full label.
    if len(label) > 120:
        label = f"{label[:111]}_{hashlib.sha1(label.encode()).hexdigest()[:8]}"
    return mode, label


def evaluation_directory(root, vehicle, controllers):
    """Choose a distinct folder under logs/evaluations for one play run."""
    mode, test = evaluation_identity(vehicle, controllers)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return Path(root) / mode / test / stamp


def array(value):
    """Convert a logging snapshot; missing quantities stay missing."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def serial(value):
    """Convert NumPy values to JSON types; non-finite floats become null."""
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def metrics(records):
    """Compute vector RMSE over all samples, retaining failure counts and survival."""
    if not records:
        return {"samples": 0, "position_rmse_m": None, "velocity_rmse_m_s": None}
    pos = np.asarray([r['position_error_world_m'] for r in records], dtype=float)
    vel = np.asarray([r['velocity_error_world_m_s'] for r in records], dtype=float)
    result = {"samples": len(records),
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(pos**2, axis=1)))),
            "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
            "failures": sum(bool(r['failed']) for r in records),
            "observed_duration_s": sum(r["dt_s"] for r in records)}
    episodes = {}
    for r in records:
        entry = episodes.setdefault(str(r.get('episode', 0)), {'observed_duration_s': 0., 'failed': False})
        entry['observed_duration_s'] += r['dt_s']
        entry['failed'] |= bool(r['failed'])
    result['episodes'] = episodes
    result['failure_episodes'] = sum(x['failed'] for x in episodes.values())
    actions = [r.get('action_raw') for r in records]
    commands = [r.get('command') for r in records]
    result['raw_action_outside_unit_fraction'] = None
    result['command_rate_rms_rad_s2'] = None
    result['rotor_speed_squared_integral_rad2_s'] = None
    if all(x is not None for x in actions):
        a = np.asarray(actions, dtype=float)
        if np.isfinite(a).all():
            result['raw_action_outside_unit_fraction'] = float(np.mean(np.abs(a) > 1))
    rates = []
    for previous, current in zip(records, records[1:]):
        if previous.get('episode') != current.get('episode'):
            continue
        if previous.get('command') is None or current.get('command') is None:
            continue
        count = len(current.get('rotor_actual_rad_s', []))
        dt = current['state_time_s'] - previous['state_time_s']
        if count and dt > 0:
            rates.append((np.asarray(current['command'][:count])-np.asarray(previous['command'][:count])) / dt)
    if rates and np.isfinite(rates).all():
        result['command_rate_rms_rad_s2'] = float(np.sqrt(np.mean(np.square(rates))))
    actual = [r.get('rotor_actual_rad_s') for r in records]
    if all(x is not None for x in actual) and np.isfinite(actual).all():
        result['rotor_speed_squared_integral_rad2_s'] = float(sum(np.sum(np.square(x))*r['dt_s'] for x,r in zip(actual,records)))
    return result



def rotate(q, vector):
    """Rotate vectors with scalar-first unit quaternions, preserving batch axes."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz = q[..., 1:]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + q[..., :1] * vector)


def aggregate_wrench(force, torque, positions, attitudes, coms, masses):
    """Sum applied model wrenches at assembly COM, in world axes; no gravity."""
    world_coms = positions + rotate(attitudes, coms)
    center = np.sum(world_coms * masses[..., None], axis=1) / masses.sum(axis=1)[:, None]
    fw, tw = rotate(attitudes, force), rotate(attitudes, torque)
    # PhysX applies each force at its part CoM (positions=None), so the lever arm starts there
    return fw.sum(axis=1), (tw + np.cross(world_coms-center[:, None, :], fw)).sum(axis=1), center


class EvaluationLog:
    """Write the physics audit into the current evaluation run directory."""

    def __init__(self, output, args, controllers, physics, vehicle, timing=None):
        """Create raw.jsonl and metadata.json; ``timing`` maps controller name to its control rates."""
        self.folder = Path(output)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = (self.folder / 'raw.jsonl').open('x', encoding='utf-8')
        self.records = {}
        self.episode = 0
        mode, test = evaluation_identity(vehicle, controllers)
        schema = {"version": 3, "revision": "aligned-com-review-v2", "evaluation_mode": mode,
                  "test_id": test, "collisions": "disabled in play_multi, including ground; failure is not a contact detector", "arguments": args, "controllers": controllers,
                  "physics": physics, "control_timing": timing or {}, "training_domain": "unknown unless supplied in controller metadata",
                  "state": "post-integration ground truth, ENU world; angular velocity FLU body",
                  "command": "last backend command, rad/s rotors then rad surfaces; applied over preceding physics interval",
                  "action_raw": "native policy order before command clipping; pretrain output is already tanh-squashed; null if unavailable",
                  "policy_to_actuator_indices": "external adapter mapping from policy motor index to vehicle motor index; null for task-defined or non-motor actions",
                  "action_clipped": "normalized policy command, null if unavailable",
                  "wrench": "model-calculated local per-part force [N] and torque [N m] at part origin; gravity excluded; not measured",
                  "aggregate_wrench": "calculated world ENU at assembly COM using pre-integration geometry; gravity excluded; forces applied at part CoM, lever arms counted once",
                  "metrics": "sqrt(mean(sum(vector_error**2))); includes failed samples; missing/nonfinite is null, never zero"}
        (self.folder / 'metadata.json').write_text(json.dumps(schema, indent=2, default=str))

    def begin_interval(self, vehicles):
        """Capture counters before world.step so pauses or duplicate callbacks fail visibly."""
        self._before = {name: getattr(v, '_state_update_count', 0) for name, v in vehicles.items()}

    def sample(self, vehicles, reset_manager, step, dt, failure_distance):
        """Record state after integration and the command from the preceding interval."""
        any_failed = False
        goal = array(reset_manager.goal_pos)
        goal_vel = array(reset_manager.goal_vel)
        for name, vehicle in vehicles.items():
            if hasattr(self, '_before'):
                count = getattr(vehicle, '_state_update_count', 0) - self._before[name]
                if count != 1 or not np.isclose(getattr(vehicle, '_last_physics_dt', float('nan')), dt):
                    raise RuntimeError(f"Invalid physics interval for {name}: post callbacks={count}, expected dt={dt}")
            backend = vehicle._backends[0]
            pos, vel = array(vehicle.state.position), array(vehicle.state.linear_velocity)
            attitude, rates = array(vehicle.state.attitude), array(vehicle.state.angular_velocity)
            raw = array(getattr(backend, '_policy_mean_raw', getattr(backend, 'last_action_raw', None)))
            clipped = array(getattr(backend, '_policy_action_clipped', getattr(backend, '_last_action_norm', None)))
            mode = getattr(backend, "_action_mode", None)
            command = array(backend.input_reference()) if mode in ("rotor_velocity", "rotor_velocity_direct") else None
            actual = array(vehicle._thrusters.velocity)
            force = array(vehicle._last_forces_local)
            torque = array(vehicle._last_torques_local)
            surface = array(getattr(vehicle, '_surface_deflections', None))
            geometry = getattr(vehicle, '_wrench_geometry', None)
            aggregate = None
            if geometry is not None:
                gp,gq,gc,gm = [array(x) for x in geometry]
                aggregate = aggregate_wrench(force, torque, gp.reshape(force.shape),
                    gq.reshape(*force.shape[:2],4), gc.reshape(force.shape), gm.reshape(force.shape[:2]))
            aero = getattr(vehicle, '_aero_wrench_local', None)
            aero = [array(x) for x in aero] if aero is not None else None
            model = getattr(vehicle, '_aerodynamics', None)
            airdata = [array(getattr(model, name, None)) for name in ('_Va','_alpha','_beta','_lift','_drag')]
            actuator_thrust = array(getattr(vehicle._thrusters, 'force', None))
            for i in range(vehicle.n_vehicles):
                ep, ev = pos[i]-goal[i], vel[i]-goal_vel[i]
                quantities = [pos[i], vel[i], attitude[i], rates[i], ep, ev, actual[i], force[i], torque[i]]
                if command is not None: quantities.append(command[i])
                if surface is not None: quantities.append(surface[i])
                failed = (not all(np.isfinite(x).all() for x in quantities)
                          or abs(np.linalg.norm(attitude[i])-1.) > 1e-3
                          or np.linalg.norm(ep) > failure_distance)
                any_failed |= failed
                item = {"controller": name, "env": i, "episode": self.episode, "state_time_s": step*dt,
                        "command_interval_start_s": (step-1)*dt, "dt_s": dt,
                        "position_world_m": pos[i], "reference_world_m": goal[i],
                        "position_error_world_m": ep, "velocity_error_world_m_s": ev,
                        "action_mode": mode, "policy_to_actuator_indices": array(getattr(backend, "_raptor_to_pegasus", None)), "action_raw": raw[i] if raw is not None else None,
                        "action_clipped": clipped[i] if clipped is not None else None,
                        "command": command[i] if command is not None else None,
                        "rotor_actual_rad_s": actual[i], "surface_actual_rad": surface[i] if surface is not None else None,
                        "part_force_local_N": force[i], "part_torque_local_Nm": torque[i],
                        "control_force_world_N": aggregate[0][i] if aggregate is not None else None,
                        "control_torque_com_world_Nm": aggregate[1][i] if aggregate is not None else None,
                        "assembly_com_world_m": aggregate[2][i] if aggregate is not None else None,
                        "aero_force_body_N": aero[0][i] if aero is not None else None,
                        "aero_torque_body_com_Nm": aero[1][i] if aero is not None else None,
                        "airspeed_m_s": [airdata[0][i]] if airdata[0] is not None else None,
                        "alpha_rad": [airdata[1][i]] if airdata[1] is not None else None,
                        "beta_rad": [airdata[2][i]] if airdata[2] is not None else None,
                        "wing_lift_drag_N": [airdata[3][i], airdata[4][i]] if airdata[3] is not None and airdata[4] is not None else None,
                        "actuator_thrust_N": actuator_thrust[i] if actuator_thrust is not None else None,
                        "attitude_wxyz": attitude[i],
                        "angular_velocity_body_rad_s": rates[i],
                        "failed": failed, "failure_reason": "nonfinite_quaternion_or_position_limit" if failed else None}
                self.stream.write(json.dumps(serial(item), allow_nan=False)+'\n')
                # Keep only metrics channels in memory; full actuation remains in the raw file.
                self.records.setdefault((name, i), []).append({k:item[k] for k in ('state_time_s','dt_s','position_error_world_m','velocity_error_world_m_s','failed','episode','action_raw','command','rotor_actual_rad_s')})
        return any_failed

    def end_episode(self, time_s, reason):
        """Append the end of the current episode to events.jsonl."""
        with (self.folder / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'episode': self.episode, 'time_s': time_s, 'reason': reason})+'\n')
        self.episode += 1

    def close(self):
        """Close raw.jsonl and write the per-controller, per-vehicle metrics.json."""
        self.stream.close()
        summaries = {f'{name}/{i}': metrics(values) for (name,i),values in self.records.items()}
        (self.folder / 'metrics.json').write_text(json.dumps(serial(summaries), indent=2, allow_nan=False))
        print(f'Evaluation records: {self.folder}')
