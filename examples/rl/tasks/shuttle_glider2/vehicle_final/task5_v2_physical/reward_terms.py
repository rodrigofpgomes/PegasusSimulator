"""Stationary reward components, independently testable without Isaac Sim.

Variant 1: tracking, hover rate damping, effort, smoothness and light neutral cost.
Variant 2: variant 1 plus air-data validity gates and soft physical margins.
Variant 3: variant 2 plus modest roll/yaw offload derived from observable refs.
No temporal second difference and no CAPS are enabled here.
Original task author: Rodrigo Gomes; original license: BSD-3-Clause.
"""
import math
import torch


def smooth_gate(value, start, full):
    if full <= start:
        raise ValueError("gate full threshold must exceed start threshold")
    u = ((value - start) / (full - start)).clamp(0.0, 1.0)
    return u.square() * (3.0 - 2.0 * u)


def compute_reward_terms(cfg, pos_error, vel_error, angular_body_rate,
                         alpha, beta, airspeed, goal_vel_w, goal_acc_w,
                         action, prev_action, rotor_velocity, min_w, max_w,
                         rotor_constants, surfaces_norm, surface_step, control_dt):
    """Return weighted costs and unweighted diagnostics; input batch is (N,...).

    Effort is a thrust proxy (omega squared), not measured electrical power.
    Each family's temporal cost is applied exactly once.
    """
    if cfg.reward_variant not in (1, 2, 3):
        raise ValueError("reward_variant must be 1, 2 or 3")
    pos_norm = torch.linalg.norm(pos_error, dim=1)
    vel_norm = torch.linalg.norm(vel_error, dim=1)
    rotor_unit = ((rotor_velocity - min_w.unsqueeze(0))
                  / (max_w - min_w).clamp_min(1e-6).unsqueeze(0)).clamp(0.0, 1.0)
    thrust_unit = rotor_unit[:, :4].square()
    quad_effort = thrust_unit.mean(dim=1)
    puller_effort = rotor_unit[:, 4].square()
    rotor_delta = (action[:, :5] - prev_action[:, :5]).square().mean(dim=1)
    surface_step_max = math.radians(cfg.surface_rate_max_deg_s) * control_dt
    surface_rate = (surface_step / surface_step_max).square().mean(dim=1)
    ref_speed = torch.linalg.norm(goal_vel_w, dim=1)
    ref_acc = torch.linalg.norm(goal_acc_w, dim=1)
    hover_gate = (1.0 - smooth_gate(ref_speed, 0.3, 2.0)) * (1.0 - smooth_gate(ref_acc, 0.2, 1.0))
    hover_rate = (angular_body_rate / cfg.hover_rate_scale).square().mean(dim=1)
    terms = {
        "pos": cfg.w_pos * pos_norm,
        "vel": cfg.w_vel * vel_norm / cfg.reward_velocity_scale,
        "hover_ang_rate": cfg.w_hover_ang_rate * hover_gate * hover_rate,
        "quad_effort": cfg.w_quad_effort * quad_effort,
        "puller_effort": cfg.w_puller_effort * puller_effort,
        "hover_puller_effort": cfg.w_hover_puller_effort * hover_gate * puller_effort,
        "d_rotor": cfg.w_d_rotor * rotor_delta,
        "surf": cfg.w_surf * surfaces_norm.square().mean(dim=1),
        "surf_rate": cfg.w_surf_rate * surface_rate,
    }
    zero = torch.zeros_like(pos_norm)
    for key in ("beta", "alpha", "rotor_margin", "elevator_margin", "ctrl_offload", "yaw_offload"):
        terms[key] = zero

    # These are physical forces, irrespective of reward normalization.
    lift_force = rotor_constants[:4].unsqueeze(0) * rotor_velocity[:, :4].square()
    lift_max = rotor_constants[:4] * max_w[:4].square()
    lift_min = rotor_constants[:4] * min_w[:4].square()
    lower_margin = lift_force - lift_min.unsqueeze(0)
    upper_margin = lift_max.unsqueeze(0) - lift_force
    roll_diff = 0.5 * (thrust_unit[:, 1] + thrust_unit[:, 2] - thrust_unit[:, 0] - thrust_unit[:, 3])
    pitch_diff = 0.5 * (thrust_unit[:, 0] + thrust_unit[:, 2] - thrust_unit[:, 1] - thrust_unit[:, 3])
    yaw_diff = 0.5 * (thrust_unit[:, 2] + thrust_unit[:, 3] - thrust_unit[:, 0] - thrust_unit[:, 1])
    gate = smooth_gate(airspeed, cfg.aero_on_speed, cfg.aero_full_speed)

    if cfg.reward_variant >= 2:
        beta_cost = (beta / math.radians(cfg.beta_scale_deg)).square().clamp(max=9.0)
        a_hi = math.radians(cfg.alpha_soft_max_deg)
        a_lo = math.radians(cfg.alpha_soft_min_deg)
        a_scale = math.radians(cfg.alpha_scale_deg)
        alpha_cost = (((alpha - a_hi).clamp_min(0.0) / a_scale).square()
                      + ((a_lo - alpha).clamp_min(0.0) / a_scale).square()).clamp(max=9.0)
        margin = cfg.rotor_margin_newtons
        margin_cost = (((margin - lower_margin).clamp_min(0.0) / margin).square()
                       + ((margin - upper_margin).clamp_min(0.0) / margin).square()).mean(dim=1)
        soft_ratio = cfg.elevator_soft_ratio
        elevator_cost = ((surfaces_norm[:, 0].abs() - soft_ratio).clamp_min(0.0)
                         / (1.0 - soft_ratio)).square()
        terms["beta"] = cfg.w_beta * gate * beta_cost
        terms["alpha"] = cfg.w_alpha * gate * alpha_cost
        terms["rotor_margin"] = cfg.w_rotor_margin * gate * margin_cost
        terms["elevator_margin"] = cfg.w_elevator_margin * gate * elevator_cost

    if cfg.reward_variant >= 3:
        speed = torch.linalg.norm(goal_vel_w[:, :2], dim=1)
        cross_z = goal_vel_w[:, 0] * goal_acc_w[:, 1] - goal_vel_w[:, 1] * goal_acc_w[:, 0]
        radial_acc = cross_z.abs() / speed.clamp_min(1e-6)
        turn_gate = smooth_gate(radial_acc, cfg.offload_turn_on_acc, cfg.offload_turn_full_acc)
        moving_gate = smooth_gate(speed, 2.0, 5.0)
        terms["ctrl_offload"] = cfg.w_ctrl_offload * gate * turn_gate * roll_diff.square()
        terms["yaw_offload"] = cfg.w_yaw_offload * gate * moving_gate * (1.0 - turn_gate) * yaw_diff.square()

    diagnostics = dict(pos_error=pos_norm, vel_error=vel_norm,
                       quad_effort=quad_effort, puller_effort=puller_effort,
                       roll_diff=roll_diff, pitch_diff=pitch_diff, yaw_diff=yaw_diff,
                       command_change_rms=torch.sqrt(rotor_delta),
                       rotor_lower_margin=lower_margin.min(dim=1).values,
                       rotor_upper_margin=upper_margin.min(dim=1).values)
    return terms, diagnostics
