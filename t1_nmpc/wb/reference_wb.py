# t1_nmpc/wb/reference_wb.py
"""Velocity-command -> per-node (x_ref, u_ref) for the WB walking MPC.
Faithful to t1_controller MpcTargetTrajectoriesCalculator + SwitchedModelReferenceManager:
0.8 command filter, heading rotation, two-phase base-pose blend (knots at t0, t0+0.7H, t0+H),
nominal posture + gait-phase arm swing. comm = [vx, vy, height, wz].

NOTE: the gravity-split stance wrench (u_ref fz = mg/n_stance) is an acados-ONLY addition, NOT a port
of any OCS2 term — OCS2's input reference is strictly all-zero (MpcTargetTrajectoriesCalculator.cpp:79,155).
It is a deliberate single-RTI gravity-comp prior: it hands the one QP a weight-supporting wrench target so
the first (and only) linearization already supports body weight. OCS2 reaches the same via iterated
equality projection across SQP iters. Kept on purpose; flagged here so it is not mistaken for faithful. (audit 2026-06-25)"""
from __future__ import annotations

import numpy as np

_G = 9.81
# T1 joint-posture indices within q_joints (config_wb MPC_JOINT_NAMES order):
_L_SHOULDER_P, _R_SHOULDER_P, _L_ELBOW_P, _R_ELBOW_P = 0, 7, 2, 9


def filter_command(prev: np.ndarray, cmd: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    return alpha * np.asarray(prev, float) + (1.0 - alpha) * np.asarray(cmd, float)


def _integrate_pose(pose, vel3, height, dT):
    tp = pose.copy()
    tp[0] += vel3[0] * dT; tp[1] += vel3[1] * dT
    tp[2] = height
    tp[3] += vel3[2] * dT; tp[4] = 0.0; tp[5] = 0.0
    return tp


def _arm_phase(gait, t):
    """getPhaseVariable: LF -> 0.5*frac, RF -> 0.5+0.5*frac; STANCE holds the boundary value."""
    from .gait_wb import LF, RF
    dur = gait.duration
    phase = (t / dur) % 1.0
    edges = np.concatenate(([0.0], gait.event_phases, [1.0]))
    idx = int(np.searchsorted(gait.event_phases, phase, side="right"))
    lo, hi = edges[idx], edges[idx + 1]
    frac = (phase - lo) / (hi - lo)
    mode = int(gait.mode_sequence[idx])
    if mode == LF:
        return 0.5 * frac
    if mode == RF:
        return 0.5 + 0.5 * frac
    return 0.5 if idx <= 1 else 0.0          # STANCE: hold the boundary (after-LF -> 0.5, after-RF -> 0)


def build_reference(x_meas, comm_filt, gait, t0, node_times, cfg, model):
    x_meas = np.asarray(x_meas, float)
    vx, vy, height, wz = (float(c) for c in comm_filt)
    yaw = float(x_meas[3])
    c, s = np.cos(yaw), np.sin(yaw)
    vgx = c * vx - s * vy
    vgy = s * vx + c * vy
    target_base_vel = np.array([vgx, vgy, 0.0, wz, 0.0, 0.0])
    vx_local = c * vgx + s * vgy             # ~ commanded body-frame vx (arm-swing scale)

    cur_pose = np.array([x_meas[0], x_meas[1], height, yaw, 0.0, 0.0])
    base_vel = x_meas[33:39]
    H = cfg.horizon                          # 1.1 (faithful), even though node grid spans tf=1.085
    t_mid = 0.7 * H
    avg1 = np.array([(base_vel[0] + vgx) / 2, (base_vel[1] + vgy) / 2, (base_vel[5] + wz) / 2])
    pose_mid = _integrate_pose(cur_pose, avg1, height, t_mid)
    pose_fin = _integrate_pose(pose_mid, np.array([vgx, vgy, wz]), height, H - t_mid)
    knot_t = np.array([t0, t0 + t_mid, t0 + H])
    knot_pose = np.stack([cur_pose, pose_mid, pose_fin])

    nominal = model.nominal_state()
    posture0 = nominal[6:33].copy()
    mg = model.total_mass() * _G
    A = cfg.arm_swing_amplitude

    node_times = np.asarray(node_times, float)
    x_ref = np.zeros((len(node_times), 68))
    u_ref = np.zeros((max(len(node_times) - 1, 0), 40))
    for k, tk in enumerate(node_times):
        pose_k = np.array([np.interp(tk, knot_t, knot_pose[:, j]) for j in range(6)])
        posture = posture0.copy()
        gcf = np.sin(2 * np.pi * (_arm_phase(gait, tk) - cfg.arm_swing_phase_offset)) * vx_local
        posture[_L_SHOULDER_P] += -A * gcf; posture[_R_SHOULDER_P] += +A * gcf
        posture[_L_ELBOW_P] += -A * gcf;    posture[_R_ELBOW_P] += +A * gcf
        xr = nominal.copy()
        xr[0:6] = pose_k; xr[6:33] = posture; xr[33:39] = target_base_vel
        xr[39:66] = 0.0; xr[66:68] = 0.0
        x_ref[k] = xr
        if k < len(u_ref):
            lf, rf = gait.contact_flags(tk)
            n_st = int(lf) + int(rf)
            if n_st > 0:
                if lf:
                    u_ref[k, 2] = mg / n_st
                if rf:
                    u_ref[k, 8] = mg / n_st
    return x_ref, u_ref
