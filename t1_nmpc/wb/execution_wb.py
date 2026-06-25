"""Whole-body joint-command extraction + tau_ff (no WBC; single-layer, wb_humanoid-faithful).

Unlike the centroidal port (qd_des from the input joint-velocity block), the WB state CARRIES the
joint velocities, so qd_des is the planned velocity STATE. tau_ff = inverse dynamics of the planned
(x,u) at node 0 (= model.joint_torque). Reuses pd_torque from t1_nmpc.execution.
"""
from __future__ import annotations

import numpy as np

from ..config import JointCommand
from ..execution import pd_torque  # noqa: F401  (re-exported for the runner)

# WB state slices
_QJ = slice(6, 33)     # q_joints (27)
_VJ = slice(39, 66)    # v_joints (27)


def to_joint_command_wb(result, cfg, model, sample_ahead_s: float = 0.005) -> JointCommand:
    """q_des/qd_des/tau_ff/wrenches ALL sampled at t+sample_ahead_s (linear interp between shooting
    nodes). Faithful to OCS2 MpcMrtJointController.cpp:256-262, which feeds the SAME look-ahead
    (state,input) pair into computeJointTorques AND the PD references — not node 0. tau_ff =
    model.joint_torque(x@t+dt, u@t+dt), the ID torque realizing the planned joint accel + contact
    wrenches at the resampled point."""
    nt = result.node_times if getattr(result, "node_times", None) is not None else (np.arange(cfg.N + 1) * cfg.dt)
    tq = nt[0] + sample_ahead_s
    xq = np.array([np.interp(tq, nt, result.x_traj[:, j]) for j in range(result.x_traj.shape[1])])
    u_src = result.u_phys_traj if getattr(result, "u_phys_traj", None) is not None else result.u_traj
    uq = np.array([np.interp(tq, nt[:cfg.N], u_src[:, j]) for j in range(u_src.shape[1])])
    tau_ff = model.joint_torque(xq, uq)
    return JointCommand(
        q_des=np.ascontiguousarray(xq[_QJ], dtype=np.float64),
        qd_des=np.ascontiguousarray(xq[_VJ], dtype=np.float64),
        kp=cfg.kp,
        kd=cfg.kd,
        tau_ff=np.ascontiguousarray(tau_ff, dtype=np.float64),
        wrench_l=np.ascontiguousarray(uq[0:6], dtype=np.float64),
        wrench_r=np.ascontiguousarray(uq[6:12], dtype=np.float64),
    )
