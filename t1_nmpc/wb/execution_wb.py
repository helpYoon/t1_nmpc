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
    s = sample_ahead_s / cfg.dt
    lo = int(np.floor(s)); hi = min(lo + 1, cfg.N); a = s - lo
    lo_u = min(lo, cfg.N - 1); hi_u = min(lo + 1, cfg.N - 1)
    xq = (1.0 - a) * result.x_traj[lo] + a * result.x_traj[hi]
    uq = (1.0 - a) * result.u_traj[lo_u] + a * result.u_traj[hi_u]
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
