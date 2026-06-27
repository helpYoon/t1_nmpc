"""MuJoCo ↔ pinocchio free-flyer state mapping for the aligator kinodynamic MPC.

Quaternion convention
---------------------
MuJoCo:    qpos[3:7]  = (w, x, y, z)
Pinocchio: q[3:7]     = (x, y, z, w)   (JointModelFreeFlyer canonical storage)

Joint layout (§A.5 order)
--------------------------
MuJoCo  qpos[7:36] / qvel[6:35] = 29 joints; head joints (AAHead_yaw, Head_pitch)
         are at local index 0 and 1 = qpos[7:9] / qvel[6:8].
MPC 27   qpos[9:36] / qvel[8:35] (27 = 29 minus the 2 head joints).

Free-flyer state x (67,) = [q(34), v(33)]
------------------------------------------
  q = [pos(3), quat_xyzw(4), joints(27)]
  v = [lin_world(3), ang_local(3), joint_vel(27)]

Angular velocity: MuJoCo qvel[3:6] is angular velocity in the LOCAL body frame,
which is exactly what pinocchio's FreeFlyer tangent v[3:6] expects — no conversion.

Cross-checked against sim/wb_state.py::wb_state_estimate and
sim/mujoco_runtime.py::_pin_q_v, which derive the joint slices for the euler-ZYX
(WBModel) path using qpos[9:36] / qvel[8:35] for the 27 MPC joints (wb_state.py:18
uses q_pin[8:35] which originates from q_j = qpos[7:36] shifted by the two head
joints at local indices 0,1 → same qpos[9:36]).
"""
from __future__ import annotations

import numpy as np

from .aligator_exec import extract_tau_ff
from ..config import JointCommand

# ---------------------------------------------------------------------------
# Fixed index slices (§A.5 order, verified against wb_state.py:18 / _pin_q_v)
# ---------------------------------------------------------------------------
# MuJoCo layout: qpos[7:36] = 29 body joints in §A.5 order; head at 7:9
_MJ_Q_JOINTS = slice(9, 36)   # 27 MPC joint positions  (skip head at 7:9)
_MJ_V_JOINTS = slice(8, 35)   # 27 MPC joint velocities (skip head at 6:8)

# Free-flyer state x (67,): q[0:34] = [pos(3), quat(4), joints(27)]
#                             v[34:67] = [lin(3), ang(3), jvel(27)]
_FF_Q_JOINTS = slice(7, 34)   # x[7:34]  = q_joints (27)
_FF_V_JOINTS = slice(40, 67)  # x[40:67] = v_joints (27)  (34 + 6 = 40)


def mujoco_to_freeflyer(rt, am) -> np.ndarray:
    """Build the pinocchio free-flyer state x (nq+nv = 67,) from MuJoCo data.

    Parameters
    ----------
    rt  : MujocoRuntime  (rt.mj_data is the live data object)
    am  : AligatorModel  (used for nq/nv shape assertions only)

    Returns
    -------
    x   : np.ndarray shape (67,) = [q(34), v(33)]
    """
    d = rt.mj_data

    # --- configuration q (34,) ---
    pos = np.array(d.qpos[0:3], dtype=np.float64)
    # MuJoCo (w, x, y, z) → Pinocchio (x, y, z, w)
    qw, qx, qy, qz = float(d.qpos[3]), float(d.qpos[4]), float(d.qpos[5]), float(d.qpos[6])
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    q_joints = np.array(d.qpos[_MJ_Q_JOINTS], dtype=np.float64)   # 27
    q = np.concatenate([pos, quat_xyzw, q_joints])                 # (34,)

    # --- velocity v (33,) ---
    lin_vel = np.array(d.qvel[0:3], dtype=np.float64)     # world-frame linear vel
    ang_vel = np.array(d.qvel[3:6], dtype=np.float64)     # LOCAL body-frame angular vel
    v_joints = np.array(d.qvel[_MJ_V_JOINTS], dtype=np.float64)   # 27
    v = np.concatenate([lin_vel, ang_vel, v_joints])               # (33,)

    assert q.shape == (am.nq,) and v.shape == (am.nv,)
    return np.concatenate([q, v])


def freeflyer_command(am, x_meas: np.ndarray, res, wb_cfg) -> JointCommand:
    """Build a JointCommand from the aligator MPC result for the MuJoCo transport.

    Uses measured joint positions / velocities as the PD reference so that
    kp*(q_des - q_meas) ≈ 0 and the feedforward tau_ff (from RNEA) carries the
    load.  This mirrors the crocoddyl execution path's use of the planned state at
    node 0 when the look-ahead is small (execution_wb.py::to_joint_command_wb with
    sample_ahead_s=0.005 and a 27-joint WBModel).

    Parameters
    ----------
    am       : AligatorModel
    x_meas   : np.ndarray (67,)  — current free-flyer state from mujoco_to_freeflyer
    res      : AligatorResult    — latest MPC result (.us[0] used)
    wb_cfg   : WBConfig          — supplies kp/kd (27,) and joint dimensions

    Returns
    -------
    JointCommand with 27-element arrays (compatible with MujocoTransport.write_command)
    """
    u0 = np.asarray(res.us[0], dtype=np.float64)
    tau_ff, wl, wr = extract_tau_ff(am, x_meas, u0)

    # PD reference = the PLANNED next-node joint state (not measured), so kp*(q_des - q_meas) actively
    # pulls the joints toward the planned motion (e.g. the lifting swing leg). q_des=measured gave zero
    # PD assist and left tau_ff alone unable to realize the planned foot lift.
    x_des = np.asarray(res.xs[1] if len(res.xs) > 1 else res.xs[0], dtype=np.float64)
    q_joints = np.ascontiguousarray(x_des[_FF_Q_JOINTS], dtype=np.float64)  # (27,)
    v_joints = np.ascontiguousarray(x_des[_FF_V_JOINTS], dtype=np.float64)  # (27,)

    return JointCommand(
        q_des=q_joints,
        qd_des=v_joints,
        kp=wb_cfg.kp,
        kd=wb_cfg.kd,
        tau_ff=np.ascontiguousarray(tau_ff, dtype=np.float64),
        wrench_l=np.ascontiguousarray(wl, dtype=np.float64),
        wrench_r=np.ascontiguousarray(wr, dtype=np.float64),
    )
