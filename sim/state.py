# sim/state.py
"""Whole-body MuJoCo state estimate + reset (extracted from the deleted acados stand gate;
reused by mujoco_transport and the crocoddyl stand gate)."""
from __future__ import annotations

import numpy as np
import mujoco

from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def wb_state_estimate(rt: MujocoRuntime) -> np.ndarray:
    """68-d WB state from sim: [q_base(6), q_joints(27), v_base(6), v_joints(27), s, v_s].
    q_pin/v_pin are euler-zyx (35,); the 27 MPC joints = the 29 minus the 2 head joints (idx 6:8)."""
    q_pin, v_pin = rt._pin_q_v()
    x = np.zeros(68, dtype=np.float64)
    x[0:6] = q_pin[0:6]
    x[6:33] = q_pin[8:35]
    x[33:39] = v_pin[0:6]
    x[39:66] = v_pin[8:35]
    return x


def wb_reset(rt: MujocoRuntime, wb_cfg) -> None:
    """Spawn at the WB nominal posture (head=0 + 27 MPC joints) above the floor and PD-settle the
    feet onto it."""
    q0 = MJ_JOINT_QPOS0
    njp29 = np.zeros(29); njp29[2:29] = np.asarray(wb_cfg.nominal_joint_pos, dtype=np.float64)
    kp = np.asarray(rt.cfg.kp, dtype=np.float64); kd = np.asarray(rt.cfg.kd, dtype=np.float64)
    d = rt.mj_data
    d.qpos[:] = 0.0; d.qvel[:] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[q0:q0 + 29] = njp29
    d.qpos[2] = wb_cfg.nominal_base_height + 0.10
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    for _ in range(int(round(0.6 * rt.cfg.physics_hz))):
        q = np.array(d.qpos[q0:q0 + 29]); qd = np.array(d.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
        rt._apply_torque(kp * (njp29 - q) - kd * qd)
        rt.step_physics()
    d.qvel[:] = 0.0
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    rt.t = 0.0
