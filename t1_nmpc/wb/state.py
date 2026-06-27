"""MuJoCo <-> pinocchio FreeFlyer state map for the reduced (head-locked, 27-joint) model.

MuJoCo qpos=[pos3, quat_wxyz4, joints29], qvel=[lin_WORLD3, ang_LOCAL3, jvel29].
pinocchio reduced q=[pos3, quat_xyzw4, joints27], v=[lin_LOCAL3, ang_LOCAL3, jvel27].
Base linear velocity differs by frame -> rotate by R(q)^T (MuJoCo->pin).
The 2 head joints (MuJoCo actuated indices 0,1) are dropped."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..robot.config import MPCConfig, JointCommand

# MuJoCo actuated-joint order: [head2, Larm7, Rarm7, waist, Lleg6, Rleg6] (29).
# Reduced pinocchio drops the 2 head joints -> select MuJoCo actuated indices 2..28.
MUJOCO_TO_PIN_JOINTS = np.arange(2, 29)        # (27,)


def mujoco_to_freeflyer(qpos, qvel, model) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64); qvel = np.asarray(qvel, dtype=np.float64)
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    q = np.empty(model.nq)
    q[0:3] = qpos[0:3]; q[3:7] = [qx, qy, qz, qw]
    q[7:] = qpos[7:][MUJOCO_TO_PIN_JOINTS]      # 27 controlled joints
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    v = np.empty(model.nv)
    v[0:3] = R.T @ qvel[0:3]; v[3:6] = qvel[3:6]
    v[6:] = qvel[6:][MUJOCO_TO_PIN_JOINTS]      # 27 controlled joint velocities
    return np.concatenate([q, v])


def freeflyer_to_mujoco_joints(x, model):
    x = np.asarray(x, dtype=np.float64)
    q = x[:model.nq]; v = x[model.nq:model.nq + model.nv]
    return q[7:].copy(), v[6:].copy()           # 27 each


def extract_command(x1, tau0, cfg: MPCConfig, rm) -> JointCommand:
    """q_des/qd_des from planned node 1; tau_ff from node 0 (post-hoc RNEA joint torque)."""
    q_des, qd_des = freeflyer_to_mujoco_joints(x1, rm.model)
    return JointCommand(
        q_des=q_des, qd_des=qd_des,
        tau_ff=np.asarray(tau0, dtype=np.float64),
        kp=np.asarray(cfg.kp, dtype=np.float64),
        kd=np.asarray(cfg.kd, dtype=np.float64),
    )
