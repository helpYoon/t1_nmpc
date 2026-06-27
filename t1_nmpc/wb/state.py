"""MuJoCo <-> pinocchio FreeFlyer state map (single source of truth) + command extraction.

MuJoCo: qpos=[pos3, quat_wxyz4, joints29], qvel=[lin_WORLD3, ang_LOCAL3, jvel29].
pinocchio FreeFlyer: q=[pos3, quat_xyzw4, joints29], v=[lin_LOCAL3, ang_LOCAL3, jvel29].
Base linear velocity differs by frame -> rotate by R(q)^T (MuJoCo->pin) / R(q) (pin->MuJoCo)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..robot.config import MPCConfig, JointCommand


def mujoco_to_freeflyer(qpos, qvel, model) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64); qvel = np.asarray(qvel, dtype=np.float64)
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    q = np.empty(model.nq); q[0:3] = qpos[0:3]; q[3:7] = [qx, qy, qz, qw]; q[7:] = qpos[7:]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    v = np.empty(model.nv); v[0:3] = R.T @ qvel[0:3]; v[3:6] = qvel[3:6]; v[6:] = qvel[6:]
    return np.concatenate([q, v])


def freeflyer_to_mujoco(x, model):
    x = np.asarray(x, dtype=np.float64)
    q = x[:model.nq]; v = x[model.nq:model.nq + model.nv]
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    qpos = np.empty(model.nq); qpos[0:3] = q[0:3]; qpos[3:7] = [qw, qx, qy, qz]; qpos[7:] = q[7:]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    qvel = np.empty(model.nv); qvel[0:3] = R @ v[0:3]; qvel[3:6] = v[3:6]; qvel[6:] = v[6:]
    return qpos, qvel


def extract_command(retracted: dict, cfg: MPCConfig) -> JointCommand:
    """tau_ff from node 0; q_des/qd_des from the planned next node (1)."""
    return JointCommand(
        q_des=np.asarray(retracted["q_sol"][1][7:], dtype=np.float64),
        qd_des=np.asarray(retracted["v_sol"][1][6:], dtype=np.float64),
        tau_ff=np.asarray(retracted["tau_sol"][0], dtype=np.float64),
        kp=np.asarray(cfg.kp, dtype=np.float64),
        kd=np.asarray(cfg.kd, dtype=np.float64),
    )
