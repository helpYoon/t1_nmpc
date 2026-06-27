"""Reduced (head-locked) T1 FreeFlyer pinocchio model + one 6D sole frame per foot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .config import MPCConfig, T1_URDF_PATH, ANKLE_ROLL_FRAMES, LOCKED_JOINTS


@dataclass
class RobotModel:
    model: pin.Model
    data: pin.Data
    sole_frame_ids: tuple[int, int]
    foot_joint_placements: tuple[tuple[int, pin.SE3], tuple[int, pin.SE3]]
    mass: float
    trunk_frame_id: int
    tau_max: np.ndarray            # (27,)
    half_extents: tuple[float, float]


def load_model(cfg: MPCConfig) -> RobotModel:
    full = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    if full.nq != 36 or full.nv != 35:
        raise ValueError(f"expected full nq=36 nv=35, got {full.nq}/{full.nv}")
    lock_ids = [full.getJointId(n) for n in LOCKED_JOINTS]
    model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
    if model.nq != 34 or model.nv != 33:
        raise ValueError(f"expected reduced nq=34 nv=33, got {model.nq}/{model.nv}")

    sole_offset = np.array([0.005, 0.0, cfg.sole_z], dtype=np.float64)  # sole_z = -0.030
    sole_ids, placements = [], []
    for ankle in ANKLE_ROLL_FRAMES:
        afid = model.getFrameId(ankle)
        parent_joint = model.frames[afid].parentJoint
        ankle_placement = model.frames[afid].placement           # ankle frame wrt parent joint
        t = ankle_placement.act(sole_offset)
        jMf = pin.SE3(np.eye(3), t)                               # sole frame wrt parent joint
        frame = pin.Frame(f"{ankle}_sole", parent_joint, afid, jMf, pin.FrameType.OP_FRAME)
        sole_ids.append(model.addFrame(frame))
        placements.append((parent_joint, jMf))

    data = model.createData()
    mass = float(pin.computeTotalMass(model, data))
    trunk_fid = model.getFrameId("Trunk")
    tau_max = np.asarray(model.effortLimit[6:], dtype=np.float64).copy()
    return RobotModel(model, data, tuple(sole_ids), tuple(placements), mass, trunk_fid,
                      tau_max, (cfg.half_len, cfg.half_width))


def nominal_q(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    q[0:3] = [0.0, 0.0, cfg.nominal_base_height]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]                                 # quat xyzw identity
    q[7:] = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)   # 27 values
    return q


def nominal_x(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    return np.concatenate([nominal_q(cfg, model), np.zeros(model.nv)])
