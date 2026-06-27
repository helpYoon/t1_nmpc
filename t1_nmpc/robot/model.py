"""T1 FreeFlyer pinocchio model + 8 foot-corner contact frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .config import MPCConfig, T1_URDF_PATH, ANKLE_ROLL_FRAMES


@dataclass
class RobotModel:
    model: pin.Model
    data: pin.Data
    corner_frame_ids: tuple[int, ...]
    foot_center_frame_ids: tuple[int, ...]
    mass: float
    trunk_frame_id: int
    tau_max: np.ndarray   # (29,)


def load_model(cfg: MPCConfig) -> RobotModel:
    model = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"expected nq=36 nv=35, got {model.nq}/{model.nv}")

    corner_ids = []
    for ankle in ANKLE_ROLL_FRAMES:
        fid = model.getFrameId(ankle)
        parent_joint = model.frames[fid].parentJoint
        parent_placement = model.frames[fid].placement   # ankle frame wrt its parent joint
        for cx in cfg.corner_x:
            for cy in cfg.corner_y:
                t = parent_placement.act(np.array([cx, cy, cfg.corner_z], dtype=np.float64))
                placement = pin.SE3(np.eye(3), t)
                name = f"{ankle}_corner_{cx:+.4f}_{cy:+.4f}"
                frame = pin.Frame(name, parent_joint, fid, placement, pin.FrameType.OP_FRAME)
                corner_ids.append(model.addFrame(frame))

    center_ids = []
    cx = (cfg.corner_x[0] + cfg.corner_x[1]) / 2.0
    for ankle in ANKLE_ROLL_FRAMES:
        fid = model.getFrameId(ankle)
        parent_joint = model.frames[fid].parentJoint
        parent_placement = model.frames[fid].placement
        t = parent_placement.act(np.array([cx, 0.0, cfg.corner_z], dtype=np.float64))
        frame = pin.Frame(f"{ankle}_center", parent_joint, fid, pin.SE3(np.eye(3), t),
                          pin.FrameType.OP_FRAME)
        center_ids.append(model.addFrame(frame))

    data = model.createData()
    mass = float(pin.computeTotalMass(model, data))
    trunk_fid = model.getFrameId("Trunk")
    tau_max = np.asarray(model.effortLimit[6:], dtype=np.float64).copy()
    return RobotModel(model, data, tuple(corner_ids), tuple(center_ids), mass, trunk_fid, tau_max)


def nominal_q(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    q[0:3] = [0.0, 0.0, cfg.nominal_base_height]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]            # quat xyzw identity
    q[7:] = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)
    return q


def nominal_x(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    return np.concatenate([nominal_q(cfg, model), np.zeros(model.nv)])
