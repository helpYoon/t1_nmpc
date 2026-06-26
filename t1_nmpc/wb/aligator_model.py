"""Faithful free-flyer T1 model for the aligator kinodynamic OCP (same reduction as WBModel,
but pin.JointModelFreeFlyer base instead of the euler composite). Validated: nq=34, nv=33, nu=39."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics
from .model_wb import _HEAD_JOINTS, MPC_JOINT_NAMES, CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS

@dataclass
class AligatorModel:
    model: object
    space: object
    foot_ids: list
    mass: float
    nq: int
    nv: int
    ndx: int

def build_aligator_model(wb_cfg) -> AligatorModel:
    full = pin.buildModelFromUrdf(wb_cfg.urdf_path, pin.JointModelFreeFlyer())
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in _HEAD_JOINTS], pin.neutral(full))
    assert tuple(model.names[2:]) == MPC_JOINT_NAMES, model.names[2:]
    model.armature[6:] = np.asarray(wb_cfg.armature, float)
    off = pin.SE3.Identity(); off.translation = np.ascontiguousarray(wb_cfg.contact_frame_offset, float)
    foot_ids = [model.addFrame(pin.Frame(fn, model.getJointId(pj), off, pin.FrameType.OP_FRAME))
                for fn, pj in zip(CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS)]
    space = manifolds.MultibodyPhaseSpace(model)
    mass = float(sum(I.mass for I in model.inertias))
    return AligatorModel(model, space, foot_ids, mass, model.nq, model.nv, space.ndx)

def make_ode(am: AligatorModel, contact_flags, FS: int = 6):
    cs = pin.StdVec_Bool(); [cs.append(bool(b)) for b in contact_flags]
    ci = pin.StdVec_Index(); [ci.append(int(i)) for i in am.foot_ids]
    return dynamics.KinodynamicsFwdDynamics(am.space, am.model, np.array([0., 0., -9.81]), cs, ci, FS)

def nominal_stand_x(am: AligatorModel, wb_cfg) -> np.ndarray:
    q = pin.neutral(am.model)
    q[2] = wb_cfg.nominal_base_height
    q[7:] = np.asarray(wb_cfg.nominal_joint_pos, float)
    return np.concatenate([q, np.zeros(am.nv)])
