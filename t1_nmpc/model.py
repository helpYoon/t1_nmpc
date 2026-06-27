"""Pinocchio model load for Booster T1 (translation + SphericalZYX floating base).

The floating base is JointModelTranslation (3 minimal coords) composed with
JointModelSphericalZYX (3 minimal Euler coords), so nq == nv == 35 and the
state mapping x[6:41] <-> q_pin is identity on the shared 35 (base pose 6 + joints 29).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pinocchio as pin

from .config import MPCConfig

# Canonical T1 URDF path (single source of truth; imported by Phase-2 tests).
T1_URDF_PATH = "/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/robot_models/booster_t1/t1_description/urdf/t1.urdf"

# §A.5 — the single source of truth for joint order (29 names, Pinocchio DFS).
EXPECTED_JOINT_NAMES: Tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)

ROOT_LINK = "Trunk"
CONTACT_FRAME_NAMES = ("foot_l_contact", "foot_r_contact")
CONTACT_PARENT_JOINTS = ("Left_Ankle_Roll", "Right_Ankle_Roll")


@dataclass
class RobotModel:
    model: pin.Model
    data: pin.Data
    mass: float
    contact_frame_ids: tuple
    torso_frame_id: int
    joint_names: tuple
    n_joints: int = 29


def _build_floating_base() -> pin.JointModelComposite:
    """JointModelTranslation (xyz) composed with JointModelSphericalZYX (yaw,pitch,roll)."""
    root = pin.JointModelComposite()
    root.addJoint(pin.JointModelTranslation())
    root.addJoint(pin.JointModelSphericalZYX())
    return root


def load_model(urdf_path: str, cfg: MPCConfig) -> RobotModel:
    root_joint = _build_floating_base()
    model = pin.buildModelFromUrdf(urdf_path, root_joint)

    # nq == nv == 35 (translation 3 + sphericalZYX 3 + 29 joints), minimal coords.
    if model.nq != 35 or model.nv != 35:
        raise ValueError(
            f"Expected nq==nv==35 (translation+sphericalZYX float base), "
            f"got nq={model.nq}, nv={model.nv}"
        )

    # --- joint-order validation (§A.5) ---
    # model.names = ['universe', <root composite joint>, <29 joints...>]
    loaded = tuple(model.names[2:])
    if loaded != EXPECTED_JOINT_NAMES:
        raise ValueError(
            "Joint order mismatch vs §A.5.\n"
            f"  expected: {EXPECTED_JOINT_NAMES}\n"
            f"  loaded:   {loaded}"
        )

    # --- add the two contact frames under the ankle-roll joints ---
    # Offset is expressed in the parent ankle-roll *joint* frame, so the Frame
    # placement is relative to the joint (parent frame = the joint's body frame).
    offset = pin.SE3.Identity()
    offset.translation = np.ascontiguousarray(cfg.contact_frame_offset, dtype=np.float64)
    contact_ids = []
    for fname, parent_joint in zip(CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS):
        jid = model.getJointId(parent_joint)
        frame = pin.Frame(fname, jid, offset, pin.FrameType.OP_FRAME)
        fid = model.addFrame(frame)
        contact_ids.append(fid)

    torso_fid = model.getFrameId(ROOT_LINK)
    if torso_fid >= model.nframes:
        raise ValueError(f"Torso link '{ROOT_LINK}' not found in model")

    data = model.createData()
    mass = float(sum(inertia.mass for inertia in model.inertias))

    return RobotModel(
        model=model,
        data=data,
        mass=mass,
        contact_frame_ids=tuple(contact_ids),
        torso_frame_id=torso_fid,
        joint_names=EXPECTED_JOINT_NAMES,
    )
