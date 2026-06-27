import numpy as np
import pinocchio as pin
import pytest

from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import (
    load_model,
    RobotModel,
    EXPECTED_JOINT_NAMES,
    CONTACT_FRAME_NAMES,
)

URDF = (
    "/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/"
    "robot_models/booster_t1/t1_description/urdf/t1.urdf"
)


@pytest.fixture(scope="module")
def rm() -> RobotModel:
    return load_model(URDF, make_config())


def _standing_q(rm: RobotModel) -> np.ndarray:
    """q_pin (35,) = x[6:41]: base pose + nominal joints, standing at 0.62 m."""
    cfg = make_config()
    q = np.zeros(35, dtype=np.float64)
    q[0:3] = [0.0, 0.0, cfg.nominal_base_height]   # base xyz
    q[3:6] = [0.0, 0.0, 0.0]                        # ZYX-Euler yaw,pitch,roll
    q[6:35] = cfg.nominal_joint_pos                 # 29 joints
    return q


def test_dimensions_nq_nv(rm):
    assert rm.model.nq == 35
    assert rm.model.nv == 35
    assert rm.n_joints == 29


def test_mass_within_tolerance(rm):
    # contract: within 0.1 of 34.513 ; actual URDF link-mass sum = 34.513469
    assert abs(rm.mass - 34.513) < 0.1
    assert abs(rm.mass - 34.513469) < 1e-4


def test_joint_order_matches_A5(rm):
    assert rm.joint_names == EXPECTED_JOINT_NAMES
    # model.names[2:] = the 29 joints after 'universe' and the float-base joint
    assert tuple(rm.model.names[2:]) == EXPECTED_JOINT_NAMES


def test_joint_order_mismatch_raises(monkeypatch):
    # Corrupt the expected-order tuple in the module and confirm load_model raises.
    import t1_nmpc.robot.model as model_mod

    bad = ("WRONG_FIRST_JOINT",) + EXPECTED_JOINT_NAMES[1:]
    monkeypatch.setattr(model_mod, "EXPECTED_JOINT_NAMES", bad)
    with pytest.raises(ValueError, match="Joint order mismatch"):
        load_model(URDF, make_config())


def test_contact_frames_added_and_named(rm):
    assert len(rm.contact_frame_ids) == 2
    for fid, name in zip(rm.contact_frame_ids, CONTACT_FRAME_NAMES):
        assert rm.model.frames[fid].name == name


def test_contact_frame_placement_offset_at_standing(rm):
    # The contact frame sits at cfg.contact_frame_offset (0.01, 0, -0.027) in the
    # ankle-roll joint frame -> the world placement difference between the contact
    # frame and its parent ankle-roll joint must equal that offset (identity rotation).
    cfg = make_config()
    q = _standing_q(rm)
    offset = cfg.contact_frame_offset
    for fid, parent_joint in zip(rm.contact_frame_ids, ("Left_Ankle_Roll", "Right_Ankle_Roll")):
        jid = rm.model.getJointId(parent_joint)
        pin.forwardKinematics(rm.model, rm.data, q)
        pin.updateFramePlacements(rm.model, rm.data)
        oMf_contact = rm.data.oMf[fid]
        oMj = rm.data.oMi[jid]                       # parent joint world placement
        jMf = oMj.inverse() * oMf_contact            # contact in parent-joint frame
        np.testing.assert_allclose(jMf.translation, offset, atol=1e-9)
        np.testing.assert_allclose(jMf.rotation, np.eye(3), atol=1e-9)
