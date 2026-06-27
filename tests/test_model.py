import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_q, nominal_x


def test_reduced_model_dims():
    rm = load_model(make_config())
    assert rm.model.nq == 34 and rm.model.nv == 33
    assert rm.model.njoints == 29           # universe + root + 27 actuated
    assert rm.tau_max.shape == (27,)
    assert "AAHead_yaw" not in list(rm.model.names)
    assert "Head_pitch" not in list(rm.model.names)


def test_sole_frames_present_and_placed():
    rm = load_model(make_config())
    assert len(rm.sole_frame_ids) == 2
    for fid in rm.sole_frame_ids:
        assert rm.model.frames[fid].type == pin.FrameType.OP_FRAME
    # foot_joint_placements parent the sole frames at the ankle-roll joints
    assert len(rm.foot_joint_placements) == 2
    for (jid, jMf) in rm.foot_joint_placements:
        assert isinstance(jMf, pin.SE3)
        np.testing.assert_allclose(jMf.translation, [0.005, 0.0, -0.030], atol=1e-9)
    assert rm.half_extents == (0.1065, 0.05)


def test_nominal_consistency():
    cfg = make_config(); rm = load_model(cfg)
    q = nominal_q(cfg, rm.model)
    assert q.shape == (34,)
    np.testing.assert_allclose(q[2], cfg.nominal_base_height)
    np.testing.assert_allclose(q[3:7], [0, 0, 0, 1])    # quat xyzw identity
    x = nominal_x(cfg, rm.model)
    assert x.shape == (67,)
    assert np.allclose(x[34:], 0.0)


def test_mass_positive():
    rm = load_model(make_config())
    assert 20.0 < rm.mass < 60.0
