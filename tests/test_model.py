import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config, JOINT_NAMES
from t1_nmpc.robot.model import load_model, nominal_q

def test_model_build_corners_mass():
    cfg = make_config()
    rm = load_model(cfg)
    assert rm.model.nq == 36 and rm.model.nv == 35
    assert tuple(rm.model.names[2:]) == JOINT_NAMES          # joint order
    assert len(rm.corner_frame_ids) == 8
    assert abs(rm.mass - 34.5135) < 1e-3
    assert rm.tau_max.shape == (29,) and np.all(rm.tau_max > 0)
    # 8 corners coplanar at the ground at nominal stand
    q = nominal_q(cfg, rm.model)
    pin.forwardKinematics(rm.model, rm.data, q)
    pin.updateFramePlacements(rm.model, rm.data)
    zs = np.array([rm.data.oMf[c].translation[2] for c in rm.corner_frame_ids])
    assert zs.max() - zs.min() < 1e-6                         # coplanar
    assert abs(zs.mean()) < 2e-3                              # ~on the ground
    # corners share 2 parent joints, 4 each
    parents = [rm.model.frames[c].parentJoint for c in rm.corner_frame_ids]
    assert len(set(parents)) == 2 and all(parents.count(p) == 4 for p in set(parents))
