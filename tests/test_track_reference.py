import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model
from t1_nmpc.wb.reference import MotionPlanReference

PLAN = "data/motion_plan.pkl"


def _ref():
    cfg = make_track_config()
    rm = load_model(cfg)
    return MotionPlanReference(PLAN, cfg, rm), cfg, rm


def test_mapping_fk_roundtrip():
    """q_ref → FK at hand frames must match the plan's hand_xyz to < 1 mm (validates base+arm map)."""
    ref, cfg, rm = _ref()
    m = rm.model; d = m.createData()
    lh, rh = rm.hand_frame_ids
    import pickle
    plan = pickle.load(open(PLAN, "rb"))
    for si in (0, 3, 6):
        seg = plan["segments"][si]
        for k in (0, 40, 80):
            q = ref.frame_to_xref(seg, k)
            pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
            assert np.linalg.norm(d.oMf[lh].translation - seg["position"]["left_hand_xyz"][k]) < 1e-3
            assert np.linalg.norm(d.oMf[rh].translation - seg["position"]["right_hand_xyz"][k]) < 1e-3


def test_mapping_sign_and_base():
    ref, cfg, rm = _ref()
    import pickle
    seg = pickle.load(open(PLAN, "rb"))["segments"][2]
    k = 30
    q = ref.frame_to_xref(seg, k)
    P = seg["position"]
    assert abs(q[2] - P["trunk_height"][k]) < 1e-9           # base z = trunk_height
    assert abs(q[7 + 16] - (-P["trunk_yaw"][k])) < 1e-9      # Waist = -trunk_yaw
    assert abs(q[7 + 17] - P["trunk_pitch"][k]) < 1e-9       # L_Hip_Pitch = trunk_pitch
    assert abs(q[7 + 23] - P["trunk_pitch"][k]) < 1e-9       # R_Hip_Pitch = trunk_pitch (broadcast)
    assert np.allclose(q[7 + 2:7 + 9], P["left_arm"][k])     # arm order, no reindex


def test_grasp_events():
    ref, cfg, rm = _ref()
    # held_objs: seg0[] 1[L] 2[L] 3[L,R] 4[L,R] 5[R] 6[R] 7[]
    # left flips at seg1 start (grasp) and seg5 start (release); right at seg3 and seg7.
    assert len(ref.events[0]) == 2 and len(ref.events[1]) == 2
    # events are increasing phase times within the plan duration
    for h in (0, 1):
        assert all(0 < t < ref.duration_phase for t in ref.events[h])
