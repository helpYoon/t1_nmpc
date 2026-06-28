import numpy as np
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model


def test_track_config_defaults():
    cfg = make_track_config()
    assert cfg.nodes == 8
    assert cfg.dt_min == 0.04 and cfg.dt_max == 0.04
    assert cfg.Q_diag.shape == (cfg.ndx,)        # 70
    assert cfg.time_scale == 5.0
    assert cfg.w_hand == 400.0
    assert cfg.grasp_halfwidth > 0.0
    assert cfg.track_warm_iters == 5


def test_track_config_override():
    cfg = make_track_config(nodes=8, time_scale=3.0)
    assert cfg.nodes == 8 and cfg.time_scale == 3.0


def test_hand_frame_ids_resolve():
    cfg = make_track_config()
    rm = load_model(cfg)
    lh, rh = rm.hand_frame_ids
    assert rm.model.frames[lh].name == "left_hand_link"
    assert rm.model.frames[rh].name == "right_hand_link"
