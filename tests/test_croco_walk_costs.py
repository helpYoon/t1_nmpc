# tests/test_croco_walk_costs.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb import reference_wb


def test_build_reference_66_shape_and_drops_path_slots():
    cfg = make_wb_config(); wb = WBModel(cfg)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height
    x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    nt = np.arange(cfg.N + 1) * cfg.dt
    xr = reference_wb.build_reference_66(x0, np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]),
                                         SLOW_WALK, 0.0, nt, cfg, wb)
    assert xr.shape == (cfg.N + 1, 66)
    # forward command -> base x advances across the horizon
    assert xr[-1, 0] > xr[0, 0]
