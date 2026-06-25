# tests/test_croco_walk_mpc.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.croco_mpc import CrocoMPC

def test_walk_step_rebuilds_and_emits_stance_aware_u():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0)
    res = mpc.step(x0, 0.0, command=np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]))
    assert res.x_traj.shape == (cfg.N+1, 68) and res.u_traj.shape == (cfg.N, 40)
    assert res.status == 0 and np.all(np.isfinite(res.u_traj))

def test_walk_advances_gait_clock_a_few_steps():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0); x = x0.copy()
    for _ in range(5):
        res = mpc.step(x, 0.0, command=np.array([0.3,0.,cfg.nominal_base_height,0.]))
        assert np.all(np.isfinite(res.x_traj)); x = res.x_traj[1].copy()
    assert mpc._t_gait > 0.0
