# tests/test_croco_mpc.py
import numpy as np, pinocchio as pin
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC

def _mpc_x0():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height
    x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    return cfg, wb, mpc, x0

def test_step_returns_compatible_result():
    cfg, wb, mpc, x0 = _mpc_x0()
    mpc.reset(x0)
    res = mpc.step(x0, 0.0)
    assert res.x_traj.shape == (cfg.N + 1, 68)
    assert res.u_traj.shape == (cfg.N, 40)
    assert res.status == 0
    assert np.all(np.isfinite(res.x_traj)) and np.all(np.isfinite(res.u_traj))
    assert mpc.last_solve_s > 0.0

def test_single_rti_holds_stand_over_a_few_steps():
    cfg, wb, mpc, x0 = _mpc_x0()
    mpc.reset(x0)
    x = x0.copy()
    for _ in range(5):
        res = mpc.step(x, 0.0)
        x = res.x_traj[1].copy()                 # advance along the plan (no sim)
    assert np.linalg.norm(x[:3] - x0[:3]) < 0.05  # didn't run away
