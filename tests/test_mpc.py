import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.mpc import WholeBodyMPC

def test_mpc_reset_and_step():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0)
    assert res.command.tau_ff.shape == (29,)
    assert res.command.q_des.shape == (29,) and res.command.qd_des.shape == (29,)
    assert res.constr_viol < 1e-3
    fz = res.forces0.reshape(8, 3)[:, 2]
    assert abs(fz.sum() - rm.mass * 9.81) / (rm.mass * 9.81) < 0.05
    # a second warm-started step also converges
    res2 = mpc.step(x0)
    assert res2.constr_viol < 1e-3
