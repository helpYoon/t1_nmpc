import numpy as np
from t1_nmpc.wb.config import make_wb_config
from t1_nmpc.wb.config import make_aligator_config
from t1_nmpc.wb.ode import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.mpc import AligatorMPC

def test_stand_step_warmstarts_and_holds_fz():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    mpc = AligatorMPC(cfg, al, am)
    x = nominal_stand_x(am, cfg); mpc.reset(x); mg = am.mass * 9.81
    res = None
    for _ in range(5):                      # warm-started repeated solves
        res = mpc.step(x, 0.0, command=np.array([0., 0., cfg.nominal_base_height, 0.]))
    u0 = np.asarray(res.us[0]); fz = u0[2] + u0[8]
    assert res.status == 0 and 0.9 < fz / mg < 1.1
    assert mpc.last_solve_s > 0
