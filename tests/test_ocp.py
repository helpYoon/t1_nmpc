import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.ocp import StandOCP

def test_stand_ocp_converges():
    cfg = make_config()
    rm = load_model(cfg)
    ocp = StandOCP(cfg, rm)
    ocp.set_weights()
    x0 = nominal_x(cfg, rm.model)
    ocp.set_x_init(x0)
    sol_fn = ocp.solve_function(max_iter=cfg.fatrop_max_iter)
    # cold start: warm param = current opti.x initial (zeros for DX, gravity-comp handled inside)
    sol_x = np.array(sol_fn(x0, cfg.Q_diag, cfg.R_diag, ocp.x_initial())).flatten()
    g, lbg, ubg = ocp.g_data()(sol_x, ocp.opti.value(ocp.opti.p))
    cv = StandOCP.constr_viol_inf(np.array(g).flatten(), np.array(lbg).flatten(), np.array(ubg).flatten())
    assert cv < 1e-4, f"constraint violation too high: {cv}"
    out = ocp.retract(sol_x)
    fz = np.array(out["forces_sol"][0]).reshape(8, 3)[:, 2]
    assert abs(fz.sum() - rm.mass * 9.81) / (rm.mass * 9.81) < 0.05   # vertical balance
    assert np.all(fz > -1e-6)                                          # unilateral
