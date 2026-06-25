from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.ocp_wb import make_ocp, COMPILE_FLAGS_DISCRETE

def test_default_is_discrete_reduced_o2():
    cfg = make_wb_config()
    ocp, _ = make_ocp(cfg)
    assert ocp.solver_options.integrator_type == "DISCRETE"
    assert ocp.solver_options.ext_fun_compile_flags == COMPILE_FLAGS_DISCRETE
    assert "-fno-schedule-insns2" in COMPILE_FLAGS_DISCRETE
    assert "-fno-gcse" in COMPILE_FLAGS_DISCRETE

def test_erk_still_available():
    cfg = make_wb_config()
    ocp, _ = make_ocp(cfg, discrete=False)
    assert ocp.solver_options.integrator_type == "ERK"

import numpy as np
import casadi as cs
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_DT
from t1_nmpc.wb.ocp_wb import _rk4

def test_pdt_param_layout_and_default():
    # P_DT is a valid scalar index inside the param vector
    assert isinstance(P_DT, int) and 0 <= P_DT < N_PARAM_WB

def test_rk4_param_dt_equals_const_dt_at_nominal():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu)
    f_const = cs.Function("fc", [x, u], [_rk4(m, x, u, cfg.dt)])
    f_param = cs.Function("fp", [x, u], [_rk4(m, x, u, cs.SX(cfg.dt))])
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu); u0[2] = u0[8] = m.total_mass() * 9.81 / 2
    np.testing.assert_allclose(np.array(f_const(x0, u0)).ravel(),
                               np.array(f_param(x0, u0)).ravel(), atol=1e-12)
