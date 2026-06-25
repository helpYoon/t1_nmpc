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
