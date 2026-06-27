import numpy as np
import aligator
from t1_nmpc.wb.config import make_wb_config
from t1_nmpc.wb.config import make_aligator_config
from t1_nmpc.wb.ode import build_aligator_model, make_ode, nominal_stand_x
from t1_nmpc.wb.ocp import make_stage, build_problem


def test_hard_cone_stand_holds_fz_equals_mg():
    cfg = make_wb_config()
    al = make_aligator_config()
    am = build_aligator_model(cfg)
    x = nominal_stand_x(am, cfg)
    mg = am.mass * 9.81
    schedule = [[True, True]] * al.N
    prob = build_problem(am, cfg, al, x, x, schedule, [[] for _ in range(al.N)])
    s = aligator.SolverProxDDP(1e-4, 1e-2, max_iters=30, verbose=aligator.QUIET)
    s.setup(prob)
    ode = make_ode(am, [True, True])
    u_grav = np.zeros(ode.nu)
    u_grav[2] = u_grav[8] = mg / 2
    s.run(prob, [x.copy() for _ in range(al.N + 1)], [u_grav.copy() for _ in range(al.N)])
    u0 = np.asarray(s.results.us[0])
    fz = u0[2] + u0[8]
    assert 0.95 < fz / mg < 1.05, f"force-shed: fz/mg={fz/mg:.3f}"
    assert np.asarray(s.results.xs[-1])[2] > 0.6


def test_soft_toggle_builds():
    cfg = make_wb_config()
    al = make_aligator_config()
    al.hard_cones = False
    am = build_aligator_model(cfg)
    x = nominal_stand_x(am, cfg)
    st = make_stage(am, cfg, al, [True, True], x, [], make_ode(am, [True, True]))
    assert st.nu == 39
