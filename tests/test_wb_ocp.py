import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.ocp_wb import make_ocp, build_solver
from t1_nmpc.wb.cost_wb import N_PARAM_WB


def test_make_ocp_shapes():
    cfg = make_wb_config()
    ocp, bundle = make_ocp(cfg)
    assert ocp.model.x.shape[0] == 68 and ocp.model.u.shape[0] == 40
    assert ocp.solver_options.N_horizon == 31
    assert abs(ocp.solver_options.tf - 1.085) < 1e-9


def test_build_and_solve_stand():
    cfg = make_wb_config()
    ocp, bundle = make_ocp(cfg)
    solver = build_solver(ocp)
    model = bundle.model
    x0 = model.nominal_state()
    u_ref = np.zeros(40)
    fz = model.total_mass() * 9.81 / 2.0
    u_ref[2] = fz; u_ref[8] = fz
    p = np.concatenate([x0, u_ref])
    assert p.shape[0] == N_PARAM_WB

    N = cfg.N
    for k in range(N + 1):
        solver.set(k, "p", p)
        solver.set(k, "x", x0)
    for k in range(N):
        solver.set(k, "u", u_ref)
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    status = solver.solve()
    assert status in (0, 2), f"solve status {status}"   # 0 converged, 2 = max_iter (single RTI)
    xtraj = np.array([solver.get(k, "x") for k in range(N + 1)])
    utraj = np.array([solver.get(k, "u") for k in range(N)])
    assert np.all(np.isfinite(xtraj)) and np.all(np.isfinite(utraj))
    assert np.allclose(xtraj[0], x0, atol=1e-6)          # node 0 pinned to x0
