"""WalkOCP regression: stand case (all-stance) and walk case (LF swing) both solve.

The compiled solver_fn takes the per-tick schedules/commands as ARGUMENTS (not
set_value) so the SAME function is reused across MPC ticks (Task 6). set_value is
still used so g_data() / opti.value(opti.p) report the matching parameter vector.
"""
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.ocp import WalkOCP
from t1_nmpc.wb.gait import WalkGait, StandGait


def _solve(ocp, cfg, rm, x0, contact, swing, max_iter=200):
    ocp.set_weights(); ocp.set_x_init(x0)
    ocp.set_schedules(contact, swing); ocp.set_base_vx(0.0)
    ft = np.zeros((2 * cfg.n_feet, cfg.nodes))
    ocp.set_footstep_targets(ft)
    fn = ocp.solve_function(max_iter)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, contact, swing, 0.0, ft,
                      ocp.x_initial())).flatten()
    g, lbg, ubg = ocp.g_data()(sol, ocp.opti.value(ocp.opti.p))
    return ocp.retract(sol), WalkOCP.constr_viol_inf(
        np.array(g).flatten(), np.array(lbg).flatten(), np.array(ubg).flatten())


def test_walkocp_stand_case_solves():
    cfg = make_config(); rm = load_model(cfg); ocp = WalkOCP(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    ct, sw = StandGait(cfg).schedules(0.0)
    out, cv = _solve(ocp, cfg, rm, x0, ct, sw)
    assert cv < 1e-2, f"stand OCP CV {cv:.2e}"
    fz = sum(out["forces_sol"][0][3 * c + 2] for c in range(8))
    assert 0.9 <= fz / (rm.mass * 9.81) <= 1.1          # gravity supported


def test_walkocp_walk_case_solves_and_lifts():
    cfg = make_config(); rm = load_model(cfg); ocp = WalkOCP(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    ct, sw = WalkGait(cfg).schedules(0.0)                # LF swing at the front of the horizon
    out, cv = _solve(ocp, cfg, rm, x0, ct, sw, max_iter=300)
    assert cv < 1e-2, f"walk OCP CV {cv:.2e}"
    # swing foot (Left = corners 0-3) carries ~zero force at an interior swing node
    swing_node = 5
    f_left = sum(abs(out["forces_sol"][swing_node][3 * c + k])
                 for c in range(4) for k in range(3))
    assert f_left < 5.0, f"swing-foot force not ~0: {f_left}"
