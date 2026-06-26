# tests/test_croco_walk_problem.py
import numpy as np, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK, STANCE_GAIT
from t1_nmpc.wb.croco_walk import WalkOCP


def _ctx():
    cfg = make_wb_config(); wb = WBModel(cfg)
    x0_68 = np.zeros(68); x0_68[2] = cfg.nominal_base_height
    x0_68[6:6 + cfg.n_joints] = cfg.nominal_joint_pos
    return cfg, wb, x0_68


def test_walkocp_fixed_nu_both_contacts_every_node():
    """The persistent template keeps a FIXED control dimension (both contacts on every node) so one
    solver can be reused; gait modes are realised by toggling contact status, not by changing nu."""
    cfg, wb, x0_68 = _ctx()
    ocp = WalkOCP(cfg, wb)
    assert len(ocp.problem.runningModels) == cfg.N
    nus = {m.nu for m in ocp.problem.runningModels}
    assert nus == {wb.model.nv + 12}                      # nv + 2*6, every node


def test_walkocp_update_toggles_swing_contact():
    """After update at a single-support gait phase, the horizon must contain a node with exactly one
    active contact (the swing foot's contact deactivated)."""
    cfg, wb, x0_68 = _ctx()
    ocp = WalkOCP(cfg, wb)
    comm = np.array([0.3, 0., cfg.nominal_base_height, 0.])
    ocp.update(x0_68[:66], 0.3, comm, SLOW_WALK, x0_68)    # t_gait inside a SLOW_WALK swing window
    n_active = []
    for m in ocp.problem.runningModels:
        c = m.differential.contacts
        n_active.append(int(c.getContactStatus("c0")) + int(c.getContactStatus("c1")))
    assert 1 in n_active                                  # at least one single-support node
    assert 2 in n_active                                  # and at least one double-support node


def test_walkocp_well_conditioned_at_feasible_stance():
    """REGRESSION (barrier/solver compatibility): QuadraticBarrier friction/CoP keeps the OCP
    well-conditioned, so a single RTI from quasiStatic at a feasible double-support config leaves a
    SMALL KKT residual. The exact log-relaxed barrier (removed) bounced to ~1e4 here."""
    cfg, wb, x0_68 = _ctx()
    ocp = WalkOCP(cfg, wb)
    comm = np.array([0., 0., cfg.nominal_base_height, 0.])
    ocp.update(x0_68[:66], 0.0, comm, STANCE_GAIT, x0_68)
    s = crocoddyl.SolverIntro(ocp.problem); s.setCallbacks([])
    xs = [x0_68[:66].copy() for _ in range(cfg.N + 1)]
    us = list(ocp.problem.quasiStatic([x0_68[:66].copy() for _ in range(cfg.N)]))
    s.solve(xs, us, 1, False, 1e-9)
    assert np.all(np.isfinite(np.asarray(s.xs)))
    assert s.stoppingCriteria() < 100.0
