# tests/test_croco_walk_problem.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.croco_problem import T1ProblemBuilder

def _b():
    cfg = make_wb_config(); wb = WBModel(cfg); b = T1ProblemBuilder(cfg, wb)
    x0_68 = np.zeros(68); x0_68[2] = cfg.nominal_base_height
    x0_68[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    return cfg, wb, b, x0_68

def test_build_walk_problem_per_node_stance_matches_gait():
    cfg, wb, b, x0_68 = _b()
    # pick t_gait inside a single-support window of SLOW_WALK so the horizon has DS and SS nodes
    prob = b.build_walk_problem(x0_68[:66], 0.3, np.array([0.3,0.,cfg.nominal_base_height,0.]), SLOW_WALK, x0_68)
    assert len(prob.runningModels) == cfg.N
    nus = {m.nu for m in prob.runningModels}
    assert nus <= {wb.model.nv + 6, wb.model.nv + 12}    # SS=39 and/or DS=45 present
    # at least one single-support node exists in a SLOW_WALK horizon
    assert (wb.model.nv + 6) in nus

def test_build_walk_problem_solves():
    cfg, wb, b, x0_68 = _b()
    prob = b.build_walk_problem(x0_68[:66], 0.3, np.array([0.3,0.,cfg.nominal_base_height,0.]), SLOW_WALK, x0_68)
    s = crocoddyl.SolverIntro(prob); s.setCallbacks([])
    xs = [x0_68[:66].copy() for _ in range(cfg.N+1)]; us = list(prob.quasiStatic([x0_68[:66].copy() for _ in range(cfg.N)]))
    s.solve(xs, us, 30, False, 1e-9)
    assert np.all(np.isfinite(np.asarray(s.xs)))

def test_walk_problem_well_conditioned_at_feasible_stance():
    """REGRESSION (relaxed-barrier sign convention): crocoddyl cone residuals encode each
    inequality as a one-sided bound on r (FrictionCone/CoP rows are feasible at r<=0). At a
    feasible double-support config those barriers must sit in their safe region, so a single RTI
    from quasiStatic leaves a SMALL KKT residual. The earlier sign-flipped barrier (applied to the
    raw residual as if h=r>=0) penalised the FEASIBLE cone region, producing ~1e4 gradients and an
    ill-conditioned OCP that diverged."""
    from t1_nmpc.wb.gait_wb import STANCE_GAIT
    cfg, wb, b, x0_68 = _b()
    prob = b.build_walk_problem(x0_68[:66], 0.0, np.array([0.,0.,cfg.nominal_base_height,0.]), STANCE_GAIT, x0_68)
    s = crocoddyl.SolverIntro(prob); s.setCallbacks([])
    xs = [x0_68[:66].copy() for _ in range(cfg.N+1)]; us = list(prob.quasiStatic([x0_68[:66].copy() for _ in range(cfg.N)]))
    s.solve(xs, us, 1, True, 1e-9)
    assert np.all(np.isfinite(np.asarray(s.xs)))
    assert s.stoppingCriteria() < 100.0    # was ~1.1e4 with the sign-flipped barrier
