# tests/test_croco_problem.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_problem import T1ProblemBuilder

def _builder_x0():
    cfg = make_wb_config(); wb = WBModel(cfg)
    b = T1ProblemBuilder(cfg, wb)
    q0 = pin.neutral(wb.model); q0[2] = cfg.nominal_base_height
    q0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    x0 = np.concatenate([q0, np.zeros(wb.model.nv)])
    return cfg, wb, b, x0

def test_stand_problem_shape_and_nu_nh():
    cfg, wb, b, x0 = _builder_x0()
    prob = b.build_stand_problem(x0)
    assert len(prob.runningModels) == cfg.N
    d = prob.runningModels[0]
    assert d.nu == wb.model.nv + 12                  # double support
    assert d.differential.nh == 18                   # 6 underactuated + 12 contact

def test_stand_problem_solves_holds_contact():
    cfg, wb, b, x0 = _builder_x0()
    prob = b.build_stand_problem(x0)
    solver = crocoddyl.SolverIntro(prob)
    xs = [x0.copy() for _ in range(cfg.N + 1)]
    us = prob.quasiStatic([x0.copy() for _ in range(cfg.N)])
    solver.solve(xs, us, 80, False, 1e-9)
    assert np.all(np.isfinite(np.asarray(solver.xs)))
    drift = max(np.linalg.norm(np.asarray(solver.xs)[k][:3] - x0[:3]) for k in range(cfg.N + 1))
    assert drift < 0.05                              # stand holds (<5 cm)
