import numpy as np
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb.ocp import OCPBuilder


def _builder():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn, OCPBuilder(cfg, rm, dyn)


def test_build_double_support_stage():
    cfg, rm, dyn, b = _builder()
    stage, handles = b.build_stage((True, True))
    assert stage.nu == 45 and stage.ndx1 == 66
    # constraints: rnea(6) + 2*(wrenchcone 8 + contactvel 6) = 6 + 28 = 34 constraint rows.
    # num_dual additionally includes the gap-closing dynamics co-state block (ndx2 = 66),
    # so the constraint-row count the arithmetic targets is constraints.total_dim.
    assert stage.constraints.total_dim == 6 + 2 * (8 + 6)
    assert stage.num_dual == stage.ndx2 + 6 + 2 * (8 + 6)
    assert handles["swing"] == []        # no swinging feet


def test_build_swing_stage():
    cfg, rm, dyn, b = _builder()
    stage, handles = b.build_stage((False, True))   # LF swing, RF stance
    # rnea(6) + LF(swingwrench 6 + swingz 1) + RF(wrenchcone 8 + contactvel 6) = 27 constraint rows.
    # num_dual = constraint rows + gap-closing dynamics co-state block (ndx2 = 66).
    assert stage.constraints.total_dim == 6 + (6 + 1) + (8 + 6)
    assert stage.num_dual == stage.ndx2 + 6 + (6 + 1) + (8 + 6)
    # add order: rnea(idx0), LF swingwrench(idx1), LF swing-z(idx2), RF wrenchcone(idx3), RF contactvel(idx4)
    assert handles["swing"] == [(0, 2)]              # (foot_index, constraint-stack index of swing-z)
    # the recorded index must point at the sliced swing-z residual (nr==1) inside the stage's stack
    assert stage.constraints.funcs[2].nr == 1


def test_build_problem_integrity():
    cfg, rm, dyn, b = _builder()
    x0 = nominal_x(cfg, rm.model)
    modes = [(True, True)] * cfg.nodes
    problem, handles = b.build_problem(modes, x0)
    assert problem.num_steps == cfg.nodes
    assert len(handles) == cfg.nodes
    problem.checkIntegrity()                          # raises if malformed
