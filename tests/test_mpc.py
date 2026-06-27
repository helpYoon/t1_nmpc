import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait, WalkGait
from t1_nmpc.wb.mpc import AligatorMPC


def test_reset_solves_stand():
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0, t=0.0)
    assert res.constr_viol < 1e-2
    assert res.command.tau_ff.shape == (27,)
    assert res.forces0.shape == (12,)


def test_warm_step_converges_fast_stand():
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    iters = []
    for _ in range(5):
        res = mpc.step(x0, t=0.0)
        iters.append(res.num_iters)
        assert res.constr_viol < 1e-2
    assert iters[-1] <= 5                      # warm convergence
