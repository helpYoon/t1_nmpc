import numpy as np
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.track_mpc import TrackingMPC

PLAN = "data/motion_plan.pkl"


def test_step_returns_command_and_advances():
    cfg = make_track_config()
    rm = load_model(cfg)
    mpc = TrackingMPC(cfg, rm, PLAN)
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0, 0.0)
    assert res.command.q_des.shape == (29,)
    assert res.command.qd_des.shape == (29,)
    assert res.command.tau_ff.shape == (29,)
    assert res.solve_time > 0.0
    assert np.all(np.isfinite(res.command.tau_ff))
    # a later tick warm-starts and still returns finite commands
    res2 = mpc.step(x0, 1.0)
    assert np.all(np.isfinite(res2.command.q_des))
    assert mpc.duration_wall > 10.0           # ~14.8s * time_scale
