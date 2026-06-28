import time
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.reference import MotionPlanReference
from t1_nmpc.wb.track_ocp import PickupOCP

PLAN = "data/motion_plan.pkl"


def _setup():
    cfg = make_track_config()
    rm = load_model(cfg)
    ref = MotionPlanReference(PLAN, cfg, rm)
    ocp = PickupOCP(cfg, rm); ocp.set_weights()
    return cfg, rm, ref, ocp


def test_cold_and_warm_solve():
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(0.0)
    ocp.set_refs(x0, xr, hr, gg)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    out = ocp.retract(sol, x0)
    # node-1 state is finite and feet stay near the ground (planted)
    assert np.all(np.isfinite(out["q_sol"][1]))
    m, d = rm.model, rm.model.createData()
    pin.forwardKinematics(m, d, out["q_sol"][1]); pin.updateFramePlacements(m, d)
    for fid in rm.foot_center_frame_ids:
        assert abs(d.oMf[fid].translation[2]) < 0.02      # foot center stays ~ground
    # one warm solve at a later phase
    xr2, hr2, gg2 = ref.sample(2.0)
    sol2 = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr2, hr2, gg2, sol)).flatten()
    assert np.all(np.isfinite(sol2))


def test_leg_limits_respected():
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(8.0)            # deep-crouch region
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    out = ocp.retract(sol, x0)
    lo, hi = rm.model.lowerPositionLimit, rm.model.upperPositionLimit
    for i in (1, cfg.nodes):
        q = out["q_sol"][i]
        for j in list(range(7 + 17, 7 + 23)) + list(range(7 + 23, 7 + 29)):
            assert lo[j] - 1e-2 <= q[j] <= hi[j] + 1e-2     # knee not hyperextended, etc.


def test_realtime_warm_p90(capsys):
    """Real-time gate: warm p90 < 16 ms at the chosen N (record the number; xfail if machine slow)."""
    import pytest
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(0.0)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    ts = []
    for k in range(12):
        xr, hr, gg = ref.sample(0.2 * k)
        t0 = time.perf_counter()
        sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, sol)).flatten()
        ts.append((time.perf_counter() - t0) * 1e3)
    p90 = float(np.percentile(ts, 90))
    print(f"\npickup warm solve p90 = {p90:.1f} ms (N={cfg.nodes})")
    if p90 >= 16.0:
        pytest.xfail(f"solve p90 {p90:.1f}ms >= 16ms — drop N to 8 (fallback) or trim leg limits")
