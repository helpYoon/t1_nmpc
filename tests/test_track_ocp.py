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


def test_realtime_warm_capped(capsys):
    """The RTI warm cap (cfg.track_warm_iters=5) BOUNDS worst-case warm solve time — uncapped, the
    reach/grasp phases spike to 100s of ms-seconds. Real-time PASS/FAIL is the Task 6 closed-loop;
    here we verify the cap keeps every warm solve bounded + finite and record the numbers."""
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.track_warm_iters)
    xr, hr, gg = ref.sample(0.0)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    ts = []
    for k in range(15):
        xr, hr, gg = ref.sample(0.7 * k)
        t0 = time.perf_counter()
        sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, sol)).flatten()
        ts.append((time.perf_counter() - t0) * 1e3)
    assert np.all(np.isfinite(sol))
    p90, mx = float(np.percentile(ts, 90)), float(np.max(ts))
    print(f"\npickup warm solve (cap={cfg.track_warm_iters}, N={cfg.nodes}): p90={p90:.1f} max={mx:.1f} ms")
    assert mx < 150.0
