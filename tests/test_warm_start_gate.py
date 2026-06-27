import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait
from t1_nmpc.wb.mpc import AligatorMPC


def test_warm_start_gate_stand():
    """Receding-horizon warm-start gate (stand / all-double-support).

    Verifies warm-started ProxDDP ticks stay cheap (few iters) and feasible (CV<=1e-2) across
    >=15 receding ticks under small disturbances — i.e. the warm primal+dual carry works in
    closed loop. The contact-switch (single-support) gate is deferred to the follow-up walk
    plan; this OCP's single-support nodes are ill-posed without the CoM-sway/footstep references
    built there (cold double-support CV 1e-5 in 1 iter; cold full-walk CV 0.08 oscillating)."""
    cfg = make_config()
    rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x = nominal_x(cfg, rm.model)
    mpc.reset(x)
    rng = np.random.default_rng(0)
    cvs, iters = [], []
    for _ in range(20):                                       # >= 15 receding ticks
        x_meas = x.copy()
        x_meas[:3] += rng.normal(0, 2e-3, 3)                  # base position jitter
        x_meas[cfg.nq:cfg.nq + 6] += rng.normal(0, 2e-3, 6)  # base velocity jitter
        res = mpc.step(x_meas, t=0.0)
        cvs.append(res.constr_viol); iters.append(res.num_iters)
        x = np.asarray(mpc._warm[0][1], dtype=np.float64)     # advance to planned node 1
    assert max(cvs) <= 1e-2, f"max CV {max(cvs):.2e} exceeds 1e-2"
    assert max(iters) <= 5, f"max warm iters {max(iters)} exceeds 5"
