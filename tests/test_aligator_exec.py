import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.aligator_model import build_aligator_model, make_ode, nominal_stand_x
from t1_nmpc.wb.aligator_exec import extract_tau_ff

def test_tau_ff_base_rows_vanish():
    cfg = make_wb_config(); am = build_aligator_model(cfg)
    ode = make_ode(am, [True, True]); nu = ode.nu
    x = nominal_stand_x(am, cfg)
    u0 = np.zeros(nu); u0[2] = u0[8] = am.mass * 9.81 / 2
    tau_ff, wl, wr = extract_tau_ff(am, x, u0)
    assert tau_ff.shape[0] == am.nv - 6 == 27
    assert np.allclose(wl, u0[0:6]) and np.allclose(wr, u0[6:12])
    # base-6 generalized force must vanish (structural underactuation consistency)
    # (extract recomputes tau internally; re-derive base rows here for the assert)
    import pinocchio as pin
    rdata = am.model.createData(); q = x[:am.nq]; v = x[am.nq:]
    od = ode.createData(); ode.forward(x, u0, od)
    a = np.asarray(od.xdot)[am.nv:].copy(); a[6:] = u0[12:]
    tau = pin.rnea(am.model, rdata, q, v, a)
    pin.computeJointJacobians(am.model, rdata, q); pin.framesForwardKinematics(am.model, rdata, q)
    for k, fid in enumerate(am.foot_ids):
        J = pin.getFrameJacobian(am.model, rdata, fid, pin.LOCAL_WORLD_ALIGNED)
        tau -= J.T @ u0[k*6:(k+1)*6]
    assert np.linalg.norm(tau[:6]) < 1e-6
