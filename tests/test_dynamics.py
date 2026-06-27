import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_q
from t1_nmpc.wb.dynamics import WBDynamics

def _numeric_rnea(model, data, ee, q, v, a, forces):
    pin.framesForwardKinematics(model, data, q)
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force(np.zeros(6)))
    for idx, fid in enumerate(ee):
        jid = model.frames[fid].parentJoint
        trans = model.frames[fid].placement.translation
        R = data.oMi[jid].rotation.T
        fl = R @ forces[idx*3:idx*3+3]
        fext[jid] = pin.Force(fext[jid].vector + np.concatenate([fl, np.cross(trans, fl)]))
    return pin.rnea(model, data, q, v, a, fext)

def test_rnea_matches_pinocchio():
    cfg = make_config(); rm = load_model(cfg)
    dyn = WBDynamics(rm.model, rm.corner_frame_ids)
    assert dyn.nf == 24 and dyn.nj == 29
    q = nominal_q(cfg, rm.model)
    rng = np.random.default_rng(0)
    v = 0.05*rng.standard_normal(rm.model.nv); a = 0.1*rng.standard_normal(rm.model.nv)
    forces = rng.uniform(-10, 10, 24); forces[2::3] += 42.0
    tau_sym = np.array(dyn.rnea_dynamics()(q, v, a, forces)).flatten()
    tau_ref = _numeric_rnea(rm.model, rm.data, rm.corner_frame_ids, q, v, a, forces)
    assert np.max(np.abs(tau_sym - tau_ref)) < 1e-8

def test_state_roundtrip():
    cfg = make_config(); rm = load_model(cfg)
    dyn = WBDynamics(rm.model, rm.corner_frame_ids)
    x = np.concatenate([nominal_q(cfg, rm.model), np.zeros(rm.model.nv)])
    dx = np.zeros(70)
    x2 = np.array(dyn.state_integrate()(x, dx)).flatten()
    assert np.allclose(x2, x, atol=1e-12)
    d = np.array(dyn.state_difference()(x, x)).flatten()
    assert np.allclose(d, 0.0, atol=1e-12)
