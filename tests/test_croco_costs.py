# tests/test_croco_costs.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb import croco_costs
from t1_nmpc.wb.croco_costs import _control_weights

def _ctx():
    cfg = make_wb_config(); wb = WBModel(cfg)
    state = crocoddyl.StateMultibody(wb.model)
    act = crocoddyl.ActuationModelFloatingBase(state)
    return cfg, wb, state, act

def test_build_costs_double_support_dims():
    cfg, wb, state, act = _ctx()
    nv = wb.model.nv
    nu = nv + 12                                  # double support
    x_ref = np.zeros(state.nx); x_ref[2] = cfg.nominal_base_height
    x_ref[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    assert costs.nu == nu
    names = set(costs.costs.todict().keys())
    assert {"xreg", "ureg", "tau_lim", "joint_lim", "com"} <= names
    assert sum(n.startswith("wrenchcone_") for n in names) == len(list(wb.contact_fids))

def test_ureg_activation_weights_match_control_weights():
    cfg, wb, state, act = _ctx()
    nv = wb.model.nv
    nc = 12  # double-support
    nu = nv + nc
    x_ref = np.zeros(state.nx)
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    activation = costs.costs["ureg"].cost.activation
    expected = _control_weights(nv, nc, np.asarray(cfg.R, float))
    assert np.allclose(np.asarray(activation.weights), expected)

def test_state_weight_comes_from_config_Q():
    cfg, wb, state, act = _ctx()
    nu = wb.model.nv + 12
    x_ref = np.zeros(state.nx)
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    act_model = costs.costs["xreg"].cost.activation
    assert np.allclose(np.asarray(act_model.weights), cfg.Q[:66])

def test_control_weights_slice_mapping():
    """Directly verify the R-index→w-index mapping in _control_weights.
    A round-trip test through build_costs cannot catch a pure slice-offset bug because
    the real cfg.R has identical values in many slots; a synthetic arange exposes it."""
    R = np.arange(40, dtype=float)
    w = _control_weights(33, 12, R)
    assert w.shape == (45,)
    # base-acceleration slots -> tiny regulariser
    assert np.all(w[0:6] == 1e-6)
    # joint qdd -> R[12:39]
    assert np.allclose(w[6:33], R[12:39])
    # left foot wrench -> R[0:6]
    assert np.allclose(w[33:39], R[0:6])
    # right foot wrench -> R[6:12]
    assert np.allclose(w[39:45], R[6:12])
