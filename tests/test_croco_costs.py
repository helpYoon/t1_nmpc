# tests/test_croco_costs.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb import croco_costs

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
    assert {"xreg", "ureg", "tau_lim", "joint_lim"} <= names
    assert any(n.startswith("wrenchcone_") for n in names)   # one per stance foot

def test_state_weight_comes_from_config_Q():
    cfg, wb, state, act = _ctx()
    nu = wb.model.nv + 12
    x_ref = np.zeros(state.nx)
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    act_model = costs.costs["xreg"].cost.activation
    assert np.allclose(np.asarray(act_model.weights), cfg.Q[:66])
