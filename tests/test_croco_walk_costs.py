# tests/test_croco_walk_costs.py
import numpy as np
import crocoddyl
import pinocchio as pin
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb import reference_wb
from t1_nmpc.wb import croco_costs


def _ctx():
    cfg = make_wb_config(); wb = WBModel(cfg)
    state = crocoddyl.StateMultibody(wb.model); act = crocoddyl.ActuationModelFloatingBase(state)
    q0 = pin.neutral(wb.model); q0[2] = cfg.nominal_base_height
    q0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    data = wb.model.createData(); pin.framesForwardKinematics(wb.model, data, q0)
    planted = {f: data.oMf[f].copy() for f in wb.contact_fids}
    x0 = np.concatenate([q0, np.zeros(wb.model.nv)])
    return cfg, wb, state, act, planted, x0


def test_walk_costs_single_support_has_swing_and_relaxed_terms():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids
    nu = wb.model.nv + 6                                  # single support
    swing = dict(fid=R, z=0.05, w_z=1e3)
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L], cfg,
                                    swing=swing, planted=planted, walk=True)
    names = set(costs.costs.todict().keys())
    assert {"xreg", "ureg", "tau_lim", "swing_z", "swing_vel", "swing_flat",
            f"friction_{L}", f"cop_{L}", f"stance_z_{L}", f"stance_flat_{L}"} <= names
    assert not any(n.startswith("wrenchcone") for n in names)   # walk uses split friction/CoP


def test_walk_costs_m0_path_unchanged():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids
    nu = wb.model.nv + 12
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L, R], cfg)  # walk=False default
    names = set(costs.costs.todict().keys())
    assert any(n.startswith("wrenchcone") for n in names)       # M0 WrenchCone preserved
    assert not any(n.startswith("swing") for n in names)


def test_walk_costs_terminal_is_qfinal():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids; nu = wb.model.nv + 12
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L, R], cfg,
                                    planted=planted, walk=True, terminal=True)
    names = list(costs.costs.todict().keys())
    assert names == ["xreg"]                                    # state-only terminal
    w = np.asarray(costs.costs["xreg"].cost.activation.weights)
    assert np.allclose(w, cfg.Q_final[:66] * cfg.terminal_scale)


def test_build_reference_66_shape_and_drops_path_slots():
    cfg = make_wb_config(); wb = WBModel(cfg)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height
    x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    nt = np.arange(cfg.N + 1) * cfg.dt
    xr = reference_wb.build_reference_66(x0, np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]),
                                         SLOW_WALK, 0.0, nt, cfg, wb)
    assert xr.shape == (cfg.N + 1, 66)
    # forward command -> base x advances across the horizon
    assert xr[-1, 0] > xr[0, 0]
