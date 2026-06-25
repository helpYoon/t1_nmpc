import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.reference_wb import filter_command, build_reference

cfg = make_wb_config(); model = WBModel(cfg)
H = 0.6734
NT = np.arange(cfg.N + 1) * cfg.dt        # node times from t0=0


def test_filter_law():
    out = filter_command(np.zeros(4), np.array([0.3, 0.0, H, 0.0]))
    np.testing.assert_allclose(out, [0.06, 0.0, 0.2 * H, 0.0])   # 0.8*0 + 0.2*cmd


def test_forward_displacement_two_phase():
    x0 = model.nominal_state()
    xref, _ = build_reference(x0, np.array([0.3, 0.0, H, 0.0]), SLOW_WALK, 0.0, NT, cfg, model)
    # two-phase: 0.15 m/s over 0.7*1.1=0.77 then 0.3 over 0.33 -> ~0.2145 m at horizon end
    assert 0.18 < xref[-1, 0] < 0.25 and xref[-1, 0] > xref[0, 0]
    np.testing.assert_allclose(xref[5, 33:39], [0.3, 0.0, 0.0, 0.0, 0.0, 0.0])  # base-vel slot = cmd


def test_heading_rotation():
    x0 = model.nominal_state(); x0[3] = np.pi / 2          # yaw = 90 deg
    xref, _ = build_reference(x0, np.array([0.3, 0.0, H, 0.0]), SLOW_WALK, 0.0, NT, cfg, model)
    np.testing.assert_allclose(xref[3, 33:35], [0.0, 0.3], atol=1e-6)   # vx command -> +vy world


def test_gravity_split_per_node_stance():
    x0 = model.nominal_state()
    _, uref = build_reference(x0, np.array([0.3, 0.0, H, 0.0]), SLOW_WALK, 0.0, NT, cfg, model)
    mg = model.total_mass() * 9.81
    # node at t=0.3 is LF (left stance only) -> full mg on left fz (u[2]); right fz (u[8]) = 0
    k = int(round(0.3 / cfg.dt))
    np.testing.assert_allclose(uref[k, 2], mg, rtol=1e-6); assert uref[k, 8] == 0.0
    # node at t=0.75 is STANCE -> split mg/2 each
    k2 = int(round(0.75 / cfg.dt))
    np.testing.assert_allclose([uref[k2, 2], uref[k2, 8]], [mg / 2, mg / 2], rtol=1e-6)


def test_arm_swing_counter_phase_and_vx_gated():
    x0 = model.nominal_state()
    xr_walk, _ = build_reference(x0, np.array([0.3, 0.0, H, 0.0]), SLOW_WALK, 0.0, NT, cfg, model)
    xr_stand, _ = build_reference(x0, np.array([0.0, 0.0, H, 0.0]), SLOW_WALK, 0.0, NT, cfg, model)
    # joint posture index: L_Shoulder_Pitch=6+0, R_Shoulder_Pitch=6+7 (state q_joints start at 6)
    base = model.nominal_state()
    dL = xr_walk[3, 6 + 0] - base[6 + 0]; dR = xr_walk[3, 6 + 7] - base[6 + 7]
    assert dL * dR < 0 or (dL == 0 and dR == 0)            # counter-phase (or zero at a node where gcf=0)
    assert np.allclose(xr_stand[:, 6:33], base[6:33])      # vx=0 -> no arm swing
