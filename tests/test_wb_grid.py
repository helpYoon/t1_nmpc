import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.gait_wb import SLOW_WALK, STANCE_GAIT
from t1_nmpc.wb.grid_wb import event_aligned_grid

cfg = make_wb_config()
T = cfg.N * cfg.dt

def _check_basic(nt, t0):
    assert nt.shape == (cfg.N + 1,)
    assert np.all(np.diff(nt) > 0)                       # strictly increasing
    assert abs(nt[0] - t0) < 1e-12 and abs(nt[-1] - (t0 + T)) < 1e-9
    assert abs(np.diff(nt).sum() - T) < 1e-9

def test_uniform_when_no_switch():
    # STANCE_GAIT never switches -> exact uniform grid
    nt = event_aligned_grid(0.0, STANCE_GAIT, cfg)
    _check_basic(nt, 0.0)
    np.testing.assert_allclose(nt, np.arange(cfg.N + 1) * cfg.dt, atol=1e-12)

def test_switches_land_on_nodes():
    t0 = 0.2
    nt = event_aligned_grid(t0, SLOW_WALK, cfg)
    _check_basic(nt, t0)
    # SLOW_WALK's real contact-mode-change switches inside [0.2, 0.2+T]: 0.65 (LF->STANCE), 0.85 (STANCE->RF)
    for s in (0.65, 0.85):
        assert np.min(np.abs(nt - s)) < 1e-9, f"switch {s} not on a node"

def test_dt_stays_near_nominal():
    nt = event_aligned_grid(0.2, SLOW_WALK, cfg)
    d = np.diff(nt)
    assert d.min() > 0.4 * cfg.dt and d.max() < 1.8 * cfg.dt  # round-per-segment keeps dt ~ nominal

def test_switch_near_t0_dropped_no_tiny_interval():
    # a switch within 0.5*dt of t0 is dropped (not forced onto a node) so no sub-0.5*dt interval forms
    s0 = SLOW_WALK.switch_times_in(0.0, 5.0)[0]
    nt = event_aligned_grid(s0 - 0.005, SLOW_WALK, cfg)        # 0.005 < 0.5*dt
    _check_basic(nt, s0 - 0.005)
    assert np.diff(nt).min() > 0.4 * cfg.dt                    # no tiny interval
    assert np.min(np.abs(nt - s0)) > 1e-6                      # near-t0 switch NOT forced onto a node

def test_interior_switch_lands_on_node():
    # a real mode-change switch comfortably inside the horizon DOES land exactly on a node
    s0 = SLOW_WALK.switch_times_in(0.0, 5.0)[0]
    t0 = s0 - 0.3
    nt = event_aligned_grid(t0, SLOW_WALK, cfg)
    _check_basic(nt, t0)
    assert np.min(np.abs(nt - s0)) < 1e-9
