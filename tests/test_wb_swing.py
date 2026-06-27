# tests/test_wb_swing.py
import numpy as np
from t1_nmpc.wb.gait import SLOW_WALK, SWING_HEIGHT

# slow_walk: right foot (side=1) swings during LF=[0.0, 0.65]; left foot (side=0) during RF=[0.85, 1.5].
G = SLOW_WALK


def test_stance_foot_is_flat_and_impact_one():
    assert G.swing_z(0.3, side=0) == (0.0, 0.0, 0.0)     # left in stance during LF
    assert G.impact_proximity(0.3, side=0) == 1.0


def test_swing_endpoints_right_foot():
    z0, zd0, _ = G.swing_z(0.0 + 1e-6, side=1)           # liftoff
    z1, zd1, _ = G.swing_z(0.65 - 1e-6, side=1)          # touchdown
    assert abs(z0 - 0.0) < 2e-3 and zd0 > 0              # starts at ground, rising
    assert abs(z1 - (-0.001)) < 3e-3 and zd1 < 0         # ends slightly below ground, descending


def test_swing_apex_height():
    zmid, zdmid, _ = G.swing_z((0.0 + 0.65) / 2.0, side=1)
    assert abs(zmid - SWING_HEIGHT) < 0.01               # ~0.08 at mid (full-length swing, scaling=1)
    assert abs(zdmid) < 1e-6                             # zero vertical velocity at apex


def test_impact_proximity_u_shape():
    assert G.impact_proximity(0.02, side=1) > 0.5        # near liftoff -> ~1
    assert G.impact_proximity(0.325, side=1) < 0.05      # mid-swing -> ~0.005
    assert G.impact_proximity(0.63, side=1) > 0.5        # near touchdown -> ~1
