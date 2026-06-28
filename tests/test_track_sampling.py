import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model
from t1_nmpc.wb.reference import MotionPlanReference

PLAN = "data/motion_plan.pkl"


def _ref(**kw):
    cfg = make_track_config(**kw)
    rm = load_model(cfg)
    return MotionPlanReference(PLAN, cfg, rm), cfg, rm


def test_sample_shapes_and_clamp():
    ref, cfg, rm = _ref()
    xr, hr, gg = ref.sample(0.0)
    assert xr.shape == (71, cfg.nodes + 1)
    assert hr.shape == (6, cfg.nodes + 1)
    assert gg.shape == (2, cfg.nodes + 1)
    # far past the end -> clamps to the final frame (q part equals last frame q)
    xr_end, _, _ = ref.sample(ref.duration_phase * 10.0 * cfg.time_scale)
    assert np.allclose(xr_end[:36, 0], ref.q_frame[-1], atol=1e-6)


def test_velocity_scales_with_time_scale():
    """Doubling time_scale halves the reference velocities (same path, slower)."""
    ref2, cfg2, _ = _ref(time_scale=2.0)
    ref4, cfg4, _ = _ref(time_scale=4.0)
    # sample mid-motion at the SAME phase (t_wall = phase * time_scale)
    phase = 3.0
    x2, _, _ = ref2.sample(phase * 2.0, time_scale=2.0)
    x4, _, _ = ref4.sample(phase * 4.0, time_scale=4.0)
    v2 = x2[36:, 0]; v4 = x4[36:, 0]
    # positions identical (same phase), velocities halved
    assert np.allclose(x2[:36, 0], x4[:36, 0], atol=1e-5)
    assert np.linalg.norm(v4) < np.linalg.norm(v2)
    assert np.allclose(v4 * 2.0, v2, rtol=5e-2, atol=2e-3) or np.linalg.norm(v2) < 1e-6


def test_grasp_gate_fires_near_event():
    ref, cfg, rm = _ref(time_scale=1.0)
    te = ref.events[0][0]               # first left event (phase time)
    _, _, gg = ref.sample(te)           # t_wall = te (time_scale 1) -> node 0 at phase te
    assert gg[0, 0] == 1.0              # left gate hot at the event
    _, _, gg_far = ref.sample(0.0)
    assert gg_far[0, 0] == 0.0          # not hot at t=0
