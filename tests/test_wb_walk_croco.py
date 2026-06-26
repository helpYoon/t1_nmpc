# tests/test_wb_walk_croco.py
"""M1 acceptance gate: single-RTI CrocoMPC(gait=SLOW_WALK) closed-loop forward walk.

Run with:
    env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_wb_walk_croco.py -v -m slow

Marked `slow` — 12-second simulation, not suitable for fast CI.
"""
import pytest
from sim.wb_walk_croco import run_wb_walk_croco


@pytest.mark.slow
def test_m1_walk_forward_no_fall():
    """Gate: single-RTI walk must not fall and must advance > 1 m in 12 s at vx=0.3 m/s."""
    m = run_wb_walk_croco(duration_s=12.0, vx=0.3)
    print(f"\nWALK_GATE={m}")
    assert m["n_solver_failures"] == 0, (
        f"solver failed {m['n_solver_failures']} times — single-RTI diverged")
    assert m["peak_tilt_rad"] < 0.2, (
        f"robot fell (peak_tilt={m['peak_tilt_rad']:.4f} rad >= 0.2)")
    assert m["final_base_z"] > 0.85 * 0.6734, (
        f"base z collapsed (final_z={m['final_base_z']:.4f} m < {0.85*0.6734:.4f})")
    assert m["com_advance_m"] > 1.0, (
        f"did not advance (com_advance={m['com_advance_m']:.4f} m < 1.0)")
