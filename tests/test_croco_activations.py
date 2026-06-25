# tests/test_croco_activations.py
import numpy as np, crocoddyl
from t1_nmpc.wb.croco_activations import RelaxedBarrier

def test_relaxed_barrier_penalizes_violation_with_correct_sign():
    rb = RelaxedBarrier(1, mu=0.2, delta=5.0)
    d = rb.createData()
    # h large & positive (deep inside feasible): small penalty
    rb.calc(d, np.array([100.0])); v_ok = d.a_value
    # h small/negative (violation): much larger penalty, and gradient pushes h UP (negative Ar)
    rb.calc(d, np.array([-1.0])); v_bad = d.a_value
    assert v_bad > v_ok
    rb.calcDiff(d, np.array([-1.0]))
    assert float(np.asarray(d.Ar).ravel()[0]) < 0.0          # d(penalty)/dh < 0 -> increasing h reduces penalty
    assert float(np.asarray(d.Arr).ravel()[0]) > 0.0         # convex

def test_relaxed_barrier_multidim():
    rb = RelaxedBarrier(3, 0.1, 0.03); d = rb.createData()
    rb.calc(d, np.array([1.0, 0.5, -0.1])); assert np.isfinite(d.a_value)
    rb.calcDiff(d, np.array([1.0, 0.5, -0.1]))
    assert np.asarray(d.Ar).shape[0] == 3 and np.asarray(d.Arr).shape == (3, 3)
