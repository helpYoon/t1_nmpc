# tests/test_croco_activations.py
"""RelaxedBarrier is BOUNDS-AWARE: it takes the crocoddyl residual bounds (lb, ub)
and applies the OCS2 relaxed-barrier shape to the feasibility margin h>=0, where
  ub finite -> h = ub - r   (feasible r <= ub)
  lb finite -> h = r - lb   (feasible r >= lb)
This matches how crocoddyl cone residuals (FrictionCone/CoPSupport/WrenchCone) encode
each inequality as a one-sided bound on r, NOT as h=r>=0. Applying the barrier to the
raw residual r (the old bug) penalised the FEASIBLE region for ub=0 rows."""
import numpy as np
from t1_nmpc.wb.croco_activations import RelaxedBarrier

INF = np.inf


def _num_grad(rb, r, eps=1e-6):
    g = np.zeros_like(r)
    d = rb.createData()
    for i in range(len(r)):
        rp = r.copy(); rp[i] += eps; rb.calc(d, rp); vp = d.a_value
        rm = r.copy(); rm[i] -= eps; rb.calc(d, rm); vm = d.a_value
        g[i] = (vp - vm) / (2 * eps)
    return g


def test_feasible_cone_residual_has_small_gradient():
    """REGRESSION: a friction-cone row is feasible at r=-67 (ub=0 => h=67 deep inside).
    The old (sign-flipped) barrier produced a huge penalty/gradient here; the correct
    bounds-aware barrier produces a tiny one."""
    rb = RelaxedBarrier(np.array([-INF]), np.array([0.0]), mu=0.2, delta=5.0)
    d = rb.createData()
    rb.calc(d, np.array([-67.0]))
    rb.calcDiff(d, np.array([-67.0]))
    assert d.a_value < 1.0                                   # tiny penalty, not ~1e3
    assert abs(float(np.asarray(d.Ar).ravel()[0])) < 0.1     # tiny gradient, not ~1e3


def test_ub_finite_row_pushes_r_away_from_upper_bound():
    """ub=0 row near the boundary (r=-0.01 => h=0.01<delta): cost rises as r->ub,
    so gradient dv/dr is POSITIVE (raising r toward ub is penalised)."""
    rb = RelaxedBarrier(np.array([-INF]), np.array([0.0]), mu=0.1, delta=0.03)
    d = rb.createData()
    rb.calcDiff(d, np.array([-0.01]))
    assert float(np.asarray(d.Ar).ravel()[0]) > 0.0          # pushes r down, away from ub=0
    assert float(np.asarray(d.Arr).ravel()[0]) > 0.0         # convex


def test_lb_finite_row_pushes_r_away_from_lower_bound():
    """lb=0 row (friction normal-force row) near the boundary (r=+0.01 => h=0.01<delta):
    gradient is NEGATIVE (raising r away from lb reduces penalty)."""
    rb = RelaxedBarrier(np.array([0.0]), np.array([INF]), mu=0.2, delta=5.0)
    d = rb.createData()
    rb.calcDiff(d, np.array([0.01]))
    assert float(np.asarray(d.Ar).ravel()[0]) < 0.0          # pushes r up, away from lb=0


def test_numerical_gradient_and_hessian():
    """Finite-difference check of Ar (vs value) and Arr (vs Ar) for a mix of
    ub-finite and lb-finite rows at feasible interior points."""
    lb = np.array([-INF, -INF, 0.0])
    ub = np.array([0.0, 0.0, INF])
    rb = RelaxedBarrier(lb, ub, mu=0.15, delta=0.5)
    d = rb.createData()
    r = np.array([-3.0, -0.7, 4.0])                          # all feasible interior
    rb.calcDiff(d, r)
    Ar = np.asarray(d.Ar).ravel().copy()
    assert np.allclose(Ar, _num_grad(rb, r), atol=1e-4)
    # Hessian is diagonal; check each diagonal entry against d(Ar_i)/dr_i
    Arr = np.asarray(d.Arr).copy()
    eps = 1e-6
    for i in range(3):
        rp = r.copy(); rp[i] += eps; rb.calcDiff(d, rp); arp = np.asarray(d.Ar).ravel()[i]
        rm = r.copy(); rm[i] -= eps; rb.calcDiff(d, rm); arm = np.asarray(d.Ar).ravel()[i]
        assert abs(Arr[i, i] - (arp - arm) / (2 * eps)) < 1e-3


def test_c2_continuity_at_margin_delta():
    """Value continuous across the margin h=delta (ub=0 row: h=delta => r=-delta)."""
    rb = RelaxedBarrier(np.array([-INF]), np.array([0.0]), mu=0.2, delta=5.0)
    d = rb.createData()
    rb.calc(d, np.array([-(5.0 - 1e-7)])); v_h_above = d.a_value   # h = 5 + 1e-7
    rb.calc(d, np.array([-(5.0 + 1e-7)])); v_h_below = d.a_value   # h = 5 - 1e-7
    assert abs(v_h_above - v_h_below) < 1e-4


def test_penalty_monotone_in_margin():
    """Penalty rises as the margin h shrinks (constraint approached). ub=0 row: h=-r."""
    rb = RelaxedBarrier(np.array([-INF]), np.array([0.0]), mu=0.2, delta=5.0)
    d = rb.createData()
    rb.calc(d, np.array([-10.0])); v_h10 = d.a_value
    rb.calc(d, np.array([-1.0]));  v_h1 = d.a_value
    rb.calc(d, np.array([-0.1]));  v_h01 = d.a_value
    assert v_h01 > v_h1 > v_h10
