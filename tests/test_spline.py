import casadi as ca
from t1_nmpc.wb.spline import get_spline_vel_z


def _vz(phase):
    p = ca.MX.sym("p")
    e = get_spline_vel_z(p, swing_period=0.6, h_max=0.08, v_liftoff=0.05, v_touchdown=-0.05)
    f = ca.Function("vz", [p], [e])
    return float(f(phase))


def test_spline_shape():
    # liftoff rising, ~0 at apex, descending at touchdown
    assert _vz(0.0) > 0.0
    assert abs(_vz(0.5)) < 1e-6
    assert _vz(1.0) < 0.0


def test_spline_symbolic():
    p = ca.MX.sym("p")
    e = get_spline_vel_z(p, 0.6, 0.08, 0.05, -0.05)
    f = ca.Function("f", [p], [e])
    assert float(f(0.0)) > 0 and float(f(1.0)) < 0
