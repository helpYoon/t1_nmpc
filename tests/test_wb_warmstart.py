# tests/test_wb_warmstart.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.mpc_wb import shift_warmstart

cfg = make_wb_config()
N, nx, nu, dt = cfg.N, cfg.nx, cfg.nu, cfg.dt


def _distinct():
    x = np.arange((N + 1) * nx, dtype=float).reshape(N + 1, nx)
    u = np.arange(N * nu, dtype=float).reshape(N, nu)
    return x, u


def test_integer_shift_is_shift_by_one():
    # one node elapsed = the oracle case the probe validated -> x_guess[j] == x_prev[j+1], last held
    x, u = _distinct()
    xg, ug = shift_warmstart(x, u, 0.0, dt, cfg)
    for j in range(N):
        np.testing.assert_array_equal(xg[j], x[j + 1])
    np.testing.assert_array_equal(xg[N], x[N])
    for j in range(N - 1):
        np.testing.assert_array_equal(ug[j], u[j + 1])
    np.testing.assert_array_equal(ug[N - 1], u[N - 1])


def test_fractional_shift_interpolates():
    # half a node elapsed (the real-loop regime) -> midpoint of consecutive nodes
    x, u = _distinct()
    xg, ug = shift_warmstart(x, u, 0.0, 0.5 * dt, cfg)
    for j in range(N):
        np.testing.assert_allclose(xg[j], 0.5 * (x[j] + x[j + 1]))
    np.testing.assert_array_equal(xg[N], x[N])
