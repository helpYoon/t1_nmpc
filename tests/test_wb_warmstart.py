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
        np.testing.assert_allclose(xg[j], x[j + 1], atol=1e-10)
    np.testing.assert_allclose(xg[N], x[N], atol=1e-10)
    for j in range(N - 1):
        np.testing.assert_allclose(ug[j], u[j + 1], atol=1e-10)
    np.testing.assert_allclose(ug[N - 1], u[N - 1], atol=1e-10)


def test_fractional_shift_interpolates():
    # half a node elapsed (the real-loop regime) -> midpoint of consecutive nodes
    x, u = _distinct()
    xg, ug = shift_warmstart(x, u, 0.0, 0.5 * dt, cfg)
    for j in range(N):
        np.testing.assert_allclose(xg[j], 0.5 * (x[j] + x[j + 1]))
    np.testing.assert_allclose(xg[N], x[N], atol=1e-10)


def test_warmstart_identity_on_same_grid():
    cfg2 = make_wb_config()
    nt = np.arange(cfg2.N + 1) * cfg2.dt
    x_prev = np.cumsum(np.ones((cfg2.N + 1, cfg2.nx)), axis=0)
    u_prev = np.cumsum(np.ones((cfg2.N, cfg2.nu)), axis=0)
    xg, ug = shift_warmstart(x_prev, u_prev, nt, nt, cfg2)   # identical grids -> identity
    np.testing.assert_allclose(xg, x_prev, atol=1e-9)
    np.testing.assert_allclose(ug, u_prev, atol=1e-9)


def test_warmstart_interpolates_onto_nonuniform_grid():
    cfg2 = make_wb_config()
    nt_prev = np.arange(cfg2.N + 1) * cfg2.dt
    # a non-uniform target grid spanning the same horizon
    nt_now = np.sort(np.concatenate(([0.0], np.cumsum(np.random.RandomState(0).uniform(
        0.5, 1.5, cfg2.N)))))
    nt_now = nt_now / nt_now[-1] * (cfg2.N * cfg2.dt)
    x_prev = (np.linspace(0, 1, cfg2.N + 1)[:, None] * np.ones((1, cfg2.nx)))  # linear in time
    u_prev = (np.linspace(0, 1, cfg2.N)[:, None] * np.ones((1, cfg2.nu)))
    xg, ug = shift_warmstart(x_prev, u_prev, nt_prev, nt_now, cfg2)
    assert xg.shape == (cfg2.N + 1, cfg2.nx) and ug.shape == (cfg2.N, cfg2.nu)
    assert np.all(np.isfinite(xg)) and np.all(np.isfinite(ug))
    # linear field -> interpolation matches the analytic line at the new node times
    expected_x = (nt_now / (cfg2.N * cfg2.dt))[:, None] * np.ones((1, cfg2.nx))
    np.testing.assert_allclose(xg, expected_x, atol=1e-9)
