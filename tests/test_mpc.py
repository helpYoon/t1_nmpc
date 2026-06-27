"""WholeBodyMPC (gait-aware) regression: stand holds, walk tick solves, base-vx not attenuated.

The driver builds ONE WalkOCP solver_fn (8-arg) and feeds per-tick gait schedules,
base-velocity command and footstep targets each step. Default gait is StandGait so the
M0 stand path is preserved.
"""
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait, WalkGait
from t1_nmpc.wb.mpc import WholeBodyMPC


def _max_base_vx(res):
    """Max planned base-local forward (x) velocity over the horizon."""
    return max(float(v[0]) for v in res.planned["v_sol"])


def test_stand_still_holds():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=StandGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    r = mpc.step(x0, t=0.0)
    assert r.constr_viol < 1e-2, f"stand CV {r.constr_viol:.2e}"
    assert r.command.tau_ff.shape == (29,)
    assert r.command.q_des.shape == (29,) and r.command.qd_des.shape == (29,)
    fz = r.forces0[2::3].sum()
    assert 0.9 <= fz / (rm.mass * 9.81) <= 1.1, fz / (rm.mass * 9.81)
    # base-vx command is 0 at stand -> planned forward velocity stays ~0
    assert abs(_max_base_vx(r)) < 0.05, _max_base_vx(r)


def test_walk_tick_solves():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    r = mpc.step(x0, t=0.0)
    assert r.constr_viol < 1e-2, f"walk CV {r.constr_viol:.2e}"
    assert r.command.tau_ff.shape == (29,)


def test_warm_started_second_tick_solves():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    mpc.step(x0, t=0.0)
    r2 = mpc.step(x0, t=cfg.dt_min)          # advance the schedule one tick
    assert r2.constr_viol < 1e-2, f"warm tick CV {r2.constr_viol:.2e}"


def test_forward_velocity_not_attenuated():
    """With a forward command, the planned base velocity must reach a meaningful
    fraction of base_vx_des. Before the fix the optimum was ~0.09*base_vx (the 2000
    state-tracking weight fought the 200 base-vx term); after the fix the dedicated
    base-vx term governs forward velocity."""
    base_vx = 0.3
    cfg = make_config(base_vx_des=base_vx); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    r = mpc.step(x0, t=0.0)
    assert r.constr_viol < 1e-2, f"walk CV {r.constr_viol:.2e}"
    frac = _max_base_vx(r) / base_vx
    assert frac >= 0.7, f"forward velocity attenuated: {frac:.2f} of base_vx"
