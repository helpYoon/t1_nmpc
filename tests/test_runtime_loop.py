# tests/test_runtime_loop.py
"""Smoke test for the refined run_loop (control_loop.py) with the CrocoMPC backend.

Verifies:
  - run_loop returns with n_fail == 0 (MPC solver never reports failure)
  - median_tot_ms > 0.0  (mpc.last_solve_s is read and populated — Task-5 refine)
  - n_solves > 0         (MPC thread ran at least one solve)

Short duration (0.5 s) keeps CI fast.  The longer stand-quality gate lives in
test_wb_stand_croco.py; this test is distinct: it specifically exercises the
control_loop's last_solve_s read-path and status-flag handling.
"""
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.wb.croco_mpc import CrocoMPC
from t1_nmpc.runtime.control_loop import run_loop


def test_run_loop_croco_smoke():
    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=60.0)
    mpc = CrocoMPC(cfg, wb)
    result = run_loop(transport, mpc, duration_s=0.5, control_hz=60.0)

    assert result["n_fail"] == 0, f"MPC solver reported failures: {result['n_fail']}"
    assert result["median_tot_ms"] > 0.0, "median_tot_ms should be > 0 (last_solve_s not read?)"
    assert result["n_solves"] > 0, "MPC thread produced no solves"
