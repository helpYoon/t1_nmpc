from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.ocp_wb import make_ocp, build_solver
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.runtime.control_loop import run_loop


def test_threaded_loop_runs_and_reports():
    """The threaded loop runs closed-loop in MuJoCo and reports timing. Whether the stand HOLDS at
    the real async solve rate is the MEASURED RESULT (out["held"]) — not asserted here, since
    single-RTI needs ~60Hz and the real solve may be slower. measure_deploy.py reports the verdict."""
    cfg = make_wb_config(); model = WBModel(cfg)
    mpc = WholeBodyMPC(cfg, model, solver=build_solver(make_ocp(cfg)[0]))
    out = run_loop(MujocoTransport(cfg), mpc, duration_s=1.0, control_hz=500.0, cores=(0, 1))
    assert out["n_fail"] == 0                      # solver stays healthy (status in {0,2})
    assert out["n_solves"] > 0 and out["median_tot_ms"] > 0.0
    assert out["effective_mpc_hz"] > 0
    assert "held" in out                           # the stand-holding verdict is reported, not asserted
