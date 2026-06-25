"""Honest deployment-timing measurement: the threaded loop in MuJoCo with the MPC on a pinned
isolated core and the control/MRT loop real-time-paced at 500Hz. Reports median/p95 acados time_tot
(the deployable C-solve cost), the async MPC rate, and whether the stand HOLDS at the real solve
speed. The async+MRT architecture can hold a stand below a 60Hz solve, so `held` — not the 16.7ms
solve budget — is the real deployment verdict for M0."""
import json
import os
os.chdir("/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc")

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.ocp_wb import make_ocp, build_solver
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.runtime.control_loop import run_loop


def main(duration_s: float = 5.0, control_hz: float = 500.0, mpc_cores=(2, 3, 4, 5), ctrl_core: int = 1):
    # MPC thread pinned to a POOL so its OMP workers (OMP_NUM_THREADS env) spread; control/MRT on its
    # own core. Set OMP_NUM_THREADS to match len(mpc_cores), and ACADOS_WB_CODEGEN_DIR to an OMP build.
    cfg = make_wb_config(); model = WBModel(cfg)
    mpc = WholeBodyMPC(cfg, model, solver=build_solver(make_ocp(cfg)[0]))
    out = run_loop(MujocoTransport(cfg), mpc, duration_s=duration_s, control_hz=control_hz,
                   cores=(list(mpc_cores), ctrl_core))
    out["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    out["solve_60hz_capable"] = bool(out["median_tot_ms"] <= 16.7)   # raw-solve budget (synchronous req)
    out["baseline_singlethread_tot_ms"] = 22.8
    out["deployable_stand"] = bool(out["held"])                      # async+MRT holds even when solve>16.7ms
    print("DEPLOY_MEASURE=" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
