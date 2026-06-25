"""LIVE MuJoCo viewer for the closed-loop WB forward walk (CPU MuJoCo + CPU acados MPC).

No GPU<->compute contention here (the acados solve is CPU; the viewer renders on the GPU in its own
thread), so launch_passive runs fine. It advances slower than real-time because each ~50 ms MPC solve
dominates -- good for watching the fall. Real-time-paced between solves so the 2 kHz physics bursts
aren't a blur. Same control law as sim/wb_walk_gate.

Run in a terminal WITH a display:
  cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
  PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n t1mpc python -u sim/wb_walk_view.py [vx] [duration_s] [speed]

  speed<1 = slow-mo (e.g. 0.5). Close the window to quit.
"""
from __future__ import annotations

import time

import numpy as np
import mujoco
import mujoco.viewer

from t1_nmpc.config import make_config
from t1_nmpc.model import load_model, T1_URDF_PATH
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime
from sim.wb_stand_gate import wb_state_estimate, _wb_reset, _sample_plan, _HEAD_KP, _HEAD_KD
from sim._sim_util import tilt_from_quat_wxyz


def run_view(vx: float = 0.3, duration_s: float = 10.0, speed: float = 1.0):
    sample_ahead_s = 0.005

    ccfg = make_config(mpc_hz=60.0)
    rt = MujocoRuntime(ccfg, load_model(T1_URDF_PATH, ccfg))
    wb_cfg = make_wb_config(); wb_model = WBModel(wb_cfg)
    mpc = WholeBodyMPC(wb_cfg, wb_model)          # loads the cached solver (rebuilds if the hash changed)

    _wb_reset(rt, wb_cfg)
    mpc.set_command([vx, 0.0, 0.0, 0.0, 0.0])
    x0 = wb_state_estimate(rt); mpc.reset(x0)
    res = mpc.step(x0, rt.t)
    x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t

    n_phys = int(round(duration_s * ccfg.physics_hz))
    cdecim, mdecim = rt.control_decim, rt.mpc_decim
    kp, kd = wb_cfg.kp, wb_cfg.kd
    dt_phys = 1.0 / ccfg.physics_hz
    sync_every = max(1, int(round(ccfg.physics_hz / 60.0)))    # ~60 fps render

    print(f"vx={vx}  duration={duration_s}s  speed={speed}x  (solve ~50ms/tick -> runs sub-real-time). "
          f"Close window to quit.", flush=True)
    with mujoco.viewer.launch_passive(rt.mj_model, rt.mj_data) as viewer:
        t_wall0 = time.perf_counter()
        for k in range(n_phys):
            if not viewer.is_running():
                break
            if k % mdecim == 0 and k > 0:
                x_meas = wb_state_estimate(rt)
                res = mpc.step(x_meas, rt.t)
                x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t
            if k % cdecim == 0:
                q_pin, v_pin = rt._pin_q_v()
                q_meas = q_pin[8:35]; qd_meas = v_pin[8:35]
                x_star, u_star = _sample_plan(x_plan, u_plan, rt.t + sample_ahead_s - t_solve, wb_cfg.dt, wb_cfg.N)
                tau_ff = wb_model.joint_torque(x_star, u_star)
                tau_wb = kp * (x_star[6:33] - q_meas) + kd * (x_star[39:66] - qd_meas) + tau_ff
                tau29 = np.zeros(29); tau29[2:29] = tau_wb
                tau29[0:2] = _HEAD_KP * (0.0 - q_pin[6:8]) - _HEAD_KD * v_pin[6:8]
                rt._apply_torque(tau29)
            rt.step_physics()
            if k % sync_every == 0:
                viewer.sync()
                target = t_wall0 + ((k + 1) * dt_phys) / max(speed, 1e-6)   # real-time pace between solves
                dt_sleep = target - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
        d = rt.mj_data
        print(f"done: net_bx={float(d.qpos[0]):.3f}m  final_z={float(d.qpos[2]):.3f}  "
              f"tilt={tilt_from_quat_wxyz(d.qpos[3:7]):.3f}rad", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live MuJoCo viewer for the WB walk (vx=0 -> stand).")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="<1 = slow-mo")
    a = ap.parse_args()
    run_view(a.vx, a.duration, a.speed)


if __name__ == "__main__":
    main()
