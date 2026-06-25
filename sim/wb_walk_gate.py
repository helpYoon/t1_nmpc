"""M1 GATE — whole-body closed-loop FORWARD WALK in MuJoCo.

Reuses the proven wb_stand_gate harness (wb_state_estimate / _wb_reset / _sample_plan, MujocoRuntime,
60 Hz MPC, MRT t+5ms resample, kp/kd + tau_ff) but drives a forward velocity command so the SLOW_WALK
gait engages, and records the base x/y trajectory for the walking metrics. Single-thread/deterministic
(behavior gate; the deployed async loop is for timing, not behavior).

Run FOREGROUND:
  ... conda run -n t1mpc python sim/wb_walk_gate.py
"""
from __future__ import annotations

import json

import numpy as np

from t1_nmpc.config import make_config
from t1_nmpc.model import load_model, T1_URDF_PATH
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime
from sim.wb_stand_gate import wb_state_estimate, _wb_reset, _sample_plan, _HEAD_KP, _HEAD_KD
from sim._sim_util import tilt_from_quat_wxyz

SLOW_WALK_PERIOD = 1.7


def run_wb_walk(duration_s: float = 10.0, vx: float = 0.3, sample_ahead_s: float = 0.005,
                mpc=None) -> dict:
    cmd = np.array([vx, 0.0, 0.0, 0.0, 0.0])
    ccfg = make_config(mpc_hz=60.0)                  # 60 Hz MPC (single-RTI is rate-dependent)
    rt = MujocoRuntime(ccfg, load_model(T1_URDF_PATH, ccfg))

    if mpc is None:
        wb_cfg = make_wb_config(); wb_model = WBModel(wb_cfg)
        mpc = WholeBodyMPC(wb_cfg, wb_model)         # loads the cached walking solver (Task 8 build)
    else:
        wb_cfg, wb_model = mpc.cfg, mpc.model

    _wb_reset(rt, wb_cfg)
    mpc.set_command(cmd)                             # speed>1e-3 -> SLOW_WALK gait
    x0 = wb_state_estimate(rt)
    mpc.reset(x0)
    res = mpc.step(x0, rt.t)
    x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t

    n_phys = int(round(duration_s * ccfg.physics_hz))
    cdecim, mdecim = rt.control_decim, rt.mpc_decim
    n_fail = 0
    tilts, base_z, base_x, base_y, solve_tot = [], [], [], [], []
    kp, kd = wb_cfg.kp, wb_cfg.kd

    for k in range(n_phys):
        if k % mdecim == 0 and k > 0:
            x_meas = wb_state_estimate(rt)
            res = mpc.step(x_meas, rt.t)             # gait clock = rt.t -> SLOW_WALK advances
            if res.status not in (0, 2):
                n_fail += 1
            x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t
            try:
                solve_tot.append(float(mpc.solver.get_stats("time_tot")) * 1e3)
            except Exception:
                pass
        if k % cdecim == 0:
            q_pin, v_pin = rt._pin_q_v()
            q_meas = q_pin[8:35]; qd_meas = v_pin[8:35]
            x_star, u_star = _sample_plan(x_plan, u_plan, rt.t + sample_ahead_s - t_solve, wb_cfg.dt, wb_cfg.N)
            q_des, qd_des = x_star[6:33], x_star[39:66]
            tau_ff = wb_model.joint_torque(x_star, u_star)
            tau_wb = kp * (q_des - q_meas) + kd * (qd_des - qd_meas) + tau_ff
            tau29 = np.zeros(29)
            tau29[2:29] = tau_wb
            tau29[0:2] = _HEAD_KP * (0.0 - q_pin[6:8]) - _HEAD_KD * v_pin[6:8]
            rt._apply_torque(tau29)
        rt.step_physics()
        d = rt.mj_data
        tilts.append(tilt_from_quat_wxyz(d.qpos[3:7]))
        base_z.append(float(d.qpos[2])); base_x.append(float(d.qpos[0])); base_y.append(float(d.qpos[1]))

    peak_tilt = float(np.max(tilts)); final_z = float(base_z[-1])
    nominal = float(wb_cfg.nominal_base_height)
    mean_vx = (base_x[-1] - base_x[0]) / duration_s
    lateral_pkpk = float(max(base_y) - min(base_y))
    n_steps = int(duration_s / SLOW_WALK_PERIOD * 2) if n_fail == 0 else 0   # 2 steps/cycle
    passed = bool(peak_tilt < 0.2 and final_z > 0.85 * nominal and n_fail == 0
                  and mean_vx > 0.20 and lateral_pkpk < 0.10)
    return {
        "mean_vx": round(mean_vx, 3), "peak_tilt_rad": round(peak_tilt, 4),
        "lateral_pkpk_m": round(lateral_pkpk, 4), "n_steps": n_steps, "n_fail": n_fail,
        "final_base_z": round(final_z, 4),
        "median_acados_tot_ms": round(float(np.median(solve_tot)), 2) if solve_tot else None,
        "passed": passed,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WB forward-walk gate: headless PASS/metrics, or live viewer with --view.")
    ap.add_argument("--vx", type=float, default=0.3, help="forward velocity command (0 -> stand)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--view", action="store_true", help="open the live MuJoCo viewer instead of printing metrics")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (<1 = slow-mo)")
    a = ap.parse_args()
    if a.view:
        from sim.wb_walk_view import run_view      # live viewer (same control law); needs a display
        run_view(a.vx, a.duration, a.speed)
    else:
        print("WALK_GATE=" + json.dumps(run_wb_walk(duration_s=a.duration, vx=a.vx)))
