"""M0 GATE — whole-body closed-loop STAND in MuJoCo.

Reuses MujocoRuntime for the sim model + helpers (_pin_q_v euler-zyx, _apply_torque, step_physics)
but drives the WHOLE-BODY MPC (68-d state) via a small adapter. Decoupled rates (physics 2000 / PD
500 / MPC 40); the WB solve (~52ms) makes this SUB-REAL-TIME but the sim is decoupled, so it still
validates the FORMULATION: does the whole-body MPC hold a stand where the centroidal port couldn't?

Run FOREGROUND:
  ... conda run -n t1mpc python sim/wb_stand_gate.py
"""
from __future__ import annotations

import json

import numpy as np
import mujoco

from t1_nmpc.config import make_config
from t1_nmpc.model import load_model, T1_URDF_PATH
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from t1_nmpc.wb.execution_wb import to_joint_command_wb, pd_torque
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0
from sim._sim_util import tilt_from_quat_wxyz, upright_ok

_HEAD_KP, _HEAD_KD = 20.0, 0.5   # light open-loop hold for the 2 head joints (not in the MPC)


def wb_state_estimate(rt: MujocoRuntime) -> np.ndarray:
    """68-d WB state from sim: [q_base(6), q_joints(27), v_base(6), v_joints(27), s, v_s].
    q_pin/v_pin are euler-zyx (35,); the 27 MPC joints = the 29 minus the 2 head joints (idx 6:8)."""
    q_pin, v_pin = rt._pin_q_v()
    x = np.zeros(68, dtype=np.float64)
    x[0:6] = q_pin[0:6]
    x[6:33] = q_pin[8:35]
    x[33:39] = v_pin[0:6]
    x[39:66] = v_pin[8:35]
    return x


def _wb_reset(rt: MujocoRuntime, wb_cfg) -> None:
    """Spawn at the WB nominal posture (head=0 + 27 MPC joints) above the floor and PD-settle the
    feet onto it — like MujocoRuntime.reset_to_nominal but for the WB crouch."""
    q0 = MJ_JOINT_QPOS0
    njp29 = np.zeros(29); njp29[2:29] = np.asarray(wb_cfg.nominal_joint_pos, dtype=np.float64)
    kp = np.asarray(rt.cfg.kp, dtype=np.float64); kd = np.asarray(rt.cfg.kd, dtype=np.float64)
    d = rt.mj_data
    d.qpos[:] = 0.0; d.qvel[:] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[q0:q0 + 29] = njp29
    d.qpos[2] = wb_cfg.nominal_base_height + 0.10
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    for _ in range(int(round(0.6 * rt.cfg.physics_hz))):
        q = np.array(d.qpos[q0:q0 + 29]); qd = np.array(d.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
        rt._apply_torque(kp * (njp29 - q) - kd * qd)
        rt.step_physics()
    d.qvel[:] = 0.0
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    rt.t = 0.0


def _sample_plan(x_plan, u_plan, dt_off, node_dt, N):
    """Linear-interp the planned (x,u) trajectory at dt_off seconds after the solve — the advancing
    sample of MRT_BASE::evaluatePolicy. x_plan (N+1,68), u_plan (N,40)."""
    s = max(0.0, dt_off) / node_dt
    lo = min(int(np.floor(s)), N - 1)
    a = min(max(s - lo, 0.0), 1.0)
    x_star = (1.0 - a) * x_plan[lo] + a * x_plan[lo + 1]
    u_star = (1.0 - a) * u_plan[lo] + a * u_plan[min(lo + 1, N - 1)]
    return x_star, u_star


def run_wb_stand(duration_s: float = 5.0, cmd=None, sample_ahead_s: float = 0.005,
                 id_feedforward: bool = True, mpc=None, kick_s=None, kick=None) -> dict:
    cmd = np.zeros(5) if cmd is None else np.asarray(cmd, dtype=np.float64)
    # WB MPC runs at 60 Hz (t1_controller's mpcDesiredFrequency) — single-RTI is rate-dependent:
    # the MPC-only rollout is STABLE at 60 Hz but DIVERGES at 40 Hz (1 SQP iter can't track the
    # larger per-tick drift). The centroidal default (40 Hz) is what collapsed the first M0 attempt.
    ccfg = make_config(mpc_hz=60.0)
    rt = MujocoRuntime(ccfg, load_model(T1_URDF_PATH, ccfg))

    if mpc is None:
        wb_cfg = make_wb_config()
        wb_model = WBModel(wb_cfg)
        mpc = WholeBodyMPC(wb_cfg, wb_model)        # builds/loads the cached -O2 solver
    else:
        wb_cfg, wb_model = mpc.cfg, mpc.model       # reuse an injected MPC (e.g. cached-solver)

    _wb_reset(rt, wb_cfg)
    mpc.set_command(cmd)
    x0 = wb_state_estimate(rt)
    mpc.reset(x0)
    res = mpc.step(x0, rt.t)
    x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t

    n_phys = int(round(duration_s * ccfg.physics_hz))
    cdecim, mdecim = rt.control_decim, rt.mpc_decim
    n_fail = 0
    max_tau = 0.0
    tilts, base_z = [], []
    solve_ms = []
    solve_tot_ms = []          # acados-internal time_tot = the C-solve cost (deployable) vs the python wall solve_ms
    kp, kd = wb_cfg.kp, wb_cfg.kd

    for k in range(n_phys):
        if k % mdecim == 0 and k > 0:
            x_meas = wb_state_estimate(rt)
            res = mpc.step(x_meas, rt.t)
            if res.status not in (0, 2):   # 2 = single-RTI max_iter (normal); 1/3/4 = real solver failure
                n_fail += 1
            x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t
            solve_ms.append(res.solve_time * 1e3)
            try:
                solve_tot_ms.append(float(mpc.solver.get_stats("time_tot")) * 1e3)
            except Exception:
                pass
        if k % cdecim == 0:
            # faithful MRT execution: resample the plan at the ADVANCING t+5ms every control tick;
            # tau_ff = inverse dynamics of the PLANNED (x*,u*) — pure feedforward, the only measured-
            # state coupling is the kp/kd PD (matches MpcMrtJointController; fixes the held-ref limit
            # cycle + the measured-state feedforward resonance).
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
            max_tau = max(max_tau, float(np.max(np.abs(tau_wb))))
        if kick_s is not None and k == int(round(kick_s * ccfg.physics_hz)):
            rt.mj_data.qvel[0:6] = rt.mj_data.qvel[0:6] + np.asarray(kick, dtype=np.float64)  # base-vel perturbation
        rt.step_physics()
        d = rt.mj_data
        tilts.append(tilt_from_quat_wxyz(d.qpos[3:7]))
        base_z.append(float(d.qpos[2]))

    peak_tilt = float(np.max(tilts))
    min_z = float(np.min(base_z))
    final_z = float(base_z[-1])
    final_tilt = float(tilts[-1])
    out = {
        "duration_s": duration_s,
        "peak_tilt_rad": round(peak_tilt, 4),
        "final_tilt_rad": round(final_tilt, 4),
        "min_base_z": round(min_z, 4),
        "final_base_z": round(final_z, 4),
        "n_solver_failures": n_fail,
        "max_abs_tau": round(max_tau, 2),
        "median_solve_ms": round(float(np.median(solve_ms)), 1) if solve_ms else None,
        "median_acados_tot_ms": round(float(np.median(solve_tot_ms)), 2) if solve_tot_ms else None,
        "upright_end": bool(upright_ok(final_z, final_tilt, wb_cfg.nominal_base_height)),
        "PASS": bool(peak_tilt < 0.20 and final_z > 0.85 * wb_cfg.nominal_base_height and n_fail == 0),
    }
    return out


if __name__ == "__main__":
    print("WB_M0=" + json.dumps(run_wb_stand()))
