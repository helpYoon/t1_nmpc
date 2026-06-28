"""Closed-loop MuJoCo floor-pickup under the whole-body RNEA tracking MPC.

Mirrors sim/stand.py: physics @2000Hz, PD @500Hz, MPC @50Hz (ZOH). The phase clock is the loop
counter * physics_dt (NOT mj_data.time, which includes the reset settle). time_scale slows the
tracked motion. --realtime paces the loop to wall-clock and reports the real-time factor."""
from __future__ import annotations

import time
import numpy as np
import mujoco
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig, make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.track_mpc import TrackingMPC
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def _tilt_deg(qpos):
    R = pin.Quaternion(qpos[3], qpos[4], qpos[5], qpos[6]).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _tilt_from_quat_xyzw(qx, qy, qz, qw):
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _measured_grf_z(m, d):
    total, f6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        frame = d.contact[i].frame.reshape(3, 3)
        total += (frame.T @ f6[:3])[2]
    return abs(total)


def _hand_err_cm(rm, qpos_mj, mpc, t_wall):
    """L2 hand position error vs the (anchored) reference, cm, max over both hands."""
    from t1_nmpc.wb.state import mujoco_to_freeflyer
    x = mujoco_to_freeflyer(qpos_mj, np.zeros(rm.model.nv), rm.model)
    q = x[:rm.model.nq]
    m, d = rm.model, rm.model.createData()
    pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    _, hand_ref, _ = mpc.ref.sample(t_wall)
    lh, rh = rm.hand_frame_ids
    el = np.linalg.norm(d.oMf[lh].translation - hand_ref[0:3, 0])
    er = np.linalg.norm(d.oMf[rh].translation - hand_ref[3:6, 0])
    return 100.0 * max(el, er)


def run_pickup(cfg: MPCConfig, plan_path: str = "data/motion_plan.pkl",
               duration: float | None = None, realtime: bool = False) -> dict:
    rm = load_model(cfg)
    rt = MujocoRuntime(cfg, rm)
    rt.reset_to_nominal()
    mpc = TrackingMPC(cfg, rm, plan_path)
    mpc.reset(nominal_x(cfg, rm.model))
    mg = rm.mass * 9.81
    dur = mpc.duration_wall + 0.5 if duration is None else duration
    n_steps = int(round(dur * cfg.physics_hz))

    cmd = None
    solve_ms, fz_ratios, reltilts, hand_errs = [], [], [], []
    fell, completed = False, False
    FALL_RELTILT = 20.0     # the motion legitimately leans to ~70deg; a FALL is measured tilt diverging
                            # from the REFERENCE tilt by > this (validated: tracking stays within ~2deg).
    t_start = time.perf_counter()
    for k in range(n_steps):
        t_wall = k * rt.physics_dt
        if k % rt.mpc_decim == 0:
            x = rt.freeflyer_state(rm.model)
            res = mpc.step(x, t_wall)
            cmd = res.command
            solve_ms.append(res.solve_time * 1e3)
            fz_ratios.append(res.forces0.reshape(8, 3)[:, 2].sum() / mg)
            xr, _, gg = mpc.ref.sample(t_wall)
            qx, qy, qz, qw = xr[3:7, 0]
            reltilt = _tilt_deg(rt.mj_data.qpos) - _tilt_from_quat_xyzw(qx, qy, qz, qw)
            reltilts.append(reltilt)
            if abs(reltilt) > FALL_RELTILT:               # diverged from the reference lean -> fall
                fell = True
                break
            if gg[:, 0].max() > 0.5:                       # hand error only at a hot grasp gate
                hand_errs.append(_hand_err_cm(rm, rt.mj_data.qpos, mpc, t_wall))
        if k % rt.control_decim == 0 and cmd is not None:
            q = np.array(rt.mj_data.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
            qd = np.array(rt.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
            rt._apply_torque(cmd.tau_ff + cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.qd_des - qd))
        rt.step_physics()
        if rt.mj_data.qpos[2] < 0.25:                      # base collapsed -> fall
            fell = True
            break
        if realtime:
            target = t_start + (k + 1) * rt.physics_dt
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    else:
        completed = (t_wall >= mpc.duration_wall - 1e-6)
    wall = time.perf_counter() - t_start
    return {
        "fz_ratio_p50": float(np.median(fz_ratios)) if fz_ratios else 0.0,
        "max_reltilt_deg": float(np.max(np.abs(reltilts))) if reltilts else 0.0,
        "hand_err_grasp_max_cm": float(np.max(hand_errs)) if hand_errs else 0.0,
        "solve_p50_ms": float(np.percentile(solve_ms, 50)) if solve_ms else 0.0,
        "solve_p90_ms": float(np.percentile(solve_ms, 90)) if solve_ms else 0.0,
        "solve_max_ms": float(np.max(solve_ms)) if solve_ms else 0.0,
        "fell": fell, "completed": completed,
        "rt_factor": (n_steps * rt.physics_dt) / wall if wall > 0 else 0.0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_scale", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--gif", type=str, default=None)
    a = ap.parse_args()
    cfg = make_track_config(time_scale=a.time_scale)
    print(run_pickup(cfg, duration=a.duration, realtime=a.realtime))
