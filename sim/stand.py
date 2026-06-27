"""Closed-loop MuJoCo stand under the whole_body_rnea (Fatrop) MPC.

Control: at mpc_hz solve -> JointCommand (q_des, qd_des, tau_ff); at control_hz apply
tau = tau_ff + kp*(q_des - q) - kd*(qd_des - qd); physics at physics_hz."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.mpc import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def _tilt_deg(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))   # angle of body-z from world-z


def run_stand(cfg: MPCConfig, duration: float = 2.0, view: bool = False, gif: str = None) -> dict:
    rm = load_model(cfg)
    rt = MujocoRuntime(cfg, rm)
    rt.reset_to_nominal()
    mpc = WholeBodyMPC(cfg, rm)
    mpc.reset(nominal_x(cfg, rm.model))

    cmd = None
    solve_ms, fz_ratios, tilts = [], [], []
    mg = rm.mass * 9.81
    n_steps = int(round(duration * cfg.physics_hz))
    fell = False
    for k in range(n_steps):
        if k % rt.mpc_decim == 0:                                   # MPC tick (ZOH)
            x = rt.freeflyer_state(rm.model)
            res = mpc.step(x)
            cmd = res.command
            solve_ms.append(res.solve_time * 1e3)
            fz_ratios.append(res.forces0.reshape(8, 3)[:, 2].sum() / mg)
        if k % rt.control_decim == 0 and cmd is not None:           # control tick
            q = np.array(rt.mj_data.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
            qd = np.array(rt.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
            tau = cmd.tau_ff + cmd.kp * (cmd.q_des - q) - cmd.kd * (cmd.qd_des - qd)
            rt._apply_torque(tau)
        rt.step_physics()
        tilts.append(_tilt_deg(rt.mj_data.qpos))
        if rt.mj_data.qpos[2] < 0.3 or tilts[-1] > 45.0:            # fell
            fell = True
            break
    return {
        "fz_ratio_p50": float(np.median(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_min": float(np.min(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_max": float(np.max(fz_ratios)) if fz_ratios else 0.0,
        "max_tilt_deg": float(np.max(tilts)) if tilts else 0.0,
        "fell": fell,
        "solve_p90_ms": float(np.percentile(solve_ms, 90)) if solve_ms else 0.0,
        "t_end": float(rt.t),
    }


if __name__ == "__main__":
    import argparse
    from t1_nmpc.robot.config import make_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    a = ap.parse_args()
    print(run_stand(make_config(), duration=a.duration))
