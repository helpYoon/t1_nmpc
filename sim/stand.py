"""Closed-loop MuJoCo stand under the whole_body_rnea (Fatrop) MPC.

Control: at mpc_hz solve -> JointCommand (q_des, qd_des, tau_ff); at control_hz apply
tau = tau_ff + kp*(q_des - q) + kd*(qd_des - qd); physics at physics_hz."""
from __future__ import annotations

import time

import numpy as np
import mujoco
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.mpc import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def _tilt_deg(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))   # angle of body-z from world-z


def _measured_grf_z(m, d) -> float:
    """MuJoCo-measured vertical ground reaction [N]: sum of world-frame vertical contact force
    over all contacts. mj_contactForce returns force in the contact frame (f6[0]=normal,
    f6[1:3]=tangent); contact.frame rows are those axes in world coords, so f_world = frame.T @ f6[:3].
    At a static stand this reads ~ m*g (the floor holding the robot up)."""
    total = 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        frame = d.contact[i].frame.reshape(3, 3)     # rows: normal, tangent1, tangent2 (world)
        f_world = frame.T @ f6[:3]
        total += f_world[2]
    return abs(total)


def run_stand(cfg: MPCConfig, duration: float = 4.0, view: bool = False,
              realtime: bool = True) -> dict:
    """Closed-loop stand. If view=True, open MuJoCo's interactive passive viewer; with
    realtime=True the loop is paced to wall-clock (closing the window stops the run)."""
    rm = load_model(cfg)
    rt = MujocoRuntime(cfg, rm)
    rt.reset_to_nominal()
    mpc = WholeBodyMPC(cfg, rm)
    mpc.reset(nominal_x(cfg, rm.model))

    cmd = None
    solve_ms, fz_ratios, grf_ratios, tilts = [], [], [], []
    mg = rm.mass * 9.81
    n_steps = int(round(duration * cfg.physics_hz))
    render_decim = max(1, int(round(cfg.physics_hz / 60.0)))       # ~60 fps viewer sync
    fell = False

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(rt.mj_model, rt.mj_data)
        viewer.sync()                                              # show the settled stand
    wall_start = time.perf_counter()
    try:
        for k in range(n_steps):
            if viewer is not None and not viewer.is_running():     # user closed the window
                break
            if k % rt.mpc_decim == 0:                              # MPC tick (ZOH)
                x = rt.freeflyer_state(rm.model)
                res = mpc.step(x)
                cmd = res.command
                solve_ms.append(res.solve_time * 1e3)
                fz_ratios.append(res.forces0.reshape(8, 3)[:, 2].sum() / mg)
                grf_ratios.append(_measured_grf_z(rt.mj_model, rt.mj_data) / mg)   # measured PLANT GRF
            if k % rt.control_decim == 0 and cmd is not None:      # control tick
                q = np.array(rt.mj_data.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
                qd = np.array(rt.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
                tau = cmd.tau_ff + cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.qd_des - qd)
                rt._apply_torque(tau)
            rt.step_physics()
            tilts.append(_tilt_deg(rt.mj_data.qpos))
            if viewer is not None and k % render_decim == 0:
                viewer.sync()
                if realtime:                                       # pace to wall-clock
                    ahead = (k + 1) * rt.physics_dt - (time.perf_counter() - wall_start)
                    if ahead > 0:
                        time.sleep(ahead)
            if rt.mj_data.qpos[2] < 0.3 or tilts[-1] > 45.0:       # fell
                fell = True
                break
    finally:
        if viewer is not None:
            viewer.close()
    return {
        "fz_ratio_p50": float(np.median(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_min": float(np.min(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_max": float(np.max(fz_ratios)) if fz_ratios else 0.0,
        "grf_ratio_p50": float(np.median(grf_ratios)) if grf_ratios else 0.0,
        "grf_ratio_min": float(np.min(grf_ratios)) if grf_ratios else 0.0,
        "grf_ratio_max": float(np.max(grf_ratios)) if grf_ratios else 0.0,
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
    ap.add_argument("--view", action="store_true", help="open the MuJoCo interactive viewer")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="run as fast as possible instead of pacing the viewer to wall-clock")
    a = ap.parse_args()
    print(run_stand(make_config(), duration=a.duration, view=a.view, realtime=a.realtime))
