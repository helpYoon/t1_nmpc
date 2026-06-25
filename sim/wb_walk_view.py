"""LIVE MuJoCo viewer for the closed-loop WB forward walk (CPU MuJoCo + CPU acados MPC).

No GPU<->compute contention here (the acados solve is CPU; the viewer renders on the GPU in its own
thread), so launch_passive runs fine. It advances slower than real-time because each ~50 ms MPC solve
dominates -- good for watching the fall. Real-time-paced between solves so the 2 kHz physics bursts
aren't a blur. Same control law as sim/wb_walk_gate.

Run in a terminal WITH a display:
  cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
  PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n t1mpc python -u sim/wb_walk_view.py [--vx V] [--duration S] [--speed X]

  --speed <1 = slow-mo (e.g. --speed 0.5). Close the window to quit. --vx 0 -> stand.

Record a GIF (off-screen render of the same run, GIF plays back at true real-time):
  ... python -u sim/wb_walk_view.py --vx 0.3 --duration 6 --record out.gif
  Add --no-view to record head-less (no window: robust over SSH / avoids a second GL context),
  and --fps to set the GIF playback rate (default 20). The recording camera defaults to a FRONT
  view (--azimuth 0); use --azimuth 90 for a side view, 180 for the back, or any angle in between.
"""
from __future__ import annotations

import contextlib
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

# Off-screen GIF render size (fits MuJoCo's default 640x480 offscreen buffer; small enough to send).
_REC_W, _REC_H = 480, 360


def _make_recorder(rt, fps: float, azimuth: float, elevation: float):
    """Off-screen renderer + base-tracking camera + capture cadence for GIF recording.

    Returns (renderer, camera, rec_decim, gif_frame_ms). rec_decim is the physics-step stride between
    captured frames; gif_frame_ms is the per-frame GIF duration set so playback is true real-time
    regardless of how sub-real-time the solve ran. azimuth 0 = front (look at the chest), 90/270 =
    side, 180 = back. Raises if the GL context can't be created.
    """
    rt.mj_model.vis.global_.offwidth = max(int(rt.mj_model.vis.global_.offwidth), _REC_W)
    rt.mj_model.vis.global_.offheight = max(int(rt.mj_model.vis.global_.offheight), _REC_H)
    renderer = mujoco.Renderer(rt.mj_model, height=_REC_H, width=_REC_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])   # body of the base free joint -> keep robot in frame
    cam.distance, cam.azimuth, cam.elevation = 3.2, azimuth, elevation
    rec_decim = max(1, int(round(rt.cfg.physics_hz / max(fps, 1.0))))
    gif_frame_ms = int(round(1000.0 * rec_decim / rt.cfg.physics_hz))
    return renderer, cam, rec_decim, gif_frame_ms


def run_view(vx: float = 0.3, duration_s: float = 10.0, speed: float = 1.0,
             record: str | None = None, fps: float = 20.0, view: bool = True,
             azimuth: float = 0.0, elevation: float = -12.0):
    if not view and record is None:
        raise SystemExit("--no-view with no --record produces nothing to watch or save.")
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

    renderer = cam = None
    rec_decim, gif_frame_ms, frames = 1, 50, []
    if record is not None:
        try:
            renderer, cam, rec_decim, gif_frame_ms = _make_recorder(rt, fps, azimuth, elevation)
        except Exception as e:                                  # GL context failure -> actionable hint
            hint = " try --no-view," if view else " set MUJOCO_GL=egl (or osmesa),"
            raise SystemExit(f"could not start the off-screen renderer ({type(e).__name__}: {e});{hint} "
                             f"then re-run.")

    print(f"vx={vx}  duration={duration_s}s  speed={speed}x  (solve ~50ms/tick -> runs sub-real-time)."
          + (f"  recording -> {record} @ {1000/gif_frame_ms:.0f}fps." if record else "")
          + ("  Close window to quit." if view else ""), flush=True)

    viewer_cm = mujoco.viewer.launch_passive(rt.mj_model, rt.mj_data) if view else contextlib.nullcontext()
    with viewer_cm as viewer:
        if view:                                               # start the live window at the same framing
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, azimuth, elevation
        t_wall0 = time.perf_counter()
        for k in range(n_phys):
            if view and not viewer.is_running():
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
            if record is not None and k % rec_decim == 0:
                renderer.update_scene(rt.mj_data, camera=cam)
                frames.append(renderer.render().copy())        # copy: render() reuses its buffer
            if view and k % sync_every == 0:
                viewer.sync()
                target = t_wall0 + ((k + 1) * dt_phys) / max(speed, 1e-6)   # real-time pace between solves
                dt_sleep = target - time.perf_counter()
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
        d = rt.mj_data
        print(f"done: net_bx={float(d.qpos[0]):.3f}m  final_z={float(d.qpos[2]):.3f}  "
              f"tilt={tilt_from_quat_wxyz(d.qpos[3:7]):.3f}rad", flush=True)

    if renderer is not None:
        renderer.close()
    if record is not None:
        if not frames:
            print("no frames captured -> nothing saved.", flush=True)
        else:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(record, save_all=True, append_images=imgs[1:],
                         duration=gif_frame_ms, loop=0, disposal=2, optimize=True)
            print(f"saved {len(frames)} frames -> {record}  "
                  f"({_REC_W}x{_REC_H}, {gif_frame_ms}ms/frame ~{1000/gif_frame_ms:.0f}fps)", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live MuJoCo viewer for the WB walk (vx=0 -> stand).")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="<1 = slow-mo")
    ap.add_argument("--record", type=str, default=None, metavar="OUT.gif",
                    help="also record an off-screen GIF of the run (plays back at real-time)")
    ap.add_argument("--fps", type=float, default=20.0, help="GIF playback frame rate (default 20)")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="record head-less without opening a window (use with --record)")
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="recording camera azimuth deg: 0=front, 90/270=side, 180=back (default 0)")
    ap.add_argument("--elevation", type=float, default=-12.0,
                    help="recording camera elevation deg (default -12; more negative looks down)")
    a = ap.parse_args()
    run_view(a.vx, a.duration, a.speed, record=a.record, fps=a.fps, view=a.view,
             azimuth=a.azimuth, elevation=a.elevation)


if __name__ == "__main__":
    main()
