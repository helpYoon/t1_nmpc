# sim/wb_stand_croco.py
"""Closed-loop M0 stand: CrocoMPC drives MuJoCo.

Two modes:
  * default (no flag): the headless acceptance GATE via the async control loop -> prints STAND_GATE.
  * --view / --record: a SYNCHRONOUS loop (solve -> command -> step physics -> render) for the live
    MuJoCo CPU viewer and/or an off-screen front-view GIF. Single-threaded, so it also avoids the
    GIL contention the async loop hits with crocoddyl (which, unlike acados, does not release the GIL).
"""
from __future__ import annotations
import argparse, json

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.runtime.control_loop import run_loop


def run_wb_stand_croco(duration_s: float = 5.0, control_hz: float = 60.0) -> dict:
    """Headless acceptance gate (async loop). Returns the STAND_GATE metrics dict."""
    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=control_hz)
    mpc = CrocoMPC(cfg, wb)
    return run_loop(transport, mpc, duration_s=duration_s, control_hz=control_hz)


def run_wb_stand_view(duration_s: float = 10.0, mpc_hz: float = 40.0,
                      record: str | None = None, fps: float = 20.0, view: bool = True,
                      azimuth: float = 0.0, elevation: float = -12.0, speed: float = 1.0) -> dict:
    """Synchronous closed-loop stand with a live viewer and/or an off-screen GIF.

    Drives the loop at the MuJoCo runtime's REAL control rate (cfg.control_hz, 500 Hz) so sim-time
    and wall-time advance together (`--view` paces to real-time, `--speed`<1 = slow-mo). The MPC
    re-solves at `mpc_hz` (~40 Hz), holding the command between solves (the per-tick PD provides the
    high-rate feedback). Reuses the gate control law (CrocoMPC.step -> to_joint_command_wb ->
    transport.write_command). azimuth 0 = front (chest), 90/270 = side, 180 = back. Returns
    {n_fail, final_base_z, tilt}.
    """
    import time
    import mujoco
    from t1_nmpc.wb.execution_wb import to_joint_command_wb
    from sim._sim_util import tilt_from_quat_wxyz

    if not view and record is None:
        raise SystemExit("--no-view with no --record produces nothing to watch or save.")

    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=mpc_hz)
    mpc = CrocoMPC(cfg, wb)
    rt = transport.rt
    x0 = transport.read_state(); mpc.reset(x0)
    # each transport.write_command advances cfg.control_decim physics steps = one control period.
    ctrl_hz = float(rt.cfg.control_hz)                            # 500 Hz: the real per-tick sim rate
    dt_ctrl = 1.0 / ctrl_hz
    solve_every = max(1, int(round(ctrl_hz / float(rt.cfg.mpc_hz))))   # re-solve the MPC every N ticks
    n_ctrl = int(round(duration_s * ctrl_hz))
    sample_ahead_s = 0.005

    # --- off-screen recorder (front-tracking camera), GIF plays back at true real-time ---
    renderer = cam = None
    rec_decim, gif_frame_ms, frames = 1, 50, []
    if record is not None:
        _REC_W, _REC_H = 480, 360
        rt.mj_model.vis.global_.offwidth = max(int(rt.mj_model.vis.global_.offwidth), _REC_W)
        rt.mj_model.vis.global_.offheight = max(int(rt.mj_model.vis.global_.offheight), _REC_H)
        try:
            renderer = mujoco.Renderer(rt.mj_model, height=_REC_H, width=_REC_W)
        except Exception as e:                                   # GL context failure -> actionable hint
            hint = " try --no-view," if view else " set MUJOCO_GL=egl (or osmesa),"
            raise SystemExit(f"could not start the off-screen renderer ({type(e).__name__}: {e});{hint} "
                             f"then re-run.")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])         # base free-joint body -> keep robot in frame
        cam.distance, cam.azimuth, cam.elevation = 3.2, azimuth, elevation
        rec_decim = max(1, int(round(ctrl_hz / max(fps, 1.0))))
        gif_frame_ms = int(round(1000.0 * rec_decim / ctrl_hz))

    print(f"stand viewer: duration={duration_s}s  loop@{ctrl_hz:.0f}Hz  solve@{rt.cfg.mpc_hz:.0f}Hz  speed={speed}x"
          + (f"  recording -> {record} @ {1000/gif_frame_ms:.0f}fps" if record else "")
          + ("  (close the window or Ctrl-C to quit)" if view else ""), flush=True)

    # Do NOT use launch_passive as a context manager / call viewer.close(): its teardown can block in
    # C, which hangs the process and swallows Ctrl-C. Leave the viewer open and force-exit at the end
    # of main() (os._exit) — the OS reclaims the viewer thread + window cleanly.
    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(rt.mj_model, rt.mj_data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, azimuth, elevation

    n_fail = 0
    render_every = max(1, int(round(ctrl_hz / 60.0)))            # live sync ~60 fps (not every 500 Hz tick)
    res = mpc.step(transport.read_state(), transport.now())      # initial plan
    cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)
    t_wall0 = time.perf_counter()
    try:
        for k in range(n_ctrl):
            if view and not viewer.is_running():
                break
            if k % solve_every == 0:                             # re-solve at ~mpc_hz; ZOH the cmd between
                res = mpc.step(transport.read_state(), transport.now())
                if res.status != 0:
                    n_fail += 1
                cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)
            transport.write_command(cmd)                         # one control period (control_decim physics steps); PD re-reads q_meas
            if record is not None and k % rec_decim == 0:
                renderer.update_scene(rt.mj_data, camera=cam)
                frames.append(renderer.render().copy())          # copy: render() reuses its buffer
            if view and k % render_every == 0:
                viewer.sync()
            if view:                                             # pace to real wall-clock (cumulative target self-corrects after a solve spike)
                target = t_wall0 + ((k + 1) * dt_ctrl) / max(speed, 1e-6)
                slack = target - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        print("\ninterrupted -> stopping.", flush=True)

    d = rt.mj_data
    final_z, tilt = float(d.qpos[2]), float(tilt_from_quat_wxyz(d.qpos[3:7]))
    print(f"done: final_z={final_z:.3f}m  tilt={tilt:.4f}rad  n_fail={n_fail}", flush=True)

    if renderer is not None:
        if not frames:
            print("no frames captured -> nothing saved.", flush=True)
        else:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(record, save_all=True, append_images=imgs[1:],
                         duration=gif_frame_ms, loop=0, disposal=2, optimize=True)
            print(f"saved {len(frames)} frames -> {record}  "
                  f"(480x360, {gif_frame_ms}ms/frame ~{1000/gif_frame_ms:.0f}fps)", flush=True)
        renderer.close()
    return {"n_fail": n_fail, "final_base_z": final_z, "tilt": tilt}


def main():
    ap = argparse.ArgumentParser(description="Closed-loop M0 stand (gate by default; --view/--record to watch).")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--control-hz", type=float, default=60.0, help="gate (headless async) control rate")
    ap.add_argument("--mpc-hz", type=float, default=40.0,
                    help="viewer MPC re-solve rate; the loop itself runs at the runtime's 500 Hz control rate")
    ap.add_argument("--view", action="store_true", help="open the live MuJoCo CPU viewer (synchronous loop)")
    ap.add_argument("--record", type=str, default=None, metavar="OUT.gif",
                    help="record an off-screen front-view GIF (implies the synchronous loop)")
    ap.add_argument("--fps", type=float, default=20.0, help="GIF playback frame rate (default 20)")
    ap.add_argument("--no-view", dest="view_off", action="store_true",
                    help="with --record: render head-less, no window (use over SSH / set MUJOCO_GL=egl)")
    ap.add_argument("--speed", type=float, default=1.0, help="<1 = slow-mo (viewer only)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="camera azimuth: 0=front, 90/270=side, 180=back")
    ap.add_argument("--elevation", type=float, default=-12.0, help="camera elevation deg (default -12)")
    args = ap.parse_args()

    want_view = (args.view or args.record is not None) and not args.view_off
    if want_view or args.record is not None:
        run_wb_stand_view(args.duration, mpc_hz=args.mpc_hz, record=args.record, fps=args.fps,
                          view=want_view, azimuth=args.azimuth, elevation=args.elevation, speed=args.speed)
    else:
        m = run_wb_stand_croco(args.duration, args.control_hz)
        print("STAND_GATE=" + json.dumps(m))


if __name__ == "__main__":
    main()
    # Force a clean exit: the MuJoCo passive viewer (and OMP worker pools) can keep non-daemon
    # threads alive whose teardown blocks; os._exit bypasses them so the shell prompt returns.
    import os
    os._exit(0)
