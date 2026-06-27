"""Aligator walk runner: headless metrics, offscreen GIF render, and interactive MuJoCo viewer.

  python -m sim.wb_walk_aligator --gif out.gif --duration 3   # offscreen GIF (headless OK)
  python -m sim.wb_walk_aligator --view --speed 0.5           # live viewer (needs a display)
  python -m sim.wb_walk_aligator                              # headless metrics only
"""
from __future__ import annotations
import argparse, dataclasses, time
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model
from t1_nmpc.wb.aligator_mpc import AligatorMPC
from t1_nmpc.wb.aligator_state import mujoco_to_freeflyer, freeflyer_command
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from sim._sim_util import tilt_from_quat_wxyz


def _build(vx, max_iter, n_horizon=None, threads=None):
    cfg = make_wb_config()
    al = make_aligator_config()
    if max_iter is not None:
        al = dataclasses.replace(al, max_iters=int(max_iter))
    if n_horizon is not None:
        al = dataclasses.replace(al, N=int(n_horizon))
    if threads is not None:
        al = dataclasses.replace(al, num_threads=int(threads))
    am = build_aligator_model(cfg)
    tp = MujocoTransport(cfg, mpc_hz=40.0)
    rt = tp.rt
    mpc = AligatorMPC(cfg, al, am, gait=SLOW_WALK, v_cmd=(float(vx), 0.0, 0.0))
    x0 = mujoco_to_freeflyer(rt, am)
    mpc.reset(x0)
    return cfg, al, am, tp, rt, mpc


def _step_loop(cfg, am, tp, rt, mpc, duration_s, on_tick=None):
    """Closed-loop walk; calls on_tick(k) each control step. Returns a metrics dict."""
    mg = am.mass * 9.81
    se = max(1, int(round(rt.cfg.control_hz / 40.0)))
    ctrl_dt = 1.0 / rt.cfg.control_hz
    x0 = mujoco_to_freeflyer(rt, am)
    res = mpc.step(x0, 0.0)
    x_start, y0 = float(rt.mj_data.qpos[0]), float(rt.mj_data.qpos[1])
    fz_min, t_fall, solve_ms = 1.0, None, []
    n = int(round(duration_s * rt.cfg.control_hz))
    for k in range(n):
        x = mujoco_to_freeflyer(rt, am)
        if k % se == 0:
            res = mpc.step(x, tp.now())
            solve_ms.append(mpc.last_solve_s * 1e3)
        tp.write_command(freeflyer_command(am, x, res, cfg))
        u0 = np.asarray(res.us[0]); fz_min = min(fz_min, (u0[2] + u0[8]) / mg)
        d = rt.mj_data
        if t_fall is None and (d.qpos[2] < 0.45 or float(tilt_from_quat_wxyz(d.qpos[3:7])) > 0.5):
            t_fall = k * ctrl_dt
        if on_tick is not None:
            on_tick(k)
    d = rt.mj_data
    return {"t_fall": t_fall, "fz_min_ratio": float(fz_min),
            "com_adv": float(d.qpos[0]) - x_start, "y_drift": float(d.qpos[1]) - y0,
            "final_z": float(d.qpos[2]), "solve_ms_p90": float(np.percentile(solve_ms, 90))}


def run_wb_walk_aligator(duration_s=4.0, vx=0.3, max_iter=None, n_horizon=None) -> dict:
    cfg, al, am, tp, rt, mpc = _build(vx, max_iter, n_horizon)
    m = _step_loop(cfg, am, tp, rt, mpc, duration_s)
    print("aligator walk: t_fall=%s  fz_min/mg=%.2f  com_adv=%+.2f  y_drift=%+.3f  solve_p90=%.1fms"
          % ("survived(%gs)" % duration_s if m["t_fall"] is None else "%.2fs" % m["t_fall"],
             m["fz_min_ratio"], m["com_adv"], m["y_drift"], m["solve_ms_p90"]), flush=True)
    return m


def render_gif(out_path, duration_s=3.0, vx=0.3, max_iter=None, threads=None, fps=30, w=480, h=360):
    import mujoco
    from PIL import Image
    cfg, al, am, tp, rt, mpc = _build(vx, max_iter, threads=threads)
    renderer = mujoco.Renderer(rt.mj_model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])
    cam.distance, cam.azimuth, cam.elevation = 3.0, 270.0, -12.0
    render_every = max(1, int(round(rt.cfg.control_hz / fps)))
    frames = []

    def tick(k):
        if k % render_every == 0:
            renderer.update_scene(rt.mj_data, cam)
            frames.append(renderer.render())

    m = _step_loop(cfg, am, tp, rt, mpc, duration_s, on_tick=tick)
    frames += [frames[-1]] * int(0.6 * fps)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote %s  frames=%d  t_fall=%s  y_drift=%+.3f" %
          (out_path, len(frames), m["t_fall"], m["y_drift"]), flush=True)
    return m


def run_view(duration_s=12.0, vx=0.3, speed=0.5, max_iter=None, azimuth=270.0, elevation=-12.0):
    import mujoco, mujoco.viewer as mj_viewer
    cfg, al, am, tp, rt, mpc = _build(vx, max_iter)
    viewer = mj_viewer.launch_passive(rt.mj_model, rt.mj_data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, azimuth, elevation
    se = max(1, int(round(rt.cfg.control_hz / 40.0)))
    dt_ctrl = 1.0 / rt.cfg.control_hz
    render_every = max(1, int(round(rt.cfg.control_hz / 60.0)))
    print(f"aligator walk viewer: maxit={mpc.al.max_iters} speed={speed}x (close window/Ctrl-C to quit)", flush=True)
    res = mpc.step(mujoco_to_freeflyer(rt, am), 0.0)
    t0 = time.perf_counter()
    try:
        for k in range(int(round(duration_s * rt.cfg.control_hz))):
            if not viewer.is_running():
                break
            x = mujoco_to_freeflyer(rt, am)
            if k % se == 0:
                res = mpc.step(x, tp.now())
            tp.write_command(freeflyer_command(am, x, res, cfg))
            if k % render_every == 0:
                viewer.sync()
            target = t0 + ((k + 1) * dt_ctrl) / max(speed, 1e-6)
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)
    d = rt.mj_data
    print("done: z=%.3f tilt=%.3f y_drift=%+.3f" %
          (float(d.qpos[2]), float(tilt_from_quat_wxyz(d.qpos[3:7])), float(d.qpos[1])), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Aligator T1 walk (headless metrics / GIF / live viewer).")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=None, help="override ProxDDP max_iters")
    ap.add_argument("--N", type=int, default=None, help="override horizon node count")
    ap.add_argument("--threads", type=int, default=None, help="LQ threads (1 = serial; walk needs serial)")
    ap.add_argument("--view", action="store_true", help="live MuJoCo viewer (needs a display)")
    ap.add_argument("--gif", type=str, default=None, help="render an offscreen GIF to this path")
    ap.add_argument("--speed", type=float, default=0.5, help="viewer playback speed")
    args = ap.parse_args()
    if args.view:
        run_view(args.duration, args.vx, speed=args.speed, max_iter=args.max_iter)
    elif args.gif:
        render_gif(args.gif, args.duration, args.vx, max_iter=args.max_iter, threads=args.threads)
    else:
        run_wb_walk_aligator(args.duration, args.vx, max_iter=args.max_iter, n_horizon=args.N)


if __name__ == "__main__":
    main()
