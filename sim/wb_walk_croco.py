# sim/wb_walk_croco.py
"""Closed-loop M1 walk gate + deviation telemetry.

Modes:
  * default (no --view): synchronous headless gate -> prints WALK_GATE={...}.
  * --view: synchronous real-time loop with the live MuJoCo CPU viewer.

Control law (both modes): CrocoMPC(gait=SLOW_WALK).step(x_meas, t, command) ->
to_joint_command_wb -> transport.write_command. MPC re-solves at ~40 Hz (decimated);
PD wraps the same command at 500 Hz. t = transport.now() (real accumulated sim time).
"""
from __future__ import annotations
import argparse, json
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.runtime.mujoco_transport import MujocoTransport


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _foot_positions(rt, wb):
    """Post-step foot positions (world XYZ) via pinocchio FK from MuJoCo state."""
    import pinocchio as pin
    q_pin35, _ = rt._pin_q_v()
    # WBModel uses reduced model (no head): 33-dim q = [pos3, euler3, joints27]
    q33 = np.concatenate([q_pin35[0:6], q_pin35[8:35]])
    pin.forwardKinematics(wb.model, wb.data, q33)
    pin.updateFramePlacements(wb.model, wb.data)
    return [np.array(wb.data.oMf[fid].translation) for fid in wb.contact_fids]


def _telemetry_tick(k, rt, wb, transport, mpc_gait,
                    tilts, base_zs, com_xs,
                    foot_pos_init, prev_stance_ref,
                    stance_slips_xy, stance_sinks_z, swing_z_errs,
                    n_steps_ref):
    """Collect telemetry for one tick (after write_command has advanced physics).
    Returns (curr_stance, n_steps) after updating all lists in-place.
    """
    from sim._sim_util import tilt_from_quat_wxyz

    d = rt.mj_data
    t_now = transport.now()

    tilts.append(float(tilt_from_quat_wxyz(d.qpos[3:7])))
    base_zs.append(float(d.qpos[2]))
    com_xs.append(float(d.qpos[0]))

    foot_pos = _foot_positions(rt, wb)
    curr_stance = mpc_gait.contact_flags(t_now)
    prev_stance = prev_stance_ref[0]
    n_steps = n_steps_ref[0]

    # On first tick: seed stance anchors
    if k == 0:
        for side in range(2):
            if curr_stance[side]:
                foot_pos_init[side] = foot_pos[side].copy()
    else:
        # Detect contact switches
        for side in range(2):
            if curr_stance[side] != prev_stance[side]:
                if curr_stance[side]:          # touchdown -> new stance anchor
                    foot_pos_init[side] = foot_pos[side].copy()
                else:                          # liftoff -> one step taken
                    n_steps += 1
                    foot_pos_init[side] = None

    # Stance slip / sink
    for side in range(2):
        if curr_stance[side] and foot_pos_init[side] is not None:
            delta = foot_pos[side] - foot_pos_init[side]
            stance_slips_xy.append(float(np.linalg.norm(delta[:2])) * 1000.0)
            stance_sinks_z.append(float(abs(delta[2])) * 1000.0)
        elif not curr_stance[side]:
            ref_z = mpc_gait.swing_z(t_now, side)[0]
            swing_z_errs.append(float(abs(foot_pos[side][2] - ref_z)) * 1000.0)

    prev_stance_ref[0] = curr_stance
    n_steps_ref[0] = n_steps
    return curr_stance


def _build_metrics(tilts, base_zs, com_xs, solve_ms_list,
                   n_solver_failures, n_steps,
                   stance_slips_xy, stance_sinks_z, swing_z_errs,
                   ctrl_hz) -> dict:
    com_advance = com_xs[-1] - com_xs[0] if com_xs else 0.0
    total_t = len(com_xs) / ctrl_hz
    return dict(
        n_solver_failures=int(n_solver_failures),
        peak_tilt_rad=round(float(np.max(tilts)), 4) if tilts else 0.0,
        final_base_z=round(float(base_zs[-1]), 4) if base_zs else 0.0,
        com_advance_m=round(float(com_advance), 4),
        mean_vx=round(float(com_advance / total_t) if total_t > 0 else 0.0, 4),
        n_steps=int(n_steps),
        stance_slip_mm=round(float(np.percentile(stance_slips_xy, 95)), 3) if stance_slips_xy else 0.0,
        stance_sink_mm=round(float(np.percentile(stance_sinks_z, 95)), 3) if stance_sinks_z else 0.0,
        swing_z_err_mm=round(float(np.mean(swing_z_errs)), 3) if swing_z_errs else 0.0,
        median_solve_ms=round(float(np.median(solve_ms_list)), 2) if solve_ms_list else 0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_wb_walk_croco(duration_s: float = 12.0, vx: float = 0.3,
                      mpc_hz: float = 40.0) -> dict:
    """Synchronous headless closed-loop walk gate.  Returns WALK_GATE metrics dict.

    Reuses the ``run_wb_stand_view`` control-law pattern (solve -> to_joint_command_wb ->
    write_command) but with CrocoMPC(gait=SLOW_WALK) and full deviation telemetry.
    No viewer, no real-time pacing — runs as fast as the solver allows.
    """
    from t1_nmpc.wb.execution_wb import to_joint_command_wb

    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=mpc_hz)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    rt = transport.rt

    x0 = transport.read_state()
    mpc.reset(x0)

    ctrl_hz = float(rt.cfg.control_hz)   # 500 Hz
    solve_every = max(1, int(round(ctrl_hz / mpc_hz)))
    n_ctrl = int(round(duration_s * ctrl_hz))
    sample_ahead_s = 0.005

    command = np.array([vx, 0.0, float(cfg.nominal_base_height), 0.0])

    # Initial plan (t=0 before first write_command advances physics)
    res = mpc.step(x0, transport.now(), command=command)
    cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)

    # Telemetry accumulators
    tilts, base_zs, com_xs, solve_ms_list = [], [], [], []
    n_solver_failures = 0
    foot_pos_init = [None, None]
    prev_stance_ref = [mpc.gait.contact_flags(transport.now())]
    stance_slips_xy, stance_sinks_z, swing_z_errs = [], [], []
    n_steps_ref = [0]

    for k in range(n_ctrl):
        x_meas = transport.read_state()

        if k % solve_every == 0:
            res = mpc.step(x_meas, transport.now(), command=command)
            if res.status != 0:
                n_solver_failures += 1
            solve_ms_list.append(mpc.last_solve_s * 1e3)
        # U2: resample the command at the LIVE clock every control tick (march along the plan)
        cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s, t_now=transport.now())

        transport.write_command(cmd)

        _telemetry_tick(k, rt, wb, transport, mpc.gait,
                        tilts, base_zs, com_xs,
                        foot_pos_init, prev_stance_ref,
                        stance_slips_xy, stance_sinks_z, swing_z_errs,
                        n_steps_ref)

    return _build_metrics(tilts, base_zs, com_xs, solve_ms_list,
                          n_solver_failures, n_steps_ref[0],
                          stance_slips_xy, stance_sinks_z, swing_z_errs,
                          ctrl_hz)


def run_wb_walk_view(duration_s: float = 12.0, vx: float = 0.3,
                     mpc_hz: float = 40.0, speed: float = 1.0,
                     azimuth: float = 270.0, elevation: float = -12.0) -> dict:
    """Synchronous walk viewer: same control law, paced to real-time.

    azimuth=270 = right-side view (good for forward-walk).  Close window or Ctrl-C to quit.
    Returns the same WALK_GATE metrics dict as run_wb_walk_croco.
    """
    import time
    import mujoco
    import mujoco.viewer as mj_viewer
    from t1_nmpc.wb.execution_wb import to_joint_command_wb
    from sim._sim_util import tilt_from_quat_wxyz

    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=mpc_hz)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    rt = transport.rt

    x0 = transport.read_state()
    mpc.reset(x0)

    ctrl_hz = float(rt.cfg.control_hz)
    dt_ctrl = 1.0 / ctrl_hz
    solve_every = max(1, int(round(ctrl_hz / mpc_hz)))
    n_ctrl = int(round(duration_s * ctrl_hz))
    sample_ahead_s = 0.005
    render_every = max(1, int(round(ctrl_hz / 60.0)))

    command = np.array([vx, 0.0, float(cfg.nominal_base_height), 0.0])

    res = mpc.step(x0, transport.now(), command=command)
    cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)

    # Do NOT use launch_passive as a context manager / call viewer.close(): its teardown can block
    # in C, which hangs the process. Leave open and force-exit via os._exit in main().
    viewer = mj_viewer.launch_passive(rt.mj_model, rt.mj_data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = int(rt.mj_model.jnt_bodyid[0])
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, azimuth, elevation

    print(f"walk viewer: duration={duration_s}s  loop@{ctrl_hz:.0f}Hz  "
          f"solve@{mpc_hz:.0f}Hz  speed={speed}x  vx={vx}  "
          "(close window or Ctrl-C to quit)", flush=True)

    tilts, base_zs, com_xs, solve_ms_list = [], [], [], []
    n_solver_failures = 0
    foot_pos_init = [None, None]
    prev_stance_ref = [mpc.gait.contact_flags(transport.now())]
    stance_slips_xy, stance_sinks_z, swing_z_errs = [], [], []
    n_steps_ref = [0]

    t_wall0 = time.perf_counter()
    try:
        for k in range(n_ctrl):
            if not viewer.is_running():
                break
            x_meas = transport.read_state()

            if k % solve_every == 0:
                res = mpc.step(x_meas, transport.now(), command=command)
                if res.status != 0:
                    n_solver_failures += 1
                solve_ms_list.append(mpc.last_solve_s * 1e3)
                cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)

            transport.write_command(cmd)

            _telemetry_tick(k, rt, wb, transport, mpc.gait,
                            tilts, base_zs, com_xs,
                            foot_pos_init, prev_stance_ref,
                            stance_slips_xy, stance_sinks_z, swing_z_errs,
                            n_steps_ref)

            if k % render_every == 0:
                viewer.sync()
            # Pace to real wall-clock (cumulative target self-corrects after a solve spike)
            target = t_wall0 + ((k + 1) * dt_ctrl) / max(speed, 1e-6)
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)

    d = rt.mj_data
    print(f"done: z={float(d.qpos[2]):.3f}m  "
          f"tilt={float(tilt_from_quat_wxyz(d.qpos[3:7])):.4f}rad  "
          f"n_fail={n_solver_failures}  steps={n_steps_ref[0]}", flush=True)

    return _build_metrics(tilts, base_zs, com_xs, solve_ms_list,
                          n_solver_failures, n_steps_ref[0],
                          stance_slips_xy, stance_sinks_z, swing_z_errs,
                          ctrl_hz)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop M1 walk gate (headless by default; --view to watch).")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="simulation duration [s]")
    ap.add_argument("--vx", type=float, default=0.3,
                    help="forward command velocity [m/s]")
    ap.add_argument("--mpc-hz", type=float, default=40.0,
                    help="MPC re-solve rate [Hz]")
    ap.add_argument("--view", action="store_true",
                    help="open the live MuJoCo CPU viewer (synchronous, paced)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="<1 = slow-mo (viewer only)")
    ap.add_argument("--azimuth", type=float, default=270.0,
                    help="camera azimuth: 0=front, 90=left, 270=right (default)")
    ap.add_argument("--elevation", type=float, default=-12.0,
                    help="camera elevation [deg]")
    args = ap.parse_args()

    if args.view:
        run_wb_walk_view(args.duration, args.vx, args.mpc_hz,
                         speed=args.speed, azimuth=args.azimuth,
                         elevation=args.elevation)
    else:
        m = run_wb_walk_croco(args.duration, args.vx, args.mpc_hz)
        print("WALK_GATE=" + json.dumps(m))


if __name__ == "__main__":
    main()
    # Flush before hard exit: os._exit bypasses the normal atexit / stdout flush,
    # so any buffered print output must be flushed explicitly first.
    # os._exit is still needed to kill OMP / MuJoCo viewer threads that block normal exit.
    import os, sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
