"""Async-MPC + fast-MRT control loop. MPC thread (pinned, free-running) solves on the latest state
SNAPSHOT and publishes the latest plan; control thread (pinned, REAL-TIME paced) resamples the plan,
builds the joint command, and drives the transport. Hand-off is lock-free atomic ref publication
(assign a fresh tuple/array to a 1-slot cell; the GIL makes each STORE/LOAD whole). acados solve()
releases the GIL, so the two threads overlap.

Real-time pacing is load-bearing: the control thread steps physics at wall-clock control_hz, so the
MPC thread's wall solve-rate equals its SIM rate (a faithful deployment measurement) — and a mostly-
idle paced control thread minimizes cross-core cache contention with the MPC thread.

Whether the stand HOLDS at the resulting async rate is the MEASURED RESULT (single-RTI needs ~60Hz;
if the real solve is slower, it falls) — not a precondition.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np


def _pin(cores) -> None:
    """Pin the calling thread to a core (int) or a SET of cores (iterable). A set is needed for the
    OMP MPC thread so its libgomp workers (OMP_NUM_THREADS) spread across the pool instead of being
    trapped on one core."""
    try:
        s = {cores} if isinstance(cores, int) else set(cores)
        os.sched_setaffinity(0, s)
    except (AttributeError, OSError, TypeError, ValueError):
        pass        # non-Linux / bad cores: run unpinned


def run_loop(transport, mpc, *, duration_s, control_hz, cores=(0, 1), sample_ahead_s=0.005) -> dict:
    from ..wb.execution_wb import to_joint_command_wb
    cfg = mpc.cfg
    x0 = transport.read_state()
    mpc.reset(x0)
    res0 = mpc.step(x0, transport.now())
    state_cell = [x0]                                  # control -> MPC: 1-slot, atomic array-ref store
    plan_cell = [(res0, transport.now())]              # MPC -> control: single STORE of a (res, t) tuple
    tot_ms: list[float] = []
    stop = threading.Event()
    n_fail = 0

    def mpc_thread():
        nonlocal n_fail
        _pin(cores[0])
        while not stop.is_set():
            snap = state_cell[0]                        # atomic ref read (fresh array)
            res = mpc.step(snap, transport.now())
            if res.status != 0:
                n_fail += 1
            plan_cell[0] = (res, transport.now())       # single atomic STORE of a fresh tuple
            tot_ms.append(float(mpc.last_solve_s) * 1e3)

    th = threading.Thread(target=mpc_thread, daemon=True); th.start()

    _pin(cores[1])
    tilts, base_z = [], []
    period = 1.0 / control_hz
    n_ctrl = int(round(duration_s * control_hz))
    wall0 = time.monotonic()
    next_t = wall0
    for _ in range(n_ctrl):
        x = transport.read_state()
        state_cell[0] = x                              # publish snapshot (single atomic STORE)
        res, _t = plan_cell[0]                         # single atomic LOAD of the (res, t) tuple
        cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=sample_ahead_s)
        transport.write_command(cmd)
        d = transport.rt.mj_data if hasattr(transport, "rt") else None
        if d is not None:
            from sim._sim_util import tilt_from_quat_wxyz
            tilts.append(tilt_from_quat_wxyz(d.qpos[3:7])); base_z.append(float(d.qpos[2]))
        next_t += period                               # real-time pace: wall-time tracks sim-time
        slack = next_t - time.monotonic()
        if slack > 0:
            time.sleep(slack)
    wall = max(time.monotonic() - wall0, 1e-9)
    stop.set(); th.join(timeout=1.0)

    nominal = float(cfg.nominal_base_height)
    peak_tilt = float(np.max(tilts)) if tilts else None
    final_z = float(base_z[-1]) if base_z else None
    held = bool(peak_tilt is not None and peak_tilt < 0.2 and final_z > 0.85 * nominal and n_fail == 0)
    return {
        "median_tot_ms": round(float(np.median(tot_ms)), 2) if tot_ms else 0.0,
        "p95_tot_ms": round(float(np.percentile(tot_ms, 95)), 2) if tot_ms else 0.0,
        "n_solves": len(tot_ms), "n_fail": n_fail,
        "effective_mpc_hz": round(len(tot_ms) / wall, 1),
        "effective_control_hz": round(n_ctrl / wall, 1),
        "peak_tilt_rad": round(peak_tilt, 4) if peak_tilt is not None else None,
        "final_base_z": round(final_z, 4) if final_z is not None else None,
        "held": held,
    }
