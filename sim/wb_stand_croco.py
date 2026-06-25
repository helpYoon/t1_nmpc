# sim/wb_stand_croco.py
"""Closed-loop M0 stand: CrocoMPC drives MuJoCo via the reused transport + control loop.
Acceptance gate equivalent to the acados M0 (peak_tilt, base_z, no failures)."""
from __future__ import annotations
import argparse, json

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.runtime.control_loop import run_loop


def run_wb_stand_croco(duration_s: float = 5.0, control_hz: float = 60.0) -> dict:
    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=control_hz)
    mpc = CrocoMPC(cfg, wb)
    return run_loop(transport, mpc, duration_s=duration_s, control_hz=control_hz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--control-hz", type=float, default=60.0)
    args = ap.parse_args()
    m = run_wb_stand_croco(args.duration, args.control_hz)
    print("STAND_GATE=" + json.dumps(m))
