"""MuJoCo transport: wraps MujocoRuntime for the threaded loop. Single-threaded access (control
thread only) -> no MjData lock needed; the MPC thread works on published snapshots, never on mjData."""
from __future__ import annotations

import numpy as np

from ..robot.config import JointCommand, make_config
from ..robot.model import load_model, T1_URDF_PATH
from ..wb.config import WBConfig
from sim.mujoco_runtime import MujocoRuntime
from sim.state import wb_state_estimate, wb_reset as _wb_reset

_HEAD_KP, _HEAD_KD = 20.0, 0.5


class MujocoTransport:
    def __init__(self, wb_cfg: WBConfig, mpc_hz: float = 60.0):
        self.wb_cfg = wb_cfg
        ccfg = make_config(mpc_hz=mpc_hz)
        self.rt = MujocoRuntime(ccfg, load_model(T1_URDF_PATH, ccfg))
        _wb_reset(self.rt, wb_cfg)
        self.control_decim = self.rt.control_decim   # physics steps per control tick

    def read_state(self) -> np.ndarray:
        return wb_state_estimate(self.rt)             # fresh array

    def write_command(self, cmd: JointCommand) -> None:
        q_pin, v_pin = self.rt._pin_q_v()
        q_meas, qd_meas = q_pin[8:35], v_pin[8:35]
        tau_wb = cmd.kp * (cmd.q_des - q_meas) + cmd.kd * (cmd.qd_des - qd_meas) + cmd.tau_ff
        tau29 = np.zeros(29); tau29[2:29] = tau_wb
        tau29[0:2] = _HEAD_KP * (0.0 - q_pin[6:8]) - _HEAD_KD * v_pin[6:8]
        for _ in range(self.control_decim):
            self.rt._apply_torque(tau29)
            self.rt.step_physics()

    def now(self) -> float:
        return float(self.rt.t)
