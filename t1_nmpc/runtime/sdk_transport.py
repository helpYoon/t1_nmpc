"""SDK transport (robot-only, UNTESTED). The booster SDK is a Python pub/sub: publish a LowCmd of
per-joint MotorCmd(q, dq, tau, kp, kd, weight). Imports WITHOUT the SDK present; construction raises
if the SDK is unavailable. The B1JointIndex<->27-MPC-joint map is the highest-risk piece — re-verify
on-robot before any motion."""
from __future__ import annotations

import numpy as np

from ..config import JointCommand
from ..wb.config_wb import WBConfig

# SDK B1JointIndex names in SDK order for the 27 MPC joints (head excluded). VERIFY ON-ROBOT.
SDK_JOINT_ORDER = [
    # left arm, right arm, waist, left leg, right leg — fill from B1JointIndex enum on the robot.
    # Placeholder ORDER is the MPC joint order (MPC_JOINT_NAMES); confirm it matches B1JointIndex.
]


class SdkTransport:
    def __init__(self, wb_cfg: WBConfig):
        try:
            import booster_robotics_sdk_python as sdk  # noqa: F401  (lazy; robot-only)
        except Exception as e:
            raise RuntimeError("booster SDK unavailable — SdkTransport is robot-only") from e
        self.sdk = sdk
        self.wb_cfg = wb_cfg
        sdk.ChannelFactory.Instance().Init(0)
        self.pub = sdk.B1LowCmdPublisher(); self.pub.InitChannel()
        # TODO(on-robot): subscribers for LowState + odometer; build SDK_JOINT_ORDER from B1JointIndex.
        self._motor = [sdk.MotorCmd() for _ in range(sdk.B1JointCnt)]

    def read_state(self) -> np.ndarray:
        # UNTESTED: assemble x68 = [base(6), q_joints(27), v_base(6), v_joints(27), s, v_s] from
        # LowState (q/dq) + odometer (base pose/vel), reordering via SDK_JOINT_ORDER. Robot-only.
        raise NotImplementedError("SdkTransport.read_state — wire on-robot")

    def write_command(self, cmd: JointCommand) -> None:
        sdk = self.sdk
        low = sdk.LowCmd(); low.cmd_type = sdk.LowCmdType.PARALLEL; low.motor_cmd = self._motor
        for i, name_idx in enumerate(SDK_JOINT_ORDER):
            mc = low.motor_cmd[name_idx]
            mc.q = float(cmd.q_des[i]); mc.dq = float(cmd.qd_des[i]); mc.tau = float(cmd.tau_ff[i])
            mc.kp = float(cmd.kp[i]); mc.kd = float(cmd.kd[i]); mc.weight = 1.0
        self.pub.Write(low)

    def now(self) -> float:
        import time
        return time.monotonic()
