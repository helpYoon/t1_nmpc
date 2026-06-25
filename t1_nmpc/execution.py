"""PD torque (wb_humanoid-faithful, no WBC, tau_ff=0).

Reference: CentroidalMpcMrtJointController.cpp:181-213.
"""
import numpy as np

from .config import JointCommand


def pd_torque(jc: JointCommand, q_meas: np.ndarray, qd_meas: np.ndarray) -> np.ndarray:
    """tau = kp(q_des - q) + kd(qd_des - qd) + tau_ff   (29,). RobotJointAction.h."""
    return jc.kp * (jc.q_des - q_meas) + jc.kd * (jc.qd_des - qd_meas) + jc.tau_ff
