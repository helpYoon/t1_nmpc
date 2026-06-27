"""MPCConfig: Booster T1 numbers for the aligator whole_body_rnea controller (reduced model).

Geometry/pose/limits trace to t1_controller (data only). Weights re-dimensioned to the
reduced 27-joint model (NOT traced to t1_controller — logged as a divergence)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
T1_URDF_PATH = os.path.join(_ASSETS, "t1_description", "urdf", "t1.urdf")
T1_PACKAGE_DIRS = (_ASSETS,)

LOCKED_JOINTS: Tuple[str, str] = ("AAHead_yaw", "Head_pitch")
# reduced pinocchio joint order (head removed): [Larm7, Rarm7, waist1, Lleg6, Rleg6]
JOINT_NAMES: Tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
ANKLE_ROLL_FRAMES = ("Left_Ankle_Roll", "Right_Ankle_Roll")
# index ranges within JOINT_NAMES (used by arm_to_nominal weighting)
ARM_JOINT_SLICE = slice(0, 14)
LEG_JOINT_SLICE = slice(15, 27)


def _nominal_joint_pos() -> np.ndarray:
    # 27 = [Larm7, Rarm7, waist, Lleg6, Rleg6] (head dropped)
    return np.array(
        [0.5, -1.0, 0, -1.4, 0, 0, 0]
        + [0.5, 1.0, 0, 1.4, 0, 0, 0]
        + [0]
        + [-0.05, 0, 0, 0.10, -0.05, 0]
        + [-0.05, 0, 0, 0.10, -0.05, 0],
        dtype=np.float64,
    )


def _kp() -> np.ndarray:
    return np.array([20] * 14 + [200] + [200, 200, 200, 200, 50, 50] * 1
                    + [200, 200, 200, 200, 50, 50], dtype=np.float64)


def _kd() -> np.ndarray:
    return np.array([0.5] * 14 + [5.0] + [5, 5, 5, 5, 3, 3] + [5, 5, 5, 5, 3, 3],
                    dtype=np.float64)


def _Q_diag() -> np.ndarray:
    # ndx=66 = [base_pos(6), joint_pos(27), base_vel(6), joint_vel(27)]
    base_pos = np.array([0, 0, 1000, 10000, 10000, 0], dtype=np.float64)   # x,y,yaw free
    joint_pos = np.concatenate([[100] * 14, [200], [300] * 6, [300] * 6])  # 27
    base_vel = np.array([2000, 2000, 1000, 1000, 1000, 2000], dtype=np.float64)
    joint_vel = np.concatenate([[10] * 14, [10], [2] * 6, [2] * 6])        # 27
    return np.concatenate([base_pos, joint_pos, base_vel, joint_vel])       # 66


def _R_diag() -> np.ndarray:
    # nu=45 = [a(33), W_L(6), W_R(6)]
    a_w = np.full(33, 1e-3)
    wrench_w = np.array([5e-4, 5e-4, 5e-4, 1e-3, 1e-3, 1e-3])              # per foot 6D
    return np.concatenate([a_w, wrench_w, wrench_w])


@dataclass(frozen=True)
class MPCConfig:
    # dims
    n_joints: int = 27
    nq: int = 34
    nv: int = 33
    nx: int = 67
    ndx: int = 66
    n_feet: int = 2
    nf: int = 12
    na: int = 33
    nu: int = 45

    # horizon (uniform; matches t1_controller)
    nodes: int = 31
    dt: float = 0.035

    # nominal stand
    nominal_base_height: float = 0.6734
    nominal_joint_pos: np.ndarray = field(default_factory=_nominal_joint_pos)
    robot_mass: float = 31.0          # reference only; live value from the model

    # sole geometry
    sole_z: float = -0.030
    half_len: float = 0.1065          # X half-extent
    half_width: float = 0.05          # Y half-extent
    friction_mu: float = 0.4

    # gait (walk; t1_controller verbatim)
    gait_cycle: float = 1.4
    switching_times: Tuple[float, ...] = (0.0, 0.6, 0.7, 1.3, 1.4)
    swing_height: float = 0.08
    v_liftoff: float = 0.05
    v_touchdown: float = -0.05

    # weights / gains
    Q_diag: np.ndarray = field(default_factory=_Q_diag)
    R_diag: np.ndarray = field(default_factory=_R_diag)
    arm_weight_scale: float = 50.0    # multiplies arm joint_pos weights in arm_to_nominal
    kp: np.ndarray = field(default_factory=_kp)
    kd: np.ndarray = field(default_factory=_kd)

    # aligator solver
    al_tol: float = 1e-3              # target_tol (constraint sat) for the AL loop
    mu_init: float = 1e-2            # initial AL penalty
    cold_max_iters: int = 50
    warm_max_iters: int = 6
    torque_limit_weight: float = 1e-3  # soft torque-limit penalty (§3.8)

    # execution rates
    pd_hz: float = 500.0
    physics_hz: float = 2000.0


def make_config(**overrides) -> MPCConfig:
    c = MPCConfig(**overrides)
    assert c.nx == c.nq + c.nv == 67
    assert c.ndx == 2 * c.nv == 66
    assert c.nf == 6 * c.n_feet == 12
    assert c.nu == c.na + c.nf == 45
    assert c.na == c.nv
    assert c.nominal_joint_pos.shape == (27,)
    assert c.Q_diag.shape == (c.ndx,)
    assert c.R_diag.shape == (c.nu,)
    assert c.kp.shape == (27,) and c.kd.shape == (27,)
    assert len(JOINT_NAMES) == 27
    return c


@dataclass
class JointCommand:
    """27-joint command: tau = tau_ff + kp*(q_des-q) + kd*(qd_des-qd)."""
    q_des: np.ndarray    # (27,)
    qd_des: np.ndarray   # (27,)
    tau_ff: np.ndarray   # (27,)
    kp: np.ndarray       # (27,)
    kd: np.ndarray       # (27,)
