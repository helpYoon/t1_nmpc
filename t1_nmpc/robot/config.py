"""MPCConfig: Booster T1 numbers for the whole_body_rnea (CasADi+Fatrop) controller.

Geometry/pose/limits trace to t1_controller (data only, not formulation). Weights are
re-dimensioned to T1's 29 joints (NOT traced to t1_controller — logged as a divergence)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
T1_URDF_PATH = os.path.join(_ASSETS, "t1_description", "urdf", "t1.urdf")
T1_PACKAGE_DIRS = (_ASSETS,)  # so 'package://t1_description/...' resolves

# §A.5 / pinocchio joint order: [head2, Larm7, Rarm7, waist1, Lleg6, Rleg6]
JOINT_NAMES: Tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
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


def _nominal_joint_pos() -> np.ndarray:
    return np.array(
        [0, 0]
        + [0.5, -1.0, 0, -1.4, 0, 0, 0]
        + [0.5, 1.0, 0, 1.4, 0, 0, 0]
        + [0]
        + [-0.05, 0, 0, 0.10, -0.05, 0]
        + [-0.05, 0, 0, 0.10, -0.05, 0],
        dtype=np.float64,
    )


def _kp() -> np.ndarray:
    return np.array(
        [20, 20] + [20] * 14 + [200]
        + [200, 200, 200, 200, 50, 50]
        + [200, 200, 200, 200, 50, 50], dtype=np.float64)


def _kd() -> np.ndarray:
    return np.array(
        [0.2, 0.2] + [0.5] * 14 + [5.0]
        + [5, 5, 5, 5, 3, 3] + [5, 5, 5, 5, 3, 3], dtype=np.float64)


def _Q_diag() -> np.ndarray:
    # state delta weights, ndx=70 = [base_pos(6), joint_pos(29), base_vel(6), joint_vel(29)]
    base_pos = np.array([0, 0, 1000, 10000, 10000, 0], dtype=np.float64)   # x,y,yaw free
    joint_pos = np.concatenate([
        [50, 50], [100] * 14, [200],
        [300] * 6, [300] * 6,
    ])
    base_vel = np.array([2000, 2000, 1000, 1000, 1000, 2000], dtype=np.float64)
    joint_vel = np.concatenate([[10, 10], [10] * 14, [10], [2] * 6, [2] * 6])
    return np.concatenate([base_pos, joint_pos, base_vel, joint_vel])


def _R_diag() -> np.ndarray:
    # input weights for the FULL width (na+nf+nj = 88): [a(35), forces(24), tau_j(29)]
    return np.concatenate([
        [1e-3] * 35,
        [5e-4] * 24,
        [1e-4] * 2, [1e-2] * 14, [1e-3], [1e-4] * 12,
    ])


@dataclass(frozen=True)
class MPCConfig:
    # dimensions / horizon
    nodes: int = 14
    tau_nodes: int = 3
    dt_min: float = 0.02
    dt_max: float = 0.06
    n_joints: int = 29
    nq: int = 36
    nv: int = 35
    nx: int = 71
    ndx: int = 70
    n_corners: int = 8
    nf: int = 24
    na: int = 35

    # nominal stand
    nominal_base_height: float = 0.6734
    nominal_joint_pos: np.ndarray = field(default_factory=_nominal_joint_pos)
    robot_mass: float = 34.5135   # reference; the live value comes from the model

    # corner geometry (ankle-frame offsets)
    corner_x: Tuple[float, float] = (-0.1015, 0.1115)
    corner_y: Tuple[float, float] = (-0.05, 0.05)
    corner_z: float = -0.030

    # friction
    friction_mu: float = 0.4

    # weights
    Q_diag: np.ndarray = field(default_factory=_Q_diag)
    R_diag: np.ndarray = field(default_factory=_R_diag)

    # per-joint PD
    kp: np.ndarray = field(default_factory=_kp)
    kd: np.ndarray = field(default_factory=_kd)

    # Fatrop options (single max_iter cap; warm-started ticks converge well under it)
    fatrop_max_iter: int = 50
    fatrop_tol: float = 1e-3
    fatrop_mu_init: float = 1e-4

    # execution rates
    mpc_hz: float = 50.0
    control_hz: float = 500.0
    physics_hz: float = 2000.0


def make_config(**overrides) -> MPCConfig:
    cfg = MPCConfig(**overrides)
    assert cfg.nx == cfg.nq + cfg.nv == 71
    assert cfg.ndx == 2 * cfg.nv == 70
    assert cfg.nf == 3 * cfg.n_corners == 24
    assert cfg.na == cfg.nv
    assert cfg.nominal_joint_pos.shape == (29,)
    assert cfg.Q_diag.shape == (cfg.ndx,)
    assert cfg.R_diag.shape == (cfg.na + cfg.nf + cfg.n_joints,)
    assert cfg.kp.shape == (29,) and cfg.kd.shape == (29,)
    return cfg


@dataclass
class JointCommand:
    """29-joint command to the control layer: tau = tau_ff + kp*(q_des-q) + kd*(qd_des-qd)."""
    q_des: np.ndarray    # (29,)
    qd_des: np.ndarray   # (29,)
    tau_ff: np.ndarray   # (29,)
    kp: np.ndarray       # (29,)
    kd: np.ndarray       # (29,)
