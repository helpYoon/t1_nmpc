"""MPCConfig: all Booster T1 numbers, ported from wb_humanoid_mpc .info files.

Sources (single source of truth — do not edit numbers here without re-checking):
  task.info       — Q, R, Q_final, task_space_costs, foot_constraint, swing_trajectory_config,
                    contacts, jointLimits, collision_constraint, joint_pd_gains, multiple_shooting, mpc
  reference.info  — defaultJointState, defaultBaseHeight, vel/height/pitch envelope
  gait.info       — walk mode_sequence / switching_times (cycle_period 1.4)
  contract §A.5   — joint order + URDF lower/upper limits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# ---- index slices (the law; §"Key cross-file invariants") ----
MOM = slice(0, 6)        # normalized centroidal momentum
BASE = slice(6, 12)      # base pose (p_xyz, theta_zyx)
JOINTS = slice(12, 41)   # 29 joint angles (state) / joint velocities (input)
WRENCH_L = slice(0, 6)   # left-foot wrench (input)
WRENCH_R = slice(6, 12)  # right-foot wrench (input)
QDJ = slice(12, 41)      # joint-velocity block of the input


# ===== concrete §B.9 vectors (built once, reused) =====

def _Q_joints() -> np.ndarray:
    # task.info Q joint block (state idx 12..40), scaling 1e0
    return np.array(
        [1, 1, 10, 20, 2, 2, 1, 1, 1,        # head(2) + L-arm(7)
         10, 20, 2, 2, 1, 1, 1,              # R-arm(7)
         0.5,                                # waist
         0.02, 0.06, 1.0, 0.02, 0.01, 0.01,  # L-leg
         0.02, 0.06, 1.0, 0.02, 0.01, 0.01], # R-leg
        dtype=np.float64,
    )


def _build_Q() -> np.ndarray:
    Q = np.zeros(41, dtype=np.float64)
    Q[0:6] = [8, 8, 15, 0, 0, 4]      # momentum
    Q[6:12] = [0, 0, 15, 0, 2, 2]     # base pose
    Q[12:41] = _Q_joints()
    return Q  # task.info scaling 1e0 → already final


def _build_R() -> np.ndarray:
    R = np.zeros(41, dtype=np.float64)
    wrench = np.array([0.05, 0.05, 0.01, 0.05, 0.05, 0.2], dtype=np.float64)
    R[0:6] = wrench
    R[6:12] = wrench
    jv = np.array(
        [100, 100, 200, 100, 100, 100, 100, 100, 100,
         200, 100, 100, 100, 100, 100, 100, 100,
         20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
        dtype=np.float64,
    )
    R[12:41] = jv
    return R * 1e-3  # task.info R scaling 1e-3


def _build_Q_final() -> np.ndarray:
    Qf = np.zeros(41, dtype=np.float64)
    Qf[0:6] = [25, 25, 25, 0, 0, 25]
    Qf[6:12] = [0, 0, 20, 0, 2, 2]
    Qf[12:41] = _Q_joints()  # joint block identical to Q
    return Qf  # scaling 1e0


def _build_kp() -> np.ndarray:
    return np.array(
        [20, 20] + [20] * 14 + [200] +
        [200, 200, 200, 200, 50, 50] +
        [200, 200, 200, 200, 50, 50],
        dtype=np.float64,
    )


def _build_kd() -> np.ndarray:
    return np.array(
        [0.2, 0.2] + [0.5] * 14 + [5.0] +
        [5, 5, 5, 5, 3, 3] + [5, 5, 5, 5, 3, 3],
        dtype=np.float64,
    )


def _build_nominal_joint_pos() -> np.ndarray:
    # reference.info defaultJointState, §A.5 order
    return np.array(
        [0, 0, 0.5, -1.0, 0, -1.4, 0, 0, 0,
         0.5, 1.0, 0, 1.4, 0, 0, 0, 0,
         -0.20, 0, 0, 0.40, -0.20, 0,
         -0.20, 0, 0, 0.40, -0.20, 0],
        dtype=np.float64,
    )


def _build_torso_task_weights() -> np.ndarray:
    # order: pos(3), ori(3), linvel(3), angvel(3), linacc(3), angacc(3)
    return np.array(
        [0, 0, 0,
         100, 100, 0,
         0.1, 0.1, 0.005,
         5, 5, 2,
         0, 0, 0,
         0, 0, 0],
        dtype=np.float64,
    )


def _build_swing_foot_task_weights() -> np.ndarray:
    return np.array(
        [0, 0, 0,
         1000, 1000, 0,
         10, 10, 0,
         1, 1, 0.005,
         0, 0, 0,
         0, 0, 0],
        dtype=np.float64,
    )


def _build_leg_torque_weights() -> np.ndarray:
    # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] * 1e-4
    return np.array([2, 2, 1, 8, 0.2, 0.2], dtype=np.float64) * 1e-4


# §A.5 URDF joint limits, in the 29-joint order
_JOINT_LOWER = np.array(
    [-1.57, -0.35, -3.29, -1.577, -2.234, -2.1402, -2.5821, -1.8185, -1.3614,
     -3.29, -1.7055, -2.234, -1.6844, -2.5821, -1.4261, -1.4285,
     -1.57,
     -1.8, -0.3, -1.0, 0.0, -0.87, -0.44,
     -1.8, -1.57, -1.0, 0.0, -0.87, -0.44],
    dtype=np.float64,
)
_JOINT_UPPER = np.array(
    [1.57, 1.22, 1.18, 1.7055, 2.234, 1.6844, 2.5821, 1.4261, 1.4285,
     1.1868, 1.577, 2.234, 2.1402, 2.5821, 1.8185, 1.3614,
     1.57,
     1.57, 1.57, 1.0, 2.34, 0.35, 0.44,
     1.57, 0.3, 1.0, 2.34, 0.35, 0.44],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MPCConfig:
    # --- dimensions / horizon ---
    N: int = 60
    dt: float = 0.02
    T: float = 1.2
    nx: int = 41
    nu: int = 41
    n_joints: int = 29
    n_contacts: int = 2
    nq: int = 35
    nv: int = 35

    # --- cost weights (diagonal, full length) ---
    Q: np.ndarray = field(default_factory=_build_Q)
    R: np.ndarray = field(default_factory=_build_R)
    Q_final: np.ndarray = field(default_factory=_build_Q_final)
    terminal_scale: float = 3.0

    # --- task-space cost weights ---
    torso_task_weights: np.ndarray = field(default_factory=_build_torso_task_weights)
    swing_foot_task_weights: np.ndarray = field(default_factory=_build_swing_foot_task_weights)
    leg_torque_weights: np.ndarray = field(default_factory=_build_leg_torque_weights)

    # --- gait params ---
    gait_name: str = "walk"
    cycle_period: float = 1.4
    speed_band_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.8], dtype=np.float64)
    )
    speed_band_names: Tuple[str, ...] = ("stance", "walk", "fast_walk")
    speed_band_hysteresis: float = 0.05

    # --- swing params ---
    swing_height: float = 0.08
    lift_off_velocity: float = 0.05
    touch_down_velocity: float = -0.0
    touch_down_height_offset: float = -0.001
    swing_time_scale: float = 0.4
    impact_prox_liftoff_vel: float = -0.15
    impact_prox_touchdown_vel: float = 0.3
    impact_prox_midpoint_value: float = 0.0

    # --- foot-constraint feedback gains ---
    foot_pos_err_gain_z: float = 5.0
    foot_ori_err_gain: float = 20.0
    foot_linvel_err_gain_z: float = 1.0
    foot_linvel_err_gain_xy: float = 1.0
    foot_angvel_err_gain: float = 1.0

    # --- per-joint PD gains ---
    kp: np.ndarray = field(default_factory=_build_kp)
    kd: np.ndarray = field(default_factory=_build_kd)

    # --- robot / geometry ---
    robot_mass: float = 34.5135
    nominal_base_height: float = 0.62
    nominal_trunk_pitch: float = 0.0
    contact_frame_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.0, -0.027], dtype=np.float64)
    )
    foot_rect_x: Tuple[float, float] = (-0.10, 0.10)
    foot_rect_y: Tuple[float, float] = (-0.045, 0.045)

    # --- friction / barrier ---
    friction_mu: float = 0.4
    friction_barrier_mu: float = 0.2
    friction_barrier_delta: float = 5.0
    friction_cone_reg: float = 25.0  # softening reg in sqrt(fx^2+fy^2+reg) (src FrictionForceConeConstraint)
    cop_barrier_mu: float = 0.6
    cop_barrier_delta: float = 0.03
    joint_limit_barrier_mu: float = 1200.0
    joint_limit_barrier_delta: float = 0.1
    joint_lower: np.ndarray = field(default_factory=lambda: _JOINT_LOWER.copy())
    joint_upper: np.ndarray = field(default_factory=lambda: _JOINT_UPPER.copy())
    collision_foot_sphere_r: float = 0.055
    collision_knee_sphere_r: float = 0.065
    collision_barrier_mu: float = 30000.0
    collision_barrier_delta: float = 0.05

    # --- velocity command scaling / clamps ---
    cmd_scale_vx: float = 2.0
    cmd_scale_vy: float = 1.0
    cmd_scale_wz: float = 1.0
    max_vx: float = 2.0
    max_vy: float = 1.0
    max_wz: float = 1.0
    max_delta_height: float = 0.3
    max_trunk_pitch: float = 1.5
    cmd_filter_break_freq_hz: float = 5.0

    # --- nominal posture ---
    nominal_joint_pos: np.ndarray = field(default_factory=_build_nominal_joint_pos)

    # --- solver ---
    solver_backend: str = "clarabel"
    sqp_iterations: int = 1
    delta_tol: float = 1e-4
    warm_start: bool = True
    cold_start: bool = False

    # --- execution rates ---
    mpc_hz: float = 40.0
    control_hz: float = 500.0
    physics_hz: float = 2000.0


def make_config(**overrides) -> MPCConfig:
    """Construct the canonical T1 MPCConfig, with optional field overrides.

    All arrays are validated for shape/dtype before return so a typo in an
    override surfaces immediately rather than deep in the QP assembly.
    """
    cfg = MPCConfig(**overrides)
    _validate(cfg)
    return cfg


# Aliases so every phase's tests resolve the same factory.
default_config = make_config
make_default_config = make_config
load_config = make_config


def _validate(cfg: MPCConfig) -> None:
    assert cfg.nx == 41 and cfg.nu == 41
    assert cfg.nq == 35 and cfg.nv == 35
    assert cfg.n_joints == 29 and cfg.n_contacts == 2
    assert abs(cfg.N * cfg.dt - cfg.T) < 1e-12, "N*dt must equal T"
    for name, arr, n in (
        ("Q", cfg.Q, 41), ("R", cfg.R, 41), ("Q_final", cfg.Q_final, 41),
        ("kp", cfg.kp, 29), ("kd", cfg.kd, 29),
        ("nominal_joint_pos", cfg.nominal_joint_pos, 29),
        ("joint_lower", cfg.joint_lower, 29), ("joint_upper", cfg.joint_upper, 29),
        ("torso_task_weights", cfg.torso_task_weights, 18),
        ("swing_foot_task_weights", cfg.swing_foot_task_weights, 18),
        ("leg_torque_weights", cfg.leg_torque_weights, 6),
        ("contact_frame_offset", cfg.contact_frame_offset, 3),
    ):
        assert arr.shape == (n,), f"{name} shape {arr.shape} != ({n},)"
        assert arr.dtype == np.float64, f"{name} dtype {arr.dtype} != float64"
    assert np.all(cfg.joint_lower < cfg.joint_upper), "joint_lower must be < joint_upper"
    assert len(cfg.speed_band_names) == len(cfg.speed_band_thresholds) + 1
    assert cfg.solver_backend in ("clarabel", "proxqp", "osqp")


@dataclass
class JointCommand:
    """Joint-PD command emitted to the 500 Hz layer (§B.8). All arrays length 29."""
    q_des: np.ndarray   # (29,)
    qd_des: np.ndarray  # (29,)
    kp: np.ndarray      # (29,)
    kd: np.ndarray      # (29,)
    tau_ff: np.ndarray  # (29,)
    # MPC contact wrenches [fx,fy,fz,Mx,My,Mz] per foot (world-aligned, at contact frame), for the
    # inverse-dynamics feedforward computed at control rate (wb computeJointTorques). None -> no ID FF.
    wrench_l: np.ndarray = None  # (6,)
    wrench_r: np.ndarray = None  # (6,)
