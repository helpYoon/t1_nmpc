"""WBConfig: Booster T1 whole-body MPC numbers, mirrored from t1_controller
robot_models/booster_t1/t1_mpc/config/mpc/task.info (the proven OCS2 walker).

State x in R^68 = [q_base(6: p_xyz, theta_zyx euler), q_joints(27), v_base(6:
world-lin + euler-rate, NOT omega), v_joints(27), s, v_s].  Input u in R^40 =
[W_l(6), W_r(6), qdd_joints(27), vdot_s(1)].  Joints = 27 (arms14 + waist1 +
legs12; head excluded), the canonical 29-joint order minus the two head joints.

task.info Q/R blocks are 66/39 entries (the model state/input WITHOUT the path
slots s,v_s); we pad with zeros to 68/40 — the path/contouring weights are
inactive until M2.  Source line refs are to that task.info.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple

import numpy as np
import pinocchio as pin

from t1_nmpc.model import EXPECTED_JOINT_NAMES, T1_URDF_PATH

# 27 MPC joints = canonical 29-joint order minus the two head joints.
MPC_JOINT_NAMES: Tuple[str, ...] = EXPECTED_JOINT_NAMES[2:]

# index slices into the 68-dim state
Q_BASE = slice(0, 6)
Q_JOINTS = slice(6, 33)
V_BASE = slice(33, 39)
V_JOINTS = slice(39, 66)
PATH_S, PATH_VS = 66, 67  # contouring slots (zero until M2)


def _build_Q() -> np.ndarray:  # task.info:221-294
    Q = np.zeros(68, dtype=np.float64)
    Q[Q_BASE] = [0, 0, 10, 0, 20, 20]                  # p_xyz, theta_zyx (z, pitch, roll)
    Q[Q_JOINTS] = (
        [0.4, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4]            # L-arm
        + [0.4, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4]          # R-arm
        + [2.0]                                        # waist
        + [0.05, 0.2, 0.2, 0.1, 0.1, 0.1]              # L-leg
        + [0.05, 0.2, 0.2, 0.1, 0.1, 0.1]              # R-leg
    )
    # Base-velocity weights: base x/y POSITION weight is 0 (above), so forward progress is driven by the
    # vx velocity weight AND foot placement. Tuning found vx weight alone trades in-place (3.0) vs
    # fall-forward (10) vs collapse (40) — the missing piece is a forward foot-placement cost to CATCH the
    # forward-leaning CoM (paper's below-hip foot placement). Kept at the stable 3.0 pending that cost.
    Q[V_BASE] = 3.0   # base-velocity tracking weight (task.info); forward drive is EMERGENT, not a foot-placement cost (as in t1_controller)
    Q[V_JOINTS] = [0.02] * 14 + [0.2] + [0.001] * 12   # arms, waist, legs
    return Q  # scaling 1e0; path slots stay 0


def _build_R() -> np.ndarray:  # task.info:296-347
    R = np.zeros(40, dtype=np.float64)
    wrench = [0.003, 0.003, 0.001, 0.01, 0.01, 0.1]    # [fx,fy,fz,Mx,My,Mz]
    R[0:6] = wrench
    R[6:12] = wrench
    R[12:39] = [0.05] * 14 + [0.005] * 13              # qdd: arms, waist+legs
    R[39] = 1e-3          # vdot_s: tiny regularizer (-> 1e-6 after the *1e-3 scaling; OCS2 has w_vs>0)
    return R * 1e-3       # so the projected GN Hessian is PD on range(P) under lm=0 (was: vdot_s stayed 0)


def _build_Q_final() -> np.ndarray:  # task.info:351-424
    Qf = np.zeros(68, dtype=np.float64)
    Qf[Q_BASE] = [0, 0, 10, 0, 0, 0]                   # trunk pitch/roll 0 terminally
    Qf[Q_JOINTS] = (
        [0.4, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4]
        + [0.4, 2.0, 0.4, 0.4, 0.4, 0.4, 0.4]
        + [2.0]
        + [0.05, 0.1, 0.1, 0.1, 0.1, 0.1]              # L-leg (hip roll/yaw 0.1, not 0.2)
        + [0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    Qf[V_BASE] = [3, 3, 3, 0, 0, 0]
    Qf[V_JOINTS] = [0.02] * 14 + [0.2] + [0.001] * 12
    return Qf


def _build_nominal_joint_pos() -> np.ndarray:  # task.info initialState:155-181
    return np.array(
        [0.5, -1.0, 0.0, -1.4, 0.0, 0.0, 0.0]          # L-arm
        + [0.5, 1.0, 0.0, 1.4, 0.0, 0.0, 0.0]          # R-arm
        + [0.0]                                        # waist
        + [-0.05, 0.0, 0.0, 0.1, -0.05, 0.0]           # L-leg
        + [-0.05, 0.0, 0.0, 0.1, -0.05, 0.0],          # R-leg
        dtype=np.float64,
    )


def _build_kp() -> np.ndarray:  # task.info joint_pd_gains:561-597 (head excluded)
    return np.array([20.0] * 14 + [200.0] + [200, 200, 200, 200, 50, 50] * 2, dtype=np.float64)


def _build_kd() -> np.ndarray:
    return np.array([0.5] * 14 + [5.0] + [5, 5, 5, 5, 3, 3] * 2, dtype=np.float64)


def _build_armature() -> np.ndarray:  # task.info joint_dynamics_sysid:63-88
    leg = [0.0523908, 0.0478125, 0.0478125, 0.0636012, 0.0407621, 0.0111713]
    return np.array([0.0282528] * 14 + [0.0478125] + leg * 2, dtype=np.float64)


def _build_viscous_damping() -> np.ndarray:
    leg = [0.266903, 0.52201, 0.76929, 0.161472, 1.43965, 0.0821329]
    return np.array([0.5] * 14 + [0.528635] + leg * 2, dtype=np.float64)


@lru_cache(maxsize=1)
def _urdf_props() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(effort_limit, lower, upper) for the 27 MPC joints, read from the URDF.
    effort 0 -> +inf (unconstrained). Loaded once and cached."""
    model = pin.buildModelFromUrdf(T1_URDF_PATH)  # fixed base: limits only
    n = len(MPC_JOINT_NAMES)
    eff = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    for i, nm in enumerate(MPC_JOINT_NAMES):
        jid = model.getJointId(nm)
        iq = model.joints[jid].idx_q
        iv = model.joints[jid].idx_v
        e = model.effortLimit[iv]
        eff[i] = e if e > 0 else np.inf
        lo[i] = model.lowerPositionLimit[iq]
        hi[i] = model.upperPositionLimit[iq]
    return eff, lo, hi


@dataclass(frozen=True)
class WBConfig:
    # --- dimensions / horizon ---
    nx: int = 68
    nu: int = 40
    n_joints: int = 27
    n_contacts: int = 2
    N: int = 31
    dt: float = 0.035
    horizon: float = 1.1          # nominal (task.info timeHorizon); solver tf = N*dt = 1.085

    # --- cost weights (diagonal, full length) ---
    Q: np.ndarray = field(default_factory=_build_Q)
    R: np.ndarray = field(default_factory=_build_R)
    Q_final: np.ndarray = field(default_factory=_build_Q_final)
    terminal_scale: float = 4.0   # task.info terminalCostScaling

    # --- joint-torque soft-cap (task.info joint_torque_limits_cost) ---
    jointtorque_weight: float = 1.0
    jointtorque_scale: float = 1e2

    # --- foot-constraint feedback gains (task.info foot_constraint:12-22) ---
    foot_pos_err_gain_z: float = 100.0
    foot_ori_err_gain: float = 80.0
    foot_linvel_err_gain_z: float = 10.0
    foot_linvel_err_gain_xy: float = 20.0
    foot_angvel_err_gain: float = 20.0

    # --- swing-foot vertical-constraint accel gain (task.info:19) ---
    foot_linacc_err_gain_z: float = 1.0
    # --- stance-foot hard pose feedback (OCS2 getStanceFootConstraint Ax: pos_z=100, ori=80). crocoddyl
    # ContactModel6D applies ONE uniform kp to all 6 DoF (vs OCS2's selective pos_xy=0/ori=80), so the
    # faithful 100 over-stiffens orientation+xy and tips the robot; kp=50 is the crocoddyl-equivalent
    # sweet spot (empirical: kp=50 walks ~5s, kp=100 falls @0.2s, kp=0 falls @2s). ---
    stance_contact_kp: float = 50.0
    # --- CoP tipping margin (Fix #2): QuadraticBarrier has zero interior gradient, so the optimizer
    # rides the CoP to the foot edge (zero margin). Shrink the support box to a fraction and raise the
    # weight so the CoP is kept inside an inner rectangle -> stability margin (approximates OCS2's
    # relaxed interior-point barrier without its single-RTI stiffness). ---
    cop_margin_scale: float = 1.0    # full support rectangle (OCS2 uses the full foot; no shrink)
    cop_weight: float = 20.0
    # --- swing-foot task-space cost weights (task.info:436-453: ori_xy=1e4, linvel_xy=5, angvel_xyz=2) ---
    swingfoot_cost_weights: np.ndarray = field(
        default_factory=lambda: np.array([1e4, 1e4, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
    )
    # Swing-foot VERTICAL tracking (soft form of t1_controller's foot_constraint Z Baumgarte: it tracks
    # the SwingTrajectoryPlanner z spline with position gain 100 + velocity gain 10, XY emergent). Soft
    # weights strong enough to reliably lift the foot so the MPC can place it in XY emergently.
    swingfoot_z_weight: float = 1.0e3     # swing-z POSITION tracking
    swingfoot_vz_weight: float = 1.0e2    # swing-z VELOCITY tracking (drives the lift rate)
    # --- arm swing (SwitchedModelReferenceManager.cpp:136-143) ---
    arm_swing_amplitude: float = 0.15
    arm_swing_phase_offset: float = 0.15
    # --- velocity command caps (reference.info:1-43) ---
    max_vel_x: float = 1.0
    max_vel_y: float = 0.6
    max_yaw_rate: float = 1.0

    # --- friction / CoP / joint-limit barriers (task.info contacts:456-494) ---
    # contact_proj_eps / pin_rho: dormant config retained for future backends.
    contact_proj_eps: float = 1e-6
    pin_rho: float = 1.0          # ker(P)-confined nullspace pin weight (does not bias u_phys)
    friction_mu: float = 0.4
    friction_min_nforce: float = 10.0   # U6: min normal force per stance foot (keeps it loaded)
    friction_barrier_mu: float = 0.2
    friction_barrier_delta: float = 5.0
    friction_cone_reg: float = 25.0
    cop_barrier_mu: float = 0.1
    cop_barrier_delta: float = 0.03
    joint_limit_barrier_mu: float = 1200.0
    joint_limit_barrier_delta: float = 0.1
    # FootCollisionConstraint (faithful t1_controller port): keep the two legs' collision spheres apart so the
    # emergent swing foot can't collapse to the midline and step on the stance foot. minDist = 2*radius (foot 0.14,
    # knee 0.15); piecewise-poly barrier mu/delta; ACTIVE single-support only (off in double-stance).
    foot_collision_radius: float = 0.07
    knee_collision_radius: float = 0.075
    collision_barrier_mu: float = 1500.0
    collision_barrier_delta: float = 0.04
    foot_rect_x: Tuple[float, float] = (-0.1115, 0.1115)
    foot_rect_y: Tuple[float, float] = (-0.05, 0.05)  # FAITHFUL to t1_controller contact_rectangle (task.info:473-474).
    # NOTE: the CoP cone uses a QuadraticBarrier (penalises only OUTSIDE the rectangle, no interior
    # repulsion) because the exact OCS2 log-relaxed barrier is incompatible with the single-RTI DDP
    # solver. So the CoP can ride the +-0.05 edge; if foot-roll reappears in walking, shrinking this
    # half-width (the old +-0.035 workaround) buys edge margin. Holds clean in double support.

    # --- per-joint dynamics + PD ---
    kp: np.ndarray = field(default_factory=_build_kp)
    kd: np.ndarray = field(default_factory=_build_kd)
    armature: np.ndarray = field(default_factory=_build_armature)
    viscous_damping: np.ndarray = field(default_factory=_build_viscous_damping)

    # --- limits (27, head-excluded MPC order) ---
    # torque_limit = t1_controller's effortLimit (its t1.urdf), NOT the wb_humanoid URDF the dynamics
    # load from. The two URDFs are byte-identical EXCEPT effortLimit on 11 leg/waist/wrist joints
    # (knee 130 vs 60, ankle 60 vs 12, hip 130 vs 45, waist 90 vs 30, wrists/hands 18 vs 7). The torque
    # soft-cap is a faithfulness parameter (penalty ReLU(tau^2-lim^2)/lim^2), so it must mirror
    # t1_controller's, not the dynamics-URDF's — the wb limits over-penalize leg push-off up to 25x
    # (ankle lim^2 ratio). joint pos limits ARE identical between the URDFs, so still read them. (audit 2026-06-25)
    torque_limit: np.ndarray = field(default_factory=lambda: np.array(
        [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
         90, 130, 30, 30, 130, 60, 12, 130, 30, 30, 130, 60, 12], dtype=np.float64))
    joint_lower: np.ndarray = field(default_factory=lambda: _urdf_props()[1].copy())
    joint_upper: np.ndarray = field(default_factory=lambda: _urdf_props()[2].copy())

    # --- nominal posture / geometry ---
    nominal_joint_pos: np.ndarray = field(default_factory=_build_nominal_joint_pos)
    nominal_base_height: float = 0.6734   # task.info initialState p_base_z:149
    nominal_trunk_pitch: float = 0.0
    contact_frame_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.0, -0.030], dtype=np.float64)
    )  # t1_controller task.info value: puts the sole at world z~0 at nominal_base_height
    # (the foot-at-ground geometry the ZeroAccel constraint regulates to). The reused
    # model.py uses -0.027 for the tau_ff lever; the 3 mm differs only in that torque FF.

    mpc_joint_names: Tuple[str, ...] = MPC_JOINT_NAMES
    urdf_path: str = T1_URDF_PATH


def make_wb_config(**overrides) -> WBConfig:
    """Construct the canonical T1 whole-body MPCConfig, with optional overrides."""
    cfg = WBConfig(**overrides)
    _validate(cfg)
    return cfg


def _validate(cfg: WBConfig) -> None:
    assert (cfg.nx, cfg.nu, cfg.n_joints) == (68, 40, 27)
    assert abs(cfg.N * cfg.dt - 1.085) < 1e-9, "N*dt must be 1.085 (31 * 0.035)"
    for name, arr, n in (
        ("Q", cfg.Q, 68), ("R", cfg.R, 40), ("Q_final", cfg.Q_final, 68),
        ("kp", cfg.kp, 27), ("kd", cfg.kd, 27),
        ("armature", cfg.armature, 27), ("viscous_damping", cfg.viscous_damping, 27),
        ("torque_limit", cfg.torque_limit, 27),
        ("joint_lower", cfg.joint_lower, 27), ("joint_upper", cfg.joint_upper, 27),
        ("nominal_joint_pos", cfg.nominal_joint_pos, 27),
        ("contact_frame_offset", cfg.contact_frame_offset, 3),
        ("swingfoot_cost_weights", cfg.swingfoot_cost_weights, 7),
    ):
        assert arr.shape == (n,), f"{name} shape {arr.shape} != ({n},)"
        assert arr.dtype == np.float64, f"{name} dtype {arr.dtype} != float64"
    assert len(cfg.mpc_joint_names) == 27 and "AAHead_yaw" not in cfg.mpc_joint_names
    assert np.all(cfg.joint_lower < cfg.joint_upper), "joint_lower must be < joint_upper"
    assert np.all(cfg.torque_limit > 0)
