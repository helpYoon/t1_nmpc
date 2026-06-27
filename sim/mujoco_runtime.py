"""MuJoCo closed-loop runtime for t1_nmpc.

Decoupled rates: physics @ 2000 Hz, PD control @ 500 Hz, MPC @ 40 Hz (latest-policy
zero-order-hold). Mirrors wb_humanoid robot_runtime/mujoco_sim_interface but as a
single-process deterministic interleaved scheduler (gate path; threaded path reserved).

MuJoCo model facts (t1.xml): nq=36 nv=35 nu=29. Base free joint:
qpos[0:3]=p, qpos[3:7]=quat(w,x,y,z); qvel[0:3]=lin WORLD, qvel[3:6]=ang LOCAL.
Joints 1..29 (the §A.5 order) live at qpos[7:36] / qvel[6:35]. The Waist motor was ADDED to t1.xml
(matching t1_controller's 2026-05-30 bugfix: an unactuated Waist lets the trunk yaw freely -> walk
collapse); the MPC's planned Waist torque is now applied (29 actuators), mapped by name like the rest.
"""
from __future__ import annotations

import numpy as np
import mujoco
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig

T1_MJCF_PATH = ("/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/"
                "robot_models/booster_t1/t1_description/urdf/t1.xml")

# MuJoCo base/joint layout offsets (free joint = 7 qpos / 6 qvel)
MJ_JOINT_QPOS0 = 7   # joint angles start here in data.qpos
MJ_JOINT_QVEL0 = 6   # joint velocities start here in data.qvel

# Real-robot system identification: per-joint (armature [kg m^2], viscous damping [N m s/rad]).
# t1.xml ships these ALL ZERO, but the MPC tau_ff already folds in the same armature + viscous_damping
# (config._build_armature/_build_viscous_damping) -> tau_ff over-commanded the sim -> wrong GRF.
# Setting them on the sim closes both the sim-to-real gap AND the tau_ff/sim mismatch. Wrists reuse the
# arm value, per the T1 sysID. (Dry friction/frictionloss is intentionally NOT set: the MPC does not
# model it and it can cause sticking -- a separate sim-to-real step.)
_ARM_AW = 0.0282528   # arm + wrist (ARMATURE_4310)
_SYSID_DYN = {
    "AAHead_yaw":           (0.01,      0.5),
    "Head_pitch":           (0.01,      0.5),
    "Left_Shoulder_Pitch":  (_ARM_AW,   0.5),
    "Left_Shoulder_Roll":   (_ARM_AW,   0.5),
    "Left_Elbow_Pitch":     (_ARM_AW,   0.5),
    "Left_Elbow_Yaw":       (_ARM_AW,   0.5),
    "Left_Wrist_Pitch":     (_ARM_AW,   0.5),
    "Left_Wrist_Yaw":       (_ARM_AW,   0.5),
    "Left_Hand_Roll":       (_ARM_AW,   0.5),
    "Right_Shoulder_Pitch": (_ARM_AW,   0.5),
    "Right_Shoulder_Roll":  (_ARM_AW,   0.5),
    "Right_Elbow_Pitch":    (_ARM_AW,   0.5),
    "Right_Elbow_Yaw":      (_ARM_AW,   0.5),
    "Right_Wrist_Pitch":    (_ARM_AW,   0.5),
    "Right_Wrist_Yaw":      (_ARM_AW,   0.5),
    "Right_Hand_Roll":      (_ARM_AW,   0.5),
    "Waist":                (0.0478125, 0.528635),
    "Left_Hip_Pitch":       (0.0523908, 0.266903),
    "Left_Hip_Roll":        (0.0478125, 0.52201),
    "Left_Hip_Yaw":         (0.0478125, 0.76929),
    "Left_Knee_Pitch":      (0.0636012, 0.161472),
    "Left_Ankle_Pitch":     (0.0407621, 1.43965),
    "Left_Ankle_Roll":      (0.0111713, 0.0821329),
    "Right_Hip_Pitch":      (0.0523908, 0.266903),
    "Right_Hip_Roll":       (0.0478125, 0.52201),
    "Right_Hip_Yaw":        (0.0478125, 0.76929),
    "Right_Knee_Pitch":     (0.0636012, 0.161472),
    "Right_Ankle_Pitch":    (0.0407621, 1.43965),
    "Right_Ankle_Roll":     (0.0111713, 0.0821329),
}

# Real-robot dry-friction sysID: per-joint (static [N m], dynamic/kinetic [N m]). MuJoCo's native
# dof_frictionloss is a SINGLE Coulomb value (can't do static != dynamic), so we apply a custom
# Stribeck force instead: |tau| = Fc + (Fs-Fc)*exp(-(v/v_s)^2), opposing motion (smoothed by tanh).
# Wrists reuse the arm value, per the sysID. (Viscous friction is already dof_damping in _SYSID_DYN.)
_SYSID_FRICTION = {
    "AAHead_yaw":           (0.5,      0.0),
    "Head_pitch":           (0.5,      0.0),
    "Left_Shoulder_Pitch":  (0.5,      0.0),
    "Left_Shoulder_Roll":   (0.5,      0.0),
    "Left_Elbow_Pitch":     (0.5,      0.0),
    "Left_Elbow_Yaw":       (0.5,      0.0),
    "Left_Wrist_Pitch":     (0.5,      0.0),
    "Left_Wrist_Yaw":       (0.5,      0.0),
    "Left_Hand_Roll":       (0.5,      0.0),
    "Right_Shoulder_Pitch": (0.5,      0.0),
    "Right_Shoulder_Roll":  (0.5,      0.0),
    "Right_Elbow_Pitch":    (0.5,      0.0),
    "Right_Elbow_Yaw":      (0.5,      0.0),
    "Right_Wrist_Pitch":    (0.5,      0.0),
    "Right_Wrist_Yaw":      (0.5,      0.0),
    "Right_Hand_Roll":      (0.5,      0.0),
    "Waist":                (0.459068, 0.247026),
    "Left_Hip_Pitch":       (0.486176, 0.480596),
    "Left_Hip_Roll":        (0.880781, 0.626236),
    "Left_Hip_Yaw":         (0.238063, 0.213108),
    "Left_Knee_Pitch":      (0.998922, 0.722987),
    "Left_Ankle_Pitch":     (0.71953,  2.88e-5),
    "Left_Ankle_Roll":      (0.209926, 0.0226177),
    "Right_Hip_Pitch":      (0.486176, 0.480596),
    "Right_Hip_Roll":       (0.880781, 0.626236),
    "Right_Hip_Yaw":        (0.238063, 0.213108),
    "Right_Knee_Pitch":     (0.998922, 0.722987),
    "Right_Ankle_Pitch":    (0.71953,  2.88e-5),
    "Right_Ankle_Roll":     (0.209926, 0.0226177),
}


def _quat_wxyz_to_zyx_euler(qw, qx, qy, qz) -> np.ndarray:
    """MuJoCo (w,x,y,z) world quat -> ZYX-intrinsic Euler (theta_z, theta_y, theta_x)."""
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    # ZYX intrinsic: yaw=atan2(R10,R00), pitch=asin(-R20), roll=atan2(R21,R22)
    theta_z = np.arctan2(R[1, 0], R[0, 0])
    theta_y = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    return np.array([theta_z, theta_y, theta_x], dtype=np.float64)


def _euler_zyx_rates_from_local_angvel(euler_zyx, w_local) -> np.ndarray:
    """ZYX-Euler angle rates from body-local angular velocity (matches Pinocchio v_pin[3:6])."""
    tz, ty, tx = euler_zyx
    cy = np.cos(ty)
    # ponytail: gimbal guard; walking never hits ty=+-pi/2 (accepted, same as wb)
    cy = np.sign(cy) * max(abs(cy), 1e-6)
    # E maps local angular velocity -> [thetadot_z, thetadot_y, thetadot_x]
    sx, cx = np.sin(tx), np.cos(tx)
    E = np.array([
        [0.0, sx / cy, cx / cy],
        [0.0, cx, -sx],
        [1.0, sx * np.tan(ty), cx * np.tan(ty)],
    ], dtype=np.float64)
    return E @ np.asarray(w_local, dtype=np.float64)


class MujocoRuntime:
    def __init__(self, cfg: MPCConfig, model, mjcf_path: str = T1_MJCF_PATH,
                 apply_joint_friction: bool = False):
        self.cfg = cfg
        self.robot_model = model               # t1_nmpc RobotModel (Pinocchio); may be None for layout-only tests
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = 1.0 / cfg.physics_hz   # 0.0005
        self._apply_sysid_dynamics()                        # real armature + viscous damping (xml ships 0)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.physics_dt = 1.0 / cfg.physics_hz
        self.control_decim = int(round(cfg.physics_hz / cfg.control_hz))   # 4
        self.mpc_decim = int(round(cfg.physics_hz / cfg.mpc_hz))           # 50

        self._build_actuator_map()

        # Custom Stribeck dry-friction (static+dynamic sysID). OFF by default: it is a disturbance the
        # MPC tau_ff does NOT model, so it stays out of the walk-control baseline; flip on for sim-to-real.
        self.apply_joint_friction = apply_joint_friction
        self.stribeck_vel = 0.05            # rad/s, static->kinetic transition width (sysID/HW tunable)
        self.friction_smooth_vel = 0.01     # rad/s, tanh smoothing so there's no v=0 chatter
        self._build_joint_friction()

        self.qpos_trace = []
        self.t = 0.0

    # ---- actuator mapping (29 actuators -> joint-local idx; Waist actuated) ----
    def _build_actuator_map(self):
        # State joint-local order is the §A.5 order == MuJoCo joints 1..29 order.
        mj_joint_names = []
        for j in range(1, self.mj_model.njnt):    # skip free joint 0
            nm = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            mj_joint_names.append(nm)
        name_to_local = {nm: i for i, nm in enumerate(mj_joint_names)}
        self.act_to_state_idx = []
        for a in range(self.mj_model.nu):
            anm = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self.act_to_state_idx.append(name_to_local[anm])

    def _apply_sysid_dynamics(self):
        """Stamp the real-robot armature + viscous damping (_SYSID_DYN) onto the MuJoCo dofs by joint
        name, so the sim plant matches both reality and the MPC tau_ff. t1.xml ships them zero."""
        for nm, (arm, damp) in _SYSID_DYN.items():
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid < 0:
                raise ValueError(f"sysid joint '{nm}' not found in MuJoCo model")
            adr = self.mj_model.jnt_dofadr[jid]
            self.mj_model.dof_armature[adr] = arm
            self.mj_model.dof_damping[adr] = damp

    def _build_joint_friction(self):
        """Per-dof (static, dynamic) dry-friction arrays from _SYSID_FRICTION (base dofs -> 0)."""
        nv = self.mj_model.nv
        self._fric_static = np.zeros(nv)
        self._fric_dynamic = np.zeros(nv)
        for nm, (fs, fc) in _SYSID_FRICTION.items():
            adr = self.mj_model.jnt_dofadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)]
            self._fric_static[adr] = fs
            self._fric_dynamic[adr] = fc

    def _joint_friction_torque(self) -> np.ndarray:
        """Stribeck dry friction opposing joint motion: |tau| = Fc + (Fs-Fc)*exp(-(v/v_s)^2),
        signed by tanh(v/v_t) (smooth through v=0). Returns an nv-vector (base dofs = 0)."""
        v = self.mj_data.qvel
        mag = self._fric_dynamic + (self._fric_static - self._fric_dynamic) * np.exp(-(v / self.stribeck_vel) ** 2)
        return -mag * np.tanh(v / self.friction_smooth_vel)

    # ---- reset / state ----
    def reset_to_nominal(self):
        """Place the robot standing with its feet ON the floor.

        `nominal_base_height` (0.62) is the MPC *reference* height, NOT the physical feet-on-floor
        height of the crouch (~0.66). Spawning the base at 0.62 sinks the feet ~4-5 cm through the
        floor (z=0), and MuJoCo's contact response launches the robot into the air (it flips → the
        solver fails on tick 1). Instead we spawn ABOVE the floor and let gravity settle the feet
        onto it under a joint-PD hold — robust for any joint config, and the settled stand matches
        the MPC's crouch reference.
        """
        q0 = MJ_JOINT_QPOS0
        njp = np.asarray(self.cfg.nominal_joint_pos, dtype=np.float64)
        kp = np.asarray(self.cfg.kp, dtype=np.float64)
        kd = np.asarray(self.cfg.kd, dtype=np.float64)
        self.mj_data.qpos[:] = 0.0
        self.mj_data.qvel[:] = 0.0
        self.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quat (w,x,y,z)
        self.mj_data.qpos[q0:q0 + 29] = njp
        self.mj_data.qpos[2] = self.cfg.nominal_base_height + 0.10   # spawn clearly above the floor
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for _ in range(int(round(0.6 * self.cfg.physics_hz))):       # ~0.6 s physics-only settle
            q = np.array(self.mj_data.qpos[q0:q0 + 29])
            qd = np.array(self.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
            self._apply_torque(kp * (njp - q) - kd * qd)            # PD hold to nominal joints
            self.step_physics()
        self.mj_data.qvel[:] = 0.0                                   # drop residual settle velocity
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.t = 0.0
        self.qpos_trace = [np.array(self.mj_data.qpos)]

    def _pin_q_v(self):
        """Pinocchio generalized coords/vels (35,) from MuJoCo: q=[pos,euler_zyx,joints],
        v=[lin_world, euler_zyx_rates, joint_vel]. Shared by state_estimate + the ID feedforward."""
        d = self.mj_data
        euler = _quat_wxyz_to_zyx_euler(d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6])
        q_j = np.array(d.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
        qd_j = np.array(d.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
        q_pin = np.concatenate([np.array(d.qpos[0:3]), euler, q_j])
        # base lin vel: MuJoCo free-joint qvel[0:3] is WORLD linear vel == v_pin[0:3] (NO rotation)
        # base ang: ZYX-Euler rates from body-local angular velocity
        v_pin = np.concatenate([np.array(d.qvel[0:3]),
                                _euler_zyx_rates_from_local_angvel(euler, d.qvel[3:6]), qd_j])
        return q_pin, v_pin

    def freeflyer_state(self, pin_model):
        """Measured FreeFlyer x[71] for the whole_body_rnea MPC (NOT the euler _pin_q_v)."""
        from t1_nmpc.wb.state import mujoco_to_freeflyer
        return mujoco_to_freeflyer(self.mj_data.qpos, self.mj_data.qvel, pin_model)

    # ---- control ----
    def _apply_torque(self, tau29: np.ndarray):
        for a, j in enumerate(self.act_to_state_idx):
            self.mj_data.ctrl[a] = tau29[j]

    def step_physics(self):
        if self.apply_joint_friction:
            self.mj_data.qfrc_applied[:] = self._joint_friction_torque()
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.t = float(self.mj_data.time)

