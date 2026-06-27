"""WBModel: whole-body RBD for the T1 via pinocchio.casadi (cpin).

Float base = JointModelComposite(Translation + SphericalZYX), identical to the
reference (t1_controller createPinocchioModel.cpp:64-65) and to t1_nmpc/model.py.
With this base the MPC state maps DIRECTLY to pinocchio coords — no quaternion
or LOCAL-frame conversion:

    pin q (33) = [pos(3), theta_zyx(3), q_joints(27)]   == state[0:33]
    pin v (33) = [v_world(3), eulerrate_zyx(3), v_joints(27)] == state[33:66]

(SphericalZYX's tangent IS the euler-angle rate, hence the reference's
"euler derivatives, not angular velocity" base velocity.)

Provides CasADi Functions M(q), nle(q,v), Jl(q), Jr(q) (contact Jacobians at
foot_l/r_contact, LOCAL_WORLD_ALIGNED) + numeric wrappers and pin oracles.
"""
from __future__ import annotations

import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from t1_nmpc.robot.model import CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS
from .config_wb import WBConfig, MPC_JOINT_NAMES

_HEAD_JOINTS = ("AAHead_yaw", "Head_pitch")


def _build_base() -> "pin.JointModelComposite":
    root = pin.JointModelComposite()
    root.addJoint(pin.JointModelTranslation())
    root.addJoint(pin.JointModelSphericalZYX())
    return root


def _symm(M):
    """CRBA fills only the upper triangle; mirror to a full symmetric matrix."""
    if isinstance(M, np.ndarray):
        return np.triu(M) + np.triu(M, 1).T
    return cs.triu(M) + cs.triu(M, False).T  # casadi triu(x, include_diagonal); False = strict upper


class WBModel:
    def __init__(self, cfg: WBConfig):
        self.cfg = cfg
        full = pin.buildModelFromUrdf(cfg.urdf_path, _build_base())
        lock = [full.getJointId(n) for n in _HEAD_JOINTS]
        model = pin.buildReducedModel(full, lock, pin.neutral(full))
        assert tuple(model.names[2:]) == MPC_JOINT_NAMES, model.names[2:]
        # reflected rotor inertia on the 27 actuated joints (base dofs 0..5 untouched)
        model.armature[6:] = np.asarray(cfg.armature, dtype=np.float64)

        offset = pin.SE3.Identity()
        offset.translation = np.ascontiguousarray(cfg.contact_frame_offset, dtype=np.float64)
        self.contact_fids = []
        for fname, parent in zip(CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS):
            jid = model.getJointId(parent)
            self.contact_fids.append(
                model.addFrame(pin.Frame(fname, jid, offset, pin.FrameType.OP_FRAME))
            )

        self.model = model
        self.data = model.createData()
        self.nq = model.nq  # 33
        self.nv = model.nv  # 33
        self._mass = float(sum(I.mass for I in model.inertias))

        # --- symbolic cpin functions ---
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        cq = cs.SX.sym("q", model.nq)
        cv = cs.SX.sym("v", model.nv)

        cpin.crba(cmodel, cdata, cq)
        Msx = _symm(cdata.M)
        nle = cpin.nonLinearEffects(cmodel, cdata, cq, cv)
        cpin.computeJointJacobians(cmodel, cdata, cq)
        cpin.updateFramePlacements(cmodel, cdata)
        Jl = cpin.getFrameJacobian(cmodel, cdata, self.contact_fids[0], pin.LOCAL_WORLD_ALIGNED)
        Jr = cpin.getFrameJacobian(cmodel, cdata, self.contact_fids[1], pin.LOCAL_WORLD_ALIGNED)

        self.M_fun = cs.Function("M", [cq], [Msx])
        self.nle_fun = cs.Function("nle", [cq, cv], [nle])
        self.Jl_fun = cs.Function("Jl", [cq], [Jl])
        self.Jr_fun = cs.Function("Jr", [cq], [Jr])

        # --- per-foot kinematics for the ZeroAcceleration / CoP constraints (LWA frame) ---
        # twist = getFrameVelocity (J v), accel = getFrameClassicalAcceleration (J a + drift),
        # pose = [world position(3), small-angle orientation-error-wrt-ground(3)], Rf = foot rotation.
        ca = cs.SX.sym("a", model.nv)
        cpin.forwardKinematics(cmodel, cdata, cq, cv, ca)
        cpin.updateFramePlacements(cmodel, cdata)
        self.foot_kin_fun = []
        self.foot_R_fun = []
        for fid in self.contact_fids:
            velm = cpin.getFrameVelocity(cmodel, cdata, fid, pin.LOCAL_WORLD_ALIGNED)
            accm = cpin.getFrameClassicalAcceleration(cmodel, cdata, fid, pin.LOCAL_WORLD_ALIGNED)
            twist = cs.vertcat(velm.linear, velm.angular)
            accel = cs.vertcat(accm.linear, accm.angular)
            oMf = cdata.oMf[fid]
            Rf = oMf.rotation
            nfz = Rf @ cs.DM([0, 0, 1])           # foot z-axis in world
            orient_err = cs.vertcat(nfz[1], -nfz[0], cs.SX(0.0))   # small-angle dist-to-ground-plane
            pose = cs.vertcat(oMf.translation, orient_err)
            self.foot_kin_fun.append(cs.Function("footkin", [cq, cv, ca], [twist, accel, pose]))
            self.foot_R_fun.append(cs.Function("footR", [cq], [Rf]))

        # --- collision points for the faithful FootCollisionConstraint port (t1_controller) ---
        # 3x10 world positions. Foot corners = foot_center + R_foot @ [+/-corner_off, 0, 0], corner_off =
        # 0.6 * foot_rect_x_max (createPinocchioModel.cpp::addCollisionCenterFrames). Column order:
        # [footc_l, foot_l_p1(+x), foot_l_p2(-x), footc_r, foot_r_p1(+x), foot_r_p2(-x), ankle_l, ankle_r, knee_l, knee_r].
        corner_off = 0.6 * float(cfg.foot_rect_x[1])
        coll_cols = []
        for fid in self.contact_fids:                       # contact_fids = [left, right]
            ctr = cdata.oMf[fid].translation; Rc = cdata.oMf[fid].rotation
            coll_cols += [ctr, ctr + Rc @ cs.DM([corner_off, 0.0, 0.0]),
                               ctr + Rc @ cs.DM([-corner_off, 0.0, 0.0])]
        for n in ("Left_Ankle_Roll", "Right_Ankle_Roll", "Left_Knee_Pitch", "Right_Knee_Pitch"):
            coll_cols.append(cdata.oMf[model.getFrameId(n)].translation)
        self.collision_pts_fun = cs.Function("collpts", [cq], [cs.horzcat(*coll_cols)])   # 3x10

        # numeric flow map + joint torque, f(x,u) for tests / rollout / execution
        xsx = cs.SX.sym("x", 68)
        usx = cs.SX.sym("u", 40)
        self._flow_fun = cs.Function("flow", [xsx, usx], [self.flow_expr(xsx, usx)])
        self._tau_fun = cs.Function("tau", [xsx, usx], [self.joint_torque_expr(xsx, usx)])

    # --- shared dynamics terms (block-diagonal RBD inversion) ---
    def _dyn_terms(self, x, u):
        """Returns (v, qdd_j, M, nle, tau_ext, vdot_base) as CasADi exprs.

        Base accel: the lin (M[:3,:3]) and ang (M[3:6,3:6]) 3x3 blocks are inverted
        SEPARATELY, dropping the lin-ang off-diagonal coupling
        (DynamicsHelperFunctions.cpp:205-215; deliberate)."""
        nq = self.nq  # 33
        q = x[0:nq]
        v = x[nq:2 * nq]              # state[33:66] = [v_base(6), v_joints(27)]
        qdd_j = u[12:39]
        M = self.M_fun(q)
        nle = self.nle_fun(q, v)
        Jl = self.Jl_fun(q)
        Jr = self.Jr_fun(q)
        tau_ext = cs.mtimes(Jl.T, u[0:6]) + cs.mtimes(Jr.T, u[6:12])   # (33,) generalized contact force
        inter = tau_ext[0:6] - nle[0:6] - cs.mtimes(M[0:6, 6:nq], qdd_j)
        vdot_lin = cs.solve(M[0:3, 0:3], inter[0:3])
        vdot_ang = cs.solve(M[3:6, 3:6], inter[3:6])
        return v, qdd_j, M, nle, tau_ext, cs.vertcat(vdot_lin, vdot_ang)

    def flow_expr(self, x, u):
        """xdot = f(x,u), x in R^68, u in R^40 (CasADi expression)."""
        v, qdd_j, _M, _nle, _te, vdot_base = self._dyn_terms(x, u)
        # dq/dt = v (exact: euler base + revolute joints), then accels, then path slots
        return cs.vertcat(v, vdot_base, qdd_j, x[67], u[39])

    def joint_torque_expr(self, x, u):
        """tau on the 27 actuated joints (CasADi): inverse dynamics of the planned
        (x,u) restricted to joint rows, plus viscous damping. = the reference
        computeJointTorques: (M*accel + nle - J^T W)[6:] + viscous_damping*v_joints."""
        nq = self.nq
        v, qdd_j, M, nle, tau_ext, vdot_base = self._dyn_terms(x, u)
        accel = cs.vertcat(vdot_base, qdd_j)                  # (33,)
        tau_full = cs.mtimes(M, accel) + nle - tau_ext
        visc = cs.DM(np.asarray(self.cfg.viscous_damping)) * v[6:nq]
        return tau_full[6:nq] + visc

    def flow(self, x, u) -> np.ndarray:
        return np.asarray(self._flow_fun(np.asarray(x, float), np.asarray(u, float))).ravel()

    def joint_torque(self, x, u) -> np.ndarray:
        return np.asarray(self._tau_fun(np.asarray(x, float), np.asarray(u, float))).ravel()

    def nominal_state(self) -> np.ndarray:
        """Stand: base at nominal height, nominal joint posture, zero velocity, s=v_s=0."""
        x = np.zeros(68, dtype=np.float64)
        x[2] = self.cfg.nominal_base_height
        x[6:33] = self.cfg.nominal_joint_pos
        return x

    # --- accessors / numeric wrappers ---
    def neutral_q(self) -> np.ndarray:
        return pin.neutral(self.model)

    def total_mass(self) -> float:
        return self._mass

    def M(self, q) -> np.ndarray:
        return np.asarray(self.M_fun(q))

    def nle(self, q, v) -> np.ndarray:
        return np.asarray(self.nle_fun(q, v)).ravel()

    def Jl(self, q) -> np.ndarray:
        return np.asarray(self.Jl_fun(q))

    def Jr(self, q) -> np.ndarray:
        return np.asarray(self.Jr_fun(q))

    # --- numeric pinocchio oracles (for tests) ---
    def M_numeric_pin(self, q) -> np.ndarray:
        M = pin.crba(self.model, self.data, np.asarray(q, dtype=np.float64))
        return _symm(M)

    def nle_numeric_pin(self, q, v) -> np.ndarray:
        return pin.nonLinearEffects(
            self.model, self.data, np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
        )
