"""M1 faithfulness de-risk spike (THROWAWAY). Verify the two faithful-from-start mechanisms in
Crocoddyl before designing M1:
  A) custom RelaxedBarrier ActivationModel (mu*log(h+sqrt(h^2+delta^2))) used in a friction-cone cost.
  B) a HARD swing-foot z-velocity equality constraint added to ContactInvDynamics, enforced by SolverIntro
     (the SwingLegVerticalConstraint analog), leaving the swing foot's xy FREE (emergent placement).
Single-support node: LEFT stance, RIGHT swing.
"""
from __future__ import annotations
import numpy as np
import pinocchio as pin
import crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel

np.set_printoptions(precision=4, suppress=True, linewidth=140)
cfg = make_wb_config(); wb = WBModel(cfg); model = wb.model; data = model.createData()
L_fid, R_fid = wb.contact_fids
nv = model.nv
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
LWA = pin.LOCAL_WORLD_ALIGNED

q0 = pin.neutral(model); q0[2] = cfg.nominal_base_height
q0[6:6 + cfg.n_joints] = cfg.nominal_joint_pos
x0 = np.concatenate([q0, np.zeros(nv)])
pin.framesForwardKinematics(model, data, q0)
L_pl = pin.SE3(np.eye(3), data.oMf[L_fid].translation.copy())

# ---------------------------------------------------------------- A: RelaxedBarrier activation
class RelaxedBarrier(crocoddyl.ActivationModelAbstract):
    """OCS2 RelaxedBarrierPenalty: per element, h>=0 desired. value = -mu*log(h)  for h>delta,
    quadratic continuation for h<=delta. Penalizes constraint VIOLATION (h small/negative)."""
    def __init__(self, nr, mu, delta):
        crocoddyl.ActivationModelAbstract.__init__(self, nr)
        self.mu = float(mu); self.delta = float(delta)
    def calc(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        v = np.where(h > d, -mu * np.log(np.maximum(h, 1e-12)),
                     mu * (0.5 * ((h - 2 * d) / d) ** 2 - np.log(d)))
        data.a_value = float(np.sum(v))
    def calcDiff(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        dv = np.where(h > d, -mu / np.maximum(h, 1e-12), mu * (h - 2 * d) / d ** 2)
        d2 = np.where(h > d, mu / np.maximum(h, 1e-12) ** 2, mu / d ** 2 * np.ones_like(h))
        data.Ar = dv
        np.fill_diagonal(np.asarray(data.Arr), d2)

print("========== A: RelaxedBarrier activation in a friction-cone cost ==========")
try:
    nu_ss = nv + 6                                # single support: 1 contact
    cone = crocoddyl.FrictionCone(np.eye(3), float(cfg.friction_mu), 4, False)
    fric_res = crocoddyl.ResidualModelContactFrictionCone(state, L_fid, cone, nu_ss, False)
    rb = RelaxedBarrier(fric_res.nr, cfg.friction_barrier_mu, cfg.friction_barrier_delta)
    fric_cost = crocoddyl.CostModelResidual(state, rb, fric_res)   # exercised in-node in Part B
    print(f"[A] friction-cone residual nr={fric_res.nr}, RelaxedBarrier(mu={cfg.friction_barrier_mu},"
          f" delta={cfg.friction_barrier_delta}) built OK (cost object constructed)")
    okA_build = True
except Exception as e:
    import traceback; traceback.print_exc(); okA_build = False

# ---------------------------------------------------------------- B: hard swing-z velocity constraint
class SwingZVel(crocoddyl.ResidualModelAbstract):
    """1D residual = (swing foot LWA z-velocity) - vz_ref. Numerical Jacobian (spike: prove the mechanism)."""
    def __init__(self, state, fid, nu, vz_ref):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu, True, True, False)
        self.fid = int(fid); self.vz_ref = float(vz_ref)
        self._m = state.pinocchio; self._d = self._m.createData(); self._nq = state.nq; self._nv = state.nv
    def calc(self, data, x, u=None):
        q = np.asarray(x[:self._nq]); v = np.asarray(x[self._nq:])
        pin.forwardKinematics(self._m, self._d, q, v); pin.updateFramePlacements(self._m, self._d)
        data.r[0] = float(pin.getFrameVelocity(self._m, self._d, self.fid, LWA).linear[2]) - self.vz_ref
    def calcDiff(self, data, x, u=None):
        q = np.asarray(x[:self._nq]); v = np.asarray(x[self._nq:])
        pin.computeForwardKinematicsDerivatives(self._m, self._d, q, v, np.zeros(self._nv))
        vpq, vpv = pin.getFrameVelocityDerivatives(self._m, self._d, self.fid, LWA)
        Rx = np.concatenate([np.asarray(vpq)[2, :], np.asarray(vpv)[2, :]])   # d(z-vel)/d[q,v]
        if data.Rx.ndim == 1:
            data.Rx[:] = Rx
        else:
            data.Rx[0, :] = Rx

print("\n========== B: hard swing-foot-z-velocity constraint + SolverIntro ==========")
try:
    DT, N = cfg.dt, 8
    foot_pl = {L_fid: L_pl}
    gains = np.array([cfg.foot_pos_err_gain_z, cfg.foot_linvel_err_gain_xy])

    def make_ss_node(terminal=False):
        nu = nv + 6
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        contacts.addContact("0_cL", crocoddyl.ContactModel6D(state, L_fid, L_pl, LWA, nu, gains))
        costs = crocoddyl.CostModelSum(state, nu)
        xreg = crocoddyl.ResidualModelState(state, x0, nu)
        costs.addCost("xreg", crocoddyl.CostModelResidual(
            state, crocoddyl.ActivationModelWeightedQuad(cfg.Q[:66]), xreg), 1.0)
        costs.addCost("ureg", crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)), 1e-3)
        # faithful friction cost (relaxed barrier) on the stance foot
        cone = crocoddyl.FrictionCone(np.eye(3), float(cfg.friction_mu), 4, False)
        fr = crocoddyl.ResidualModelContactFrictionCone(state, L_fid, cone, nu, False)
        costs.addCost("fric", crocoddyl.CostModelResidual(
            state, RelaxedBarrier(fr.nr, cfg.friction_barrier_mu, cfg.friction_barrier_delta), fr), 1.0)
        # HARD swing-z constraint on the RIGHT (swinging) foot
        cons = crocoddyl.ConstraintModelManager(state, nu)
        cons.addConstraint("swingz", crocoddyl.ConstraintModelResidual(
            state, SwingZVel(state, R_fid, nu, vz_ref=0.05)))     # lift at 0.05 m/s (faithful LIFTOFF_VEL)
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(state, actuation, contacts, costs, cons)
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else DT), dam

    run0, dam0 = make_ss_node()
    print(f"[B] single-support node: nu={dam0.nu} nh(eq incl swingz)={dam0.nh} ng={dam0.ng}")
    running = [make_ss_node()[0] for _ in range(N)]
    term, _ = make_ss_node(terminal=True)
    prob = crocoddyl.ShootingProblem(x0, running, term)
    solver = crocoddyl.SolverIntro(prob); solver.setCallbacks([])
    xs = [x0.copy() for _ in range(N + 1)]; us = list(prob.quasiStatic([x0.copy() for _ in range(N)]))
    solver.solve(xs, us, 50, False, 1e-9)
    xs = np.asarray(solver.xs)
    # check the swing-z constraint is satisfied: right-foot z-velocity ~ 0.05 along the trajectory
    zerr = []
    for x in xs:
        q = x[:model.nq]; v = x[model.nq:]
        pin.forwardKinematics(model, data, q, v); pin.updateFramePlacements(model, data)
        zerr.append(pin.getFrameVelocity(model, data, R_fid, LWA).linear[2] - 0.05)
    # swing foot xy should be FREE (moved): check right-foot xy displacement
    pin.framesForwardKinematics(model, data, xs[0][:model.nq]); r0 = data.oMf[R_fid].translation.copy()
    pin.framesForwardKinematics(model, data, xs[-1][:model.nq]); r1 = data.oMf[R_fid].translation.copy()
    print(f"[B] converged stop={solver.stoppingCriteria():.2e}  finite={np.all(np.isfinite(xs))}")
    print(f"[B] swing-z constraint residual: max|vz-0.05|={max(abs(np.array(zerr))):.2e}  (want ~0 = ENFORCED)")
    print(f"[B] swing foot moved: dz={r1[2]-r0[2]*1:.3f}m (lifts)  dxy={np.linalg.norm(r1[:2]-r0[:2]):.3f}m (xy free)")
    okB = np.all(np.isfinite(xs)) and max(abs(np.array(zerr))) < 1e-2
except Exception as e:
    import traceback; traceback.print_exc(); okB = False

print("\n========== M1 FAITHFUL-MECHANISM VERDICT ==========")
print(f"  A RelaxedBarrier activation builds : {'PASS' if okA_build else 'FAIL'}")
print(f"  B hard swing-z constraint enforced : {'PASS' if okB else 'FAIL'}")
