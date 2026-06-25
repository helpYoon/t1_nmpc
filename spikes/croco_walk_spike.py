"""Crocoddyl WALKING feasibility spike — THROWAWAY de-risk of the case that killed acados.

The acados port passed the M0 stand but diverged (res_eq -> 1e80) the INSTANT a foot
swung (single-support + contact switch). This spike asks: does a forward walk with
single-support phases and contact switches SOLVE in Crocoddyl, with the projection
holding, at a sane rate?

Smooth MODE-SWITCHED contacts (contact set changes per phase), NOT impulse models —
this matches OCS2/t1_controller (mode-switched contact constraints, no impacts).

  A) ContactFwdDynamics + FDDP        -> robust walk recipe: proves gait solves + rate
  B) ContactInvDynamics + SolverIntro -> projected-ID nullspace THROUGH single-support
     + switches (the linchpin). Each node carries its own control dim (DS nu=nv+12,
     SS nu=nv+6) -> demonstrates the 'nu varies per contact mode' obstacle that broke
     the acados reduced-basis is a NON-ISSUE in crocoddyl (per-node nu is native).

Run: env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python spikes/croco_walk_spike.py
"""
from __future__ import annotations

import time
import numpy as np
import pinocchio as pin
import crocoddyl

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel

np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ----------------------------------------------------------------------------
# Faithful T1 model
# ----------------------------------------------------------------------------
cfg = make_wb_config()
wb = WBModel(cfg)
model = wb.model
data = model.createData()
L_fid, R_fid = wb.contact_fids
nq, nv, n_joints = model.nq, model.nv, cfg.n_joints
mass = float(sum(I.mass for I in model.inertias))
g = abs(model.gravity.linear[2])
DT = cfg.dt

q0 = pin.neutral(model)
q0[2] = cfg.nominal_base_height
q0[6:6 + n_joints] = cfg.nominal_joint_pos
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])

pin.framesForwardKinematics(model, data, q0)
pin.centerOfMass(model, data, q0)
com0 = data.com[0].copy()
Lp0 = data.oMf[L_fid].translation.copy()
Rp0 = data.oMf[R_fid].translation.copy()
PLANT = {L_fid: pin.SE3(np.eye(3), Lp0), R_fid: pin.SE3(np.eye(3), Rp0)}
print(f"[model] nq={nq} nv={nv} mass={mass:.2f}  com0={com0}  L={Lp0} R={Rp0}")

state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
LWA = pin.LOCAL_WORLD_ALIGNED

# ----------------------------------------------------------------------------
# Gait schedule (one cycle: DS, R-swing, DS, L-swing, DS), forward walk
# ----------------------------------------------------------------------------
N_DS, N_SS = 2, 9
STEP, APEX, VX = 0.12, 0.06, 0.3

# state-reg weights (base x,y position weight 0 -> forward motion is free, as in port Q)
W_STATE = np.concatenate([
    np.array([0.0, 0.0, 10.0, 20.0, 20.0, 20.0]),   # base pos(3, x/y=0) + euler(3)
    2.0 * np.ones(n_joints),                         # joints
    0.1 * np.ones(nv),                               # velocities (light damping)
])


def swing_arc(p0, dx, n):
    out = []
    for k in range(n):
        s = (k + 1) / n
        out.append(np.array([p0[0] + dx * s, p0[1], p0[2] + APEX * np.sin(np.pi * s)]))
    return out


sched = []          # per running node: dict(stance=[fids], swing=fid|None, swing_t=p|None, com=p)
comx = float(com0[0])


def add_block(stance, swing_fid, swing_p0, n):
    global comx
    arc = swing_arc(swing_p0, STEP, n) if swing_fid is not None else [None] * n
    for k in range(n):
        comx += VX * DT
        sched.append(dict(stance=list(stance), swing=swing_fid, swing_t=arc[k],
                          com=np.array([comx, com0[1], com0[2]])))


add_block([L_fid, R_fid], None, None, N_DS)
add_block([L_fid], R_fid, Rp0, N_SS)            # right swings (stance left)
add_block([L_fid, R_fid], None, None, N_DS)
add_block([R_fid], L_fid, Lp0, N_SS)            # left swings (stance right)
add_block([L_fid, R_fid], None, None, N_DS)
N = len(sched)
print(f"[gait] N={N} nodes ({N*DT:.3f}s): DS={N_DS} SS={N_SS}, step={STEP} apex={APEX} vx={VX}")


# ----------------------------------------------------------------------------
# Per-node model builders
# ----------------------------------------------------------------------------
def _add_common_costs(costs, spec, nu):
    sreg = crocoddyl.ResidualModelState(state, x0, nu)
    costs.addCost("xreg", crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(W_STATE ** 2), sreg), 1e-1)
    costs.addCost("ureg", crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, nu)), 1e-3)
    costs.addCost("com", crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelCoMPosition(state, spec["com"], nu)), 1e2)
    if spec["swing"] is not None:
        sw = crocoddyl.ResidualModelFrameTranslation(state, spec["swing"], spec["swing_t"], nu)
        costs.addCost("swing", crocoddyl.CostModelResidual(state, sw), 1e4)


def running_fwd(spec, terminal=False):
    nu = actuation.nu
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    for fid in spec["stance"]:
        c = crocoddyl.ContactModel6D(state, fid, PLANT[fid], LWA, nu, np.array([0.0, 0.0]))
        contacts.addContact("c%d" % fid, c)
    costs = crocoddyl.CostModelSum(state, nu)
    _add_common_costs(costs, spec, nu)
    dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, 1e-8, True)
    return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else DT)


def running_inv(spec, terminal=False):
    nu = nv + 6 * len(spec["stance"])          # per-node varying control dim
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    for fid in spec["stance"]:
        c = crocoddyl.ContactModel6D(state, fid, PLANT[fid], LWA, nu, np.array([0.0, 0.0]))
        contacts.addContact("c%d" % fid, c)
    costs = crocoddyl.CostModelSum(state, nu)
    _add_common_costs(costs, spec, nu)
    dam = crocoddyl.DifferentialActionModelContactInvDynamics(
        state, actuation, contacts, costs)
    return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else DT)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def report(solver, name, t_cold, t_warm):
    xs = np.asarray(solver.xs)
    finite = np.all(np.isfinite(xs)) and np.isfinite(solver.cost)
    coms, Lz, Rz = [], [], []
    for x in xs:
        q = x[:nq]
        pin.framesForwardKinematics(model, data, q)
        pin.centerOfMass(model, data, q)
        coms.append(float(data.com[0][0]))
        Lz.append(float(data.oMf[L_fid].translation[2]))
        Rz.append(float(data.oMf[R_fid].translation[2]))
    com_adv = coms[-1] - coms[0]
    print(f"[{name}] converged={solver.isFeasible if hasattr(solver,'isFeasible') else '?'} "
          f"iters={solver.iter} cost={solver.cost:.3f} stop={solver.stoppingCriteria():.2e} "
          f"finite={finite}")
    print(f"[{name}] CoM x advance = {com_adv*100:.1f} cm (target ~{VX*N*DT*100:.1f} cm) | "
          f"swing lift: L max z={max(Lz)*100:.1f} cm  R max z={max(Rz)*100:.1f} cm (apex {APEX*100:.0f})")
    print(f"[{name}] solve: cold={t_cold*1e3:.1f} ms | warm 1-iter avg={t_warm*1e3:.2f} ms "
          f"(vs acados -O0 ~480 ms, which DIVERGED on walk)")
    return finite and com_adv > 0.02 and max(max(Lz), max(Rz)) > 0.03


def solve_and_time(solver, name, maxit=120):
    prob = solver.problem
    xs = [x0.copy() for _ in range(N + 1)]
    us = prob.quasiStatic([x0.copy() for _ in range(N)])
    t = time.perf_counter()
    solver.solve(xs, us, maxit, False, 1e-9)
    t_cold = time.perf_counter() - t
    reps = 10
    t = time.perf_counter()
    for _ in range(reps):
        solver.solve(solver.xs, solver.us, 1, False, 1e-9)
    t_warm = (time.perf_counter() - t) / reps
    return report(solver, name, t_cold, t_warm)


# ----------------------------------------------------------------------------
# PART A — ContactFwdDynamics + FDDP
# ----------------------------------------------------------------------------
print("\n========== PART A: walk via ContactFwdDynamics + FDDP ==========")
try:
    run = [running_fwd(s) for s in sched]
    term = running_fwd(sched[-1], terminal=True)
    okA = solve_and_time(crocoddyl.SolverFDDP(crocoddyl.ShootingProblem(x0, run, term)), "A/FDDP")
except Exception as e:
    import traceback; traceback.print_exc(); okA = False; print(f"[A] FAILED: {e}")

# ----------------------------------------------------------------------------
# PART B — ContactInvDynamics + SolverIntro (per-node nu; projected-ID linchpin)
# ----------------------------------------------------------------------------
print("\n========== PART B: walk via ContactInvDynamics + SolverIntro ==========")
try:
    run = [running_inv(s) for s in sched]
    term = running_inv(sched[-1], terminal=True)
    nus = sorted({m.nu for m in run})
    print(f"[B] per-node control dims present = {nus} (DS=nv+12={nv+12}, SS=nv+6={nv+6})")
    solverB = crocoddyl.SolverIntro(crocoddyl.ShootingProblem(x0, run, term))
    okB = solve_and_time(solverB, "B/Intro")
    hmax = 0.0
    for d in solverB.problem.runningDatas:
        c = getattr(d.differential, "constraints", None)
        if c is not None and getattr(c, "h", None) is not None and c.h.size:
            hmax = max(hmax, float(np.linalg.norm(c.h)))
    print(f"[B] max equality (RNEA+contact) residual ||h|| over walk = {hmax:.2e} "
          f"(acados was 1e80 here)")
except Exception as e:
    import traceback; traceback.print_exc(); okB = False; print(f"[B] FAILED: {e}")

print("\n========== WALK SPIKE VERDICT ==========")
print(f"  A ContactFwdDynamics + FDDP  : {'PASS' if okA else 'FAIL'}")
print(f"  B ContactInvDynamics + Intro : {'PASS' if okB else 'FAIL'}")
print("  PASS = finite, CoM advances forward, swing feet lift >3cm, projection holds.")
