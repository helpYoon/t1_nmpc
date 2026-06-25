"""Crocoddyl feasibility spike — THROWAWAY de-risking probe (not project code).

Linchpin question for the acados->Crocoddyl backend pivot:
  Can Crocoddyl + Pinocchio represent the *real* T1 whole-body model, hold a
  double-support stand with 6D foot contacts, and solve it at a SANE rate
  (i.e. nowhere near the acados -O0 480 ms/solve), with the nullspace
  inverse-dynamics projection (SolverIntro) actually working?

Two parts:
  A) ContactFwdDynamics + FDDP   -> proves model+contacts+stand+solve-rate (stable API)
  B) ContactInvDynamics + SolverIntro -> proves the PROJECTED-ID nullspace mechanism
     (the native replacement for the acados state-input equality projection)

Reuses the port's faithful pinocchio model (t1_nmpc.wb.model_wb.WBModel): composite
Translation+SphericalZYX base, head locked (27 joints), foot_l/r_contact frames at the
ankle-roll joints. Run:
  env -u PYTHONPATH conda run -n t1mpc python spikes/croco_stand_spike.py
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
# 1. Faithful T1 model (reuse the port's loader)
# ----------------------------------------------------------------------------
cfg = make_wb_config()
wb = WBModel(cfg)
model = wb.model
data = model.createData()
foot_fids = list(wb.contact_fids)             # [left, right]
nq, nv = model.nq, model.nv                   # 33, 33
mass = float(sum(I.mass for I in model.inertias))
g = abs(model.gravity.linear[2])
print(f"[model] nq={nq} nv={nv} mass={mass:.3f} kg  weight={mass*g:.1f} N  "
      f"foot_fids={foot_fids}  njoints={len(model.names)-2}")

# Nominal double-support stand: base at nominal height, identity orientation, nominal joints.
q0 = pin.neutral(model)
q0[2] = cfg.nominal_base_height
q0[6:6 + cfg.n_joints] = cfg.nominal_joint_pos
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])

pin.framesForwardKinematics(model, data, q0)
foot_pl = [data.oMf[fid].copy() for fid in foot_fids]
for nm, pl in zip(("L", "R"), foot_pl):
    print(f"[stand] foot {nm} world pos = {pl.translation}")

# ----------------------------------------------------------------------------
# 2. crocoddyl scaffolding
# ----------------------------------------------------------------------------
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
print(f"[croco] state.nx={state.nx} state.ndx={state.ndx} actuation.nu={actuation.nu}")

LWA = pin.LOCAL_WORLD_ALIGNED
DT, N = cfg.dt, cfg.N


def make_contacts(nu):
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    for nm, fid, pl in zip(("L", "R"), foot_fids, foot_pl):
        c6 = crocoddyl.ContactModel6D(state, fid, pl, LWA, nu, np.array([0.0, 0.0]))
        contacts.addContact("foot_" + nm, c6)
    return contacts


def make_costs(nu):
    """State-reg toward the stand + control-reg toward zero."""
    costs = crocoddyl.CostModelSum(state, nu)
    # state weights: hold base pose + joints, damp all velocities
    w = np.concatenate([
        np.array([0, 0, 10.0, 20, 20, 20]),          # base pos(3) + euler(3)
        2.0 * np.ones(cfg.n_joints),                  # joints
        1.0 * np.ones(nv),                            # all velocities
    ])
    state_res = crocoddyl.ResidualModelState(state, x0, nu)
    state_act = crocoddyl.ActivationModelWeightedQuad(w ** 2)
    costs.addCost("xreg", crocoddyl.CostModelResidual(state, state_act, state_res), 1.0)
    ctrl_res = crocoddyl.ResidualModelControl(state, nu)
    costs.addCost("ureg", crocoddyl.CostModelResidual(state, ctrl_res), 1e-3)
    return costs


def run_solver(solver, name, maxiter_cold=60):
    problem = solver.problem
    xs = [x0.copy() for _ in range(N + 1)]
    us = problem.quasiStatic([x0.copy() for _ in range(N)])
    # cold solve
    t = time.perf_counter()
    ok = solver.solve(xs, us, maxiter_cold, False, 1e-9)
    t_cold = time.perf_counter() - t
    # warm single-iteration (RTI-like) timing, averaged
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        solver.solve(solver.xs, solver.us, 1, False, 1e-9)
    t_warm = (time.perf_counter() - t) / reps
    print(f"[{name}] converged={ok} iters={solver.iter} cost={solver.cost:.4f} "
          f"stop={solver.stoppingCriteria():.2e}")
    print(f"[{name}] solve: cold({maxiter_cold}it)={t_cold*1e3:.1f} ms | "
          f"warm 1-iter avg={t_warm*1e3:.2f} ms  (vs acados -O0 ~480 ms)")
    # drift of the stand over the horizon (base pos)
    drift = max(np.linalg.norm(np.asarray(solver.xs)[k][:3] - x0[:3]) for k in range(N + 1))
    print(f"[{name}] max base-pos drift over horizon = {drift*1e3:.2f} mm")
    return ok, t_warm


# ----------------------------------------------------------------------------
# 3. PART A — ContactFwdDynamics + FDDP (stable, proves model/contacts/rate)
# ----------------------------------------------------------------------------
print("\n========== PART A: ContactFwdDynamics + FDDP ==========")
try:
    nu_a = actuation.nu
    dam_a = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, make_contacts(nu_a), make_costs(nu_a), 1e-8, True)
    run_a = crocoddyl.IntegratedActionModelEuler(dam_a, DT)
    term_a = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, make_contacts(nu_a), make_costs(nu_a), 1e-8, True), 0.0)
    prob_a = crocoddyl.ShootingProblem(x0, [run_a] * N, term_a)
    # contact force sanity at node 0
    d0 = run_a.createData()
    us_qs = prob_a.quasiStatic([x0.copy() for _ in range(N)])
    run_a.calc(d0, x0, us_qs[0])
    fz = 0.0
    for nm in ("L", "R"):
        fr = d0.differential.multibody.contacts.contacts["foot_" + nm].f
        fz += fr.linear[2]
    print(f"[A] sum vertical contact force at stand = {fz:.1f} N "
          f"(expect ~weight {mass*g:.1f} N)")
    okA, _ = run_solver(crocoddyl.SolverFDDP(prob_a), "A/FDDP")
except Exception as e:
    import traceback; traceback.print_exc()
    okA = False
    print(f"[A] FAILED: {e}")

# ----------------------------------------------------------------------------
# 4. PART B — ContactInvDynamics + SolverIntro (the PROJECTED-ID linchpin)
# ----------------------------------------------------------------------------
print("\n========== PART B: ContactInvDynamics + SolverIntro ==========")
try:
    # ContactInvDynamics control = [accelerations(nv); contact forces(nc)] -> nu = nv + nc
    nc = 2 * 6                                    # two 6D foot contacts
    nu_b = nv + nc                                # 33 + 12 = 45
    dam_b = crocoddyl.DifferentialActionModelContactInvDynamics(
        state, actuation, make_contacts(nu_b), make_costs(nu_b))
    print(f"[B] ContactInvDynamics nu={dam_b.nu} (nv={nv}+nc={nc}) "
          f"nh(eq-constraints)={dam_b.nh} ng(ineq)={dam_b.ng}")
    run_b = crocoddyl.IntegratedActionModelEuler(dam_b, DT)
    term_b = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelContactInvDynamics(
            state, actuation, make_contacts(nu_b), make_costs(nu_b)), 0.0)
    prob_b = crocoddyl.ShootingProblem(x0, [run_b] * N, term_b)
    solver_b = crocoddyl.SolverIntro(prob_b)
    okB, _ = run_solver(solver_b, "B/Intro")
    # equality-constraint residual (the inverse-dynamics + contact nullspace projection)
    h = max(np.linalg.norm(d.differential.constraints.h) if hasattr(d.differential, "constraints") else 0.0
            for d in solver_b.problem.runningDatas)
    print(f"[B] max equality (RNEA+contact) residual ||h|| over horizon = {h:.2e} (want ~0)")
except Exception as e:
    import traceback; traceback.print_exc()
    okB = False
    print(f"[B] FAILED: {e}")

# ----------------------------------------------------------------------------
print("\n========== SPIKE VERDICT ==========")
print(f"  A ContactFwdDynamics + FDDP    : {'PASS' if okA else 'FAIL'}")
print(f"  B ContactInvDynamics + Intro   : {'PASS' if okB else 'FAIL'}")
print("  Linchpin = B (native projected-ID nullspace). A proves model+contacts+rate.")
