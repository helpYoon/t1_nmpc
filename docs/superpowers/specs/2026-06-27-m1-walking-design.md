# M1 walking — design (aligator ProxDDP + whole_body_rnea inverse dynamics)

**Date:** 2026-06-27 (rev 2 — solver pivot)
**Branch:** wb-rnea-port (stays — no new branch)
**Status:** design; supersedes the Fatrop rev 1 of this file. Plan to follow.

## 1. Problem & goal

M1 = **stable forward biped walk** for T1, in closed-loop MuJoCo. The controller backend
**pivots from Fatrop to aligator (ProxDDP / AL-DDP)**, keeping the paper's `whole_body_rnea`
**inverse-dynamics formulation** (RNEA `τ_base=0` as a hard stagewise path constraint). This is an
**in-place transformation of `wb-rnea-port`**, not a new branch and not an additive parallel
implementation — the name stays accurate (still a `whole_body_rnea` port; the solver is now aligator).

## 2. Why aligator + RNEA-ID (verified, not assumed)

An exhaustive multi-solver spike odyssey (all run on this env) settled the solver question:

- **Fatrop — ruled out (proven).** Interior-point cannot warm-start the receding walk: 8–15 iters/tick
  to CV ≤ 1e-2, never ≤ 3; the barrier freezes on warm restart. Fundamental IPM deficiency.
- **acados — ruled out.** No hard stagewise equality projection — the `τ_base=0` path constraint has to
  be soft-slacked or reformulated away (the reduced-basis-projection contraption). Can't carry the
  formulation's core requirement.
- **crocoddyl — ruled out.** Penalty only; hard equalities segfault.
- **aligator — PASSES.** ProxDDP warm-starts the RNEA-ID OCP (RNEA `τ_base=0` hard equality + hard 6D
  contact) to **CV ≤ 1e-2 in 2–4 outer iters/tick across 40 receding ticks crossing 4 contact switches**
  (max CV 1.1e-3), `al_iter=0` (warm duals carry it), **clean conditioning at switches** (no indefinite
  Hessian). The t1_controller `sqpIteration=1` regime.

**The decisive insight (re-diagnosis of the old M1 topple):** the old forward-kinodynamic aligator port
failed *not* on warm-start (its stand converges in 4 inner / 1 outer iter) but because its **velocity-level
stance contact** (`FrameVelocityResidual`) is **index-2 and not AL-convergent** in single support (floors
at prim_infeas 0.4–1.4, dual → 1e6, at any dt/iters/CoM). The prior M1 wall was therefore *partly a
non-convergent single-support OCP*, not only a missing CoM reference. **RNEA-ID fixes the root cause** —
it enforces contact at the *acceleration* level (`τ_rnea[:6]=0` is acceleration-consistent), which *is*
AL-convergent. So the formulation choice is correct, not merely viable.

## 3. Implementation directive (how this lands)

- **Stay on `wb-rnea-port`.** No new branch.
- **Structure unchanged:** `robot/ wb/ runtime/ sim/` exactly as today.
- **Rewrite the `wb/` controller Fatrop → aligator IN PLACE — NOT additive.** Remove the Fatrop pieces;
  do not keep them beside aligator:
  - **Remove (Fatrop-specific only):** `wb/ocp.py` `StandOCP` (`opti.to_function`), `wb/mpc.py`
    `WholeBodyMPC` (Fatrop), the Fatrop opts in `config.py`, `tools/codegen_solver.py`. **Keep the
    8-corner contact model** in `robot/model.py` (paper-faithful — see §4).
  - **Build the aligator port FRESH** from the spike's Route-A code + the *current* Fatrop OCP's
    formulation. **Do NOT reuse the old `aligator-port` branch** (wrong formulation: forward-kinodynamic,
    velocity-level contact).
  - **Rewrite in place:** `wb/dynamics.py` (cpin RNEA functions for the residual), `wb/ocp.py` (aligator
    `TrajOptProblem` builder), `wb/mpc.py` (aligator `SolverProxDDP` + receding `replaceStageCircular`/
    `cycleProblem`), `wb/gait.py` (biped walk schedule), `wb/state.py` (MuJoCo↔pinocchio for the reduced
    model), `robot/model.py` (head-locked reduced model + 8 corner force frames + per-foot velocity frame),
    `robot/config.py`.
- **M0 stand re-homes onto aligator** as the all-double-support case of the same OCP (the Fatrop stand is
  retired, its tests rewritten). One controller, one solver.

## 4. The formulation (Fatrop port verbatim, on aligator)

**Minimal-change principle:** the `whole_body_rnea` formulation is the **Fatrop port's, unchanged** —
8-corner contact forces, RNEA `τ_base=0` path constraint, swing-z spline, footstep heuristic, gait,
discretization. Swapping Fatrop→aligator forces **exactly three** small changes, each a proven consequence
of aligator's augmented Lagrangian (not gratuitous): **(i)** contact velocity per-foot 6D, not per-corner
(per-corner is rank-deficient — 12 eqs / 6-DOF rigid foot — *and* less paper-faithful; the paper enforces
one velocity constraint per foot); **(ii)** swing-z as a *soft* cost, not a hard equality (hard stalls the
AL outer loop — also the foot-lift workstream, §6.1); **(iii)** joint torque recovered *post-hoc* via RNEA,
no `tau_nodes` decision vars (the receding shift makes `tau_nodes` a warm-start trap). Everything else is
the Fatrop port.

- **Model:** `buildReducedModel(freeflyer T1, lock AAHead_yaw + Head_pitch)` → **27 joints, nq=34, nv=33**
  (head-lock per the standing directive). Keep the **8 corner `OP_FRAME`s** (4 per foot, the Fatrop sole
  rectangle) for the contact forces, plus the ankle-roll/foot frame for the 6D velocity constraint.
- **State** `x=[q(34), v(33)]` on `MultibodyPhaseSpace`. **Control** `u=[forces(24), a(33)]` — the **8
  corner 3D contact forces** (paper-style point-contact forces; CoP emerges from the corner spread +
  unilateral `fz≥0`), then the full generalized acceleration.
- **Dynamics:** trivial double integrator `ẋ=[v,a]` (`DoubleIntODE` + `IntegratorSemiImplEuler`, dt=0.035).
- **RNEA path constraint (every stage):** `RNEA(q, v, a, f_ext(8 corners))[:6] == 0` (base underactuation)
  as an `EqualityConstraintSet` (cpin StageFunction, exact autodiff Jacobians); the 8 corner forces
  accumulate into the shared ankle `f_ext` slot (the Fatrop RNEA already does this). Joint torque recovered
  post-hoc = `RNEA(...)[6:]`.
- **Contact:** per-corner friction cone `fz≥0` and `μ²fz² ≥ fx²+fy²` (`NegativeOrthant`; the 8 unilateral
  corners give the support polygon / CoP — no separate wrench-cone); **per-foot 6D `FrameVelocityResidual
  == 0`** for each stance foot (LWA).
- **Swing + footstep:** swing-z as a **soft `QuadraticResidualCost`** on the Baumgarte z-velocity spline
  (`get_spline_vel_z`); **footstep placement** (restored) — a Raibert target
  `p_foot_xy = stance_xy + ½·T_step·v_des + k·(v_meas − v_des)` + nominal stance half-width, as a soft cost
  on the swing foot's xy through swing. Forward command via `base_vel_des` (track `vx` in the velocity cost).
- **Solver:** `SolverProxDDP(mu_init=1e-4)`, `LQ_SOLVER_SERIAL` (cpin residuals are Python → GIL),
  `ROLLOUT_LINEAR`; warm ticks cap `max_iters≈6`, `target_tol=1e-2` (mirroring `g_max=1e-2`, `g_min=1e-6`).
- **Receding warm carry:** `replaceStageCircular(tip @ gait_t = t+N·dt)` then `cycleProblem(...)` *before*
  the solve; **shift `xs/us` one knot, carry `vs/lams` UNSHIFTED** (wrongly shifting duals diverges to CV~33).

*(The warm-start gate was proven with per-foot 6D wrench forces; with the 8-corner forces it should still
hold — warm-start is carried by RNEA + the per-foot velocity constraint + the solver, insensitive to the
force representation — but the plan's first task re-confirms the gate with the 8-corner model.)*

## 5. Discretization & gait

`dt = 0.035` uniform, `N ≈ 20` → horizon ≈ 0.7 s (the spike-proven warm-start config). Biped walk gait
`cycle = 1.4 s` (`LF[0,0.6) → dbl[0.6,0.7) → RF[0.7,1.3) → dbl[1.3,1.4)`), preserving **cycle > horizon**.
The receding loop advances the gait clock one knot per tick. (aligator handles the uniform grid cleanly;
the Fatrop-era adaptive event-aware grid is not needed for AL-DDP.) **Horizon is tunable up** toward
~1.1 s (`N ≈ 31`, the t1_controller value) for **balance anticipation** in the walk phase — AL-DDP scales
far better than the IPM did, so the longer horizon is affordable; tune it against the lateral-balance work.

## 6. The three open M1 problems (honest — the real milestone work)

The spike resolved the **solver/formulation foundation**; the *locomotion* is not solved by it:

1. **Foot-lift (swing-z).** A *hard* swing-z Baumgarte equality stalls the AL outer loop (al=2, CV~3.8e-2 —
   the index-2 pathology); a *soft* swing-z cost warm-starts cleanly but does not lift the foot at low
   iters. **Foot-lift-while-warm-startable is unsolved** — its own workstream. Candidates: a C++ swing
   residual, a position+velocity-level (non-accel) swing constraint, or a higher-weight swing cost with a
   few extra iters at swing nodes only.
2. **Lateral balance (closed loop).** The warm-start gate was an *idealized* loop (`x_meas` = the solver's
   own node-1; no MuJoCo feedback). Real closed-loop lateral balance still needs the MuJoCo test and, very
   likely, an **explicit lateral CoM-sway reference**. The OCP now *solves* single support (unlike the old
   port) — the prerequisite — but staying upright in the plant is the open milestone.
3. **Real-time speed.** ~49–90 ms/tick (Python cpin residuals → serial) vs the ~20 ms budget. The
   **iteration-count gate passes regardless**; speed needs a **C++/pinocchio-bindings RNEA residual**
   (which also re-enables `LQ_SOLVER_PARALLEL`). **Deferrable** — not an M1-in-sim blocker.

## 7. Module layout (in-place)

```
robot/
  config.py   rewrite: head-locked 27-joint dims, per-foot frame geometry, gait, aligator solver params,
              g_max/g_min, weights. (remove Fatrop opts.)
  model.py    rewrite: buildReducedModel(lock head) + KEEP 8 corner OP_FRAMEs (forces) + a per-foot frame
              (6D velocity) + mass.
wb/
  dynamics.py rewrite: cpin RNEA functions (base-6 residual w/ 8 accumulated corner forces + Jacobians +
              post-hoc torque); swing-z cpin fn.
  ocp.py      rewrite: aligator TrajOptProblem builder — DoubleIntODE + IntegratorSemiImplEuler, the
              RneaBaseResidual EqualityConstraintSet, per-corner friction cones (8), per-foot 6D
              FrameVelocityResidual, soft swing-z cost, footstep cost, tracking cost; per-stage by gait
              flags. (remove StandOCP/Fatrop.)
  mpc.py      rewrite: AligatorMPC — SolverProxDDP(serial), reset (cold), step (warm via
              replaceStageCircular+cycleProblem, xs/us shift, vs/lams carry), command extraction. (remove
              WholeBodyMPC/to_function.)
  gait.py     rewrite/extend: biped walk schedule (cycle 1.4 s) + stand (all double-support).
  state.py    rewrite: MuJoCo<->pinocchio for the reduced 27-joint model; command (q_des,qd_des,tau_ff).
runtime/      kept; repoint to the new mpc/state.
sim/
  mujoco_runtime.py  kept; state read + command for the reduced model.
  stand.py / walk.py closed-loop runners on the aligator MPC.
tools/codegen_solver.py  REMOVE (Fatrop-only, impractical).
```

## 8. Success criteria

- **Warm-start (the verified gate, re-confirmed in-tree):** ProxDDP reaches CV ≤ 1e-2 in ≤ 5 (target ≤ 3)
  outer iters/tick across ≥ 15 receding ticks incl. contact switches.
- **M0 stand (re-homed):** closed-loop MuJoCo stand holds, Σ contact f_z / (m·g) ∈ [0.9,1.1], upright.
- **M1 walk:** closed-loop advances forward ≥ ~0.5 m over ≥ 5 s without falling; feet alternate with
  confirmed lift; lateral drift bounded (< ~0.1 m); watchable in `--view`.

## 9. Incremental build

1. **Port the proven solver/formulation** into `robot/model.py`+`robot/config.py`+`wb/{dynamics,ocp,mpc}.py`
   (RNEA-ID + per-foot 6D + ProxDDP). Verify the **warm-start gate in-tree** (the spike recipe).
2. **Re-home M0 stand** on aligator (all-double-support); closed-loop MuJoCo stand passes.
3. **Foot-lift** workstream (swing-z that lifts *and* warm-starts).
4. **Forward walk + lateral balance** (closed-loop MuJoCo; add the CoM-sway reference as needed) — the gate.
5. Docs + divergences. (C++ RNEA residual + real-time speed is a *separate, later* effort.)

## 10. Divergences to log (`docs/2026-06-25-t1controller-divergences.md`)

- Solver: aligator ProxDDP (AL-DDP) — replaces the Fatrop port; chosen because IPM cannot warm-start the
  receding walk and acados cannot carry the hard stagewise equality.
- **8-corner 3D contact force model KEPT** (paper-faithful). Only the contact *velocity* constraint moves to
  **per-foot 6D** (one `FrameVelocity==0` per foot), replacing the Fatrop port's per-corner velocity
  (rank-deficient — and the paper itself is per-foot).
- **Swing-z soft** (cost), not a hard equality (hard stalls AL — the foot-lift open problem).
- **Torque post-hoc** via RNEA, no `tau_nodes` decision vars (warm-start trap); torque limits, if needed,
  as soft penalties (a paper divergence to revisit).
- **Head locked always** (27-joint model); arms locked for the first walk.

## 11. Scope / risks

- **In:** in-place Fatrop→aligator rewrite; RNEA-ID + per-foot 6D; ProxDDP receding warm-start; re-homed
  stand; foot-lift + forward-walk + closed-loop balance.
- **Out / deferred:** C++ RNEA residual + real-time (the iteration gate already passes in sim, serial);
  arm-swing (arms locked first); turning / lateral / variable-speed commands; hardware.
- **Risks:** foot-lift-while-warm-startable is unsolved (own workstream); closed-loop lateral balance is the
  genuine M1 wall (now on a *convergent* single-support OCP, but unproven in the plant); the `vs/lams`
  carry protocol is exact-or-diverges; serial cpin residuals are ~50–90 ms/tick (sim-only until C++).
```
