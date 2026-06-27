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

**Minimal-change principle:** the `whole_body_rnea` formulation stays the **paper's / Fatrop port's** —
8-corner contact forces, RNEA `τ_base=0` path constraint, **hard swing-z spline equality**, footstep
heuristic, gait. Swapping Fatrop→aligator forces only:
- **(i) contact velocity per-foot 6D, not per-corner** — a *correction toward the paper* (per-corner is
  rank-deficient on a 6-DOF rigid foot; the paper enforces one velocity constraint per foot).
- **(ii) joint torque post-hoc via RNEA with torque limits SOFT/relaxed**, not the hard `tau_nodes` box —
  the **only genuine divergence**, forced because `tau_nodes` is a warm-start trap *and* the hard box is
  cold-infeasible in dynamic phases (spike: CV~31). Logged as a divergence (§10).

**Swing-z stays a HARD equality** (paper-faithful); making it warm-start on aligator's AL is the foot-lift
workstream (§6.1) — the crux of M1, not a downgrade to soft. Everything else is the paper, verbatim.

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
- **Swing + footstep:** swing-z as a **HARD `EqualityConstraintSet`** on the Baumgarte z-velocity spline
  (`get_spline_vel_z`) — paper-faithful (`wb-mpc-locoman ocp.py:173-182`, `subject_to(... == 0)`); the
  AL-stall its hard form hit in the spike is the §6.1 foot-lift crux. **Footstep placement** (restored) —
  a Raibert target `p_foot_xy = stance_xy + ½·T_step·v_des + k·(v_meas − v_des)` + nominal stance half-width,
  as a **soft cost** on the swing foot's xy through swing. Forward command via `base_vel_des` (track `vx`).
- **Solver:** `SolverProxDDP(mu_init=1e-4)`, `LQ_SOLVER_SERIAL` (cpin residuals are Python → GIL),
  `ROLLOUT_LINEAR`; warm ticks cap `max_iters≈6`, `target_tol=1e-2` (mirroring `g_max=1e-2`, `g_min=1e-6`).
- **Receding warm carry:** `replaceStageCircular(tip @ gait_t = t+N·dt)` then `cycleProblem(...)` *before*
  the solve; **shift `xs/us` one knot, carry `vs/lams` UNSHIFTED** (wrongly shifting duals diverges to CV~33).

*(The warm-start gate was proven with per-foot 6D wrench forces; with the 8-corner forces it should still
hold — warm-start is carried by RNEA + the per-foot velocity constraint + the solver, insensitive to the
force representation — but the plan's first task re-confirms the gate with the 8-corner model.)*

## 5. Discretization & gait

`dt` base 0.035; **event-aware adaptive grid** (fine at t=0 + each contact-switch time, coarse in steady
single-support) → ~**N=17** vs ~31 uniform. The node-count reduction speeds **every** solver (the
per-iteration Riccati pass *and* the per-node cpin residual evals scale ~linearly in N) — it is *not*
Fatrop-specific, so we keep it. Biped walk gait `cycle = 1.4 s`
(`LF[0,0.6) → dbl[0.6,0.7) → RF[0.7,1.3) → dbl[1.3,1.4)`), preserving **cycle > horizon** (~0.7–1.1 s;
horizon tunable up toward 1.1 s for balance anticipation, affordable since AL-DDP scales far better than the
IPM did).

**Caveat — must validate (the grid ⊥ cheap-cycling tension):** aligator's cheap receding warm-start
(`replaceStageCircular`+`cycleProblem`) cleanly rotates a *uniform* grid; an event-aware grid's
fine-nodes-at-switches drift through the horizon as the gait advances, so it must be **gait-phase-stationary**
(dt pattern periodic with the 1.4 s cycle, horizon an integer knot count) to cycle without per-tick rebuilds.
The spike proved the warm-start gate on a *uniform* grid; the plan's first solver task **re-validates the
cyclic warm-start on the gait-phase-stationary adaptive grid**, with **uniform as the fallback** if it can't
cycle cleanly.

## 6. The three open M1 problems (honest — the real milestone work)

The spike resolved the **solver/formulation foundation**; the *locomotion* is not solved by it:

1. **Foot-lift — HARD swing-z on aligator (THE CRUX).** The paper's swing-z is a hard equality and **we keep
   it hard** (no downgrade to soft). The spike found *its* hard accel-Baumgarte residual stalls aligator's AL
   (al=2, CV~3.8e-2 — index-2), so **making the hard swing-z warm-start is the central, unsolved M1 work** —
   the one thing the verification spike did not crack. Candidates: an *input-coupled* accel-Baumgarte (the
   old port claimed this form *is* AL-enforceable), a C++ swing residual, AL/weight conditioning, or a
   position+velocity-level swing constraint. This is the primary M1 risk.
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
  dynamics.py   cpin symbolic PRIMITIVES only: whole-body RNEA (+ exact Jacobians, 8 accumulated corner
                forces, post-hoc joint torque), swing-z Baumgarte residual, frame velocities.
  constraint.py NEW (refactor — paper-auditable). Each HARD constraint a named StageFunction + ConstraintSet
                builder, **each docstring citing the paper §/eq + flagging any divergence**:
                  rnea_base       EqualityConstraintSet   [paper: base underactuation τ_rnea[:6]=0]
                  contact_velocity per-foot 6D, EqualitySet [paper: zero contact velocity, per foot]
                  friction_cone   per corner (8), NegativeOrthant [paper: friction cone; CoP via corners]
                  swing_z         HARD EqualityConstraintSet [paper: swing z-velocity spline]
                  torque_limit    SOFT/relaxed  [DIVERGENCE — paper hard tau_nodes box; §10]
                  joint_pos/vel_limit  [paper: state/input bounds]
  cost.py       NEW (refactor — paper-auditable). Each cost a named builder, **each paper-cited**:
                  state_tracking (Q) [paper Q] · input_reg (R) [paper R] · base_velocity (vx command)
                  · footstep_placement (Raibert — flagged: our addition, not in the quadruped paper).
  ocp.py        rewrite: THIN assembler — builds the aligator TrajOptProblem by wiring cost.py + constraint.py
                per stage from the gait flags (DoubleIntODE + IntegratorSemiImplEuler). No physics here.
                (remove StandOCP/Fatrop.)
  mpc.py        rewrite: AligatorMPC — SolverProxDDP(serial), reset (cold), step (warm via
                replaceStageCircular+cycleProblem, xs/us shift, vs/lams carry), command extraction. (remove
                WholeBodyMPC/to_function.)
  gait.py       rewrite/extend: biped walk schedule (cycle 1.4 s) + stand (all double-support).
  state.py      rewrite: MuJoCo<->pinocchio for the reduced 27-joint model; command (q_des,qd_des,tau_ff).
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

1. **Port the proven solver/formulation + the paper-auditable refactor** into `robot/{model,config}.py` and
   `wb/{dynamics,constraint,cost,ocp,mpc}.py` (RNEA-ID + 8-corner forces + per-foot-6D velocity + ProxDDP;
   each cost/constraint paper-cited). Verify the **warm-start gate in-tree** AND that the **event-adaptive
   gait-phase-stationary grid cycles cleanly** under `cycleProblem` (uniform fallback).
2. **Re-home M0 stand** on aligator (all-double-support); closed-loop MuJoCo stand passes.
3. **Foot-lift — make the HARD swing-z warm-start on aligator** (the crux; §6.1).
4. **Forward walk + lateral balance** (closed-loop MuJoCo; add the CoM-sway reference as needed) — the gate.
5. Docs: the **paper↔code mapping table** + divergences. (C++ RNEA residual + real-time is a separate effort.)

## 10. Divergences + the paper↔code map

Logged in `docs/2026-06-25-t1controller-divergences.md`; the plan also emits a **paper↔code mapping table**
(`docs/2026-06-27-paper-mapping.md`) listing every [arXiv:2511.19709] cost/constraint → our
`cost.py`/`constraint.py` unit → match/divergence, for at-a-glance audit.

- **Solver:** aligator ProxDDP (AL-DDP) replaces the Fatrop port — IPM cannot warm-start the receding walk;
  acados cannot carry the hard stagewise equality.
- **8-corner 3D contact force model KEPT** (paper-faithful). Contact *velocity* → **per-foot 6D** (one
  `FrameVelocity==0` per foot): a *correction* (per-corner is rank-deficient; the paper is already per-foot),
  **not** a divergence.
- **Swing-z stays a HARD equality** (paper-faithful) — no divergence; the aligator AL-stall is the §6.1
  foot-lift work, not a downgrade.
- **Torque limits SOFT/relaxed — the one real divergence:** post-hoc RNEA torque, no hard `tau_nodes` box,
  forced because `tau_nodes` is a warm-start trap *and* the hard box is cold-infeasible in dynamic phases.
- **Head locked always** (27-joint model); arms locked for the first walk.

## 11. Scope / risks

- **In:** in-place Fatrop→aligator rewrite (structure preserved, not additive); the `cost.py`/`constraint.py`
  paper-auditable refactor; RNEA-ID + 8-corner forces + per-foot-6D velocity + hard swing-z; ProxDDP receding
  warm-start; event-adaptive grid; re-homed stand; foot-lift + forward-walk + closed-loop balance.
- **Out / deferred:** C++ RNEA residual + real-time (the iteration gate already passes in sim, serial);
  arm-swing (arms locked first); turning / lateral / variable-speed commands; hardware.
- **Risks:** **HARD swing-z on aligator is unsolved — the crux** (the spike's hard form stalled the AL);
  closed-loop lateral balance is the genuine M1 wall (now on a *convergent* single-support OCP, but unproven
  in the plant); the event-adaptive grid must cycle cleanly under `cycleProblem` (uniform fallback); the
  `vs/lams` carry protocol is exact-or-diverges; serial cpin residuals are ~50–90 ms/tick (sim-only until C++).
