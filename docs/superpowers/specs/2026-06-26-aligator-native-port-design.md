# Aligator Native Walk-MPC Port — Design Spec

**Goal:** Port the T1 whole-body walk MPC from crocoddyl (FDDP/SolverIntro) to a native aligator
`KinodynamicsFwdDynamics` + `SolverProxDDP` formulation, so that hard friction/CoP constraints
eliminate the force-shed that topples the crocoddyl walk, while staying faithful to OCS2's
acceleration-based whole-body formulation and meeting the ~25 ms real-time budget.

**Architecture:** A new aligator OCP (`aligator_walk.py`) builds one `StageModel` per contact mode over
a fixed-N horizon; a persistent `SolverProxDDP` (`aligator_mpc.py`) solves it warm-started with the
parallel LQ backend, advancing the contact schedule each control tick via `replaceStageCircular` +
`cycleProblem`; `aligator_exec.py` extracts contact wrenches + a feed-forward torque. All
solver-agnostic infrastructure (gait schedule, reference generation, MuJoCo transport, config, the
Pinocchio model reduction) is reused from the crocoddyl port unchanged. The crocoddyl walk stays in
place as the working baseline and A/B comparator.

**Tech stack:** aligator 0.19.0 (conda-forge), pinocchio 4.0.0, Python 3.12 (conda env `t1mpc`).
crocoddyl 3.2.1 remains installed for the baseline.

## Global Constraints

- **Faithful T1 model:** the aligator model MUST be the same reduced model as `WBModel` (URDF
  `cfg.urdf_path`, head joints locked via `_HEAD_JOINTS`, 27 actuated joints, armature applied,
  contact frames `foot_l_contact`/`foot_r_contact` at `cfg.contact_frame_offset`) — but with a
  `pin.JointModelFreeFlyer()` base instead of the euler composite. Verified dims: nq=34, nv=33,
  nx=67, ndx=66, total mass 34.51 kg (m·g = 338.6 N).
- **Control layout (read, never hardcode):** `KinodynamicsFwdDynamics` control
  `u = [f_left(6), f_right(6), joint_accels(27)]`, `nu = 2·FS + (nv−6) = 39` for FS=6. Always read
  `ode.nu`; the earlier "nu=39/nv=33" guess only coincidentally matches the reduced model — the full
  29-joint URDF gives nu=41.
- **Real-time budget:** 25 ms per MPC solve at the chosen operating point. Validated target:
  **N=20, max_iters=2, 4 threads, parallel LQ → 12 ms mean / 15 ms p90** (stand phase).
- **Faithfulness vs OCS2:** kinodynamic (accel/wrench) control like `AccelDynamicsAD`; underactuation
  structural; friction/CoP available both as OCS2-faithful soft relaxed barriers AND as hard
  constraints (a config flag), defaulting to hard for the force-shed fix.
- **RAM safety:** no compilation (aligator/pinocchio are prebuilt conda binaries). Validation walks
  run one process at a time.

---

## 1. Background & motivation

The crocoddyl port walks forward but topples at ~2.6–4 s. The gating diagnostic (instrumented
maxiter=12 run) proved the cause is **force-shed**, not the velocity runaway: measured base velocity
tracks the plan (~0.15–0.2 m/s), but the planned total vertical force oscillates between 6 % and
154 % of m·g, repeatedly collapsing to 48–57 % → the base sinks → topple. This is the
`ActivationModelQuadraticBarrier` always-on-gradient pathology: crocoddyl FDDP cannot tolerate the
faithful relaxed log barrier (its ~1/h Hessian collapses the warm-started DDP step at every iteration
count), so the port was forced onto the QuadraticBarrier compromise, which sheds support force.

OCS2's t1_controller does not have this problem because its multiple-shooting SQP step (linear
increment, full equality-constrained KKT solve) tolerates the relaxed barrier. **The fix is a solver
whose single step accounts for the binding constraints.** mim_solvers is inequality-only and rejects
our equality constraints; aligator's `SolverProxDDP` (primal-dual augmented Lagrangian) supports hard
equality AND inequality natively. The aligator crocoddyl-compat layer (`aligator.croc`) drops the
`ConstraintManager` constraints, so a **native** port is required.

**Phase 0 (already executed, scratchpad spikes) validated the whole approach** — see Appendix A.
Headline: with hard cones, ProxDDP plans fz = exactly m·g (no shed), and the warm-started parallel-LQ
solve hits 12 ms at a useful operating point. This spec turns that proof into the production port.

## 2. Faithfulness contract

| Aspect | OCS2 t1_controller | This port |
|---|---|---|
| Dynamics | `AccelDynamicsAD`: control = [wrenches, joint accels], base accel condensed | `KinodynamicsFwdDynamics`: control = [contact forces, joint accels], base accel from centroidal momentum — **same class of formulation** |
| Underactuation | structural (base accel condensed) | structural (no τ_base constraint) |
| Contact stick | hard `zeroVelocity` equality | hard `FrameVelocityResidual` + `EqualityConstraintSet` |
| Swing foot | `zeroWrench` + `normalVelocity` equalities | swing-force regularization + `FrameTranslationResidual`/`FrameVelocityResidual` swing-z |
| Friction/CoP | **soft** relaxed barrier | toggle: hard `NegativeOrthant` (default) OR soft `RelaxedLogBarrierCost` (faithful) |
| Solver | multiple-shooting SQP + HPIPM | ProxDDP (primal-dual AL) |
| Per-step | 1 SQP iter, warm-started | small fixed budget (1–3 iters), warm-started primal+dual |

Not bit-identical (different algorithm class; AL tolerance vs exact projection). Cost weights and
constraint geometry transfer; solver-specific knobs (HPIPM params, sqpIteration, barrier μ/δ) do not.
Hard friction/CoP is *stricter* than OCS2; the soft toggle recovers OCS2's exact treatment for
comparison.

## 3. Architecture & module layout

New files under `t1_nmpc/wb/`, parallel to the `croco_*` pair:

| File | Responsibility | Depends on |
|---|---|---|
| `aligator_model.py` | build the faithful free-flyer T1 pinocchio model + contact frames + cached ids/mass | `config_wb`, `model_wb` constants (`_HEAD_JOINTS`, `MPC_JOINT_NAMES`, `CONTACT_FRAME_NAMES`, `CONTACT_PARENT_JOINTS`) |
| `aligator_walk.py` | per-contact-mode `StageModel` factory + fixed-N `TrajOptProblem` builder; hard/soft toggle | `aligator_model`, `gait_wb`, `reference_wb`, `config_aligator` |
| `aligator_mpc.py` | persistent `SolverProxDDP`; warm-start; `replaceStageCircular`+`cycleProblem` receding loop; mirrors `CrocoMPC.reset/step` | `aligator_walk`, `gait_wb`, `reference_wb` |
| `aligator_exec.py` | extract wrenches + RNEA `tau_ff` + feedback gains from the kinodynamic solution | `aligator_model`, `execution_wb` PD helpers |
| `config_aligator.py` | aligator settings: N, dt, max_iters, max_al_iters, mu_init, threads, hard/soft flag, cost weights | `config_wb` |

`aligator_mpc.CrocoMPC`-compatible interface means the existing runner, GIF, and diagnostic scripts
work by swapping the MPC object. Everything solver-agnostic is reused untouched: `gait_wb`
(contact schedule + swing trajectory), `reference_wb` (base/joint references), `config_wb`,
`mujoco_transport`, the validation harnesses.

## 4. The faithful T1 model (`aligator_model.py`)

`build_aligator_model(cfg) -> (model, space, foot_ids, mass)`:
1. `full = pin.buildModelFromUrdf(cfg.urdf_path, pin.JointModelFreeFlyer())`
2. `model = pin.buildReducedModel(full, [getJointId(n) for n in _HEAD_JOINTS], pin.neutral(full))`;
   assert `model.names[2:] == MPC_JOINT_NAMES`; `model.armature[6:] = cfg.armature`.
3. Add `foot_l_contact`/`foot_r_contact` `OP_FRAME`s at `cfg.contact_frame_offset` on the contact
   parent joints (same as `WBModel`).
4. `space = manifolds.MultibodyPhaseSpace(model)`.

State convention: x = [q(nq=34: 3 pos + quat(xyzw) + 27 joints), v(nv=33)]. Nominal stand:
`q[2]=cfg.nominal_base_height`, `q[7:]=cfg.nominal_joint_pos`, v=0.

## 5. Per-node OCP (`aligator_walk.py`)

`make_stage(contact_flags, refs, hard_cones) -> StageModel`:

- **Dynamics:** `ode = dynamics.KinodynamicsFwdDynamics(space, model, gravity[3], contact_states,
  contact_ids, FS=6)`; `disc = dynamics.IntegratorSemiImplEuler(ode, dt)`. A *separate* ode/disc per
  contact mode (the contact mask is baked at construction; nu invariant = 39).
- **Costs (`CostStack(space, nu)`):**
  - `QuadraticStateCost(space, nu, x_ref, diag(Wx))` — track the reference posture/base (Wx from
    `config_aligator`, mapped from the crocoddyl `Q`).
  - `QuadraticControlCost(space, u_ref, diag(Wu))` — regularize toward the weight-supporting input
    (`u_ref[fz slots] = m·g/nstance`), the aligator analog of the crocoddyl weight-comp ureg.
  - per **swing** foot: `FrameTranslationResidual(ndx, nu, model, p_swing_ref, fid)` wrapped in
    `QuadraticResidualCost` with z-weighted diag — tracks the gait swing-z + xy step target.
  - swing-foot **zero force**: heavy `Wu` weight on that foot's force slots `u[k·FS:(k+1)·FS]` (the
    dynamics only drop inactive forces from xdot; the u-slots stay free — must be pinned).
- **Constraints (`stage.addConstraint(residual, set)`):**
  - per **stance** foot: `FrameVelocityResidual(ndx, nu, model, Motion.Zero(), fid, LOCAL_WORLD_ALIGNED)`
    + `EqualityConstraintSet()` (foot stick).
  - per **stance** foot, friction + CoP: `CentroidalFrictionConeResidual(ndx, nu, k, mu, eps)` (nr=2)
    and `CentroidalWrenchConeResidual(ndx, nu, k, mu, L, W)` (nr=17, includes CoP rows), where k is
    the force-slot index, `L,W` the foot half-extents (`(foot_rect_*[1]-foot_rect_*[0])/2`).
  - control box: `ControlErrorResidual` + `BoxConstraint(umin, umax)` (joint-accel / force bounds).

**Hard/soft toggle** (`config_aligator.hard_cones`): same residual, different consumer —
`stage.addConstraint(cone, NegativeOrthant())` (hard, AL-enforced; feasible when residual ≤ 0) vs
`cost.addCost(RelaxedLogBarrierCost(space, cone, w, thr))` (soft, OCS2-faithful). Default hard.

## 6. Horizon assembly & receding-horizon cycle (`aligator_walk.py` + `aligator_mpc.py`)

- **Assembly:** a fixed `N` (default 20). Pre-build one `StageModel` per node of one gait *period*
  from `gait_wb`'s contact schedule + swing references (a ring buffer of models + their `StageData`).
  `problem = TrajOptProblem(x0, stages[:N], term_cost)`.
- **Per control tick (`recede(x_meas, t)`):**
  1. update the about-to-append node's swing refs from `gait_wb`/`reference_wb` at the new horizon
     end;
  2. `problem.replaceStageCircular(next_model)` (drop stage 0, append next-phase stage; N constant);
  3. `solver.cycleProblem(problem, next_model_data)` (rotate Results/Workspace/Riccati + warm-start);
  4. rotate the ring buffers;
  5. `problem.x0_init = x_meas`; `xs[0] = x_meas`; shift the primal guess one node;
  6. `solver.run(problem, xs, us, vs, lams)`; copy out `results.{xs,us,vs,lams}`.

Contact-mode change ⇒ a different `StageModel` (constraints are baked); nu/ndx identical across modes
(39/66) ⇒ swap-compatible. **Critical order:** `replaceStageCircular` before `cycleProblem` with the
matching data. **Open risk (Phase 2):** `cycleProblem` across knots whose constraint *count* changes
(stance: eq+cones; swing: none) is unproven — validate a full DS→Lswing→DS cycle first.

## 7. Solver & real-time (`aligator_mpc.py`)

```
solver = SolverProxDDP(tol=1e-3, mu_init=1e-2, max_iters=2, verbose=QUIET)
solver.linear_solver_choice = LQ_SOLVER_PARALLEL
solver.setNumThreads(4)            # + run with OMP_NUM_THREADS>=4
solver.max_al_iters = 2
solver.rollout_type = ROLLOUT_LINEAR
solver.sa_strategy  = SA_FILTER
solver.setup(problem)              # once
```
Warm-start: persist `xs/us/vs/lams` as copies; each tick set both `problem.x0_init` and `xs[0]` to
`x_meas` (the AL penalty μ stays warm between calls — do not reset except on large schedule jumps).
Operating point: **N=20, max_iters=2, 4 threads → 12 ms / 15 ms p90** (validated, stand). Iteration
budget may need to rise to 3–5 at gait transitions (variable budget); measure and tune in Phase 2.

## 8. Execution extraction (`aligator_exec.py`)

From `u0 = results.us[0]` and `x_meas = [q, v]`:
- **Wrenches:** `wrench_l = u0[0:6]`, `wrench_r = u0[6:12]` (LOCAL_WORLD_ALIGNED).
- **Feed-forward torque:** evaluate the continuous ode at (x_meas, u0); full accel
  `a = [base6 from ode.xdot[nv:] ; joint accels = u0[2·FS:]]`; `tau = rnea(model, rdata, q, v, a)`;
  for each foot `tau -= J_fid(LOCAL_WORLD_ALIGNED)ᵀ · u0[k·FS:(k+1)·FS]`; `tau_ff = tau[6:]`
  (verified `|tau[:6]|≈1e-13`). Reuse `execution_wb.pd_torque` for the PD term.
- **1 kHz interpolation:** `results.controlFeedbacks()` / `controlFeedforwards()` (not
  `getCtrlFeedbacks`). Output a `JointCommand` identical in shape to the crocoddyl path so the runner
  is unchanged.

## 9. Phase plan & success criteria

- **Phase 0 — de-risk spike. DONE (Appendix A).** Model, dynamics, hard-cone stand (fz=m·g), cone
  signs, tau_ff, no-segfault equalities, swing node, and the RT gate all validated.
- **Phase 1 — stand OCP (production modules).** Build `aligator_model/walk/mpc/exec` + config; a
  persistent warm-started double-support stand. **GO:** `0.9 ≤ fz/mg ≤ 1.1` held 5 s in closed-loop
  MuJoCo AND solve ≤ 25 ms at N=20/maxit=2/4thr. Then flip `hard_cones=False` (soft relaxed barrier)
  and record the faithfulness comparison (does soft alone also hold fz, under ProxDDP?).
- **Phase 2 — contact switching + steps.** Wire `gait_wb` into the cycle; validate one full
  DS→Lswing→DS cycle through `cycleProblem` end-to-end (the changing-constraint risk); then several
  steps. **GO:** ≥4 steps, no trunk-sink topple, fz bounded, still ≤ 25 ms (variable iteration budget
  allowed at transitions).
- **Phase 3 — full-walk parity.** Full gait + references; sustained walk; side-by-side GIF vs the
  crocoddyl baseline; document the hard-vs-soft and aligator-vs-crocoddyl comparison.

## 10. Testing strategy

- **Unit (pytest, fast):** model dims/mass/frames; `make_stage` builds for DS/Lswing/Rswing with
  correct nu/ndx and constraint counts; cone residual sign at a known-good and known-bad wrench;
  tau_ff base-rows ≈ 0; a single-node forward/Jacobian finite-check.
- **Solve-level:** a short stand horizon converges to fz≈m·g (hard) and (soft); a single gait-cycle
  problem solves without error.
- **RT:** a warm-started solve-time benchmark asserting the N=20/maxit=2/4thr operating point stays
  under budget on this machine (informational on slower hardware).
- **Closed-loop (MuJoCo, the gates):** the Phase 1/2/3 GO criteria above, via the reused transport +
  diagnostic harness (reuse the force-shed instrumentation: log fz/mg + base_z + solve_ms).
- **Baseline parity:** the crocoddyl tests stay green (no shared-code regressions).

## 11. Risks & open questions

1. **`cycleProblem` with per-knot changing constraint counts** (Phase 2 blocker): unproven that the
   dual/LQ warm-start stays valid when a knot flips stance↔swing. Validate a full cycle before
   trusting the loop; fallback = rebuild the problem each tick (slower but correct).
2. **Iteration budget at transitions:** maxit=2 holds the stand; footfall active-set jumps may need
   3–5 iters (dual warm-start is poor exactly there). Mitigation: variable budget + dual shifting
   (seed the about-to-activate foot's force multiplier from the schedule). Measure in Phase 2.
3. **RT at full walk:** stand timings have headroom (12 ms vs 25); swing + transitions cost more and
   are unmeasured. If over budget: fewer iters, smaller N, or more threads (saturates ~past 4).
4. **Swing-foot zero-force enforcement:** chosen as heavy control-reg weight; if it leaks force,
   switch to a control-slice equality (slicing an isolated force slot needs verification).
5. **Soft-barrier behavior under ProxDDP:** spike showed it stays finite (no FDDP-style collapse) but
   did not fully converge; the soft path is a comparison, not the primary, and may need μ/δ tuning.
6. **Reference/gait mapping:** `reference_wb`/`gait_wb` emit crocoddyl-state-convention quantities
   (euler base); the aligator model is quaternion free-flyer — the swing/base refs must be mapped to
   the free-flyer convention. Low risk (position-level), but explicit.

## Appendix A — Phase 0 validated facts (2026-06-26)

All on the faithful free-flyer T1 model in env `t1mpc` (aligator 0.19.0):
- Model: nq=34, nv=33, 27 joints, **nu=39**, m·g=338.6 N, frames `foot_l_contact`(63)/`foot_r_contact`(64).
- `KinodynamicsFwdDynamics` builds; `forward`/`dForward` finite.
- **Hard-cone double-support stand: fz = 338.6 N exactly (169.3/169.3), base height held, joint
  drift 1e-3 — force-shed eliminated.**
- Cone sign: `NegativeOrthant` feasible when residual ≤ 0 (good force ≤0; bad lateral force → +).
- tau_ff: `|tau[:6]| = 1.1e-13`.
- Stance zero-vel `EqualityConstraintSet` and single-support swing node both build + solve with **no
  segfault** (the crocoddyl SolverIntro state-only-equality segfault does not occur in aligator AL).
- **RT (warm-started, parallel LQ, 4 threads):** N=20/maxit=2 → 12 ms mean / 15 ms p90; N=20/maxit=1
  → 7 ms; N=30/maxit=1 → 12 ms. Parallel LQ ≈ 2.5× over serial. Under the 25 ms gate with margin.
