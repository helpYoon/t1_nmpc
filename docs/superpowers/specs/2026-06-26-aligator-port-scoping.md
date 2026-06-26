# Aligator port — scoping note

**Status:** scoping (not yet a build plan). Decision of 2026-06-26: *land the crocoddyl quad
baseline, then scope aligator.* The quad baseline is landed (see "Landed crocoddyl baseline"
below); this note scopes the move to [aligator](https://github.com/Simple-Robotics/aligator).

## Why move backends

The crocoddyl port keeps hitting one wall: **crocoddyl's FDDP cannot enforce hard constraints**,
so every OCS2 hard/relaxed constraint has to be smuggled in as a penalty — and the penalties that
are *faithful* to OCS2 destabilize warm-started DDP:

- **Relaxed log barrier (friction/CoP):** OCS2's interior-point penalty. Mathematically ported and
  FD-verified, but it **collapses the walk at t≈0.09 s at every iteration count** (bisected
  2026-06-26: maxit 20/40/80, 80 *worse* than 40 → it defines a bad optimum for a warm-started DDP
  step, not under-convergence). The ~1/h Hessian is too stiff for one DDP step; OCS2 only tolerates
  it because each SQP iteration solves a full interior-point QP (HPIPM).
- **Underactuation / equality projection** (the M1 yaw-divergence root cause): only enforceable in
  crocoddyl via `SolverIntro`'s `Hu≠0` special case; state-only equalities segfault.
- Net effect: a string of "faithful in isolation, destabilizing in our solver" compromises
  (relaxed barrier, torque-limit ×100, normal-force floor) — all deferred, none truly faithful.

Aligator removes the wall: it enforces hard stagewise equality **and** inequality constraints
natively, so the OCS2 formulation ports as *constraints*, not penalty hacks.

## Backend comparison (the question: "is aligator's backend the same as OCS2's?")

**No — different algorithm classes. OCS2's exact settings do not transfer 1:1.**

| | OCS2 (t1_controller) | Aligator | Crocoddyl (current) |
|---|---|---|---|
| Algorithm | multiple-shooting **SQP** | **ProxDDP** (proximal augmented-Lagrangian DDP) | **FDDP** |
| Step solver | **HPIPM** interior-point QP | Riccati backward pass + AL multipliers | Riccati backward pass |
| Inequalities | **soft** relaxed-log-barrier | **hard** (AL multipliers) — or soft if chosen | penalty only |
| Equalities | hard, in the QP | hard, via AL | only `SolverIntro` `Hu≠0` |

**Transfers faithfully:** cost weights (Q, R), residual/dynamics definitions, contact model,
constraint *geometry* (friction cone, CoP box, the equality). **Does NOT transfer:** anything
solver-specific — `sqpIteration`, HPIPM params, and the relaxed-barrier `mu`/`delta` (aligator uses
hard constraints, no barrier). Aligator's knobs are AL-specific: `mu_init`, the BCL penalty-update
schedule, proximal regularization, primal/dual tolerances.

## Migration path (cheap on-ramp → real win)

`conda install -c conda-forge aligator`. Aligator ships **`aligator.croc`**, which converts a
crocoddyl `ShootingProblem` → aligator `TrajOptProblem`. So the **entire existing `WalkOCP`**
(ContactInvDynamics models, residuals, contacts, costs) is reusable — swap the solver only.

1. **Spike (near-zero porting, highest information):** convert the current crocoddyl problem via
   `aligator.croc`, solve with `aligator.SolverProxDDP`. The compat carries our costs over *as
   penalties*, but ProxDDP adds proximal + AL regularization that FDDP lacks — exactly what tames a
   stiff/indefinite Hessian. **Test:** does flipping `T1_U67=relaxed` survive under ProxDDP where it
   collapses under FDDP? If yes → OCS2-faithful soft barriers almost for free.
2. **Real win (native API, incremental):** re-express friction/CoP and the underactuation/equality
   as aligator **hard `StageConstraint`s** (no barrier). Promote one constraint family at a time,
   re-validating the walk each step.

## Open questions to resolve in the spike

- Does `aligator.croc` carry `ContactInvDynamics`'s **intrinsic τ_base=0 equality** as a real
  constraint? If yes, we're more faithful than crocoddyl on day one; if it drops it, that's the
  first native-API task.
- ProxDDP solve time per MPC step vs the 25 ms budget (FDDP is already ~15× over at maxit 20).
- Warm-start semantics across mode-schedule changes (OCS2 spreads `primalSolution_`; does aligator
  expose an equivalent for the persistent single-RTI pattern we rely on?).
- Does the `aligator.croc` conversion preserve our per-cycle in-place mutation pattern
  (`changeContactStatus` / reference retargeting), or must the problem be rebuilt each cycle?

## Risks

- Compat layer may not cover every crocoddyl model feature we use (custom residuals/activations,
  ContactInvDynamics constraints) — spike will surface gaps fast.
- Real-time budget: AL outer iterations could be costlier than one FDDP pass.
- Hard friction/CoP is *stricter* than OCS2's soft relaxed barrier; if it over-constrains, fall back
  to aligator's soft/penalty mode to mirror OCS2's softness.

## Landed crocoddyl baseline (what aligator is measured against)

`croco_walk.py` defaults: `quad` friction/CoP barrier · U1 self-collision ON · U5 torque-×100 OFF ·
U6 normal-force-floor OFF · U9 stand-path box-dims fix · U2 per-tick live look-ahead. 9/9 tests pass.
Walks forward, no collapse; falls ~2.6 s from the deeper **runaway / balance** root (forward velocity
under-regulated — robot runs ahead of the planned velocity), which is independent of the barrier
choice and is the next target after (or via) the aligator move. Env toggles `T1_U67/U5/U6/U1` are
retained for the ProxDDP comparison spike.
