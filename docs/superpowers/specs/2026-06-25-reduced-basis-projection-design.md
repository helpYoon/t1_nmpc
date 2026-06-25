# Reduced-Basis State-Input Equality Projection — Design Spec

**Date:** 2026-06-25
**Topic:** Faithful port of OCS2 `projectStateInputEqualityConstraints` into the `t1_nmpc` whole-body acados NMPC.
**Status:** approved design → ready for implementation plan.

## Goal

Close the **structural single-RTI gap** in the whole-body NMPC by reproducing OCS2's state-input
equality projection inside acados, so that one SQP iteration lands on the contact-feasible manifold
(as it does in `t1_controller`) instead of leaving a large equality residual. Concretely: drive the
single-RTI equality residual `res_eq` from ~0.44 toward the converged ~3e-4, faithfully (machine
precision / first order) and without raising the iteration count.

## Background — why this is needed

`t1_controller` (OCS2 `SqpSolver`) runs `sqpIteration 1` **with `projectStateInputEqualityConstraints
true`**: each iteration eliminates the linearized state-input equality constraints out of the QP via a
range/null-space decomposition, so one iteration is accurate. Our acados port replicates the single
iteration and the line search but imposes the contact equalities as raw `con_h` rows (`lh==uh`) — acados
has **no native state-input equality projection** (its `PROJECT/MIRROR/CONVEXIFY` only regularize the
Hessian). Instrumentation (`sim/wb_walk_gate.py --log --gap-probe`) measured single-RTI `res_eq ≈ 0.44`
vs `≈ 3e-4` converged, `‖x_RTI − x_conv‖∞ ≈ 3.3`; the closed loop's yaw subsystem (base yaw + waist +
hip-yaws) diverges and the robot spins. RTI theory (Diehl–Bock–Schlöder 2005) attributes single-iteration
failure to an ill-conditioned active equality Jacobian; OCS2's projection removes exactly that
ill-conditioning at the source.

A 9-agent equivalence-verification workflow (2026-06-25) proved the existing dead `projection_wb.py` is
**not** equivalent to OCS2 (square ε-projector, ZeroWrench via a separate gate, cost/inequalities on raw
`u`, spurious nullspace pinned only by uniform LM). This spec is the **faithful rewrite**.

## Success criteria (acceptance bar)

1. **Faithful:** the projection is mathematically equivalent to OCS2's `projectStateInputEqualityConstraints`
   to machine precision (NumPy prototype: `‖u_phys_full − u_phys_OCS2_reduced‖ ≤ 1e-10`).
2. **`res_eq` closes:** single-RTI `res_eq` median drops from ~0.44 to **≤ ~1e-3 (target ~3e-4)** on
   `wb_walk_gate --log --gap-probe-every 10`, and the gap-probe `x_gap` collapses (single-RTI ≈ converged).
3. **M0 preserved:** `wb_stand_gate` still PASS (peak_tilt < 0.2, no fall, `n_fail == 0`).

The full **M1 walk gate is observed and reported but NOT gated** — D1 may be necessary-but-not-sufficient
(D-JL joint limits, the removed foot-placement, etc. are separate divergences). This spec isolates D1.

## Constraints (Global)

- **Faithful to `t1_controller`** wherever exact porting is possible; where acados forces a deviation,
  borrow the theoretically-equivalent construction and document it.
- **Single RTI:** runtime `max_iter = 1` stays (matches OCS2 `sqpIteration 1`). Raising iterations is
  **rejected** (kills the real-time budget, unfaithful). Single-RTI is also what makes axes 5–6 below
  exact, so it is *required* for equivalence.
- **Faithfulness first:** a slow solve is acceptable for this milestone; codegen/Jacobian-density
  optimization is a named, deferred follow-on (Section: Deferred), NOT an acceptance gate.

## Divergence ledger this spec must close (from the equivalence verification)

| Axis | Old (dead code) | This spec's fix |
|---|---|---|
| 1. Substitution coverage | cost/ineq on raw `u` (dynamics-only) | substitute `u_phys` into **dynamics + cost + inequalities**; remove the equality rows |
| 2. ε-regularized inverse | `Dᵀ(DDᵀ+εI)⁻¹` | **exact rank-detecting** Moore–Penrose `pinv` (SVD), drop rank-deficient rows |
| 3. Square projector, uniform LM pin | `(I−D⁺D)`, `lm·I` biases `u_phys` O(lm) | `ker(P)`-confined pin `√ρ·(I−P)(u−u_ref)`, `lm→0`, `NO_REGULARIZE` |
| 4. ZeroWrench via gate `G` | `P=(I−D⁺D)·G`, leaks swing wrench | **fold ZeroWrench rows into `D`**; project onto `ker([D_accel; D_swingZ; S])` jointly |
| 5. nonlinear-relinearize | — | exact to first order under single RTI (no change needed) |
| 6. frozen vs per-iter projection | — | exact-zero under single RTI (no change needed) |

## Architecture & components

One new idea threaded through existing files — no new subsystem. The projector is computed per node in
Python at the warm-start and passed to acados as parameters (frozen → acados never differentiates the
matrix inverse, the prior codegen blowup).

- **`t1_nmpc/wb/projection_wb.py` (rewrite).**
  - `build_projector_funcs(cfg, model)` → CasADi evaluators of the **folded, gated** contact residual
    `r(x,u,p)` and its Jacobians `D = ∂r/∂u`, `C = ∂r/∂x`. `r` stacks ZeroAccel (per stance foot, 6
    rows), SwingZ (per swing foot, 1 row), and the swing-wrench identity rows `S·u` (per swing foot, 6
    rows), with inactive rows gated to 0 by the per-node contact flags (`p[P_CONTACT]`). Fixed max size,
    gated → rank detection drops the inactive rows.
  - `compute_projector(x_node, u_node, p_node, funcs, cfg)` → `(P [nu×nu], Q [nu×nx], u_p [nu])` using
    `np.linalg.pinv` (rank-detecting SVD; **no ε term**, no `G` gate).
- **`t1_nmpc/wb/cost_wb.py` (modify).** Cost residuals evaluated at `u_phys`; **append** the `ker(P)`-pin
  residual `√ρ·(I−P)(u−u_ref)` to the CONL inner residual `y` (pure-LS rows). `(I−P)` derived in-expression.
- **`t1_nmpc/wb/constraints_wb.py` (modify).** The ZeroAccel/SwingZ rows stop being OCP `con_h` equalities
  (absorbed by the projector); `contact_residual_gated` is what `projection_wb` consumes (folded with the
  ZeroWrench rows). Friction/CoP barriers (already CONL margins in the cost) now read `u_phys`.
- **`t1_nmpc/wb/ocp_wb.py` (modify).** Substitute `u_phys` into `disc_dyn` + cost; remove the `con_h`
  equality rows (`lh/uh`) and the ZeroWrench `idxbu/lbu/ubu` box; keep `idxbx` joint-position limits; add
  the `P/Q/u_p` params; `levenberg_marquardt → 0`, `regularize_method → NO_REGULARIZE`.
- **`t1_nmpc/wb/mpc_wb.py` (modify).** `build_node_params` computes `(P_k,Q_k,u_p_k)` per node from the
  warm-start `(xg,ug)` and fills the new param slots; `step()` stores `u_phys_traj = P_k·u_k + Q_k·x_k +
  u_p_k` after the solve.
- **`t1_nmpc/wb/execution_wb.py` (modify).** Execution / `tau_ff` sample `u_phys_traj`, **not raw `u`**.
- **`t1_nmpc/mpc_result.py` (modify).** Add `u_phys_traj` field.
- **`tests/test_wb_projection.py` (new).** Prototype + unit (Stage 1 below).

**Data flow per tick (single RTI):** warm-start `(xg,ug)` → per node `compute_projector` → fill params →
acados solves the QP in raw `u` while dynamics/cost see `u_phys`; the `ker(P)`-pin makes the GN Hessian
PD; the equality is gone → extract `u`, map to `u_phys_traj` for execution.

## The projection math

At each node, linearize the gated contact constraint `r(x,u)=0` about the warm-start `(x₀,u₀)`:
`r₀ + C·(x−x₀) + D·(u−u₀) = 0`. Parametrize the feasible input by the raw decision `u` projected into
`ker(D)`:

```
u_phys = P·u + Q·x + u_p
P   = I − D⁺ D                     # orthogonal projector onto ker(D)  (range(P)=ker(D))
Q   = − D⁺ C
u_p = D⁺D·u₀ − D⁺ r₀ + D⁺ C·x₀     # so u_phys(u₀,x₀) = u₀ − D⁺r₀  (warm-start input projected onto the feasible manifold)
```

Parametrizing the free nullspace component by the *delta* `(u − u₀)` (not absolute `u`) is what gives the
`D⁺D·u₀` term and places the warm-start input on the feasible manifold — required for a sensible
linearization point. Both forms span the same `u_phys` feasible set (they differ by an element of
`ker(D)`), so `u_phys*` is identical; the delta form is the canonical one and matches the existing
`projection_wb.compute_projector` constant once `G=I` (ZeroWrench folded) and `ε→0`.

- `D⁺` is the **Moore–Penrose pseudoinverse via rank-detecting SVD** (`np.linalg.pinv`): kept singular
  values get exact `1/σ` (axis-2 fix vs `σ/(σ²+ε)`); zero/gated rows are dropped by rank detection, not
  ε-floored.
- **ZeroWrench folded in:** because the swing-wrench identity rows `S` are part of `D`, `P` projects onto
  `ker([D_accel; D_swingZ; S])` jointly (axis-4 fix), so `u_phys[swing-wrench] = 0` exactly.
- Because `D⁺` is Moore–Penrose, `P = I − D⁺D` is the symmetric idempotent orthogonal projector onto
  `ker(D)`, and `I − P = D⁺D` is the orthogonal projector onto `ker(P) = row(D)`, with
  `range(P) ⊥ ker(P)` to machine precision.

### The `ker(P)`-confined pin (axis-3 fix)

`u_phys` depends on `u` only through `range(P) = ker(D)`; the `ker(P) = row(D)` directions of `u` are free
→ singular GN Hessian. Add the cost residual

```
y_pin = √ρ · (I − P) · (u − u_ref)
```

Since `I − P` is the orthogonal projector onto `ker(P)`, `y_pin` is **exactly zero on `range(P)`** → it
pins the spurious nullspace **without biasing `u_phys`** (verification measured 3e-13 vs OCS2). The GN
Hessian on the input block becomes `PᵀRP` (PD on `range(P)`, since `R` PD) ⊕ `ρ(I−P)` (PD on `ker(P)`),
and `range(P) ⊥ ker(P)` ⇒ PD overall ⇒ `NO_REGULARIZE` is safe. This **replaces** the uniform
`levenberg_marquardt` (which biased `u_phys` by O(lm)); hence `lm → 0`.

`ρ` only needs to make the Hessian well-conditioned on `ker(P)`; in exact arithmetic any `ρ>0` leaves
`u_phys*` unchanged. Default `ρ = 1.0`, validated in the prototype (sweep over decades → `u_phys*`
invariant to 1e-10).

Under single RTI the projector is frozen at the warm-start (axes 5–6 exact), so the closed QP yields the
same physical `u_phys*` and the same R-weighted feasible minimizer as OCS2's reduced QP, to machine
precision and first order.

## acados wiring

**Parameter layout** (append to the current 119-slot vector; `nu=40`, `nx=68`):

| slot | range | size |
|---|---|---|
| existing (P_XREF…P_DT) | 0 : 119 | 119 |
| `P_PROJ_P` (P, row-major) | 119 : 1719 | nu·nu = 1600 |
| `P_PROJ_Q` (Q, row-major) | 1719 : 4439 | nu·nx = 2720 |
| `P_PROJ_UP` (u_p) | 4439 : 4479 | nu = 40 |
| **N_PARAM_WB** | | **4479** |

`(I − P)` and `u_ref` are derived (`u_ref` = existing `P_UREF`) — no extra params. The param-count jump
only feeds the deferred codegen cost; it does not affect correctness.

**Substitutions (`ocp_wb.make_ocp`):**
- `u_phys = reshape(p[P_PROJ_P],(nu,nu))@u + reshape(p[P_PROJ_Q],(nu,nx))@x + p[P_PROJ_UP]`.
- `am.disc_dyn_expr = _rk4(model, x, u_phys, p[P_DT])`.
- `build_cost_conl(x, u_phys, p, cfg, model)` so LS-tracking, joint-torque cap, and friction/CoP barriers
  all read `u_phys`; **append** `y_pin = √ρ·(I−P)(u−u_ref)` to the CONL residual `y` (identity `ψ` on
  those rows, `yref` 0).
- **Remove** `con_h` ZeroAccel/SwingZ rows (`con_h_expr`, `con_h_expr_0`, `lh_0/uh_0/lh/uh`) and the
  ZeroWrench `idxbu/lbu/ubu` box. **Keep** `idxbx` joint-position limits.

**Solver options:** `levenberg_marquardt = 0.0`; `regularize_method = "NO_REGULARIZE"`. Unchanged: `SQP`,
runtime `max_iter = 1`, `PARTIAL_CONDENSING_HPIPM`, `hpipm_mode SPEED`, `DISCRETE`, `MERIT_BACKTRACKING`
(OCS2 line-searches its single iteration — faithful), tols, warm-start, `cost_scaling = ones`.

**`mpc_wb` / `execution_wb`:**
- `build_node_params(...)`: for each node `k`, `compute_projector(xg[k], ug[k], P_node_params[k])` →
  `(P_k, Q_k, u_p_k)` → flatten into the `P_PROJ_*` slots. The projector linearizes `r` at the SAME
  `(xg[k], ug[k])` acados uses as the warm-start (consistent linearization point — matches OCS2 projecting
  at the iterate).
- `step()` after the solve: `u_phys_traj[k] = P_k·u_traj[k] + Q_k·x_traj[k] + u_p_k`; store on `MPCResult`.
  Raw `u_traj` remains the warm-start carry (`shift_warmstart` unchanged).
- `execution_wb.to_joint_command_wb` / the gates' look-ahead sampling: sample `u_phys_traj` for `tau_ff`
  and the reported wrenches (the one wiring trap — raw `u` is not the physical input).

## Test plan & acceptance gates

**Stage 1 — NumPy prototype (no codegen), `tests/test_wb_projection.py`**, at representative warm-start
nodes (double-stance, single-support mid-swing, touchdown):
- Projector identities: `P` symmetric-idempotent (≤1e-10), `D·P ≈ 0` (≤1e-9), `rank(P) = nu − rank(D)`,
  `r(x, u_phys) ≈ 0` on active rows (≤1e-9) for arbitrary `u`, `u_phys[swing-wrench] = 0` (≤1e-12).
- **OCS2 equivalence (the proof):** build the OCS2-style *reduced* QP (`R_new = ZᵀRZ` on the reduced `v`,
  `Z = nullspace basis of D`) and the *full-nu* QP (projector + `ker(P)`-pin) in NumPy; assert
  `‖u_phys_full − u_phys_reduced‖ ≤ 1e-10`.
- **Pin-doesn't-bias:** sweep `ρ ∈ {1e-3 … 1e3}`, assert `u_phys*` unchanged to ≤1e-10.

**Stage 2 — acados integration:**
- **Acceptance (res_eq):** `wb_walk_gate --log --gap-probe-every 10` → single-RTI `res_eq` median
  ≤ ~1e-3 (target ~3e-4); gap-probe `x_gap` median collapses toward 0.
- **Regression (M0):** `wb_stand_gate` PASS (peak_tilt < 0.2, no fall, `n_fail == 0`).
- **Unit:** param-layout test (`N_PARAM_WB == 4479`, slices contiguous and non-overlapping), projector
  params filled correctly per node.

**Observed, not gated:** M1 walk (does the yaw spin stop / forward progress?) — reported, not an
acceptance criterion.

**Run preamble (every python/pytest):**
`PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python …`

## Deferred (named follow-on, out of scope for acceptance)

The dense `P·u` and `Q·x` inside the nonlinear RK4 densify `∂(disc_dyn)/∂(x,u)` (~135 MB → `-O0`, ~2.5
s/solve). **Not an acceptance gate** — `res_eq`, equivalence, and M0 are measurable at any solve time.
Later options to evaluate: (a) substitute the projector into the *linearized* model only (custom
`A_new=A+BQ`, `B_new=BP`) if acados exposes it; (b) MX/sparse codegen for the projector matmuls; (c)
partial-condensing tuning; (d) revisit per-phase reduced-dim. Gets its own spec/plan when prioritized.

## Risks & open questions

- **Codegen size/time** at `N_PARAM_WB=4479` and the dense projector matmuls — expected, deferred; the
  prototype (Stage 1) de-risks the math before paying for it.
- **`NO_REGULARIZE` numerics:** if HPIPM flags indefiniteness despite the `ker(P)`-pin (floating-point
  edge), fall back to a small `ker(P)`-confined `ρ` bump — never uniform LM.
- **Rank detection at mode switches:** `np.linalg.pinv`'s `rcond` must drop exactly the gated/inactive
  rows; validate the kept rank equals the active-constraint count at each representative node (Stage 1).
- **D1 sufficiency:** even with `res_eq` closed and M0 safe, the M1 walk may still fail on other
  divergences — that is expected and explicitly out of this spec's gate.

## Note

`t1_nmpc` is a git repo (branch `master`); the environment's "not a git repo" flag refers to the stale
`t1_cmpc` cwd, not this package. Implementation tasks commit normally in `t1_nmpc`.
