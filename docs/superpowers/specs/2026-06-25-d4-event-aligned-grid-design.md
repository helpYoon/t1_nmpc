# D4 — Event-aligned variable-dt time grid (design)

**Date:** 2026-06-25
**Status:** approved design; next step = implementation plan (writing-plans).
**Context:** Tier 2 of the t1_controller faithfulness program. Closes divergence **D4** in `docs/2026-06-25-t1controller-divergences.md` — the prime suspect for the M1 closed-loop walk failure.

## Problem

The acados port uses a **fixed uniform** shooting grid: `node_times = t + arange(N+1)·dt`, `N=31`, `dt=0.035`, horizon `T = N·dt = 1.085 s`. OCS2 `t1_controller` instead uses an **event-aligned** grid (`TimeDiscretization.cpp:60-114`): a shooting node lands exactly on every contact switch in the horizon.

With the uniform grid, a contact switch (touchdown) falls **mid-interval**, quantized by up to `±dt/2 = ±17 ms` (mean 8.7 ms; ~44% of switches >10 ms off-node). In the single-RTI closed loop this forces a STANCE constraint (ZeroAccel + free wrench) onto a foot still ~2.4 mm above ground at −0.35 m/s up to 17 ms **early** → premature contact → vertical "stomp", matching the reported failure signature.

The DISCRETE integrator (hand-rolled fused RK4, kept for ~2× speed over ERK) bakes `dt` into the generated code, so the grid cannot currently vary without a rebuild.

## Goal / non-goals

**Goal:** per-tick event-aligned variable-dt grid at **fixed N=31** and **fixed horizon T**, so every in-horizon contact switch lands exactly on a shooting node, with the stage cost time-integrated faithfully (`Σ dt_k·L_k`). Runtime-variable, no per-tick rebuild.

**Success bar (agreed):** *measurable* walk improvement — reduced premature-contact stomp (lower peak tilt, foot not airborne when its STANCE constraint fires, fewer/zero solver NaNs) vs the current baseline. Full M1 gate pass is **not** required here: D1 (un-projected single-RTI conditioning) is a co-suspect and remains open until Tier 3.

**Non-goals:** D1/D2 (projection, solver riders); D-JL (joint-limit soft barrier); any change to N, horizon length, gait cadence, costs/constraints/weights, or the contouring/M2 stack.

## Approach (chosen: full event alignment)

All node-grid logic lives in a new single-purpose module `grid_wb.py`. Both the dynamics and the cost read each interval's length from one new per-stage parameter `P_DT`.

### 1. `grid_wb.py` — `event_aligned_grid(t0, gait, cfg) -> node_times`

Faithful **fixed-N adaptation** of OCS2 `timeDiscretizationWithEvents` (`TimeDiscretization.cpp:60-114`), which marches at uniform `dt` and snaps the crossing node onto each event (shortening only that pre-event interval). Pure function; returns `N+1` strictly-increasing times spanning `[t0, t0+T]` (T = `cfg.N·cfg.dt`, horizon **unchanged**), with a node exactly on every gait switch in the window.

1. `switches = gait.switch_times_in(t0, t0+T)` — interior switch times.
2. Segment boundaries `B = [t0, *switches, t0+T]`, `M = len(B)-1` segments (`M ≤ ~3 ≪ N`).
3. Per segment, `n_k = max(1, round(len_k / cfg.dt))` intervals (keeps dt ≈ nominal — OCS2's uniform-dt marching). Reconcile `Σ n_k` to exactly `N` by adding/removing one interval from the **longest** segment (never a 1-interval segment; switch nodes stay exact).
4. Place each segment's intervals uniformly; concatenate.

**Single node per switch:** the switch node takes the post-switch (stance) mode via the existing `contact_flags → searchsorted(side="right")`.

**Invariant (test):** no switch in window → exactly `t0 + arange(N+1)·dt` (a single segment of length `N·dt` gives `round(N)=N` intervals). Underpins the regression test.

**Faithfulness to OCS2 (source-grounded) + bounded divergences:**
- ✅ Cost time-scaling `psi·dt_k` matches OCS2 — `SqpSolver.cpp:387,457` scale every stage by `getIntervalDuration`.
- ✅ A node lands exactly on every in-horizon switch — the stomp fix.
- ⚠️ **No event duplication** (decided 2026-06-25). OCS2 emits a zero-length PreEvent/PostEvent pair (jump-map interval) per switch; the port uses one node (post-switch mode). Jump map is identity for this robot → the only loss is applying the swing (pre-jump) constraint at the exact pre-touchdown instant. Bounded/benign; full duplication via zero-dt intervals deferred.
- ⚠️ **Remainder spread vs short pre-event interval.** OCS2 puts one short interval right before each event; the port spreads the sub-`dt` remainder evenly across the segment (`round`-per-segment). Both keep dt ≈ nominal and the switch exact.
- ⚠️ **Fixed N=31** vs OCS2's variable count. OCS2's `dt_min` (=`10·limitEpsilon`, no real floor) is moot: `round`-per-segment never produces sub-`0.5·dt` intervals.

New `Gait.switch_times_in(t0, t1)` helper (`gait_wb.py`): the gait is periodic (`duration`, `event_phases`); switch phases are `sorted(unique([0.0] + event_phases))`; absolute switch times are `(j + sp)·duration` for integer `j` and switch-phase `sp`, intersected with `(t0, t1)`.

### 2. Per-stage `dt` parameter `P_DT`

- `cost_wb.py`: add a 1-wide `P_DT` slot to the param layout; `N_PARAM_WB += 1`.
- `ocp_wb.py`: `am.disc_dyn_expr = _rk4(model, x, u, p[P_DT])` (was `cfg.dt`); default `parameter_values[P_DT] = cfg.dt`.
- Stage `k` integrates over its real interval `dt_k = node_times[k+1] − node_times[k]`. Runtime-variable; **no rebuild** when the grid changes (only `(x,u,p)`-affecting *source* edits trigger codegen).

### 3. Cost time-scaling (faithful `Σ dt_k·L_k`)

- `build_cost_conl`: scale the **entire** stage cost by its interval — `psi_scaled = p[P_DT] · psi` — so LS tracking *and* the friction/CoP/collision barriers are time-integrated, as OCS2 does. Scaling by a positive parameter preserves convexity in `r` and the GN Hessian PSD-ness.
- `ocp_wb.py`: set `ocp.solver_options.cost_scaling = np.ones(N+1)` so the `P_DT` factor is the **single** source of time-weighting (acados otherwise defaults `cost_scaling` to the time-step array → double-scaling). Terminal node unscaled.
- At the nominal uniform grid `dt_k = 0.035 = tf/N`, `p[P_DT]·psi` with `cost_scaling=1` should equal acados's current default (`time_steps`-scaled) cost → **no retuning**. This equality assumes acados's default `cost_scaling` is the time-step array (Risk 1); if not, the nominal scale differs by a constant and the weights take a one-time constant rescale — the regression test (§Testing) detects either case.

### 4. Per-tick wiring (`mpc_wb.py`)

- `build_node_params`: `node_times = event_aligned_grid(t, gait, cfg)` (was uniform); set `P[k, P_DT] = dt_k` for `k<N` (node `N`: `P_DT` unused by dynamics; set to last `dt_k` for tidiness).
- Everything else already keys off `node_times`: `x_ref`, contact flags, swing-Z, impact, and the per-node con_h / wrench bounds. A node landing exactly on a touchdown gets the **post-switch (stance)** mode via the existing `contact_flags → searchsorted(side="right")`, so ZeroAccel activates *at* contact, not early.

### 5. Warm-start generalization (`shift_warmstart`)

The only substantively new logic. The previous solution lives on the **previous** tick's (non-uniform) grid; warm-start must interpolate `x_prev`/`u_prev` by **absolute time** onto the current grid.
- Store `self._node_times_prev` alongside `self._x_prev/_u_prev/_t_prev`.
- Generalize `shift_warmstart` to take `node_times_prev` and `node_times_now` and interpolate each state/input component at the new node times (hold-last past the end). Replaces today's uniform-shift assumption.

## Files

| File | Change |
|---|---|
| `t1_nmpc/wb/grid_wb.py` | **new** — `event_aligned_grid`, segment allocation |
| `t1_nmpc/wb/gait_wb.py` | + `Gait.switch_times_in(t0, t1)` |
| `t1_nmpc/wb/cost_wb.py` | + `P_DT` in param layout; `psi *= p[P_DT]` in `build_cost_conl` |
| `t1_nmpc/wb/ocp_wb.py` | `_rk4` reads `p[P_DT]`; `cost_scaling = ones(N+1)`; default `parameter_values[P_DT] = dt` |
| `t1_nmpc/wb/mpc_wb.py` | grid call + `P_DT` fill; store prev node times; generalized warm-start |
| `tests/test_wb_grid.py` | **new** — grid construction + invariants |

## Testing & acceptance

**Unit (`test_wb_grid.py` + additions):**
- Uniform-when-no-switch: `event_aligned_grid` == `t0 + arange(N+1)·dt` when the horizon contains no switch.
- Switch-on-node: with 1 and 2 in-horizon switches, each switch equals some `node_times[k]` within `1e-9`; `N+1` strictly-increasing nodes; `node_times[0]=t0`, `node_times[-1]=t0+T`; `Σ dt_k = T`.
- Min-dt floor: a switch within `min_dt` of `t0` is dropped (no sub-`min_dt` interval); all `dt_k ≥ min_dt`.
- `switch_times_in` correctness against `mode_at` (a switch is where `mode_at` changes).
- Contact-flag at a switch node = post-switch (stance) mode.
- Warm-start interpolation: monotone, bounded, exact at coincident grids.

**Regression (the safety net for §2/§3):**
- With all `P_DT = cfg.dt` and `cost_scaling = ones`, the OCP solve **reproduces the current (post-Tier-1) uniform-grid solve** (x_traj/u_traj within solver tol) — proves the dt-parameterization and the cost re-scaling are exact at the nominal grid. (Baseline = the present cost, i.e. with the Tier-1 torque/gate fixes and foot-placement removed, not the original.)
- M0 stand gate still **PASS**; full unit suite no new failures (current baseline: 66 pass / 10 pre-existing drift fails).

**Acceptance (measurable improvement, vs current M1 baseline `mean_vx 0.165, peak_tilt 2.22, n_fail 353`):**
- Peak tilt reduced and trending upright; `n_fail` (ACADOS_NAN) down from 353.
- Instrument check: at each tick the foot whose STANCE constraint newly activates is at height ≈ 0 (not airborne) — i.e. the premature-contact quantization is gone.
- A new `sim` instrument (or extend the walk gate) logs per-switch node-alignment error (should be ~0) and foot height at constraint activation.

## Risks & spikes

1. **acados cost-scaling semantics (spike first).** Confirm: (a) acados's *default* `cost_scaling` is the time-step array; (b) `cost_psi_expr` may reference the parameter `p`; (c) the nominal-grid regression solve is unchanged. If `p`-in-`psi` is unsupported, fall back to runtime `cost_scaling`/`time_steps` updates (set `cost_scaling[k]=dt_k` per tick). The regression test gates correctness either way.
2. **DISCRETE disc_dyn with symbolic `dt`.** `dt` is a scalar multiplier in the RK4 stages; expect minimal codegen/solve-time change. Spike: rebuild, compare median solve time to the pre-D4 ~22 ms (note: the B4 swing-foot gate already raised stand solve to ~80 ms single-thread; that is a separate D1-coupled cost, out of scope here).
3. **Warm-start across changing grids.** The main new correctness surface; covered by the interpolation unit test + the regression solve.
4. **Switch near `t0` / `t0+T`.** A switch within `<0.5·dt` of a boundary makes a `round(<0.5)=0 → max(1,·)=1`-interval segment — no degenerate interval, no special-casing needed.

## Out of scope

D1/D2 (revive projector, strip LM/PROJECT/warm-start riders), D-JL (joint-limit soft barrier), and the B4 swing-foot solve-time regression. These are tracked separately in the divergence ledger.
