# Divergences from `t1_controller` — WB acados port (corrected ledger)

**Date:** 2026-06-25 (rewritten after the adversarial faithfulness audit).
**Reference:** `src/t1_controller` (OCS2 `humanoid_nmpc/{humanoid_mpc,humanoid_common_mpc}`, config `robot_models/booster_t1/t1_mpc/config/mpc/task.info`) — the proven OCS2 T1 walker.
**This port:** `src/t1_nmpc/t1_nmpc/wb/` (acados WB MPC). Scope = **stand (M0) + forward walk (M1)**; contouring / hand-tracking (M2) is out of scope and not evaluated.

**Provenance:** this ledger replaces an earlier version that mislabeled three real bugs as "faithful" and concluded the walk failure was "not a faithfulness gap." A 14-dimension adversarial audit (each dimension extracted from both codebases, then attacked by a hostile skeptic, then given a closure plan; raw output `/tmp/.../wfsc0hx07.output`) corrected that. Every claim below is either marked **[verified]** (re-derived from the C++/task.info) or **[open]**.

**Status legend:** ✅ faithful · 🔧 CLOSED (Tier 1, this session) · 🔧 CLOSED (Tier 2, this session) · 🟠 OPEN Tier 2 · 🔴 OPEN Tier 3 · ⚪ real but low-priority.

---

## 1. Faithful — adversarially verified ✅

The OCP **problem** is a line-for-line port on these dimensions (re-derived from source + `task.info`, not from comments):

- **State / input** — `x` = base pose(6: xyz+ZYX-euler) + q_joints(27) + base-vel(6: world-lin + euler-rate) + v_joints(27) (+ inert path slots s,vₛ for M2); `u` = `[W_l(6), W_r(6), qdd_joints(27)]` (+ v̇ₛ). Composite `Translation+SphericalZYX` base, direct pin q/v mapping, head excluded (27/29). **[verified]** (cosmetic: contact frames use `OP_FRAME` vs OCS2 `FIXED_JOINT`, placement identical; collision points computed symbolically rather than as pin frames.)
- **Dynamics & integrator** — per-node exact cpin RBD; base accel via block-diagonal 6×6 mass inversion. **[verified]** The doc's old "continuous-vs-discrete RK4" framing was *imprecise in the faithful direction*: OCS2's `rk4SensitivityDiscretization` is itself a discrete linearization, mathematically identical to acados' AD-of-fused-RK4. The block-diagonal base inversion is an OCS2 modeling approximation that the port copies exactly (zero port-vs-OCS2 divergence).
- **State/input quadratic cost** (Q/R/Q_final, terminal-scale, base-velocity weight 3.0, joint-vel weights) — **[verified]** term-by-term against `task.info`.
- **ZeroAccel + SwingZ contact equalities** — `Ax`=diag(0,0,100,80,80,80), `Av`=diag(20,20,10,20,20,20), `Aa`=I; SwingZ 100/10/1. **[verified]** (OCS2 carries a stale comment claiming the orientation gains are ignored; its code applies them, matching the port.)
- **Friction & CoP relaxed-barrier penalties** — `μ·f_z − √(f_x²+f_y²+25)` (μ=0.4), CoP 4-row, RelaxedBarrier friction μ0.2/δ5.0, CoP μ0.1/δ0.03, `foot_rect` ±0.05. **[verified]** (omitted `gripperForce` term is identically 0 in OCS2.)
- **Gait & swing planner** — slow_walk schedule, switching times, mode inversion, SplineCpg swing-Z (apex 0.08), impact-proximity. **[verified]** steady-state identical (only FLY-gait branches and the one-time gait-start STANCE differ, neither exercised by slow_walk).

---

## 2. Divergences — corrected

| # | Component | OCS2 | Port | Severity | Status |
|---|---|---|---|---|---|
| **B1** | Joint-torque soft-cap limits | effortLimit from **t1_controller** URDF (knee 130, ankle 60, hip 130, waist 90) | read **wb_humanoid** URDF (knee 60, ankle 12, hip 45, waist 30) — 11 joints under-limited, over-penalizing push-off up to 25× | **material** | 🔧 CLOSED |
| **B2** | Joint-torque cost weight | GN LS weight = scaling·weight (no factor 2) | `2.0·scaling·weight` — exactly 2× | **material** | 🔧 CLOSED |
| **B3** | Execution τ_ff (deployed path) | τ_ff, q_des, qd_des all from the **same** t+5 ms look-ahead sample | τ_ff from plan **node 0**, q_des/qd_des from t+5 ms — internally inconsistent | material | 🔧 CLOSED |
| **B4** | Swing-foot cost gating | active on **both** feet, scaled only by impact-proximity (=1 in stance) | multiplied by `(1−contact)` → zeroed on the stance foot | minor | 🔧 CLOSED |
| **B5** | Reference gravity-split + vel clamp | input ref **all-zero**; command bounded upstream (1.0/0.6/1.0) | gravity-split `u_ref fz=mg/n` (acados-only); vel caps stored but unused | minor | 🔧 CLOSED |
| **B6** | Capture-point foot-placement cost | **none** (placement is emergent) | `_foot_placement_residual` (4 rows, inert at weight 0) — a latent divergence | latent | 🔧 REMOVED |
| **D-JL** | Joint position limits | **soft** two-sided PieceWisePolynomialBarrier (μ1200/δ0.1) on 27 joints | **hard** acados state box (`idxbx`), no barrier/slack; the barrier consts are dead | material | 🟠 Tier 2 |
| **D4** | Time discretization grid | **event-aligned** (a node lands on every contact switch) | fixed uniform N=31, dt=0.035 → switches fall mid-interval, quantized ±17 ms | **material** (walk suspect) | 🔧 CLOSED |
| **D1** | Contact-equality handling | null-space **projection** out of the QP → reduced unconstrained QP | con_h + per-node bounds inside HPIPM's KKT (un-projected) | **material** (not "neutral") | 🔴 Tier 3 |
| **D2** | Solver numerical riders | pure GN, **no** Hessian reg, cold start, FILTER | LM 1e-3 + regularize=PROJECT + QP warm-start + MERIT — different-conditioned single-RTI QP | material | 🔴 Tier 3 |
| **D5** | State estimator base height | lowest-foot-pinned FK (10 mm threshold) | MuJoCo **true** base pose | minor | ⚪ benign at stand |
| **D3** | Warm-start | event-aware grid + mode-clamp across switches | uniform-grid fractional time-shift + interp, no clamp | minor | ⚪ (tied to D4) |

### Closed this session (Tier 1)

- **B1/B2 — joint-torque soft-cap.** The port read `effortLimit` from the wrong URDF (the two `t1.urdf` are byte-identical except `effortLimit` on 11 leg/waist/wrist joints — verified: mass/inertia/pos-limits all identical) **and** doubled the cost weight. Compounded, the leg/ankle push-off torque OCS2 permits was penalized far more than 2×. Fix: hardcode t1_controller's `effortLimit` (`config_wb.py`), drop the `2.0` (`cost_wb.py`). Locked by `test_torque_limit_matches_t1_controller`, `test_residual_shapes_and_weights`.
- **B3 — τ_ff.** Deployed `to_joint_command_wb` now samples τ_ff from the same t+5 ms look-ahead pair as q_des/qd_des (`execution_wb.py`), matching `MpcMrtJointController.cpp:256-262`. Locked by `test_wb_execution.py`.
- **B4 — swing-foot gating.** Gate is now impact-proximity only (`cost_wb.py:_swing_foot_residual`), keeping the foot task cost on the stance foot as OCS2 does. **Cost:** because the contact equalities sit *unprojected* in the KKT (D1), the now-active stance-foot cost roughly **tripled the stand solve time (~22 → ~80 ms single-thread)**. Faithful but expensive — the cost is a symptom of D1 and should vanish once D1 is closed. Locked by `test_swing_foot_cost_active_on_moving_stance_foot`.
- **B5 — reference.** Gravity-split relabeled as an acados-only single-RTI prior (not an OCS2 term); command velocity clamp now applied (`mpc_wb.py`).
- **B6 — foot-placement removed.** The capture-point residual (weight 0, not in OCS2) was deleted; the walking residual is now **149 rows = OCS2's exact structure** (x + u + torque + swing-foot). Behavior unchanged (it was inert).

### Closed this session (Tier 2)

- **D4 — event-aligned time grid.** **Mechanism:** `grid_wb.event_aligned_grid` places a node on every in-horizon contact-mode switch; a new per-stage parameter `P_DT` is threaded into both the discrete RK4 integrator and the stage cost so the cost is the faithful time-integral, **normalized**: `psi *= p[P_DT]/cfg.dt`. Grounded in OCS2 `TimeDiscretization.cpp:60–114` and `SqpSolver.cpp:387,457`.

  **Spike finding:** acados' default `cost_scaling` for this OCP is **1** (a constant), NOT the time-step array — so the un-normalized `p[P_DT]` (≈0.035) shrank the effective cost ~28.6× relative to the fixed `levenberg_marquardt=1e-3` and collapsed the stand (211 MINSTEP solver failures). Normalizing by `cfg.dt` gives factor 1 at the nominal grid (preserves all existing tuning) while keeping relative stage weight ∝ dt_k.

  **Bounded divergences (faithful adaptation, documented):** (a) single node per switch — NOT OCS2's zero-length PreEvent/PostEvent jump duplication; the jump map is identity for this robot, so the only loss is applying the pre-jump (swing) constraint at the exact pre-touchdown instant; (b) the sub-`dt` remainder is spread evenly across each segment (round-per-segment) rather than placed as OCS2's one short pre-event interval; (c) fixed N=31 vs OCS2's variable node count; near-boundary switches (within 0.5·dt of t₀ or horizon end) are dropped and re-aligned on earlier ticks.

  **Measured results:** M0 stand still PASS (peak_tilt 0.0279 rad, 0 solver failures). M1 walk vs the pre-D4 baseline (mean_vx 0.165, peak_tilt 2.22, n_fail 353): **n_fail 353 → 122 (−65%)**, peak_tilt 2.22 → 2.16, min_foot_z_at_stance_activation ≈ 0.03 m — but the robot **still falls** (mean_vx −0.015). Full test suite: 77 pass / 11 pre-existing fails, 0 new failures.

### Open — Tier 2

- **D-JL — joint limits.** Port OCS2's soft two-sided `PieceWisePolynomialBarrier` (μ1200/δ0.1) as a `cost_wb` residual block (mirror `_foot_collision_residual`) and drop the hard `idxbx`. The dead `joint_limit_barrier_*` consts already exist.

### Open — Tier 3

- **D1/D2 — revive the projector.** The repo already contains an AD-safe affine projector (`projection_wb.py compute_projector`, `u_phys=P@u+Q@x+u_p`) — **implemented but never called** (dead code). "In-model elimination intractable (94–255 MB)" is true only for the *naive* route; the param-passed projector is tractable and reproduces OCS2's null-space elimination. Wiring it in (and stripping LM/PROJECT/QP-warm-start so the single-RTI step matches OCS2's pure-GN cold start) closes D1+D2 to a true equivalent. High effort; do last and TDD it.

---

## 3. Walk-failure analysis (corrected)

The previous §5 — *"the walk failure is not traceable to a faithfulness gap, it's a closed-loop balance issue"* — is **refuted.** Three faithfulness gaps each match the stomp/over-stride signature:

1. **D4 time-grid quantization** → premature STANCE constraint up to 17 ms early → vertical stomp. **Was Tier 2; now CLOSED this session.**
2. **B1/B2 joint-torque over-penalty** on legs/ankles → suppressed push-off → over-stride/hop. **Fixed (Tier 1).**
3. **D1 unprojected single-RTI QP** → equalities only to IPM tolerance, different conditioning. **Tier 3; now the leading remaining suspect.**

**Tier 1 measurement:** with B1–B6 fixed, **M0 stand still PASSES** (peak tilt 0.028 rad, 0 solver failures) but **M1 walk still FAILS** identically (mean_vx 0.165, peak tilt 2.22, fell at step ~2, ACADOS_NAN once the robot tilts). So fixing the torque over-penalty alone is insufficient.

**Tier 2 measurement (D4):** with the event-aligned grid implemented, solver failures dropped 65% (n_fail 353 → 122) and peak tilt improved marginally (2.22 → 2.16), confirming the quantization hypothesis. However, the closed-loop walk still fails (mean_vx −0.015) — the robot falls. D4 was a real contributor to instability but not the sole cause.

**Leading remaining suspect: D1.** With D4 closed, the un-projected single-RTI QP (D1) is now the strongest open suspect. OCS2 eliminates contact equalities via null-space projection before solving; the port solves them inside HPIPM's KKT as inequality constraints, giving different QP conditioning and equality satisfaction only to IPM tolerance. This affects every step of the walk. D2 (LM regularization, QP warm-start, MERIT filter) is entangled with D1 and should be addressed together when wiring the already-implemented projector (`projection_wb.py`).

---

## 4. Sim-side consistency (NOT MPC divergences)

- MuJoCo `t1.xml` ships armature/damping/frictionloss = 0; the sim stamps the sysID armature + viscous damping onto MuJoCo dofs so the plant matches what τ_ff already models (a fidelity fix, not a divergence — `t1_controller`'s τ_ff uses the same sysID). Optional Stribeck dry friction is default-OFF.

## 5. Bottom line

The port is **faithful on the problem** (§1) on every dimension that defines the OCP structure. The remaining divergences are concentrated in (a) data/convention bugs now fixed (B1–B6) and (b) the **solver/timing layer** (D1, D2, D-JL), where acados ≠ OCS2 forces a different mechanism. D4 (event-aligned grid) was implemented this session and measurably reduced solver failures by 65%, but walk still fails — **D1 (un-projected single-RTI conditioning) is now the leading remaining suspect** and the next lever to close.
