# Reduced-Basis State-Input Equality Projection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce OCS2's `projectStateInputEqualityConstraints` inside the t1_nmpc whole-body acados NMPC so one single-RTI iteration lands on the contact-feasible manifold, closing the structural gap (single-RTI `res_eq` ~0.44 → ~3e-4).

**Architecture:** Per node, compute an exact (rank-detecting pseudoinverse) affine projector `u_phys = P·u + Q·x + u_p` at the warm-start and pass `P,Q,u_p` as acados parameters (frozen → acados never differentiates the matrix inverse). Substitute `u_phys` into the dynamics, cost, and inequalities; remove the contact-equality `con_h` rows and the ZeroWrench box; pin the spurious nullspace with a `ker(P)`-confined cost penalty (`lm→0`, `NO_REGULARIZE`). Validate the math in a pure-NumPy prototype before paying for acados codegen.

**Tech Stack:** Python 3.10, CasADi (SX), acados (`AcadosOcp`/`AcadosOcpSolver`), HPIPM, NumPy, pinocchio, MuJoCo (gates), pytest, conda env `t1mpc`.

## Global Constraints

- **Single RTI:** runtime `max_iter = 1` stays (OCS2 `sqpIteration 1`). Do NOT raise iteration count — it is rejected (real-time budget + faithfulness) and is required for the projection equivalence.
- **Exact pseudoinverse:** use `np.linalg.pinv(D)` (rank-detecting SVD). NO `(DDᵀ+εI)⁻¹` Tikhonov term.
- **Fold ZeroWrench into the equality residual** `r` (swing-wrench identity rows), NOT a separate gate `G` and NOT input box bounds.
- **`ker(P)`-confined pin:** cost residual `√ρ·(I−P)(u−u_ref)`, `ρ = cfg.pin_rho = 1.0`; `levenberg_marquardt = 0.0`, `regularize_method = "NO_REGULARIZE"`.
- **Substitution coverage:** `u_phys` feeds dynamics AND cost AND inequalities; the ZeroAccel/SwingZ `con_h` equality rows and the ZeroWrench `idxbu` box are REMOVED.
- **Param matrix flatten convention:** column-major (`order='F'`) on the NumPy side, plain `cs.reshape(slice,(rows,cols))` on the CasADi side (column-major) — they must match. `nu=40`, `nx=68`.
- **Execution uses `u_phys`:** `tau_ff`/wrench sampling reads `result.u_phys_traj`, never raw `u`.
- **Param layout target:** `N_PARAM_WB == 4479` (119 + 1600 + 2720 + 40).
- **Run preamble (every python/pytest), run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc`:**
  `PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python …`
- **Codegen optimization is OUT OF SCOPE** (the dense `Q·x`/`P·u` densify the disc_dyn Jacobian → slow build/solve; accepted, named follow-on). The acceptance gate is measurable at any solve time.
- **Acceptance:** (1) NumPy prototype proves machine-precision equivalence to OCS2 (`≤1e-10`); (2) single-RTI `res_eq` median ≤ ~1e-3 on `wb_walk_gate --gap-probe`; (3) `wb_stand_gate` still PASS. M1 walk is observed, not gated.

## File Structure

- `t1_nmpc/wb/projection_wb.py` (**rewrite**) — `build_projector_funcs` (folded gated residual `r` + `D=∂r/∂u`, `C=∂r/∂x`); `compute_projector` (exact `pinv` → `P,Q,u_p`). Pure projection math; no acados.
- `t1_nmpc/wb/cost_wb.py` (**modify**) — projector param slots (`P_PROJ_P/Q/UP`, `N_PARAM_WB=4479`); `build_cost_conl` gains optional `u_raw,P_mat` → appends the `ker(P)`-pin residual.
- `t1_nmpc/wb/config_wb.py` (**modify**) — `pin_rho: float = 1.0`.
- `t1_nmpc/wb/ocp_wb.py` (**modify**) — reconstruct projector from params, substitute `u_phys` into dynamics+cost, remove `con_h` equality + `idxbu`, solver opts `lm→0`/`NO_REGULARIZE`, default identity projector in `pv0`.
- `t1_nmpc/wb/mpc_wb.py` (**modify**) — `step()` computes warm-start first; `build_node_params` fills `P/Q/u_p` per node from the warm-start; `step()` stores `u_phys_traj`; drop the removed `con_h`/wrench bound sets.
- `t1_nmpc/mpc_result.py` (**modify**) — add `u_phys_traj` field (default `None`).
- `t1_nmpc/wb/execution_wb.py` (**modify**) — sample `u_phys_traj`.
- `sim/wb_walk_gate.py`, `sim/wb_stand_gate.py`, `sim/wb_walk_view.py` (**modify**) — use `res.u_phys_traj` as the plan input.
- `tests/test_wb_projection.py` (**create**) — projector unit tests + the OCS2-equivalence prototype proof.
- `tests/test_wb_config_walk.py` (**modify**) — `N_PARAM_WB == 4479` + projector slice asserts.

---

## Task 1: Folded gated contact residual + Jacobians

**Files:**
- Modify: `t1_nmpc/wb/projection_wb.py` (rewrite the two functions)
- Test: `tests/test_wb_projection.py`

**Interfaces:**
- Consumes: `constraints_wb.contact_residual_gated(x,u,p,cfg,model) -> SX[14]`; `cost_wb.P_CONTACT` (slice 108:110); `cost_wb.N_PARAM_WB`.
- Produces: `build_projector_funcs(cfg, model) -> (r_fun, D_fun, C_fun)` where each is a `cs.Function([x,u,p],[·])`; `r_fun` returns the **26-row** folded residual `[contact_residual_gated(14); zerowrench(12)]`, `D_fun` returns `∂r/∂u` (26×40), `C_fun` returns `∂r/∂x` (26×68).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_projection.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_CONTACT, P_DT
from t1_nmpc.wb import projection_wb


def _p_vec(cfg, lf, rf):
    p = np.zeros(N_PARAM_WB)
    p[P_CONTACT] = [float(lf), float(rf)]
    p[P_DT] = cfg.dt
    return p


def test_folded_residual_has_zerowrench_rows():
    cfg = make_wb_config(); m = WBModel(cfg)
    r_fun, D_fun, C_fun = projection_wb.build_projector_funcs(cfg, m)
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu)
    # single support: left stance, right swing -> right foot wrench u[6:12] must appear in r
    p = _p_vec(cfg, lf=1, rf=0)
    r = np.asarray(r_fun(x0, u0, p)).ravel()
    D = np.asarray(D_fun(x0, u0, p))
    assert r.shape == (26,) and D.shape == (26, 40)
    # rows 14..19 = left ZeroWrench (left is STANCE -> gated to 0); rows 20..25 = right ZeroWrench (SWING -> active = u[6:12])
    assert np.allclose(D[14:20, :], 0.0)                     # left stance -> no ZeroWrench
    assert np.allclose(D[20:26, 6:12], np.eye(6))            # right swing -> identity on its wrench
    assert np.allclose(np.delete(D[20:26, :], np.s_[6:12], axis=1), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_wb_projection.py::test_folded_residual_has_zerowrench_rows -v`
Expected: FAIL (`build_projector_funcs` returns the old 14-row residual / wrong shape).

- [ ] **Step 3: Rewrite `projection_wb.build_projector_funcs` (+ helper)**

```python
# t1_nmpc/wb/projection_wb.py  (replace build_projector_funcs; remove the old compute_projector for now)
"""Reduced-basis state-input equality projection (faithful to OCS2 projectStateInputEqualityConstraints).

u_phys = P@u + Q@x + u_p makes the LINEARIZED contact equalities (ZeroAccel + SwingZ + ZeroWrench)
satisfied by construction. P,Q,u_p are frozen at the warm-start each tick and passed as acados params,
so acados never auto-differentiates the matrix pseudoinverse. ZeroWrench (swing-foot wrench == 0) is
FOLDED INTO the residual r (identity rows on the swing wrench), so a single exact pseudoinverse projects
onto ker([D_accel; D_swingZ; S]) jointly.
"""
from __future__ import annotations

import casadi as cs
import numpy as np

from .constraints_wb import contact_residual_gated
from .cost_wb import N_PARAM_WB, P_CONTACT


def folded_contact_residual(x, u, p, cfg, model):
    """26-row residual r(x,u,p)=0: [contact_residual_gated(14); ZeroWrench(12)].
    ZeroWrench row block for foot i = swing_i * u[6i:6i+6] (== 0 desired); gated to 0 on a stance foot."""
    r_contact = contact_residual_gated(x, u, p, cfg, model)          # 14
    zw_rows = []
    for i in (0, 1):
        swing = 1.0 - p[P_CONTACT][i]
        zw_rows.append(swing * u[6 * i:6 * i + 6])                   # 6 per foot
    return cs.vertcat(r_contact, *zw_rows)                           # 26


def build_projector_funcs(cfg, model):
    """CasADi evaluators of the folded contact residual r and its Jacobians D=dr/du, C=dr/dx."""
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu); p = cs.SX.sym("p", N_PARAM_WB)
    r = folded_contact_residual(x, u, p, cfg, model)
    return (cs.Function("proj_r", [x, u, p], [r]),
            cs.Function("proj_D", [x, u, p], [cs.jacobian(r, u)]),
            cs.Function("proj_C", [x, u, p], [cs.jacobian(r, x)]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_wb_projection.py::test_folded_residual_has_zerowrench_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/projection_wb.py tests/test_wb_projection.py && git commit -m "feat(proj): folded gated contact residual (ZeroAccel+SwingZ+ZeroWrench) + Jacobians"
```

---

## Task 2: `compute_projector` (exact rank-detecting pseudoinverse)

**Files:**
- Modify: `t1_nmpc/wb/projection_wb.py` (add `compute_projector`)
- Test: `tests/test_wb_projection.py`

**Interfaces:**
- Consumes: `build_projector_funcs` from Task 1.
- Produces: `compute_projector(x_node, u_node, p_node, funcs, cfg) -> (P, Q, u_p)` with `P` (40×40), `Q` (40×68), `u_p` (40,), all `np.float64`. `P = I − D⁺D`, `Q = −D⁺C`, `u_p = D⁺D·u₀ − D⁺r₀ + D⁺C·x₀`, `D⁺ = np.linalg.pinv(D)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_projection.py  (append)
def test_projector_identities_single_support():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu)
    u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0               # gravity-ish stance wrench
    p = _p_vec(cfg, lf=1, rf=0)
    P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
    assert P.shape == (40, 40) and Q.shape == (40, 68) and u_p.shape == (40,)
    assert np.allclose(P, P.T, atol=1e-10)                    # orthogonal projector: symmetric
    assert np.allclose(P @ P, P, atol=1e-10)                  # idempotent
    D = np.asarray(funcs[1](x0, u0, p))
    assert np.linalg.norm(D @ P) < 1e-9                       # range(P) = ker(D)
    # for an arbitrary raw u, u_phys satisfies the (active) linearized equality and zeros the swing wrench
    rng = np.random.default_rng(0); u = rng.standard_normal(40)
    u_phys = P @ u + Q @ x0 + u_p
    r_fun = funcs[0]
    # linearized residual at (x0,u_phys): r0 + D(u_phys-u0) (C term drops since x=x0)
    r0 = np.asarray(r_fun(x0, u0, p)).ravel()
    lin = r0 + D @ (u_phys - u0)
    assert np.linalg.norm(lin) < 1e-9
    assert np.allclose(u_phys[6:12], 0.0, atol=1e-10)         # right (swing) wrench == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_projection.py::test_projector_identities_single_support -v`
Expected: FAIL (`compute_projector` not defined).

- [ ] **Step 3: Add `compute_projector`**

```python
# t1_nmpc/wb/projection_wb.py  (append)
def compute_projector(x_node, u_node, p_node, funcs, cfg):
    """(P [nu×nu], Q [nu×nx], u_p [nu]) for u_phys = P@u + Q@x + u_p — the exact (rank-detecting)
    linearized projection of the folded contact equalities at the warm-start (x_node, u_node).
    P = I − D⁺D (orthogonal projector onto ker(D)); u_p places u_node on the feasible manifold."""
    r_fun, D_fun, C_fun = funcs
    x0 = np.asarray(x_node, dtype=np.float64); u0 = np.asarray(u_node, dtype=np.float64)
    r0 = np.asarray(r_fun(x0, u0, p_node)).ravel()
    D = np.asarray(D_fun(x0, u0, p_node), dtype=np.float64)        # nr × nu
    C = np.asarray(C_fun(x0, u0, p_node), dtype=np.float64)        # nr × nx
    Dp = np.linalg.pinv(D)                                         # nu × nr, rank-detecting (no Tikhonov)
    DpD = Dp @ D                                                   # nu × nu == I − P
    P = np.eye(cfg.nu) - DpD
    Q = -Dp @ C
    u_p = DpD @ u0 - Dp @ r0 + Dp @ C @ x0
    return P, Q, u_p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `… python -m pytest tests/test_wb_projection.py::test_projector_identities_single_support -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/projection_wb.py tests/test_wb_projection.py && git commit -m "feat(proj): exact rank-detecting compute_projector (P,Q,u_p)"
```

---

## Task 3: OCS2-equivalence prototype proof (machine precision)

**Files:**
- Test: `tests/test_wb_projection.py` (test-only — the Stage-1 acceptance gate)

**Interfaces:**
- Consumes: `compute_projector`, `build_projector_funcs`, `cfg.R`, `cfg.pin_rho` (Task 4 adds `pin_rho`; here use a local `RHO` constant so this task does not depend on Task 4).
- Produces: nothing (proof test).

This task proves that the full-`nu` projector + `ker(P)`-pin QP produces the **same physical input** `u_phys*` as OCS2's reduced (`ZᵀRZ`) QP, and that `u_phys*` is invariant to `ρ`.

- [ ] **Step 1: Write the test (it is the deliverable)**

```python
# tests/test_wb_projection.py  (append)
def _null_space(D):
    """Orthonormal basis of ker(D) via SVD (numpy-only; matches scipy.linalg.null_space)."""
    _u, s, vh = np.linalg.svd(D)
    tol = (np.amax(s) if s.size else 0.0) * np.finfo(s.dtype).eps * max(D.shape)
    return vh[np.sum(s > tol):].conj().T


def _solve_reduced_qp(D, r0, R, u0, u_ref):
    """OCS2-style reduced QP at dx=0: u_phys = u0 - D⁺ r0 + Z v, minimize 0.5 (u_phys-u_ref)' R (u_phys-u_ref)."""
    Dp = np.linalg.pinv(D)
    Z = _null_space(D)                                           # nu × (nu - rank(D))
    base = u0 - Dp @ r0                                          # particular solution
    a = base - u_ref
    Rz = Z.T @ R @ Z
    v = -np.linalg.solve(Rz, Z.T @ R @ a)
    return base + Z @ v


def _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho):
    """Full-nu QP at dx=0: min over raw u of 0.5(u_phys-u_ref)'R(u_phys-u_ref) + 0.5 rho ||(I-P)(u-u_ref)||^2."""
    c = Q @ x0 + u_p                                             # u_phys = P u + c
    ImP = np.eye(P.shape[0]) - P
    H = P.T @ R @ P + rho * ImP                                  # (I-P) sym-idempotent -> (I-P)'(I-P)=(I-P)
    b = -P.T @ R @ (c - u_ref) + rho * ImP @ u_ref
    u = np.linalg.solve(H, b)
    return P @ u + c


def test_full_nu_projector_matches_ocs2_reduced_qp():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    R = np.diag(np.asarray(cfg.R, dtype=np.float64))
    rng = np.random.default_rng(1); u_ref = rng.standard_normal(40)
    for lf, rf in [(1, 1), (1, 0), (0, 1)]:                      # double + each single support
        x0 = m.nominal_state(); u0 = np.zeros(40)
        u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
        p = _p_vec(cfg, lf, rf)
        D = np.asarray(funcs[1](x0, u0, p)); r0 = np.asarray(funcs[0](x0, u0, p)).ravel()
        P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
        u_red = _solve_reduced_qp(D, r0, R, u0, u_ref)
        u_full = _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho=1.0)
        assert np.linalg.norm(u_full - u_red) <= 1e-9, f"mode {lf,rf}: {np.linalg.norm(u_full-u_red)}"


def test_pin_rho_does_not_bias_u_phys():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    R = np.diag(np.asarray(cfg.R, dtype=np.float64))
    rng = np.random.default_rng(2); u_ref = rng.standard_normal(40)
    x0 = m.nominal_state(); u0 = np.zeros(40); u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
    p = _p_vec(cfg, lf=1, rf=0)
    P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
    base = _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho=1.0)
    for rho in (1e-3, 1e-1, 1e1, 1e3):
        assert np.linalg.norm(_solve_full_qp(P, Q, u_p, x0, R, u_ref, rho) - base) <= 1e-9
```

- [ ] **Step 2: Run to verify it PASSES (this is a proof, not a red-green pair)**

Run: `… python -m pytest tests/test_wb_projection.py -k "matches_ocs2 or does_not_bias" -v`
Expected: PASS. If it fails, the projector math (Task 2) is wrong — STOP and fix Task 2, do not proceed.

- [ ] **Step 3: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add tests/test_wb_projection.py && git commit -m "test(proj): machine-precision equivalence to OCS2 reduced QP + pin invariance"
```

---

## Task 4: Projector param slots + `pin_rho`

**Files:**
- Modify: `t1_nmpc/wb/cost_wb.py:32-35` (param-layout block)
- Modify: `t1_nmpc/wb/config_wb.py` (add `pin_rho` field)
- Test: `tests/test_wb_config_walk.py`

**Interfaces:**
- Produces: `cost_wb.P_PROJ_P = slice(119,1719)`, `cost_wb.P_PROJ_Q = slice(1719,4439)`, `cost_wb.P_PROJ_UP = slice(4439,4479)`, `cost_wb.N_PARAM_WB = 4479`; `cfg.pin_rho = 1.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_config_walk.py  (replace the body of test_param_layout_grown_and_contiguous)
def test_param_layout_grown_and_contiguous():
    assert cost_wb.P_XREF == slice(0, 68) and cost_wb.P_UREF == slice(68, 108)
    assert cost_wb.P_CONTACT == slice(108, 110) and cost_wb.P_SWINGZ == slice(110, 116)
    assert cost_wb.P_IMPACT == slice(116, 118) and cost_wb.P_DT == 118
    assert cost_wb.P_PROJ_P == slice(119, 1719)      # 40*40
    assert cost_wb.P_PROJ_Q == slice(1719, 4439)     # 40*68
    assert cost_wb.P_PROJ_UP == slice(4439, 4479)    # 40
    assert cost_wb.N_PARAM_WB == 4479


def test_pin_rho_default():
    from t1_nmpc.wb.config_wb import make_wb_config
    assert make_wb_config().pin_rho == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_config_walk.py::test_param_layout_grown_and_contiguous tests/test_wb_config_walk.py::test_pin_rho_default -v`
Expected: FAIL.

- [ ] **Step 3: Edit the param-layout block in `cost_wb.py`**

Replace lines 32-35 (`N_PARAM_WB = 118` … `N_PARAM_WB = N_PARAM_WB + 1`) with:

```python
N_PARAM_WB = 118
# D4 event-aligned grid: per-stage interval length dt_k (scalar); appended AFTER all existing slots.
P_DT = N_PARAM_WB            # 118
N_PARAM_WB = N_PARAM_WB + 1  # 119
# Reduced-basis projection (D1): per-node affine projector u_phys = P@u + Q@x + u_p, passed as params.
# Matrices stored COLUMN-MAJOR (order='F'); ocp_wb reconstructs via cs.reshape (also column-major). nu=40, nx=68.
P_PROJ_P = slice(N_PARAM_WB, N_PARAM_WB + 40 * 40)              # 119:1719  (P, 40x40)
N_PARAM_WB = N_PARAM_WB + 40 * 40                               # 1719
P_PROJ_Q = slice(N_PARAM_WB, N_PARAM_WB + 40 * 68)             # 1719:4439 (Q, 40x68)
N_PARAM_WB = N_PARAM_WB + 40 * 68                              # 4439
P_PROJ_UP = slice(N_PARAM_WB, N_PARAM_WB + 40)                 # 4439:4479 (u_p, 40)
N_PARAM_WB = N_PARAM_WB + 40                                    # 4479
```

In `config_wb.py`, add a field to the `WBConfig` dataclass (next to the other scalar params, e.g. after `contact_proj_eps`):

```python
    pin_rho: float = 1.0          # ker(P)-confined nullspace pin weight (does not bias u_phys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `… python -m pytest tests/test_wb_config_walk.py::test_param_layout_grown_and_contiguous tests/test_wb_config_walk.py::test_pin_rho_default -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/cost_wb.py t1_nmpc/wb/config_wb.py tests/test_wb_config_walk.py && git commit -m "feat(proj): projector param slots (N_PARAM_WB=4479) + pin_rho"
```

---

## Task 5: Cost substitution + `ker(P)`-pin

**Files:**
- Modify: `t1_nmpc/wb/cost_wb.py:147-170` (`build_cost_conl`)
- Test: `tests/test_wb_projection.py`

**Interfaces:**
- Consumes: `P_PROJ_P`, `P_UREF`, `pin_rho` (Task 4).
- Produces: `build_cost_conl(x, u, p, cfg, model, u_raw=None, P_mat=None)` — `u` is `u_phys`; when `u_raw` and `P_mat` are given, appends 40 pin rows `(I−P_mat)(u_raw − u_ref)` to `y` and `0.5·ρ·Σ pin²` to `psi`. Returns `(y, yref, psi, r)`. With `u_raw=None` the behavior is unchanged (backward compatible).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_projection.py  (append)
import casadi as cs
from t1_nmpc.wb import cost_wb


def test_cost_appends_kerp_pin_block():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu); p = cs.SX.sym("p", cost_wb.N_PARAM_WB)
    Pm = cs.reshape(p[cost_wb.P_PROJ_P], (cfg.nu, cfg.nu))
    y0, _, _, _ = cost_wb.build_cost_conl(x, u, p, cfg, m)                       # no pin
    y1, _, psi1, _ = cost_wb.build_cost_conl(x, u, p, cfg, m, u_raw=u, P_mat=Pm)  # with pin
    assert y1.shape[0] == y0.shape[0] + cfg.nu                                   # +40 pin rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_projection.py::test_cost_appends_kerp_pin_block -v`
Expected: FAIL (`build_cost_conl` takes no `u_raw`).

- [ ] **Step 3: Edit `build_cost_conl`**

Replace the signature and the residual/psi assembly so it reads (full function):

```python
def build_cost_conl(x, u, p, cfg, model, u_raw=None, P_mat=None):
    """Stage CONVEX_OVER_NONLINEAR cost; `u` is u_phys (the projected input). When (u_raw, P_mat) are
    given, append the ker(P)-confined pin √ρ·(I−P)(u_raw−u_ref) (pin rows are pure-LS; ρ does not bias
    u_phys because (I−P) is orthogonal to range(P))."""
    y_ls, yref_ls, W_ls = build_residual(x, u, p, cfg, model)
    h_bar = _contact_barrier_args(x, u, p, cfg, model)      # 10 friction/CoP margins (RelaxedBarrier)
    h_coll = _foot_collision_residual(x, p, cfg, model)     # 16 foot/knee collision margins
    blocks = [y_ls, h_bar, h_coll]
    n_ls = y_ls.shape[0]; n_bar = h_bar.shape[0]; n_coll = h_coll.shape[0]
    if u_raw is not None and P_mat is not None:
        u_ref = p[P_UREF]
        y_pin = (u_raw - u_ref) - P_mat @ (u_raw - u_ref)   # (I−P)(u_raw−u_ref)
        blocks.append(y_pin)
    y = cs.vertcat(*blocks)
    r = cs.SX.sym("r_psi", y.shape[0])
    psi = 0.5 * cs.dot(cs.DM(np.asarray(W_ls, dtype=np.float64)), r[0:n_ls] ** 2)
    bar = ([(cfg.friction_barrier_mu, cfg.friction_barrier_delta)]
           + [(cfg.cop_barrier_mu, cfg.cop_barrier_delta)] * 4) * 2
    for j, (mu, dl) in enumerate(bar):
        psi = psi + _relaxed_barrier(r[n_ls + j], float(mu), float(dl))
    for j in range(n_coll):
        psi = psi + _piecewise_poly_barrier(r[n_ls + n_bar + j], cfg.collision_barrier_mu, cfg.collision_barrier_delta)
    if u_raw is not None and P_mat is not None:
        off = n_ls + n_bar + n_coll
        psi = psi + 0.5 * float(cfg.pin_rho) * cs.sumsqr(r[off:off + cfg.nu])
    psi = (p[P_DT] / cfg.dt) * psi
    yref = np.zeros(y.shape[0])
    return y, yref, psi, r
```

Add `P_UREF` to the existing `from .cost_wb import …` usages — it is already a module-level name in `cost_wb.py`, so inside `cost_wb.py` reference it directly as `P_UREF` (no import).

- [ ] **Step 4: Run test to verify it passes**

Run: `… python -m pytest tests/test_wb_projection.py::test_cost_appends_kerp_pin_block -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/cost_wb.py tests/test_wb_projection.py && git commit -m "feat(proj): cost reads u_phys + appends ker(P)-confined pin block"
```

---

## Task 6: OCP wiring + build

**Files:**
- Modify: `t1_nmpc/wb/ocp_wb.py:56-167` (`make_ocp`)
- Test: `tests/test_wb_projection.py`

**Interfaces:**
- Consumes: `cost_wb.{P_PROJ_P,P_PROJ_Q,P_PROJ_UP,P_UREF}`, `build_cost_conl(...,u_raw,P_mat)`.
- Produces: an `AcadosOcp` whose dynamics+cost use `u_phys`, with the contact-equality `con_h` and ZeroWrench `idxbu` removed, `levenberg_marquardt=0`, `regularize_method="NO_REGULARIZE"`, `parameter_values` of length 4479 defaulting to an identity projector.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_projection.py  (append)
from t1_nmpc.wb.ocp_wb import make_ocp


def test_ocp_has_projector_params_and_no_contact_con_h():
    cfg = make_wb_config()
    ocp, _ = make_ocp(cfg)
    assert ocp.parameter_values.shape[0] == cost_wb.N_PARAM_WB == 4479
    assert ocp.solver_options.levenberg_marquardt == 0.0
    assert ocp.solver_options.regularize_method == "NO_REGULARIZE"
    # con_h contact-equality rows removed (no nonlinear constraint expr left)
    assert getattr(ocp.model, "con_h_expr", None) is None or ocp.model.con_h_expr.shape[0] == 0
    # ZeroWrench input box removed; joint-position box (idxbx) kept
    assert getattr(ocp.constraints, "idxbu", np.array([])).size == 0
    assert ocp.constraints.idxbx.size == 27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_projection.py::test_ocp_has_projector_params_and_no_contact_con_h -v`
Expected: FAIL.

- [ ] **Step 3: Edit `make_ocp`**

In `make_ocp`, after `p = cs.SX.sym("p", N_PARAM_WB)` and before the cost/dynamics build, reconstruct the projector and substitute (replace the block at lines 72-91 that currently sets `f_expl`, cost, `disc_dyn`, `con_h`):

```python
    from .cost_wb import P_PROJ_P, P_PROJ_Q, P_PROJ_UP
    # Reduced-basis projection: reconstruct the per-node affine projector (column-major; mpc_wb fills order='F').
    Pm = cs.reshape(p[P_PROJ_P], (cfg.nu, cfg.nu))
    Qm = cs.reshape(p[P_PROJ_Q], (cfg.nu, cfg.nx))
    up = p[P_PROJ_UP]
    u_phys = Pm @ u + Qm @ x + up

    f_expl = model.flow_expr(x, u_phys)
    y, yref, psi, r_psi = build_cost_conl(x, u_phys, p, cfg, model, u_raw=u, P_mat=Pm)
    y_e, yref_e, Wdiag_e = build_residual_terminal(x, p, cfg)

    am = AcadosTemplateModel()
    am.name = "t1_wb"
    xdot = cs.SX.sym("xdot", cfg.nx)
    am.x, am.u, am.xdot, am.p = x, u, xdot, p
    am.f_expl_expr = f_expl
    am.f_impl_expr = xdot - f_expl
    if discrete:
        am.disc_dyn_expr = _rk4(model, x, u_phys, p[P_DT])
    am.cost_y_expr = y
    am.cost_r_in_psi_expr = r_psi
    am.cost_psi_expr = psi
    am.cost_y_expr_e = y_e
    # NO con_h: the contact equalities are absorbed by the projector.
```

Remove the `build_con_h(...)` call and the `am.con_h_expr_0/con_h_expr` assignments; remove the `from .constraints_wb import build_con_h, NH, NBU` import (keep none of NH/NBU here). In the constraints block (lines 103-116) remove the `lh_0/uh_0/lh/uh` and the `idxbu/lbu/ubu` lines; KEEP the `idxbx/lbx/ubx` joint-position block. Update `pv0`:

```python
    x0 = model.nominal_state()
    ocp.constraints.x0 = x0
    pv0 = np.zeros(N_PARAM_WB); pv0[P_DT] = cfg.dt
    pv0[P_PROJ_P] = np.eye(cfg.nu).flatten(order="F")          # default = identity projector (u_phys=u)
    ocp.parameter_values = pv0
```

Set the solver options:

```python
    so.levenberg_marquardt = 0.0                               # pin replaces uniform LM (axis-3 fix)
    so.regularize_method = "NO_REGULARIZE"                     # GN Hessian PD via the ker(P) pin
```

(Delete the old `so.levenberg_marquardt = 1e-3` and `so.regularize_method = "PROJECT"` lines.)

- [ ] **Step 4: Run test to verify it passes** (no solver build needed — `make_ocp` only)

Run: `… python -m pytest tests/test_wb_projection.py::test_ocp_has_projector_params_and_no_contact_con_h -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/ocp_wb.py tests/test_wb_projection.py && git commit -m "feat(proj): wire u_phys into dynamics+cost, drop contact con_h/idxbu, lm->0/NO_REGULARIZE"
```

---

## Task 7: `mpc_wb` projector wiring + `u_phys_traj`

**Files:**
- Modify: `t1_nmpc/mpc_result.py` (add `u_phys_traj` field — needed by `step()` below)
- Modify: `t1_nmpc/wb/mpc_wb.py:25-43` (`build_node_params`), `:98-139` (`step`)
- Test: `tests/test_wb_projection.py`

**Interfaces:**
- Consumes: `projection_wb.{build_projector_funcs,compute_projector}`, `cost_wb.{P_PROJ_P,P_PROJ_Q,P_PROJ_UP}`.
- Produces: `MPCResult.u_phys_traj` field; `build_node_params(x_meas, node_times, comm_filt, gait, cfg, model, xg, ug, proj_funcs) -> P` (fills the projector slots from the warm-start; `xg`/`ug` are both `N+1` rows); `step()` stores `MPCResult.u_phys_traj` (N×nu).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_projection.py  (append)
from t1_nmpc.wb.mpc_wb import WholeBodyMPC


def test_step_fills_projector_params_and_u_phys_traj():
    cfg = make_wb_config(); m = WBModel(cfg)
    mpc = WholeBodyMPC(cfg, m)
    mpc.set_command([0.3, 0.0, 0.0, 0.0, 0.0])
    x0 = m.nominal_state(); mpc.reset(x0)
    res = mpc.step(x0, 0.0)
    assert res.u_phys_traj is not None and res.u_phys_traj.shape == (cfg.N, cfg.nu)
    # swing-foot wrench is zeroed by the projector on at least one node of a walking gait
    assert np.min(np.abs(res.u_phys_traj[:, 0:12]).sum(axis=1)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_projection.py::test_step_fills_projector_params_and_u_phys_traj -v`
Expected: FAIL (`u_phys_traj` is None / `build_node_params` signature).

- [ ] **Step 3a: Add the `u_phys_traj` field to `MPCResult`**

In `t1_nmpc/mpc_result.py`, add to the `MPCResult` dataclass (after `node_times`):

```python
    u_phys_traj: np.ndarray | None = None    # physical (projected) inputs P@u+Q@x+u_p for execution
```

- [ ] **Step 3b: Edit `mpc_wb.py`**

In `WholeBodyMPC.__init__`, build the projector funcs once: after `self.model = model`, add

```python
        from .projection_wb import build_projector_funcs
        self._proj_funcs = build_projector_funcs(cfg, model)
```

Change `build_node_params` to accept and use the warm-start:

```python
def build_node_params(x_meas, node_times, comm_filt, gait, cfg, model, xg, ug, proj_funcs) -> np.ndarray:
    from .cost_wb import P_PROJ_P, P_PROJ_Q, P_PROJ_UP
    from .projection_wb import compute_projector
    node_times = np.asarray(node_times, float)
    x_ref, u_ref = build_reference(x_meas, comm_filt, gait, node_times[0], node_times, cfg, model)
    P = np.zeros((cfg.N + 1, N_PARAM_WB))
    for k, tk in enumerate(node_times):
        P[k, P_XREF] = x_ref[k]
        if k < len(u_ref):
            P[k, P_UREF] = u_ref[k]
        lf, rf = gait.contact_flags(tk)
        P[k, P_CONTACT] = [float(lf), float(rf)]
        zL = gait.swing_z(tk, 0); zR = gait.swing_z(tk, 1)
        P[k, P_SWINGZ] = [zL[0], zL[1], zL[2], zR[0], zR[1], zR[2]]
        P[k, P_IMPACT] = [gait.impact_proximity(tk, 0), gait.impact_proximity(tk, 1)]
    dts = np.diff(node_times)
    P[:cfg.N, P_DT] = dts
    P[cfg.N, P_DT] = dts[-1]
    # projector per node, linearized at the warm-start (xg,ug both N+1 rows) -- the SAME point acados linearizes at.
    for k in range(cfg.N + 1):
        Pk, Qk, upk = compute_projector(xg[k], ug[k], P[k], proj_funcs, cfg)
        P[k, P_PROJ_P] = Pk.flatten(order="F")
        P[k, P_PROJ_Q] = Qk.flatten(order="F")
        P[k, P_PROJ_UP] = upk
    return P
```

Add `from .cost_wb import …, P_PROJ_P, P_PROJ_Q, P_PROJ_UP` to the existing import (or import locally as above).

In `step()`: compute the warm-start BEFORE `build_node_params`, pass `(xg,ug,self._proj_funcs)`, drop the removed `con_h`/wrench bound sets, and store `u_phys_traj`. Replace the body from `node_times = event_aligned_grid(...)` through the `return` with:

```python
        node_times = event_aligned_grid(t, self._gait, cfg)
        if self._x_prev is not None:
            xg, ug = shift_warmstart(self._x_prev, self._u_prev, self._node_times_prev, node_times, cfg)
        else:
            u0 = np.zeros(cfg.nu); u0[2] = u0[8] = self.model.total_mass() * 9.81 / 2.0
            xg = np.tile(x_meas, (cfg.N + 1, 1)); ug = np.tile(u0, (cfg.N, 1))
        ug_full = np.vstack([ug, ug[-1]])                       # N+1 rows for the terminal projector node
        P = build_node_params(x_meas, node_times, self._comm_filt, self._gait, cfg, self.model, xg, ug_full, self._proj_funcs)
        for k in range(cfg.N + 1):
            self.solver.set(k, "x", xg[k])
        for k in range(cfg.N):
            self.solver.set(k, "u", ug[k])
        for k in range(cfg.N + 1):
            self.solver.set(k, "p", P[k])
        self._last_warmstart_x, self._last_warmstart_u = xg, ug
        self.solver.constraints_set(0, "lbx", x_meas)
        self.solver.constraints_set(0, "ubx", x_meas)
        status = self.solver.solve()
        x_traj = np.array([self.solver.get(k, "x") for k in range(cfg.N + 1)])
        u_traj = np.array([self.solver.get(k, "u") for k in range(cfg.N)])
        from .cost_wb import P_PROJ_P, P_PROJ_Q, P_PROJ_UP
        u_phys_traj = np.empty((cfg.N, cfg.nu))
        for k in range(cfg.N):
            Pk = P[k, P_PROJ_P].reshape(cfg.nu, cfg.nu, order="F")
            Qk = P[k, P_PROJ_Q].reshape(cfg.nu, cfg.nx, order="F")
            u_phys_traj[k] = Pk @ u_traj[k] + Qk @ x_traj[k] + P[k, P_PROJ_UP]
        self._x_prev, self._u_prev, self._t_prev = x_traj, u_traj, t
        self._node_times_prev = node_times
        return MPCResult(x_traj=x_traj, u_traj=u_traj, feasible=(status == 0),
                         solve_time=time.perf_counter() - t0, mode_schedule=None, status=int(status),
                         node_times=node_times, u_phys_traj=u_phys_traj)
```

Remove the now-dead imports `from .constraints_wb import stage_constraint_bounds, stage_wrench_bounds` and the loop that set `lh/uh/lbu/ubu` (it is gone above). Note: the warm-start `comm_filt` filtering (`self._comm_filt = filter_command(...)`) and the `comm`/clip block stay BEFORE this block, unchanged.

- [ ] **Step 4: Run test to verify it passes** (builds the solver once — slow, minutes, due to the deferred codegen density)

Run: `… python -m pytest tests/test_wb_projection.py::test_step_fills_projector_params_and_u_phys_traj -v`
Expected: PASS (after the one-time solver build).

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/mpc_result.py t1_nmpc/wb/mpc_wb.py tests/test_wb_projection.py && git commit -m "feat(proj): mpc_wb fills per-node projector from warm-start, stores u_phys_traj"
```

---

## Task 8: Execution samples `u_phys_traj`

**Files:**
- Modify: `t1_nmpc/wb/execution_wb.py:25-29`, `sim/wb_stand_gate.py`, `sim/wb_walk_gate.py`, `sim/wb_walk_view.py`
- Test: `tests/test_wb_execution.py`

**Interfaces:**
- Consumes: `MPCResult.u_phys_traj` (field added in Task 7).
- Produces: `to_joint_command_wb` and all three gates sample `u_phys_traj` (fallback to `u_traj` when absent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wb_execution.py  (append)
def test_tau_ff_uses_u_phys_traj_when_present():
    cfg = make_wb_config(); m = WBModel(cfg)
    N = cfg.N
    x0 = m.nominal_state()
    u_raw = np.zeros(40); u_raw[2] = u_raw[8] = m.total_mass() * 9.81 / 2.0
    u_phys = u_raw.copy(); u_phys[12:39] = 0.2                      # physical input differs from raw
    x_traj = np.tile(x0, (N + 1, 1))
    u_traj = np.tile(u_raw, (N, 1))
    u_phys_traj = np.tile(u_phys, (N, 1))
    r = _Result(x_traj, u_traj, node_times=np.arange(N + 1) * cfg.dt)
    r.u_phys_traj = u_phys_traj
    jc = to_joint_command_wb(r, cfg, m, sample_ahead_s=0.005)
    np.testing.assert_allclose(jc.tau_ff, m.joint_torque(x0, u_phys), atol=1e-9)
    assert not np.allclose(jc.tau_ff, m.joint_torque(x0, u_raw), atol=1e-6)
```

(`_Result` in `tests/test_wb_execution.py` already stores arbitrary attrs; if its `__init__` does not accept `u_phys_traj`, set it as an attribute as shown.)

- [ ] **Step 2: Run test to verify it fails**

Run: `… python -m pytest tests/test_wb_execution.py::test_tau_ff_uses_u_phys_traj_when_present -v`
Expected: FAIL (execution samples raw `u_traj`).

- [ ] **Step 3: Edits** (the `MPCResult.u_phys_traj` field was added in Task 7)

`t1_nmpc/wb/execution_wb.py` — change line 28 to source the projected input:

```python
    u_src = result.u_phys_traj if getattr(result, "u_phys_traj", None) is not None else result.u_traj
    uq = np.array([np.interp(tq, nt[:cfg.N], u_src[:, j]) for j in range(u_src.shape[1])])
```

In each of `sim/wb_stand_gate.py`, `sim/wb_walk_gate.py`, `sim/wb_walk_view.py`, where the plan input is captured as `u_plan = res.u_traj` (and re-captured inside the MPC-tick branch), replace with:

```python
            u_plan = res.u_phys_traj if getattr(res, "u_phys_traj", None) is not None else res.u_traj
```

(There are TWO such assignments per gate: the initial `res = mpc.step(...)` line and the in-loop `res = mpc.step(...)` line. Update both in each file. `wb_walk_view.py` has one initial + one in-loop as well.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `… python -m pytest tests/test_wb_execution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add t1_nmpc/wb/execution_wb.py sim/wb_stand_gate.py sim/wb_walk_gate.py sim/wb_walk_view.py tests/test_wb_execution.py && git commit -m "feat(proj): execution + gates sample u_phys_traj (physical input)"
```

---

## Task 9: Acceptance — `res_eq` closes + M0 preserved

**Files:**
- No production code. Validation runs + a recorded result note in `docs/2026-06-25-t1controller-divergences.md` (D1 entry).

**Interfaces:**
- Consumes: the full integrated solver (Tasks 1–8).
- Produces: a pass/fail record against the spec's acceptance bar.

- [ ] **Step 1: Run the M0 stand regression**

Run: `… conda run -n t1mpc python sim/wb_stand_gate.py`
Expected: `WB_M0={... "PASS": true}` (peak_tilt < 0.2, `n_solver_failures` 0). If PASS is false, the projection regressed the stand — STOP and debug before claiming acceptance.

- [ ] **Step 2: Run the gap-probe `res_eq` acceptance**

Run: `… conda run -n t1mpc python sim/wb_walk_gate.py --duration 3 --vx 0.3 --log /tmp/proj_walk.npz --gap-probe-every 10`
Expected: `WALK_CONV` reports `res_eq_median ≤ ~1e-3` (target ~3e-4) and the `gap_probe.x_gap_median` collapses toward 0 (single-RTI ≈ converged) — vs the pre-projection baseline `res_eq_median ≈ 0.44`, `x_gap_median ≈ 3.3`. Record both numbers.

- [ ] **Step 3: Record the result in the divergence ledger**

Update the D1 entry in `docs/2026-06-25-t1controller-divergences.md` with the measured single-RTI `res_eq` before/after and the M0 PASS, and mark D1 CLOSED (faithful) or note the residual gap if `res_eq` did not reach target.

- [ ] **Step 4: Run the whole projection + config suite**

Run: `… conda run -n t1mpc python -m pytest tests/test_wb_projection.py tests/test_wb_config_walk.py tests/test_wb_execution.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git add docs/2026-06-25-t1controller-divergences.md && git commit -m "docs(proj): record D1 reduced-basis projection acceptance (res_eq before/after, M0 PASS)"
```

---

## Notes for the implementer

- **Observed, not gated:** after Task 9, optionally watch `sim/wb_walk_view.py --vx 0.3 --duration 6 --no-view --record /tmp/proj_walk.gif` to see whether the yaw spin stops. Report it; it is NOT an acceptance criterion (D1 may be necessary-but-not-sufficient).
- **First solver build is slow, and may need `-O0` to terminate** (the deferred codegen density: dense `Q·x`/`P·u` densify the disc_dyn Jacobian far beyond the current 18 MB). If `make` stalls at `-O2` on the dynamics objects during Task 7's build, extend the existing Makefile-rewrite trick in `ocp_wb.build_solver` (which already forces `%_hess.o: … -O0`) with a sibling line forcing the disc_dyn objects to `-O0` — find the object name via `ls .acados_wb/c_generated_code/*dyn*` (e.g. add `txt += "\n%_dyn_disc_phi_fun.o: CFLAGS := -fPIC -std=c99 -O0\n"` for each `*dyn*` object that stalls). This makes the build terminate (slow solve, ~seconds) and is the accepted, deferred cost — do NOT "fix" the density by re-introducing ε or dropping the substitution.
- **Pre-existing failures:** the suite has ~10 pre-existing failures unrelated to this work (M1 drift tests asserting the old 36-row `con_h`, mujoco waist-skip, ocp dims). Do not treat those as regressions; the Task-9 suite command lists only the files this plan touches.
