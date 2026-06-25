# D4 Event-Aligned Variable-dt Grid — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the acados WB-NMPC shooting grid event-aligned (a node lands on every in-horizon contact switch) with per-stage variable `dt`, so the STANCE constraint no longer fires while the foot is still airborne (the premature-contact "stomp").

**Architecture:** A new pure module `grid_wb.py` builds an event-aligned node grid each tick (faithful fixed-N adaptation of OCS2 `timeDiscretizationWithEvents`). One new per-stage acados parameter `P_DT` carries each interval's length into BOTH the DISCRETE RK4 integrator and the stage cost (`psi·dt_k`, faithful to OCS2's `getIntervalDuration` cost scaling). The warm-start is generalized to interpolate across changing non-uniform grids.

**Tech Stack:** Python 3.10 (conda env `t1mpc`), acados (`AcadosOcp`, DISCRETE integrator, PARTIAL_CONDENSING_HPIPM), CasADi, pinocchio.casadi, numpy, pytest.

## Global Constraints

- **Faithful to OCS2 `t1_controller`.** Grid = fixed-N adaptation of `timeDiscretizationWithEvents` (`TimeDiscretization.cpp:60-114`); cost scaling = per-interval (`SqpSolver.cpp:387,457`). Documented bounded divergences: single node per switch (no zero-length jump duplication; identity jump map → benign), remainder-spread vs short pre-event interval, fixed `N=31`.
- **Fixed dimensions unchanged:** `N=31`, `dt=0.035`, horizon `T = N·dt = 1.085`. Do NOT change N, the horizon, gait cadence, costs/weights, constraints, or any M2/contouring code.
- **No git repo** in `src/t1_nmpc` — replace each "Commit" with a **Checkpoint** (run the suite). 
- **Every command uses this preamble**, run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc`:
  `PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python ...`
  (abbreviated `$PRE` below; the empty `PYTHONPATH=` is load-bearing — keeps ROS numpy<2 pinocchio off the path).
- **acados rebuild:** any edit to `ocp_wb.py`/`cost_wb.py`/`model_wb.py`/`constraints_wb.py` changes the build hash → `build_solver` recompiles (~100 s) on next solver build. Pure-Python tests (grid, gait, warm-start) do NOT trigger it.
- **Test baseline:** current suite = 66 pass / 10 pre-existing drift fails. A failure outside that set is a real regression.

## File Structure

| File | Responsibility |
|---|---|
| `t1_nmpc/wb/grid_wb.py` | **new** — `event_aligned_grid(t0, gait, cfg) -> node_times`; pure, no acados/casadi |
| `t1_nmpc/wb/gait_wb.py` | + `Gait.switch_times_in(t0, t1) -> list[float]` |
| `t1_nmpc/wb/cost_wb.py` | + `P_DT` in param layout; `psi *= p[P_DT]` in `build_cost_conl` |
| `t1_nmpc/wb/ocp_wb.py` | `disc_dyn_expr = _rk4(model,x,u,p[P_DT])`; `cost_scaling = ones(N+1)`; default `parameter_values[P_DT]=dt` |
| `t1_nmpc/wb/mpc_wb.py` | `step` computes the grid + `P_DT`; stores prev node times; generalized `shift_warmstart` |
| `tests/test_wb_grid.py` | **new** — grid construction + invariants |
| `tests/test_wb_warmstart.py` | + non-uniform-grid interpolation test |
| `docs/2026-06-25-t1controller-divergences.md` | mark D4 closed; record the bounded divergences |

---

### Task 1: `Gait.switch_times_in` helper

**Files:**
- Modify: `t1_nmpc/wb/gait_wb.py` (add method to `Gait`, after `impact_proximity`)
- Test: `tests/test_wb_gait.py`

**Interfaces:**
- Produces: `Gait.switch_times_in(self, t0: float, t1: float) -> list[float]` — strictly-increasing absolute switch times in the open interval `(t0, t1)`. A switch is a phase where the contact mode changes: the phases `0.0` and each `event_phase`, tiled by `duration`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_wb_gait.py  (append)
import numpy as np
from t1_nmpc.wb.gait_wb import SLOW_WALK

def test_switch_times_in_matches_mode_changes():
    g = SLOW_WALK  # event_phases from [0.0, 0.65, 0.85, 1.5, 1.7], duration 1.7
    t0, t1 = 0.2, 0.2 + 1.085
    sw = g.switch_times_in(t0, t1)
    # strictly increasing, inside the open window
    assert sw == sorted(sw) and len(set(sw)) == len(sw)
    assert all(t0 < s < t1 for s in sw)
    # each returned time is where contact_flags actually changes
    for s in sw:
        before = g.contact_flags(s - 1e-4)
        after = g.contact_flags(s + 1e-4)
        assert before != after
    # and no missed switch: sampling densely finds no change outside the returned set
    ts = np.linspace(t0 + 1e-4, t1 - 1e-4, 4000)
    flags = [g.contact_flags(t) for t in ts]
    changes = sum(flags[i] != flags[i - 1] for i in range(1, len(ts)))
    assert changes == len(sw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PRE -m pytest tests/test_wb_gait.py::test_switch_times_in_matches_mode_changes -v`
Expected: FAIL — `AttributeError: 'Gait' object has no attribute 'switch_times_in'`.

- [ ] **Step 3: Implement the method**
```python
# t1_nmpc/wb/gait_wb.py  — inside class Gait, after impact_proximity
    def switch_times_in(self, t0: float, t1: float) -> list[float]:
        """Absolute contact-switch times in the open interval (t0, t1). A switch occurs at phase
        0.0 and at each event_phase, tiled by `duration` (faithful to OCS2 GaitSchedule switch times)."""
        switch_phases = np.unique(np.concatenate(([0.0], np.asarray(self.event_phases, float))))
        out: list[float] = []
        j0 = int(np.floor(t0 / self.duration))
        j1 = int(np.ceil(t1 / self.duration))
        for j in range(j0, j1 + 1):
            for sp in switch_phases:
                s = (j + sp) * self.duration
                if t0 < s < t1:
                    out.append(float(s))
        return sorted(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PRE -m pytest tests/test_wb_gait.py::test_switch_times_in_matches_mode_changes -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run: `$PRE -m pytest tests/test_wb_gait.py -q`
Expected: all gait tests PASS.

---

### Task 2: `grid_wb.event_aligned_grid`

**Files:**
- Create: `t1_nmpc/wb/grid_wb.py`
- Test: `tests/test_wb_grid.py`

**Interfaces:**
- Consumes: `gait.switch_times_in(t0, t1)` (Task 1); `cfg.N` (int), `cfg.dt` (float).
- Produces: `event_aligned_grid(t0: float, gait, cfg) -> np.ndarray` shape `(cfg.N+1,)`, strictly increasing, `node_times[0]==t0`, `node_times[-1]==t0+cfg.N*cfg.dt`, a node equal (to 1e-9) to every switch in the window.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_wb_grid.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.gait_wb import SLOW_WALK, STANCE_GAIT
from t1_nmpc.wb.grid_wb import event_aligned_grid

cfg = make_wb_config()
T = cfg.N * cfg.dt

def _check_basic(nt, t0):
    assert nt.shape == (cfg.N + 1,)
    assert np.all(np.diff(nt) > 0)                       # strictly increasing
    assert abs(nt[0] - t0) < 1e-12 and abs(nt[-1] - (t0 + T)) < 1e-9
    assert abs(np.diff(nt).sum() - T) < 1e-9

def test_uniform_when_no_switch():
    # STANCE_GAIT never switches -> exact uniform grid
    nt = event_aligned_grid(0.0, STANCE_GAIT, cfg)
    _check_basic(nt, 0.0)
    np.testing.assert_allclose(nt, np.arange(cfg.N + 1) * cfg.dt, atol=1e-12)

def test_switches_land_on_nodes():
    t0 = 0.2
    nt = event_aligned_grid(t0, SLOW_WALK, cfg)
    _check_basic(nt, t0)
    for s in SLOW_WALK.switch_times_in(t0, t0 + T):
        assert np.min(np.abs(nt - s)) < 1e-9, f"switch {s} not on a node"

def test_dt_stays_near_nominal():
    nt = event_aligned_grid(0.2, SLOW_WALK, cfg)
    d = np.diff(nt)
    assert d.min() > 0.4 * cfg.dt and d.max() < 1.8 * cfg.dt  # round-per-segment keeps dt ~ nominal

def test_switch_near_t0_no_degenerate():
    # place t0 just before a switch -> first segment rounds to >=1 interval, no zero/tiny dt
    s0 = SLOW_WALK.switch_times_in(0.0, 5.0)[0]
    nt = event_aligned_grid(s0 - 0.005, SLOW_WALK, cfg)
    _check_basic(nt, s0 - 0.005)
    assert np.diff(nt).min() > 0.3 * cfg.dt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PRE -m pytest tests/test_wb_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: t1_nmpc.wb.grid_wb`.

- [ ] **Step 3: Implement `grid_wb.py`**
```python
# t1_nmpc/wb/grid_wb.py
"""Event-aligned variable-dt shooting grid — faithful fixed-N adaptation of OCS2
timeDiscretizationWithEvents (TimeDiscretization.cpp:60-114). Marches at ~uniform dt but lands a
node exactly on every contact switch in the horizon. Single node per switch (no jump duplication;
identity jump map -> benign). Pure: no acados/casadi."""
from __future__ import annotations

import numpy as np


def event_aligned_grid(t0: float, gait, cfg) -> np.ndarray:
    N, dt = cfg.N, cfg.dt
    T = N * dt
    switches = gait.switch_times_in(t0, t0 + T)
    bounds = np.array([t0, *switches, t0 + T], dtype=np.float64)
    seg_len = np.diff(bounds)                                  # M segments
    # intervals per segment ~ uniform dt (round), >=1
    n = np.maximum(1, np.round(seg_len / dt).astype(int))
    # reconcile to exactly N by adjusting the LONGEST segment (never below 1 interval)
    while n.sum() != N:
        if n.sum() > N:
            cand = np.where(n > 1)[0]
            j = cand[np.argmax(seg_len[cand])]
            n[j] -= 1
        else:
            j = int(np.argmax(seg_len))
            n[j] += 1
    # place each segment's nodes uniformly; drop the shared right endpoint between segments
    nodes = [np.linspace(bounds[k], bounds[k + 1], n[k] + 1)[:-1] for k in range(len(n))]
    node_times = np.concatenate([*nodes, [bounds[-1]]])
    return np.ascontiguousarray(node_times, dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PRE -m pytest tests/test_wb_grid.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Checkpoint**

Run: `$PRE -m pytest tests/test_wb_grid.py tests/test_wb_gait.py -q`
Expected: PASS.

---

### Task 3: `P_DT` parameter + dt-parameterized integrator

**Files:**
- Modify: `t1_nmpc/wb/cost_wb.py` (param layout: add `P_DT`, bump `N_PARAM_WB`)
- Modify: `t1_nmpc/wb/ocp_wb.py` (`disc_dyn_expr` uses `p[P_DT]`; default `parameter_values[P_DT]=dt`)
- Test: `tests/test_wb_default_discrete.py`

**Interfaces:**
- Produces: module-level `P_DT` (int index) and updated `N_PARAM_WB` in `cost_wb.py`. `_rk4(model, x, u, dt_expr)` unchanged in signature (already takes `dt`); `ocp_wb` passes `p[P_DT]` as `dt_expr`.

- [ ] **Step 1: Write the failing test** (the integrator at `P_DT=dt` equals the old constant-dt step — pure CasADi, no rebuild)
```python
# tests/test_wb_default_discrete.py  (append)
import numpy as np
import casadi as cs
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_DT
from t1_nmpc.wb.ocp_wb import _rk4

def test_pdt_param_layout_and_default():
    # P_DT is a valid scalar index inside the param vector
    assert isinstance(P_DT, int) and 0 <= P_DT < N_PARAM_WB

def test_rk4_param_dt_equals_const_dt_at_nominal():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu)
    f_const = cs.Function("fc", [x, u], [_rk4(m, x, u, cfg.dt)])
    f_param = cs.Function("fp", [x, u], [_rk4(m, x, u, cs.SX(cfg.dt))])
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu); u0[2] = u0[8] = m.total_mass() * 9.81 / 2
    np.testing.assert_allclose(np.array(f_const(x0, u0)).ravel(),
                               np.array(f_param(x0, u0)).ravel(), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PRE -m pytest tests/test_wb_default_discrete.py::test_pdt_param_layout_and_default -v`
Expected: FAIL — `ImportError: cannot import name 'P_DT'`.

- [ ] **Step 3: Add `P_DT` to the param layout**

In `t1_nmpc/wb/cost_wb.py`, find the param-index block (the `P_XREF/P_UREF/P_CONTACT/P_SWINGZ/P_IMPACT` definitions and `N_PARAM_WB`). Append a scalar `P_DT` as the next index and grow the total by 1. Concretely, after the last existing index definition and before/at `N_PARAM_WB`:
```python
# ... existing P_IMPACT = slice(...) and the running offset ...
P_DT = N_PARAM_WB            # per-stage interval length dt_k (D4 event-aligned grid); scalar slot
N_PARAM_WB = N_PARAM_WB + 1  # grow the param vector by one
```
(Match the file's existing style: if indices are built from a running `_off` counter, set `P_DT = _off; _off += 1; N_PARAM_WB = _off`. Keep `P_DT` an `int`, not a `slice`.)

- [ ] **Step 4: Thread `p[P_DT]` into the integrator**

In `t1_nmpc/wb/ocp_wb.py`:
- import `P_DT` from `.cost_wb` (add to the existing `from .cost_wb import ...`).
- Replace the discrete-dynamics line:
```python
    if discrete:
        am.disc_dyn_expr = _rk4(model, x, u, p[P_DT])   # was: _rk4(model, x, u, cfg.dt)
```
- Set the default for `P_DT` so a fresh solver integrates at nominal dt. After `ocp.parameter_values = np.zeros(N_PARAM_WB)`:
```python
    pv0 = np.zeros(N_PARAM_WB); pv0[P_DT] = cfg.dt
    ocp.parameter_values = pv0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PRE -m pytest tests/test_wb_default_discrete.py -v`
Expected: PASS (param layout + integrator-equivalence).

- [ ] **Step 6: Checkpoint** (param-vector consumers still consistent — pure-Python cost/constraint tests)

Run: `$PRE -m pytest tests/test_wb_cost.py tests/test_wb_cost_walk.py tests/test_wb_constraints.py tests/test_wb_constraints_walk.py -q`
Expected: no NEW failures vs the baseline (the pre-existing `test_wb_constraints*` drift fails may remain; `test_wb_cost*` should PASS). If a cost/constraint test now fails on a param-index shift, the `P_DT` append broke an existing index — re-check Step 3 (P_DT must be appended AFTER all existing slots, never inserted).

---

### Task 4: Cost time-scaling by `dt_k` (faithful `Σ dt_k·L_k`)

**Files:**
- Modify: `t1_nmpc/wb/cost_wb.py` (`build_cost_conl`: `psi *= p[P_DT]`)
- Modify: `t1_nmpc/wb/ocp_wb.py` (`cost_scaling = ones(N+1)`)
- Test: `tests/test_wb_cost.py`

**Interfaces:**
- Consumes: `P_DT` (Task 3). `build_cost_conl(x, u, p, cfg, model) -> (y, yref, psi, r)` unchanged in signature; only `psi` is scaled.

- [ ] **Step 1: Write the failing test** (stage cost scales linearly with `P_DT`)
```python
# tests/test_wb_cost.py  (append)
import casadi as cs
from t1_nmpc.wb.cost_wb import build_cost_conl, P_DT

def test_stage_cost_scales_with_pdt():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", 68); u = cs.SX.sym("u", 40); p = cs.SX.sym("p", N_PARAM_WB)
    y, yref, psi, r = build_cost_conl(x, u, p, cfg, m)
    psi_fun = cs.Function("psi", [r, p], [psi])
    rv = 0.1 * np.ones(y.shape[0])
    p_a = np.zeros(N_PARAM_WB); p_a[P_DT] = cfg.dt
    p_b = np.zeros(N_PARAM_WB); p_b[P_DT] = 2.0 * cfg.dt
    va = float(psi_fun(rv, p_a)); vb = float(psi_fun(rv, p_b))
    assert abs(vb - 2.0 * va) < 1e-9 and va > 0      # cost doubles when dt doubles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PRE -m pytest tests/test_wb_cost.py::test_stage_cost_scales_with_pdt -v`
Expected: FAIL — `psi` does not depend on `p[P_DT]` yet (`vb == va`).

- [ ] **Step 3: Scale `psi` by `p[P_DT]` in `build_cost_conl`**

In `t1_nmpc/wb/cost_wb.py::build_cost_conl`, after `psi` is fully assembled (the LS term + all barrier terms) and before `return y, yref, psi, r`:
```python
    # Faithful time-integral: OCS2 scales each stage by its interval (SqpSolver.cpp:387,457).
    # Scaling the whole stage cost (LS + barriers) by the positive per-stage dt preserves convexity in r.
    psi = p[P_DT] * psi
    return y, yref, psi, r
```

- [ ] **Step 4: Set `cost_scaling = ones` so `P_DT` is the only time-weighting**

In `t1_nmpc/wb/ocp_wb.py`, in the `so = ocp.solver_options` block (near the other `so.*` settings):
```python
    # P_DT (in psi) is the SOLE time-weighting; disable acados' default cost_scaling=time_steps to
    # avoid double-scaling. Terminal (index N) unscaled. (D4)
    so.cost_scaling = np.ones(cfg.N + 1)
```

- [ ] **Step 5: Run the cost test**

Run: `$PRE -m pytest tests/test_wb_cost.py -v`
Expected: PASS (including `test_stage_cost_scales_with_pdt`).

- [ ] **Step 6: Build + M0-stand regression (the faithfulness gate for Tasks 3-4)**

This rebuilds the solver (~100 s) and proves the nominal grid is unchanged: with uniform `P_DT=dt` and `cost_scaling=ones`, the stand must still hold exactly as before.

Run: `$PRE python sim/wb_stand_gate.py`
Expected: `WB_M0={... "peak_tilt_rad": <0.05, "n_solver_failures": 0, "PASS": true}`.
If `PASS` is false or `peak_tilt` jumped: the spike assumption failed (acados default `cost_scaling` was NOT `time_steps`, so the nominal scale shifted). Remedy: the cost is now `dt`-scaled where before it was `time_steps`-scaled by acados — these are equal IFF the default was `time_steps`. If not, the absolute cost changed by a constant; confirm via the stand result and, if needed, note the constant and proceed (relative weighting is unchanged, so the stand should still hold). Record the finding in the divergence ledger.

- [ ] **Step 7: Checkpoint**

Run: `$PRE -m pytest tests/test_wb_cost.py tests/test_wb_default_discrete.py -q`
Expected: PASS.

---

### Task 5: Per-tick wiring — event-aligned grid + `P_DT` in `mpc_wb`

**Files:**
- Modify: `t1_nmpc/wb/mpc_wb.py` (`build_node_params` takes `node_times`; `step` builds the grid + fills `P_DT`; stores prev grid)
- Test: `tests/test_wb_mpc_walk.py`

**Interfaces:**
- Consumes: `event_aligned_grid(t0, gait, cfg)` (Task 2); `P_DT`, `N_PARAM_WB` (Task 3).
- Produces: `build_node_params(x_meas, node_times, comm_filt, gait, cfg, model) -> np.ndarray` — **signature changed**: `node_times` (array) replaces the scalar `t`. Sets `P[k, P_DT] = node_times[k+1]-node_times[k]` for `k<N`, and `P[N, P_DT] = P[N-1, P_DT]` (terminal slot unused by dynamics).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_wb_mpc_walk.py  (append)
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.grid_wb import event_aligned_grid
from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_DT
from t1_nmpc.wb.mpc_wb import build_node_params

def test_build_node_params_fills_pdt_and_aligns_switch():
    cfg = make_wb_config(); m = WBModel(cfg)
    t0 = 0.2
    nt = event_aligned_grid(t0, SLOW_WALK, cfg)
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(m.nominal_state(), nt, comm, SLOW_WALK, cfg, m)
    assert P.shape == (cfg.N + 1, N_PARAM_WB)
    # P_DT column == interval lengths
    np.testing.assert_allclose(P[:cfg.N, P_DT], np.diff(nt), atol=1e-12)
    # a switch lands on a node (grid invariant carried through)
    for s in SLOW_WALK.switch_times_in(t0, t0 + cfg.N * cfg.dt):
        assert np.min(np.abs(nt - s)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PRE -m pytest tests/test_wb_mpc_walk.py::test_build_node_params_fills_pdt_and_aligns_switch -v`
Expected: FAIL — `build_node_params()` still takes a scalar `t` / does not set `P_DT`.

- [ ] **Step 3: Update `build_node_params`**

In `t1_nmpc/wb/mpc_wb.py`, change the signature and body (import `event_aligned_grid` and `P_DT`):
```python
from .grid_wb import event_aligned_grid
from .cost_wb import N_PARAM_WB, P_XREF, P_UREF, P_CONTACT, P_SWINGZ, P_IMPACT, P_DT

def build_node_params(x_meas, node_times, comm_filt, gait, cfg, model) -> np.ndarray:
    """Per-node acados parameter matrix (N+1, N_PARAM_WB) on the given (possibly non-uniform)
    node_times: folded reference + contact flags + swing-Z + impact + per-stage dt (P_DT)."""
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
    P[cfg.N, P_DT] = dts[-1]                  # terminal slot unused by dynamics
    return P
```

- [ ] **Step 4: Update `step` to build the grid and store it**

In `WholeBodyMPC.step`, replace the uniform-grid `build_node_params` call. Currently:
```python
        P = build_node_params(x_meas, t, self._comm_filt, self._gait, cfg, self.model)
```
Replace with:
```python
        node_times = event_aligned_grid(t, self._gait, cfg)
        P = build_node_params(x_meas, node_times, self._comm_filt, self._gait, cfg, self.model)
```
(The per-node `solver.set(k, "p", P[k])` loop already pushes `P_DT` to acados — no change there.) Leave the warm-start call as-is for now; Task 6 generalizes it. In `reset` and `__init__`, add `self._node_times_prev = None` next to `self._x_prev/_u_prev/_t_prev`.

- [ ] **Step 5: Run test to verify it passes**

Run: `$PRE -m pytest tests/test_wb_mpc_walk.py::test_build_node_params_fills_pdt_and_aligns_switch -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint**

Run: `$PRE -m pytest tests/test_wb_mpc_walk.py -q`
Expected: PASS (or no new failures vs baseline).

---

### Task 6: Warm-start interpolation across non-uniform grids

**Files:**
- Modify: `t1_nmpc/wb/mpc_wb.py` (`shift_warmstart` signature + body; `step` passes prev/now grids; store prev grid)
- Test: `tests/test_wb_warmstart.py`

**Interfaces:**
- Produces: `shift_warmstart(x_prev, u_prev, node_times_prev, node_times_now, cfg) -> (xg, ug)` — **signature changed**: takes the previous and current node-time arrays (was scalar `t_prev, t_now`). Interpolates each state/input component by absolute time onto `node_times_now`; holds the last value past the previous horizon end.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_wb_warmstart.py  (append)
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.mpc_wb import shift_warmstart

def test_warmstart_identity_on_same_grid():
    cfg = make_wb_config()
    nt = np.arange(cfg.N + 1) * cfg.dt
    x_prev = np.cumsum(np.ones((cfg.N + 1, cfg.nx)), axis=0)
    u_prev = np.cumsum(np.ones((cfg.N, cfg.nu)), axis=0)
    xg, ug = shift_warmstart(x_prev, u_prev, nt, nt, cfg)   # identical grids -> identity
    np.testing.assert_allclose(xg, x_prev, atol=1e-9)
    np.testing.assert_allclose(ug, u_prev, atol=1e-9)

def test_warmstart_interpolates_onto_nonuniform_grid():
    cfg = make_wb_config()
    nt_prev = np.arange(cfg.N + 1) * cfg.dt
    # a non-uniform target grid spanning the same horizon
    nt_now = np.sort(np.concatenate(([0.0], np.cumsum(np.random.RandomState(0).uniform(
        0.5, 1.5, cfg.N)))) )
    nt_now = nt_now / nt_now[-1] * (cfg.N * cfg.dt)
    x_prev = (np.linspace(0, 1, cfg.N + 1)[:, None] * np.ones((1, cfg.nx)))  # linear in time
    u_prev = (np.linspace(0, 1, cfg.N)[:, None] * np.ones((1, cfg.nu)))
    xg, ug = shift_warmstart(x_prev, u_prev, nt_prev, nt_now, cfg)
    assert xg.shape == (cfg.N + 1, cfg.nx) and ug.shape == (cfg.N, cfg.nu)
    assert np.all(np.isfinite(xg)) and np.all(np.isfinite(ug))
    # linear field -> interpolation matches the analytic line at the new node times
    expected_x = (nt_now / (cfg.N * cfg.dt))[:, None] * np.ones((1, cfg.nx))
    np.testing.assert_allclose(xg, expected_x, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PRE -m pytest tests/test_wb_warmstart.py -v`
Expected: FAIL — `shift_warmstart` has the old `(x_prev, u_prev, t_prev, t_now, cfg)` signature.

- [ ] **Step 3: Generalize `shift_warmstart`**
```python
# t1_nmpc/wb/mpc_wb.py  — replace shift_warmstart
def shift_warmstart(x_prev, u_prev, node_times_prev, node_times_now, cfg):
    """Interpolate the previous primal (defined on node_times_prev) onto node_times_now by absolute
    time; hold-last past the previous horizon end. Generalizes the old uniform time-shift to the
    D4 event-aligned (non-uniform, per-tick-varying) grid."""
    tp = np.asarray(node_times_prev, float)
    tn = np.asarray(node_times_now, float)
    xg = np.empty((cfg.N + 1, cfg.nx))
    for j in range(cfg.nx):
        xg[:, j] = np.interp(tn, tp, x_prev[:, j])              # np.interp holds-last past tp[-1]
    # u defined on intervals: sample at the START of each new interval from prev interval-starts
    up_t = tp[:cfg.N]
    ug = np.empty((cfg.N, cfg.nu))
    for j in range(cfg.nu):
        ug[:, j] = np.interp(tn[:cfg.N], up_t, u_prev[:, j])
    return xg, ug
```

- [ ] **Step 4: Wire prev/now grids in `step`**

In `WholeBodyMPC.step`, update the warm-start branch:
```python
        if self._x_prev is not None:
            xg, ug = shift_warmstart(self._x_prev, self._u_prev, self._node_times_prev, node_times, cfg)
        else:
            u0 = np.zeros(cfg.nu); u0[2] = u0[8] = self.model.total_mass() * 9.81 / 2.0
            xg = np.tile(x_meas, (cfg.N + 1, 1)); ug = np.tile(u0, (cfg.N, 1))
```
At the end of `step`, store the grid alongside the trajectory:
```python
        self._x_prev, self._u_prev, self._t_prev = x_traj, u_traj, t
        self._node_times_prev = node_times
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PRE -m pytest tests/test_wb_warmstart.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint**

Run: `$PRE -m pytest tests/test_wb_warmstart.py tests/test_wb_mpc_walk.py -q`
Expected: PASS.

---

### Task 7: Integration — M0 regression, M1 measurement, full suite

**Files:**
- Modify: `sim/wb_walk_gate.py` (add a `min_foot_z_at_stance_activation` metric to the printed JSON)
- Test: the gates + full suite (manual acceptance, not a unit test)

**Interfaces:**
- Consumes: the full D4 stack (Tasks 1-6). No new public functions.

- [ ] **Step 1: Add the premature-contact instrument to the walk gate**

In `sim/wb_walk_gate.py`, inside the loop where the per-node contact flags / foot heights are available, track the minimum measured foot height at the tick where a foot's STANCE constraint newly activates, and add it to the returned dict:
```python
    # near the metrics assembly:
    result["min_foot_z_at_stance_activation"] = float(min_foot_z_at_activation)  # ~0 means no airborne stomp
```
(If the gate does not currently expose per-foot height at activation, compute it from the MuJoCo foot body z at the control tick when `contact_flags` transitions swing->stance for that foot.)

- [ ] **Step 2: M0 stand regression (must still PASS)**

Run: `$PRE python sim/wb_stand_gate.py`
Expected: `"PASS": true`, `peak_tilt_rad < 0.05`, `n_solver_failures: 0`. (Confirms the variable-dt machinery is inert at the nominal grid.)

- [ ] **Step 3: M1 walk measurement (the acceptance bar)**

Run: `$PRE python sim/wb_walk_gate.py --vx 0.3 --duration 10.0`
Expected (vs the pre-D4 baseline `mean_vx 0.165, peak_tilt 2.22, n_fail 353`):
- `peak_tilt_rad` reduced and trending upright;
- `n_fail` (ACADOS_NAN) down from 353;
- `min_foot_z_at_stance_activation` ≈ 0 (no STANCE constraint firing on an airborne foot) — the core D4 success signal.
Record the numbers. Per the spec success bar, **measurable improvement** is the pass condition; a full gate pass is not required while D1 is open.

- [ ] **Step 4: Full suite — no new failures**

Run: `$PRE -m pytest tests/ -q -p no:cacheprovider`
Expected: the new grid/warmstart/cost tests PASS; total failures ≤ the 10 pre-existing drift fails (none introduced by D4).

- [ ] **Step 5: Checkpoint** — record M0/M1 numbers in the task notes for the ledger update (Task 8).

---

### Task 8: Update the divergence ledger

**Files:**
- Modify: `docs/2026-06-25-t1controller-divergences.md`

- [ ] **Step 1: Mark D4 closed with the bounded divergences**

In the divergence table, change D4's status from `🟠 OPEN Tier 2` to `🔧 CLOSED (single-node adaptation)`. Add a "Closed this session (Tier 2)" entry stating: event-aligned variable-dt grid via per-stage `P_DT` (dynamics + faithful `psi·dt_k` cost scaling, confirmed against `SqpSolver.cpp:387,457`); grid is a fixed-N adaptation of `timeDiscretizationWithEvents`; **documented bounded divergences** — single node per switch (no zero-length jump duplication; identity jump map), remainder-spread vs short pre-event interval, fixed N=31. Record the M0/M1 measured numbers from Task 7. If Task 4 Step 6 found acados's default `cost_scaling` was not `time_steps`, note the constant-scale finding here.

- [ ] **Step 2: Checkpoint** — final read-through of the ledger for consistency.

---

## Self-Review

**Spec coverage:** grid construction (T2), `switch_times_in` (T1), `P_DT` integrator (T3), faithful cost scaling (T4), per-tick wiring (T5), warm-start generalization (T6), testing/acceptance (T7), ledger (T8) — every spec section maps to a task. ✅

**Placeholder scan:** all steps carry concrete code/commands; the one conditional ("if the gate doesn't expose foot height at activation") names exactly what to compute. ✅

**Type consistency:** `event_aligned_grid(t0, gait, cfg)->ndarray`, `switch_times_in(t0,t1)->list[float]`, `build_node_params(x_meas, node_times, ...)`, `shift_warmstart(x_prev,u_prev,node_times_prev,node_times_now,cfg)`, `P_DT:int` — used identically across tasks. ✅

**Faithfulness:** grid + cost scaling grounded in OCS2 source with citations; bounded divergences documented and carried into the ledger (T8). ✅
