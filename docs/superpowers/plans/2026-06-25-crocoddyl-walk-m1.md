# Crocoddyl Whole-Body MPC — M1 Implementation Plan (Walking)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend the M0 crocoddyl backend to a closed-loop forward walk in MuJoCo, faithful to `t1_controller`.

**Architecture:** Backward-compatible extensions to `croco_costs`/`croco_problem`/`croco_mpc` (M0 defaults preserved) + a per-node receding gait scheduler driving the already-faithful `gait_wb`/`reference_wb`. Spec: `docs/superpowers/specs/2026-06-25-crocoddyl-walk-m1-design.md`.

**Tech Stack:** Python 3.12, crocoddyl 3.2.1, pinocchio 4.0.0, mujoco 3.10, pytest. Conda env `t1mpc`.

## Global Constraints

- **Run:** `env -u PYTHONPATH conda run -n t1mpc python -m pytest <args>` from repo root `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc`.
- **M0 must keep passing.** All extensions are gated by new optional params whose defaults reproduce M0 exactly. After each task, the M0 tests (`test_croco_costs`, `test_croco_problem`, `test_croco_mpc`, `test_wb_stand_croco`) stay green.
- **Faithful values from `config_wb`/`gait_wb`** (never hardcode a value that exists there): swing weights `swingfoot_cost_weights=[1e4,1e4,5,5,2,2,2]`, `friction_mu=0.4`, `friction_barrier_mu=0.2`/`friction_barrier_delta=5.0`, `cop_barrier_mu=0.1`/`cop_barrier_delta=0.03`, `foot_rect_x=(±0.1115)`/`foot_rect_y=(±0.05)`, `foot_linvel_err_gain_xy=20`, `foot_pos_err_gain_z=100`, `foot_ori_err_gain=80`, `Q_final`, `terminal_scale=4.0`, `arm_swing_*`. Gait: `SLOW_WALK`, `swing_z`, `impact_proximity`, `contact_flags`.
- **Contact gains for walking = `[0, foot_linvel_err_gain_xy]`** (kp=0 for emergent landing + the §2 decomposition stabilization cost); the M0 stand keeps `[foot_pos_err_gain_z, foot_linvel_err_gain_xy]`.
- **Single-RTI** `maxiter=1`. Do NOT raise it to make the walk converge without escalating to the human (spec §6).
- **Deviation telemetry is mandatory** in the walk gate: stance-foot slip/sink/flatness, swing-z error, single-RTI `stop`/`‖h‖`, CoP, per-foot `fz` (spec §7).

## File Structure

| File | Disposition |
|---|---|
| `t1_nmpc/wb/croco_activations.py` | **create** — `RelaxedBarrier` |
| `t1_nmpc/wb/croco_costs.py` | **modify** — swing block, relaxed friction/CoP, stance stabilization, terminal, yawed `R_foot` (gated) |
| `t1_nmpc/wb/croco_problem.py` | **modify** — `make_node` swing/gains; `build_walk_problem` |
| `t1_nmpc/wb/croco_mpc.py` | **modify** — walk mode |
| `t1_nmpc/wb/reference_wb.py` | **modify** — `build_reference_66` 68→66 wrapper |
| `sim/wb_walk_croco.py` | **create** — walk gate + telemetry |
| `tests/test_croco_activations.py`, `tests/test_croco_walk_costs.py`, `tests/test_croco_walk_problem.py`, `tests/test_croco_walk_mpc.py`, `tests/test_wb_walk_croco.py` | **create** |

---

## Task 1: Branch + commit spec & plan

- [ ] **Step 1: Branch from master**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git checkout master && git checkout -b crocoddyl-walk-m1
```

- [ ] **Step 2: Commit spec + plan**

```bash
git add docs/superpowers/specs/2026-06-25-crocoddyl-walk-m1-design.md docs/superpowers/plans/2026-06-25-crocoddyl-walk-m1.md
git commit -m "docs(walk): M1 walking design + plan"
```
Expected: commit on `crocoddyl-walk-m1`.

---

## Task 2: `RelaxedBarrier` activation

**Files:** Create `t1_nmpc/wb/croco_activations.py`; Test `tests/test_croco_activations.py`.

**Interfaces — Produces:** `class RelaxedBarrier(crocoddyl.ActivationModelAbstract)`, `__init__(nr, mu, delta)`. Per element, desired `h ≥ 0`; penalizes violation. `calc` sets `data.a_value`; `calcDiff` sets `data.Ar`, `data.Arr`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_activations.py
import numpy as np, crocoddyl
from t1_nmpc.wb.croco_activations import RelaxedBarrier

def test_relaxed_barrier_penalizes_violation_with_correct_sign():
    rb = RelaxedBarrier(1, mu=0.2, delta=5.0)
    d = rb.createData()
    # h large & positive (deep inside feasible): small penalty
    rb.calc(d, np.array([100.0])); v_ok = d.a_value
    # h small/negative (violation): much larger penalty, and gradient pushes h UP (negative Ar)
    rb.calc(d, np.array([-1.0])); v_bad = d.a_value
    assert v_bad > v_ok
    rb.calcDiff(d, np.array([-1.0]))
    assert float(np.asarray(d.Ar).ravel()[0]) < 0.0          # d(penalty)/dh < 0 -> increasing h reduces penalty
    assert float(np.asarray(d.Arr).ravel()[0]) > 0.0         # convex

def test_relaxed_barrier_multidim():
    rb = RelaxedBarrier(3, 0.1, 0.03); d = rb.createData()
    rb.calc(d, np.array([1.0, 0.5, -0.1])); assert np.isfinite(d.a_value)
    rb.calcDiff(d, np.array([1.0, 0.5, -0.1]))
    assert np.asarray(d.Ar).shape[0] == 3 and np.asarray(d.Arr).shape == (3, 3)
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`)

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_activations.py -v`

- [ ] **Step 3: Implement** (validated in `spikes/croco_m1_faithful_spike.py`)

```python
# t1_nmpc/wb/croco_activations.py
"""Custom crocoddyl activations for the faithful T1 walking costs."""
from __future__ import annotations
import numpy as np
import crocoddyl


class RelaxedBarrier(crocoddyl.ActivationModelAbstract):
    """OCS2 RelaxedBarrierPenalty: per element, h>=0 desired.
    value = -mu*log(h)               for h > delta
          = mu*(0.5*((h-2d)/d)^2 - log(d))   for h <= delta   (quadratic continuation, C1 at h=delta).
    Penalizes constraint VIOLATION (h small/negative); gradient pushes h up."""
    def __init__(self, nr, mu, delta):
        crocoddyl.ActivationModelAbstract.__init__(self, nr)
        self.mu = float(mu); self.delta = float(delta)

    def calc(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        v = np.where(h > d, -mu * np.log(np.maximum(h, 1e-12)),
                     mu * (0.5 * ((h - 2 * d) / d) ** 2 - np.log(d)))
        data.a_value = float(np.sum(v))

    def calcDiff(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        dv = np.where(h > d, -mu / np.maximum(h, 1e-12), mu * (h - 2 * d) / d ** 2)
        d2 = np.where(h > d, mu / np.maximum(h, 1e-12) ** 2, mu / d ** 2 * np.ones_like(h))
        np.asarray(data.Ar)[:] = dv
        np.fill_diagonal(np.asarray(data.Arr), d2)
```

- [ ] **Step 4: Run — expect PASS.** Commit.

```bash
git add t1_nmpc/wb/croco_activations.py tests/test_croco_activations.py
git commit -m "feat(walk): RelaxedBarrier activation (OCS2 relaxed-barrier penalty)"
```

---

## Task 3: `reference_wb` 68→66 wrapper

**Files:** Modify `t1_nmpc/wb/reference_wb.py`; Test `tests/test_croco_walk_costs.py` (the reference part).

**Interfaces — Produces:** `build_reference_66(x_meas_68, comm_filt, gait, t0, node_times, cfg, model) -> np.ndarray (N+1, 66)`. Calls `build_reference`, drops `s,v_s` (cols 66:68), discards `u_ref`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_walk_costs.py  (reference section)
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.model import RobotModel  # via WBModel below
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb import reference_wb

def test_build_reference_66_shape_and_drops_path_slots():
    cfg = make_wb_config(); wb = WBModel(cfg)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height
    x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    nt = np.arange(cfg.N + 1) * cfg.dt
    xr = reference_wb.build_reference_66(x0, np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]),
                                         SLOW_WALK, 0.0, nt, cfg, wb)
    assert xr.shape == (cfg.N + 1, 66)
    # forward command -> base x advances across the horizon
    assert xr[-1, 0] > xr[0, 0]
```

(Note: `build_reference` needs `model.nominal_state()`, `model.total_mass()` — `WBModel` provides these; pass `wb`.)

- [ ] **Step 2: Run — expect FAIL** (`build_reference_66` missing).

- [ ] **Step 3: Implement** — append to `reference_wb.py`:

```python
def build_reference_66(x_meas, comm_filt, gait, t0, node_times, cfg, model):
    """66-dim per-node state reference for the crocoddyl walk (drops the 68-dim acados s,v_s
    path slots; the gravity-split u_ref is discarded — crocoddyl's ID supplies gravity)."""
    x_ref, _u_ref = build_reference(x_meas, comm_filt, gait, t0, node_times, cfg, model)
    return np.ascontiguousarray(x_ref[:, :66])
```

- [ ] **Step 4: Run — expect PASS.** Commit `feat(walk): reference_wb 68->66 wrapper`.

---

## Task 4: `build_costs` — swing block, relaxed friction/CoP, stance stabilization, terminal

**Files:** Modify `t1_nmpc/wb/croco_costs.py`; Test `tests/test_croco_walk_costs.py`.

**Interfaces — Produces:** extended
`build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg, swing=None, planted=None, terminal=False, walk=False)`.
- `swing`: `None` or `dict(fid=int, z=float, w_z=float)` — the swinging foot, its target z (from `gait.swing_z(t)[0]`), and the per-node weight scale (`base_swing_weight × gait.impact_proximity(t)`).
- `planted`: `dict[int, pin.SE3]` (for the yawed `R_foot` rotation), required when `walk=True`.
- `terminal`: state-only `Q_final·terminal_scale` cost.
- `walk=True`: use the faithful `FrictionCone`+`CoP`+`RelaxedBarrier` (vs M0's `WrenchCone`+`QuadraticBarrier`) + add the stance z/foot-flat stabilization costs. `walk=False` reproduces M0 exactly.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_croco_walk_costs.py`)

```python
import crocoddyl, pinocchio as pin
from t1_nmpc.wb import croco_costs

def _ctx():
    cfg = make_wb_config(); wb = WBModel(cfg)
    state = crocoddyl.StateMultibody(wb.model); act = crocoddyl.ActuationModelFloatingBase(state)
    q0 = pin.neutral(wb.model); q0[2] = cfg.nominal_base_height
    q0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    data = wb.model.createData(); pin.framesForwardKinematics(wb.model, data, q0)
    planted = {f: data.oMf[f].copy() for f in wb.contact_fids}
    x0 = np.concatenate([q0, np.zeros(wb.model.nv)])
    return cfg, wb, state, act, planted, x0

def test_walk_costs_single_support_has_swing_and_relaxed_terms():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids
    nu = wb.model.nv + 6                                  # single support
    swing = dict(fid=R, z=0.05, w_z=1e3)
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L], cfg,
                                    swing=swing, planted=planted, walk=True)
    names = set(costs.costs.todict().keys())
    assert {"xreg", "ureg", "tau_lim", "swing_z", "swing_vel", "swing_flat",
            f"friction_{L}", f"cop_{L}", f"stance_z_{L}", f"stance_flat_{L}"} <= names
    assert not any(n.startswith("wrenchcone") for n in names)   # walk uses split friction/CoP

def test_walk_costs_m0_path_unchanged():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids
    nu = wb.model.nv + 12
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L, R], cfg)  # walk=False default
    names = set(costs.costs.todict().keys())
    assert any(n.startswith("wrenchcone") for n in names)       # M0 WrenchCone preserved
    assert not any(n.startswith("swing") for n in names)

def test_walk_costs_terminal_is_qfinal():
    cfg, wb, state, act, planted, x0 = _ctx()
    L, R = wb.contact_fids; nu = wb.model.nv + 12
    costs = croco_costs.build_costs(state, act, nu, x0[:66], np.zeros(3), [L, R], cfg,
                                    planted=planted, walk=True, terminal=True)
    names = list(costs.costs.todict().keys())
    assert names == ["xreg"]                                    # state-only terminal
    w = np.asarray(costs.costs["xreg"].cost.activation.weights)
    assert np.allclose(w, cfg.Q_final[:66] * cfg.terminal_scale)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — replace the body of `build_costs` (keep `_control_weights`, `_BIG`):

```python
def _frame_vel_zero(state, fid, nu):
    return crocoddyl.ResidualModelFrameVelocity(state, fid, pin.Motion.Zero(), _LWA, nu)

def build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg,
                swing=None, planted=None, terminal=False, walk=False):
    nv = state.pinocchio.nv
    nc = 6 * len(stance_fids)
    costs = crocoddyl.CostModelSum(state, nu)

    # --- terminal: state-only Q_final*terminal_scale ---
    if terminal:
        xres = crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)
        wq = crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q_final[:66], float) * float(cfg.terminal_scale))
        costs.addCost("xreg", crocoddyl.CostModelResidual(state, wq, xres), 1.0)
        return costs

    # --- 1-5: carried from M0 (state, CoM, input-reg, torque-limit, joint-limit) ---
    costs.addCost("xreg", crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q[:66], float)),
        crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)), 1.0)
    costs.addCost("com", crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelCoMPosition(state, np.asarray(com_ref, float), nu)), 1.0)
    costs.addCost("ureg", crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(_control_weights(nv, nc, np.asarray(cfg.R, float))),
        crocoddyl.ResidualModelControl(state, nu)), 1.0)
    tau_lim = np.asarray(cfg.torque_limit, float)
    costs.addCost("tau_lim", crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-tau_lim, tau_lim)),
        crocoddyl.ResidualModelJointEffort(state, actuation, np.zeros(actuation.nu), nu, False)),
        float(cfg.jointtorque_weight))
    lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
    lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float); ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
    costs.addCost("joint_lim", crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
        crocoddyl.ResidualModelState(state, np.zeros(state.nx), nu)), float(cfg.joint_limit_barrier_mu))

    if not walk:
        # --- M0 stance: combined WrenchCone + QuadraticBarrier (unchanged) ---
        box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)
        for fid in stance_fids:
            cone = crocoddyl.WrenchCone(np.eye(3), float(cfg.friction_mu), box)
            costs.addCost(f"wrenchcone_{fid}", crocoddyl.CostModelResidual(
                state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.ResidualModelContactWrenchCone(state, fid, cone, nu, False)),
                float(cfg.friction_cone_reg))
        return costs

    # --- M1 WALK: faithful relaxed-barrier friction + CoP, yawed R_foot, stance stabilization ---
    box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)
    for fid in stance_fids:
        R_foot = np.asarray(planted[fid].rotation, float)
        fcone = crocoddyl.FrictionCone(R_foot, float(cfg.friction_mu), 4, False)
        fres = crocoddyl.ResidualModelContactFrictionCone(state, fid, fcone, nu, False)
        costs.addCost(f"friction_{fid}", crocoddyl.CostModelResidual(
            state, RelaxedBarrier(fres.nr, cfg.friction_barrier_mu, cfg.friction_barrier_delta), fres),
            float(cfg.friction_cone_reg))
        cop = crocoddyl.CoPSupport(R_foot, box)
        cres = crocoddyl.ResidualModelContactCoPPosition(state, fid, cop, nu, False)
        costs.addCost(f"cop_{fid}", crocoddyl.CostModelResidual(
            state, RelaxedBarrier(cres.nr, cfg.cop_barrier_mu, cfg.cop_barrier_delta), cres), 1.0)
        # stance z->ground + foot-flat stabilization (the kp=0 decomposition; weights seeded from gains)
        z_ground = float(planted[fid].translation[2])
        zres = crocoddyl.ResidualModelFrameTranslation(state, fid, planted[fid].translation, nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 1.], float))   # z only
        costs.addCost(f"stance_z_{fid}", crocoddyl.CostModelResidual(state, zact, zres),
                      float(cfg.foot_pos_err_gain_z))
        flat = crocoddyl.ResidualModelFramePlacement(state, fid, planted[fid], nu)
        flatact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 0., 1., 1., 1.], float))  # rot only
        costs.addCost(f"stance_flat_{fid}", crocoddyl.CostModelResidual(state, flatact, flat),
                      float(cfg.foot_ori_err_gain))

    # --- M1 swing-foot block (foot-flat, vel, z-track), x impact_proximity (folded into w_z by caller) ---
    if swing is not None:
        sfid = int(swing["fid"]); wz = float(swing["w_z"])
        sw = float(cfg.swingfoot_cost_weights[0])  # ori_xy weight 1e4 (impact scale applied by caller)
        # foot-flat orientation (FramePlacement, rotation rows only)
        fp = crocoddyl.ResidualModelFramePlacement(state, sfid, _flat_se3(planted, sfid), nu)
        fpact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 0., 1., 1., 1.], float))
        costs.addCost("swing_flat", crocoddyl.CostModelResidual(state, fpact, fp), sw)
        # lin-vel xy -> 0 (5), ang-vel -> 0 (2)
        velact = crocoddyl.ActivationModelWeightedQuad(np.array(
            [cfg.swingfoot_cost_weights[2], cfg.swingfoot_cost_weights[3], 0.,
             cfg.swingfoot_cost_weights[4], cfg.swingfoot_cost_weights[5], cfg.swingfoot_cost_weights[6]], float))
        costs.addCost("swing_vel", crocoddyl.CostModelResidual(state, velact, _frame_vel_zero(state, sfid, nu)), 1.0)
        # swing-z tracking (strong cost; replaces the hard SwingLegVerticalConstraint)
        ztarget = np.array([0., 0., float(swing["z"])], float)
        zres = crocoddyl.ResidualModelFrameTranslation(state, sfid, ztarget, nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 1.], float))
        costs.addCost("swing_z", crocoddyl.CostModelResidual(state, zact, zres), wz)
    return costs
```

Add near the top of the module: `import pinocchio as pin`, `_LWA = pin.LOCAL_WORLD_ALIGNED`, `from .croco_activations import RelaxedBarrier`, and a helper:
```python
def _flat_se3(planted, fid):
    import pinocchio as pin
    p = planted[fid].copy(); p.rotation = np.eye(3); return p   # foot-flat orientation reference
```

(Implementer note: the swing-z target uses world-frame z = `swing["z"]` directly via `ResidualModelFrameTranslation`'s xyz; if `FrameTranslation`'s ref must be a full 3-vec in world, set x,y from the current foot xy so only z is weighted — the `[0,0,1]` activation already zeroes x,y error, so the ref's x,y are irrelevant.)

- [ ] **Step 4: Run — expect PASS;** then run M0 cost tests `tests/test_croco_costs.py` (must stay green — the `walk=False` default is byte-compatible).

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_walk_costs.py tests/test_croco_costs.py -v`

- [ ] **Step 5: Commit** `feat(walk): build_costs swing + relaxed friction/CoP + stance stabilization + terminal`.

---

## Task 5: `make_node` swing/gains + `build_walk_problem`

**Files:** Modify `t1_nmpc/wb/croco_problem.py`; Test `tests/test_croco_walk_problem.py`.

**Interfaces — Produces:**
- `make_node(stance_fids, x_ref, com_ref, planted, swing=None, gains=None, terminal=False, walk=False)`.
- `build_walk_problem(x0_66, t_gait, comm_filt, gait, x_meas_68) -> ShootingProblem`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_walk_problem.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.croco_problem import T1ProblemBuilder

def _b():
    cfg = make_wb_config(); wb = WBModel(cfg); b = T1ProblemBuilder(cfg, wb)
    x0_68 = np.zeros(68); x0_68[2] = cfg.nominal_base_height
    x0_68[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    return cfg, wb, b, x0_68

def test_build_walk_problem_per_node_stance_matches_gait():
    cfg, wb, b, x0_68 = _b()
    # pick t_gait inside a single-support window of SLOW_WALK so the horizon has DS and SS nodes
    prob = b.build_walk_problem(x0_68[:66], 0.3, np.array([0.3,0.,cfg.nominal_base_height,0.]), SLOW_WALK, x0_68)
    assert len(prob.runningModels) == cfg.N
    nus = {m.nu for m in prob.runningModels}
    assert nus <= {wb.model.nv + 6, wb.model.nv + 12}    # SS=39 and/or DS=45 present
    # at least one single-support node exists in a SLOW_WALK horizon
    assert (wb.model.nv + 6) in nus

def test_build_walk_problem_solves():
    cfg, wb, b, x0_68 = _b()
    prob = b.build_walk_problem(x0_68[:66], 0.3, np.array([0.3,0.,cfg.nominal_base_height,0.]), SLOW_WALK, x0_68)
    s = crocoddyl.SolverIntro(prob); s.setCallbacks([])
    xs = [x0_68[:66].copy() for _ in range(cfg.N+1)]; us = list(prob.quasiStatic([x0_68[:66].copy() for _ in range(cfg.N)]))
    s.solve(xs, us, 30, False, 1e-9)
    assert np.all(np.isfinite(np.asarray(s.xs)))
```

- [ ] **Step 2: Run — expect FAIL** (`build_walk_problem` missing).

- [ ] **Step 3: Implement** — extend `croco_problem.py`:

```python
from .gait_wb import mode_to_stance
from . import reference_wb

class T1ProblemBuilder:
    # __init__, _planted: unchanged. Add walk gains:
    #   self._walk_gains = np.array([0.0, cfg.foot_linvel_err_gain_xy], float)   # kp=0 emergent landing

    def make_node(self, stance_fids, x_ref, com_ref, planted, swing=None, gains=None, terminal=False, walk=False):
        nu = self.nv + 6 * len(stance_fids)
        g = self._gains if gains is None else np.asarray(gains, float)
        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        for i, fid in enumerate(stance_fids):
            contacts.addContact("%d_c%d" % (i, fid),
                                crocoddyl.ContactModel6D(self.state, fid, planted[fid], _LWA, nu, g))
        costs = build_costs(self.state, self.actuation, nu, x_ref, com_ref, stance_fids, self.cfg,
                            swing=swing, planted=planted, terminal=terminal, walk=walk)
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contacts, costs)
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else self.dt)

    def build_walk_problem(self, x0_66, t_gait, comm_filt, gait, x_meas_68):
        x0 = np.asarray(x0_66, float)
        planted = self._planted(x0)
        node_times = t_gait + np.arange(self.N + 1) * self.dt
        x_ref = reference_wb.build_reference_66(x_meas_68, comm_filt, gait, t_gait, node_times, self.cfg, self.wb)
        data = self.model.createData()
        running = []
        for k in range(self.N):
            tk = float(node_times[k])
            flags = gait.contact_flags(tk)                       # (left, right) bool
            stance = [self.foot_fids[i] for i in (0, 1) if flags[i]]
            swing = None
            for i in (0, 1):
                if not flags[i]:                                 # swinging foot i
                    z = gait.swing_z(tk, i)[0]
                    w_z = float(self.cfg.swingfoot_cost_weights[0]) * 1e-1 * gait.impact_proximity(tk, i)
                    # base swing-z weight tunable; seeded here. (impact scaling folded in.)
                    swing = dict(fid=self.foot_fids[i], z=z, w_z=max(w_z, 1e-6))
            pin.centerOfMass(self.model, data, x_ref[k][:self.model.nq])
            running.append(self.make_node(stance, x_ref[k], data.com[0].copy(), planted,
                                          swing=swing, gains=self._walk_gains, walk=True))
        flagsN = gait.contact_flags(float(node_times[self.N]))
        stanceN = [self.foot_fids[i] for i in (0, 1) if flagsN[i]] or self.foot_fids
        pin.centerOfMass(self.model, data, x_ref[self.N][:self.model.nq])
        term = self.make_node(stanceN, x_ref[self.N], data.com[0].copy(), planted,
                              gains=self._walk_gains, terminal=True, walk=True)
        return crocoddyl.ShootingProblem(x0, running, term)
```

(Add `self._walk_gains` in `__init__`. Note `build_reference_66` builds the 66-dim x_ref whose `[:nq]` is the reference config for the per-node CoM.)

- [ ] **Step 4: Run — expect PASS;** then M0 `tests/test_croco_problem.py` stays green (make_node defaults unchanged). Commit `feat(walk): make_node swing/gains + build_walk_problem gait scheduler`.

---

## Task 6: `CrocoMPC` walk mode

**Files:** Modify `t1_nmpc/wb/croco_mpc.py`; Test `tests/test_croco_walk_mpc.py`.

**Interfaces — Produces:** `CrocoMPC.__init__(cfg, wb, max_iter=1, gait=None)`; when `gait` is set, `step(x_meas_68, t, command=None)` runs the walk path (rebuild + stance-aware `u_traj`). M0 (no `gait`) path unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_walk_mpc.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.croco_mpc import CrocoMPC

def test_walk_step_rebuilds_and_emits_stance_aware_u():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0)
    res = mpc.step(x0, 0.0, command=np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]))
    assert res.x_traj.shape == (cfg.N+1, 68) and res.u_traj.shape == (cfg.N, 40)
    assert res.status == 0 and np.all(np.isfinite(res.u_traj))

def test_walk_advances_gait_clock_a_few_steps():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0); x = x0.copy()
    for _ in range(5):
        res = mpc.step(x, 0.0, command=np.array([0.3,0.,cfg.nominal_base_height,0.]))
        assert np.all(np.isfinite(res.x_traj)); x = res.x_traj[1].copy()
    assert mpc._t_gait > 0.0
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — in `croco_mpc.py`:
- `__init__(..., gait=None)`: store `self.gait = gait`; `self._t_gait = 0.0`; `self._comm = np.array([0.,0.,cfg.nominal_base_height,0.])`.
- `step(self, x_meas_68, t, command=None)`: if `self.gait is None`, run the existing M0 body. Else:

```python
        x66 = np.asarray(x_meas_68, float)[:66]
        if command is not None:
            from .reference_wb import filter_command
            self._comm = filter_command(self._comm, command)
        self._t_gait += float(self.cfg.dt)                   # advance gait clock one control dt
        prob = self.builder.build_walk_problem(x66, self._t_gait, self._comm, self.gait, np.asarray(x_meas_68, float))
        self.solver = crocoddyl.SolverIntro(prob); self.solver.setCallbacks([])
        xs = [x66.copy()] + self._xs[2:] + [self._xs[-1]]
        us = list(prob.quasiStatic([x66.copy() for _ in range(self.N)]))   # re-quasiStatic: dimension-safe across nu changes
        t0 = time.perf_counter(); self.solver.solve(xs, us, self.max_iter, False, _REG)
        self.last_solve_s = time.perf_counter() - t0
        self._xs = list(self.solver.xs); self._us = list(self.solver.us)
        xs_arr = np.asarray(self.solver.xs)
        ok = bool(np.all(np.isfinite(xs_arr)) and self.solver.isFeasible)
        x_traj = np.zeros((self.N+1, 68)); x_traj[:, :66] = xs_arr
        u_traj = self._acados_layout_walk(self.solver.us, self._t_gait)
        if not ok:
            x_traj[:] = x_traj[0]; u_traj = np.zeros((self.N, 40))
        return MPCResult(x_traj=x_traj, u_traj=u_traj, feasible=ok, solve_time=self.last_solve_s,
                         mode_schedule=None, status=0 if ok else 1, node_times=t + self._node_times, u_phys_traj=None)
```

- Add `_acados_layout_walk(self, us, t_gait)` — stance-aware mapping (force → `W_l` if foot is left-stance, `W_r` if right-stance, by querying `self.gait.contact_flags(t_gait + k*dt)`):

```python
    def _acados_layout_walk(self, us, t_gait):
        out = np.zeros((self.N, 40)); dt = float(self.cfg.dt)
        for k, u in enumerate(us):
            u = np.asarray(u, float); a = u[:self.nv]; forces = u[self.nv:]
            out[k, 12:39] = a[6:33]
            flags = self.gait.contact_flags(t_gait + k * dt)      # (left, right)
            stance = [i for i in (0, 1) if flags[i]]              # contact order matches make_node enumerate
            for j, side in enumerate(stance):
                sl = slice(0, 6) if side == 0 else slice(6, 12)   # left->W_l, right->W_r
                out[k, sl] = forces[6*j:6*j+6]
        return out
```

- [ ] **Step 4: Run — expect PASS;** M0 `tests/test_croco_mpc.py` stays green (gait=None path unchanged). Commit `feat(walk): CrocoMPC walk mode (gait clock, rebuild, stance-aware u_traj)`.

---

## Task 7: Closed-loop walk gate + telemetry (M1 acceptance)

**Files:** Create `sim/wb_walk_croco.py`, `tests/test_wb_walk_croco.py`.

**Interfaces — Produces:** `run_wb_walk_croco(duration_s=12.0, vx=0.3, control_hz=500.0) -> dict` with `WALK_GATE` metrics + telemetry, driving the closed loop via the synchronous control law (reuse the viewer's loop pattern from `sim/wb_stand_croco.py`, MPC at the runtime control rate, `CrocoMPC(gait=SLOW_WALK)`, `command=[vx,0,height,0]`).

- [ ] **Step 1: Write the failing acceptance test**

```python
# tests/test_wb_walk_croco.py
import pytest
from sim.wb_walk_croco import run_wb_walk_croco

@pytest.mark.slow
def test_m1_walk_forward_no_fall():
    m = run_wb_walk_croco(duration_s=12.0, vx=0.3)
    assert m["n_solver_failures"] == 0
    assert m["peak_tilt_rad"] < 0.2
    assert m["final_base_z"] > 0.85 * 0.6734
    assert m["com_advance_m"] > 1.0           # walked forward
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

- [ ] **Step 3: Implement `sim/wb_walk_croco.py`** — a synchronous closed loop modeled on `sim/wb_stand_croco.py::run_wb_stand_view` (control law: `CrocoMPC.step(x_meas, t, command) -> to_joint_command_wb -> transport.write_command`), but:
- `mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)`; each control tick pass `command=np.array([vx,0,cfg.nominal_base_height,0])`.
- accumulate **telemetry** each tick: base tilt (`tilt_from_quat_wxyz`), base z, per-foot world z + xy (stance-foot slip/sink), foot-flatness (foot R vs world up), CoP/`fz` if available from the result, solver `status`/`last_solve_s`, swing-foot z vs `gait.swing_z`.
- return `dict(n_solver_failures, peak_tilt_rad, final_base_z, com_advance_m, mean_vx, n_steps, stance_slip_mm, stance_sink_mm, swing_z_err_mm, median_solve_ms, ...)`.
- `__main__`: `--duration --vx --view` (reuse the viewer launch + `os._exit(0)` clean-exit pattern from `wb_stand_croco.py`).

Implementer: reuse `make_wb_config`, `WBModel`, `MujocoTransport`, `to_joint_command_wb`, `tilt_from_quat_wxyz`, and the runtime rate (`transport.rt.cfg.control_hz`). Keep the loop synchronous (single-process real-time, per M0 spec §8).

- [ ] **Step 4: Run the gate.**

Run: `env -u PYTHONPATH conda run -n t1mpc python sim/wb_walk_croco.py --duration 12 --vx 0.3`
Expected: prints `WALK_GATE={...}` with `n_solver_failures=0`, forward CoM advance > 1 m, no fall.

**If single-RTI (`maxiter=1`) cannot hold the walk** (divergence / falls / slip telemetry bad): this is the flagged uncertainty (spec §6). Do NOT silently raise `maxiter`. Record the failure mode + telemetry in the task report and STOP for human review — options are better warm-start, the contact-gain escalation (custom per-axis model), or accepting >1 iteration as a deviation.

- [ ] **Step 5: Commit** `feat(walk): closed-loop M1 walk gate + deviation telemetry`.

---

## Self-Review

**Spec coverage:** §3 module changes (Tasks 2-7) ✓; §4 gait scheduler (Task 5) ✓; §5 cost terms incl. swing/relaxed-barrier/stabilization/terminal/yawed-R_foot (Task 4) ✓; §6 walk loop incl. rebuild + stance-aware u_traj + single-RTI + warm-start-via-quasiStatic (Task 6) ✓; §7 gate + telemetry (Task 7) ✓; deviations: emergent-xy (no footstep code) ✓, swing-z cost (Task 4) ✓, contact `[0,20]`+stabilization (Tasks 4,5) ✓.

**Backward-compat:** every change is gated (`walk=False`, `swing=None`, `gait=None` defaults reproduce M0); each task re-runs the M0 tests.

**Placeholder scan:** swing-z base weight + the two stabilization weights are explicitly **tuning** values (seeded from gains), driven by Task 7 telemetry — not placeholders, but flagged as tunable.

**Known-risk flags carried into the plan:** single-RTI walk convergence (Task 7 Step 4 — stop-for-review, no silent maxiter bump); warm-start across `nu` via re-`quasiStatic` (Task 6); contact-gain decomposition adequacy (Task 7 telemetry, custom-model escalation).

**Type consistency:** `build_costs(... swing, planted, terminal, walk)` signature identical across Task 4 def and Task 5 call; `swing=dict(fid,z,w_z)` shape consistent; `MPCResult` fields unchanged; `CrocoMPC(gait=)` / `step(command=)` consistent Tasks 6-7.
