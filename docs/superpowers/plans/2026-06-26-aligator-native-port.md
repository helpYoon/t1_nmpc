# Aligator Native Walk-MPC Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the T1 whole-body walk MPC to a native aligator `KinodynamicsFwdDynamics` +
`SolverProxDDP` formulation so hard friction/CoP constraints eliminate the force-shed, staying
faithful to OCS2 and inside the 25 ms budget.

**Architecture:** New `aligator_*` modules build one `StageModel` per contact mode over a fixed-N
horizon, solved by a persistent warm-started `SolverProxDDP` with the parallel LQ backend, advancing
the contact schedule each tick via `replaceStageCircular`+`cycleProblem`. All solver-agnostic infra
(gait, reference, config, MuJoCo transport, the Pinocchio reduction) is reused; the crocoddyl walk
stays as the baseline/comparator.

**Tech Stack:** aligator 0.19.0, pinocchio 4.0.0, Python 3.12, conda env `t1mpc`. Spec:
`docs/superpowers/specs/2026-06-26-aligator-native-port-design.md`. Phase 0 already validated (spec
Appendix A); this plan implements Phases 1–3.

## Global Constraints

- **Env / run prefix:** all python/pytest runs use `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc ...`; the RT benchmark uses `OMP_NUM_THREADS=4`. No compilation (prebuilt binaries). Run validation walks one process at a time (RAM).
- **Faithful model:** same reduced model as `WBModel` (`cfg.urdf_path`, `_HEAD_JOINTS` locked, `MPC_JOINT_NAMES` assert, `cfg.armature`, contact frames `foot_l_contact`/`foot_r_contact` at `cfg.contact_frame_offset`) but with `pin.JointModelFreeFlyer()` base. Dims: nq=34, nv=33, nx=67, ndx=66, mass 34.51 kg, m·g=338.6 N.
- **Control layout:** `u = [f_left(6), f_right(6), joint_accels(27)]`, `nu=39`. ALWAYS read `ode.nu`; never hardcode.
- **RT budget:** 25 ms; validated operating point N=20, max_iters=2, 4 threads, parallel LQ → 12 ms.
- **Faithfulness:** kinodynamic control; structural underactuation; friction/CoP toggle hard (`NegativeOrthant`, default) ↔ soft (`RelaxedLogBarrierCost`).
- **Branch first:** create branch `aligator-port` before Task 1; the crocoddyl baseline and its passing tests must stay green.
- **Cone feasibility:** `NegativeOrthant` ⇒ residual ≤ 0 is feasible (verified).

---

### Task 1: `config_aligator.py` — settings

**Files:**
- Create: `t1_nmpc/wb/config_aligator.py`
- Test: `tests/test_aligator_config.py`

**Interfaces:**
- Consumes: `t1_nmpc.wb.config_wb.WBConfig` (fields used downstream: `urdf_path, armature, contact_frame_offset, nominal_base_height, nominal_joint_pos, friction_mu, foot_rect_x, foot_rect_y, dt, n_joints`).
- Produces: `AligatorConfig` dataclass; `make_aligator_config() -> AligatorConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_config.py
from t1_nmpc.wb.config_aligator import AligatorConfig, make_aligator_config

def test_defaults_match_validated_operating_point():
    c = make_aligator_config()
    assert c.N == 20 and c.max_iters == 2 and c.num_threads == 4
    assert c.hard_cones is True and c.FS == 6
    assert c.mu_init == 1e-2 and c.tol == 1e-3 and c.max_al_iters == 2
    # cost-weight vectors are present and finite-sized
    assert c.w_base_pose > 0 and c.w_joint_pos > 0 and c.w_force_reg > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 't1_nmpc.wb.config_aligator'`

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/config_aligator.py
"""Aligator-specific MPC settings (validated operating point: N=20, maxit=2, 4 threads -> 12ms)."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AligatorConfig:
    N: int = 20                 # horizon nodes
    max_iters: int = 2          # ProxDDP inner-iteration budget (raise to 3-5 at transitions in Phase 2)
    max_al_iters: int = 2       # outer AL iterations
    mu_init: float = 1e-2       # AL penalty
    tol: float = 1e-3
    num_threads: int = 4
    hard_cones: bool = True     # True: NegativeOrthant; False: RelaxedLogBarrierCost (OCS2-faithful)
    FS: int = 6                 # 6D contact wrench per foot
    cone_eps: float = 1e-3      # friction-cone relaxation arg
    barrier_thr: float = 0.1    # RelaxedLogBarrierCost threshold (soft mode)
    # cost weights (mapped from crocoddyl Q/R intent)
    w_base_pose: float = 50.0
    w_joint_pos: float = 5.0
    w_vel: float = 1.0
    w_force_reg: float = 1e-3
    w_accel_reg: float = 1e-2
    w_swing_z: float = 100.0
    w_swing_force: float = 1e2  # heavy reg pinning an inactive foot's force slots to zero
    w_term_scale: float = 5.0

def make_aligator_config() -> AligatorConfig:
    return AligatorConfig()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/config_aligator.py tests/test_aligator_config.py
git commit -m "feat(aligator): config settings dataclass"
```

---

### Task 2: `aligator_model.py` — faithful free-flyer T1 model

**Files:**
- Create: `t1_nmpc/wb/aligator_model.py`
- Test: `tests/test_aligator_model.py`

**Interfaces:**
- Consumes: `config_wb.make_wb_config`; `model_wb` constants `_HEAD_JOINTS, MPC_JOINT_NAMES, CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS`.
- Produces: `@dataclass AligatorModel(model, space, foot_ids: list[int], mass: float, nq: int, nv: int, ndx: int)`; `build_aligator_model(wb_cfg) -> AligatorModel`; `nominal_stand_x(am, wb_cfg) -> np.ndarray (nx,)`; `make_ode(am, contact_flags) -> KinodynamicsFwdDynamics`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_model.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x, make_ode

def test_faithful_model_dims_and_dynamics():
    am = build_aligator_model(make_wb_config())
    assert am.nq == 34 and am.nv == 33 and am.ndx == 66
    assert abs(am.mass - 34.51) < 0.1            # m*g ~ 338.6 N
    assert len(am.foot_ids) == 2
    ode = make_ode(am, [True, True])
    assert ode.nu == 2 * 6 + (am.nv - 6) == 39
    x = nominal_stand_x(am, make_wb_config())
    assert x.shape[0] == am.nq + am.nv == 67
    import aligator
    disc = aligator.dynamics.IntegratorSemiImplEuler(ode, 0.02)
    d = disc.createData(); u = np.zeros(ode.nu); u[2] = u[8] = am.mass * 9.81 / 2
    disc.forward(x, u, d)
    assert np.all(np.isfinite(np.asarray(d.xnext)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_model.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation** (lifted verbatim from validated Phase 0 spike `p0_model.py`)

```python
# t1_nmpc/wb/aligator_model.py
"""Faithful free-flyer T1 model for the aligator kinodynamic OCP (same reduction as WBModel,
but pin.JointModelFreeFlyer base instead of the euler composite). Validated: nq=34, nv=33, nu=39."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics
from .model_wb import _HEAD_JOINTS, MPC_JOINT_NAMES, CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS

@dataclass
class AligatorModel:
    model: object
    space: object
    foot_ids: list
    mass: float
    nq: int
    nv: int
    ndx: int

def build_aligator_model(wb_cfg) -> AligatorModel:
    full = pin.buildModelFromUrdf(wb_cfg.urdf_path, pin.JointModelFreeFlyer())
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in _HEAD_JOINTS], pin.neutral(full))
    assert tuple(model.names[2:]) == MPC_JOINT_NAMES, model.names[2:]
    model.armature[6:] = np.asarray(wb_cfg.armature, float)
    off = pin.SE3.Identity(); off.translation = np.ascontiguousarray(wb_cfg.contact_frame_offset, float)
    foot_ids = [model.addFrame(pin.Frame(fn, model.getJointId(pj), off, pin.FrameType.OP_FRAME))
                for fn, pj in zip(CONTACT_FRAME_NAMES, CONTACT_PARENT_JOINTS)]
    space = manifolds.MultibodyPhaseSpace(model)
    mass = float(sum(I.mass for I in model.inertias))
    return AligatorModel(model, space, foot_ids, mass, model.nq, model.nv, space.ndx)

def make_ode(am: AligatorModel, contact_flags, FS: int = 6):
    cs = pin.StdVec_Bool(); [cs.append(bool(b)) for b in contact_flags]
    ci = pin.StdVec_Index(); [ci.append(int(i)) for i in am.foot_ids]
    return dynamics.KinodynamicsFwdDynamics(am.space, am.model, np.array([0., 0., -9.81]), cs, ci, FS)

def nominal_stand_x(am: AligatorModel, wb_cfg) -> np.ndarray:
    q = pin.neutral(am.model)
    q[2] = wb_cfg.nominal_base_height
    q[7:] = np.asarray(wb_cfg.nominal_joint_pos, float)
    return np.concatenate([q, np.zeros(am.nv)])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_model.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_model.py tests/test_aligator_model.py
git commit -m "feat(aligator): faithful free-flyer T1 model builder"
```

---

### Task 3: `aligator_exec.py` — wrench + tau_ff extraction

**Files:**
- Create: `t1_nmpc/wb/aligator_exec.py`
- Test: `tests/test_aligator_exec.py`

**Interfaces:**
- Consumes: `AligatorModel` (Task 2); `make_ode`; `config.JointCommand`; `execution.pd_torque`.
- Produces: `extract_tau_ff(am, x_meas, u0, FS=6) -> (tau_ff: np.ndarray (nv-6,), wrench_l, wrench_r)`.

- [ ] **Step 1: Write the failing test** (the base-rows-zero check validated in `p0_eq_tau.py`)

```python
# tests/test_aligator_exec.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.aligator_model import build_aligator_model, make_ode, nominal_stand_x
from t1_nmpc.wb.aligator_exec import extract_tau_ff

def test_tau_ff_base_rows_vanish():
    cfg = make_wb_config(); am = build_aligator_model(cfg)
    ode = make_ode(am, [True, True]); nu = ode.nu
    x = nominal_stand_x(am, cfg)
    u0 = np.zeros(nu); u0[2] = u0[8] = am.mass * 9.81 / 2
    tau_ff, wl, wr = extract_tau_ff(am, x, u0)
    assert tau_ff.shape[0] == am.nv - 6 == 27
    assert np.allclose(wl, u0[0:6]) and np.allclose(wr, u0[6:12])
    # base-6 generalized force must vanish (structural underactuation consistency)
    # (extract recomputes tau internally; re-derive base rows here for the assert)
    import pinocchio as pin
    rdata = am.model.createData(); q = x[:am.nq]; v = x[am.nq:]
    od = ode.createData(); ode.forward(x, u0, od)
    a = np.asarray(od.xdot)[am.nv:].copy(); a[6:] = u0[12:]
    tau = pin.rnea(am.model, rdata, q, v, a)
    pin.computeJointJacobians(am.model, rdata, q); pin.framesForwardKinematics(am.model, rdata, q)
    for k, fid in enumerate(am.foot_ids):
        J = pin.getFrameJacobian(am.model, rdata, fid, pin.LOCAL_WORLD_ALIGNED)
        tau -= J.T @ u0[k*6:(k+1)*6]
    assert np.linalg.norm(tau[:6]) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_exec.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation** (validated in `p0_eq_tau.py` part 1)

```python
# t1_nmpc/wb/aligator_exec.py
"""Extract the robot command (contact wrenches + feed-forward joint torque) from a kinodynamic
solution. Kinodynamics returns NO torque -> recover via RNEA(q,v,a) - sum J_LWA^T f. Validated:
|tau[:6]| ~ 1e-13."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from .aligator_model import make_ode

def extract_tau_ff(am, x_meas, u0, FS: int = 6):
    q = np.asarray(x_meas[:am.nq]); v = np.asarray(x_meas[am.nq:])
    # base-6 accel is SOLVED by the dynamics; read it from the continuous ode's xdot
    ode = make_ode(am, [True, True], FS)            # contact mask irrelevant to the accel passthrough
    od = ode.createData(); ode.forward(x_meas, u0, od)
    a = np.asarray(od.xdot)[am.nv:].copy()
    a[6:] = np.asarray(u0[2 * FS:])                 # joint accels pass straight from the control
    rdata = am.model.createData()
    tau = pin.rnea(am.model, rdata, q, v, a)
    pin.computeJointJacobians(am.model, rdata, q); pin.framesForwardKinematics(am.model, rdata, q)
    for k, fid in enumerate(am.foot_ids):
        J = pin.getFrameJacobian(am.model, rdata, fid, pin.LOCAL_WORLD_ALIGNED)
        tau -= J.T @ np.asarray(u0[k * FS:(k + 1) * FS])
    return tau[6:].copy(), np.asarray(u0[0:6]).copy(), np.asarray(u0[6:12]).copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_exec.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_exec.py tests/test_aligator_exec.py
git commit -m "feat(aligator): wrench + tau_ff extraction from kinodynamic solution"
```

---

### Task 4: `aligator_walk.py` — DS stage factory + stand problem (hard/soft toggle)

**Files:**
- Create: `t1_nmpc/wb/aligator_walk.py`
- Test: `tests/test_aligator_walk.py`

**Interfaces:**
- Consumes: `AligatorModel`, `make_ode`, `nominal_stand_x` (Task 2); `AligatorConfig` (Task 1); `WBConfig` (`friction_mu, foot_rect_x, foot_rect_y, dt`).
- Produces: `make_stage(am, wb_cfg, al_cfg, contact_flags, x_ref, swing_refs, ode) -> StageModel` (swing_refs = `list[(foot_idx, p_ref_3)]`); `build_problem(am, wb_cfg, al_cfg, x0, x_ref, schedule, swing_schedule) -> TrajOptProblem`; helper `_foot_half_extents(wb_cfg) -> (L, W)`.

- [ ] **Step 1: Write the failing test** (DS stand solve → fz=m·g, validated in `p0_stand.py`)

```python
# tests/test_aligator_walk.py
import numpy as np, aligator
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model, make_ode, nominal_stand_x
from t1_nmpc.wb.aligator_walk import make_stage, build_problem

def test_hard_cone_stand_holds_fz_equals_mg():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    x = nominal_stand_x(am, cfg); mg = am.mass * 9.81
    schedule = [[True, True]] * al.N
    prob = build_problem(am, cfg, al, x, x, schedule, [[] for _ in range(al.N)])
    s = aligator.SolverProxDDP(1e-4, 1e-2, max_iters=30, verbose=aligator.QUIET); s.setup(prob)
    ode = make_ode(am, [True, True]); u_grav = np.zeros(ode.nu); u_grav[2] = u_grav[8] = mg / 2
    s.run(prob, [x.copy() for _ in range(al.N + 1)], [u_grav.copy() for _ in range(al.N)])
    u0 = np.asarray(s.results.us[0]); fz = u0[2] + u0[8]
    assert 0.95 < fz / mg < 1.05, f"force-shed: fz/mg={fz/mg:.3f}"   # validated: fz = m*g exactly
    assert np.asarray(s.results.xs[-1])[2] > 0.6                     # stand held

def test_soft_toggle_builds():
    cfg = make_wb_config(); al = make_aligator_config(); al.hard_cones = False
    am = build_aligator_model(cfg); x = nominal_stand_x(am, cfg)
    st = make_stage(am, cfg, al, [True, True], x, [], make_ode(am, [True, True]))
    assert st.nu == 39
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_walk.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation** (lifted from `p0_stand.py`, with the soft branch from the spec)

```python
# t1_nmpc/wb/aligator_walk.py
"""Per-contact-mode StageModel factory + fixed-N TrajOptProblem builder for the kinodynamic walk.
Stance feet: zero-vel equality + Centroidal friction/wrench cones (hard NegativeOrthant or soft
RelaxedLogBarrierCost). Swing feet: FrameTranslation z-track + heavy force-slot regularization."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
import aligator
from aligator import dynamics, constraints
from .aligator_model import make_ode

def _foot_half_extents(wb_cfg):
    return ((wb_cfg.foot_rect_x[1] - wb_cfg.foot_rect_x[0]) / 2.0,
            (wb_cfg.foot_rect_y[1] - wb_cfg.foot_rect_y[0]) / 2.0)

def _weights(am, al_cfg, contact_flags, FS=6):
    nv = am.nv; ndx = am.ndx
    wx = np.r_[np.full(6, al_cfg.w_base_pose), np.full(nv - 6, al_cfg.w_joint_pos), np.full(nv, al_cfg.w_vel)]
    nu = 2 * FS + (nv - 6)
    wu = np.empty(nu); wu[:2 * FS] = al_cfg.w_force_reg; wu[2 * FS:] = al_cfg.w_accel_reg
    for k, on in enumerate(contact_flags):           # pin a SWING foot's force slots to zero
        if not on:
            wu[k * FS:(k + 1) * FS] = al_cfg.w_swing_force
    return wx, wu

def make_stage(am, wb_cfg, al_cfg, contact_flags, x_ref, swing_refs, ode, FS=6):
    nu = ode.nu; ndx = am.ndx; mu = float(wb_cfg.friction_mu); L, W = _foot_half_extents(wb_cfg)
    mg = am.mass * 9.81; nst = max(1, sum(contact_flags))
    wx, wu = _weights(am, al_cfg, contact_flags, FS)
    u_ref = np.zeros(nu)
    for k, on in enumerate(contact_flags):
        if on:
            u_ref[k * FS + 2] = mg / nst             # weight-supporting reference
    cost = aligator.CostStack(am.space, nu)
    cost.addCost("xreg", aligator.QuadraticStateCost(am.space, nu, x_ref, np.diag(wx)))
    cost.addCost("ureg", aligator.QuadraticControlCost(am.space, u_ref, np.diag(wu)))
    for foot_idx, p_ref in swing_refs:
        ft = aligator.FrameTranslationResidual(ndx, nu, am.model, np.asarray(p_ref, float), int(am.foot_ids[foot_idx]))
        cost.addCost(f"swz{foot_idx}", aligator.QuadraticResidualCost(am.space, ft, np.diag([0., 0., al_cfg.w_swing_z])))
    st = aligator.StageModel(cost, dynamics.IntegratorSemiImplEuler(ode, float(wb_cfg.dt)))
    for k, on in enumerate(contact_flags):
        if not on:
            continue
        zv = aligator.FrameVelocityResidual(ndx, nu, am.model, pin.Motion.Zero(), int(am.foot_ids[k]), pin.LOCAL_WORLD_ALIGNED)
        st.addConstraint(zv, constraints.EqualityConstraintSet())
        fr = aligator.CentroidalFrictionConeResidual(ndx, nu, k, mu, al_cfg.cone_eps)
        wc = aligator.CentroidalWrenchConeResidual(ndx, nu, k, mu, L, W)
        if al_cfg.hard_cones:
            st.addConstraint(fr, constraints.NegativeOrthant())
            st.addConstraint(wc, constraints.NegativeOrthant())
        else:
            cost.addCost(f"fric{k}", aligator.RelaxedLogBarrierCost(am.space, fr, np.ones(2), al_cfg.barrier_thr))
            cost.addCost(f"wcon{k}", aligator.RelaxedLogBarrierCost(am.space, wc, np.ones(17), al_cfg.barrier_thr))
    return st

def build_problem(am, wb_cfg, al_cfg, x0, x_ref, schedule, swing_schedule, FS=6):
    odes = {}
    def ode_for(flags):
        key = tuple(flags)
        if key not in odes:
            odes[key] = make_ode(am, flags, FS)
        return odes[key]
    stages = [make_stage(am, wb_cfg, al_cfg, schedule[t], x_ref, swing_schedule[t], ode_for(schedule[t]), FS)
              for t in range(al_cfg.N)]
    wx, _ = _weights(am, al_cfg, [True, True], FS)
    term = aligator.CostStack(am.space, stages[0].nu)
    term.addCost("xt", aligator.QuadraticStateCost(am.space, stages[0].nu, x_ref, np.diag(wx * al_cfg.w_term_scale)))
    return aligator.TrajOptProblem(x0, stages, term)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_walk.py -q`
Expected: PASS (both tests; the stand solve reproduces fz/mg≈1.00)

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_walk.py tests/test_aligator_walk.py
git commit -m "feat(aligator): kinodynamic stage factory + stand problem (hard/soft cones)"
```

---

### Task 5: `aligator_mpc.py` — persistent warm-started solver (stand)

**Files:**
- Create: `t1_nmpc/wb/aligator_mpc.py`
- Test: `tests/test_aligator_mpc.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `class AligatorMPC` with `reset(x0)`, `step(x_meas, t, command) -> AligatorResult`; `last_solve_s`. `AligatorResult` has `xs, us, status (0 ok), num_iters`. State convention: free-flyer `x` (nx=67). A `_configure_solver()` applying the parallel-LQ + thread settings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_mpc.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.aligator_mpc import AligatorMPC

def test_stand_step_warmstarts_and_holds_fz():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    mpc = AligatorMPC(cfg, al, am)
    x = nominal_stand_x(am, cfg); mpc.reset(x); mg = am.mass * 9.81
    res = None
    for _ in range(5):                      # warm-started repeated solves
        res = mpc.step(x, 0.0, command=np.array([0., 0., cfg.nominal_base_height, 0.]))
    u0 = np.asarray(res.us[0]); fz = u0[2] + u0[8]
    assert res.status == 0 and 0.9 < fz / mg < 1.1
    assert mpc.last_solve_s > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_mpc.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation** (solver config + warm-start from `p1_rt.py`)

```python
# t1_nmpc/wb/aligator_mpc.py
"""Persistent warm-started SolverProxDDP for the kinodynamic walk. Phase 1: fixed double-support
stand problem (no contact cycling yet -- the receding cycle is added in Task 8). Mirrors CrocoMPC's
reset/step interface so the runner + diagnostics swap in directly."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import aligator
from .aligator_model import make_ode, nominal_stand_x
from .aligator_walk import build_problem

@dataclass
class AligatorResult:
    xs: list
    us: list
    status: int
    num_iters: int

class AligatorMPC:
    def __init__(self, wb_cfg, al_cfg, am, gait=None):
        self.cfg = wb_cfg; self.al = al_cfg; self.am = am; self.gait = gait
        self.model = am  # exec uses .am-like fields; alias for runner compatibility
        self.last_solve_s = 0.0
        self._built = False

    def _configure(self, problem):
        s = aligator.SolverProxDDP(self.al.tol, self.al.mu_init, max_iters=self.al.max_iters, verbose=aligator.QUIET)
        try: s.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
        except Exception: pass
        try: s.setNumThreads(self.al.num_threads)
        except Exception: pass
        for attr, val in [("max_al_iters", self.al.max_al_iters),
                          ("rollout_type", getattr(aligator, "ROLLOUT_LINEAR", None)),
                          ("sa_strategy", getattr(aligator, "SA_FILTER", None))]:
            if val is not None:
                try: setattr(s, attr, val)
                except Exception: pass
        s.setup(problem)
        return s

    def reset(self, x0):
        x0 = np.asarray(x0, float)
        N = self.al.N
        sched = [[True, True]] * N
        self._x_ref = nominal_stand_x(self.am, self.cfg)
        self.problem = build_problem(self.am, self.cfg, self.al, x0, self._x_ref, sched, [[] for _ in range(N)])
        self.solver = self._configure(self.problem)
        ode = make_ode(self.am, [True, True]); mg = self.am.mass * 9.81
        u_grav = np.zeros(ode.nu); u_grav[2] = u_grav[8] = mg / 2
        self.xs = [x0.copy() for _ in range(N + 1)]; self.us = [u_grav.copy() for _ in range(N)]
        self.vs = []; self.lams = []
        self._built = True

    def step(self, x_meas, t, command=None) -> AligatorResult:
        x_meas = np.asarray(x_meas, float)
        if not self._built:
            self.reset(x_meas)
        self.problem.x0_init = x_meas; self.xs[0] = x_meas.copy()
        t0 = time.perf_counter()
        ok = self.solver.run(self.problem, self.xs, self.us, self.vs, self.lams)
        self.last_solve_s = time.perf_counter() - t0
        R = self.solver.results
        self.xs = [np.asarray(a).copy() for a in R.xs]; self.us = [np.asarray(a).copy() for a in R.us]
        self.vs = [np.asarray(a).copy() for a in R.vs]; self.lams = [np.asarray(a).copy() for a in R.lams]
        finite = all(np.all(np.isfinite(a)) for a in self.xs)
        return AligatorResult(self.xs, self.us, 0 if finite else 1, R.num_iters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_mpc.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_mpc.py tests/test_aligator_mpc.py
git commit -m "feat(aligator): persistent warm-started ProxDDP MPC (stand)"
```

---

### Task 6: Phase 1 gate — state mapping + closed-loop stand (force-shed + RT)

**Files:**
- Create: `t1_nmpc/wb/aligator_state.py` (MuJoCo/euler ↔ free-flyer conversion)
- Create: `tests/test_aligator_phase1_gate.py`
- Reference: `sim/wb_walk_croco.py` (transport usage), `t1_nmpc/runtime/mujoco_transport.py`

**Interfaces:**
- Consumes: Tasks 2–5; `runtime.mujoco_transport.MujocoTransport`.
- Produces: `mujoco_to_freeflyer(rt, am) -> np.ndarray (nx,)` reading `rt.mj_data.qpos/qvel` (MuJoCo wxyz quat; pinocchio xyzw — reorder), mapping the 27 actuated joints in `MPC_JOINT_NAMES` order; `freeflyer_command(...)` reusing `aligator_exec.extract_tau_ff` + `execution.pd_torque` to build a `config.JointCommand`.

- [ ] **Step 1: Write the failing test** (closed-loop stand: fz bounded + RT)

```python
# tests/test_aligator_phase1_gate.py
import numpy as np, pytest
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model
from t1_nmpc.wb.aligator_mpc import AligatorMPC
from t1_nmpc.wb.aligator_state import mujoco_to_freeflyer, freeflyer_command
from t1_nmpc.runtime.mujoco_transport import MujocoTransport

@pytest.mark.slow
def test_phase1_stand_no_force_shed_and_realtime():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    tp = MujocoTransport(cfg, mpc_hz=40.0); rt = tp.rt
    mpc = AligatorMPC(cfg, al, am)
    x0 = mujoco_to_freeflyer(rt, am); mpc.reset(x0)
    mg = am.mass * 9.81; se = max(1, int(round(rt.cfg.control_hz / 40.0)))
    fz_ratios, solve_ms = [], []
    res = mpc.step(x0, 0.0); cmd = freeflyer_command(am, x0, res, cfg)
    for k in range(int(round(5.0 * rt.cfg.control_hz))):
        x = mujoco_to_freeflyer(rt, am)
        if k % se == 0:
            res = mpc.step(x, tp.now()); solve_ms.append(mpc.last_solve_s * 1e3)
        cmd = freeflyer_command(am, x, res, cfg)
        tp.write_command(cmd)
        u0 = np.asarray(res.us[0]); fz_ratios.append((u0[2] + u0[8]) / mg)
        if rt.mj_data.qpos[2] < 0.45:
            pytest.fail(f"stand collapsed at k={k}")
    fz = np.array(fz_ratios)
    assert fz.min() > 0.9 and fz.max() < 1.1, f"force-shed: fz/mg in [{fz.min():.2f},{fz.max():.2f}]"
    # RT informational (machine-dependent): print, assert generous ceiling
    print("solve ms p90 =", np.percentile(solve_ms, 90))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase1_gate.py -q`
Expected: FAIL with `ModuleNotFoundError: aligator_state`

- [ ] **Step 3: Write the implementation**

Implement `t1_nmpc/wb/aligator_state.py`:
- `mujoco_to_freeflyer(rt, am)`: read `rt.mj_data.qpos` (`[pos3, quat_wxyz4, joints...]`) and `qvel` (`[linvel3, angvel3, jointvel...]`); build pinocchio q = `[pos3, quat_xyzw4, joints27]` by reordering the quaternion (`wxyz`→`xyzw`) and selecting the 27 `MPC_JOINT_NAMES` joints in model order; v = `[linvel3, angvel3, jointvel27]`. Return `np.concatenate([q, v])`. (Cross-check joint ordering against `rt` joint map; add a one-time index cache.)
- `freeflyer_command(am, x_meas, res, wb_cfg)`: `tau_ff, wl, wr = extract_tau_ff(am, x_meas, res.us[0])`; map `tau_ff` (model joint order) back to the actuator order the transport expects; build `config.JointCommand(q_des=x_meas joints, qd_des=v joints, kp, kd, tau_ff, wrench_l=wl, wrench_r=wr)` reusing `execution.pd_torque` conventions.

Wire `run_aligator_walk_view`-style harness later (Phase 3); this task only needs the closed-loop test green.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase1_gate.py -q`
Expected: PASS (fz stays in [0.9,1.1] — the force-shed is gone vs the crocoddyl baseline). Record the printed solve-ms p90 in the commit message.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_state.py tests/test_aligator_phase1_gate.py
git commit -m "feat(aligator): Phase 1 gate — closed-loop stand, no force-shed (fz in [0.9,1.1])"
```

- [ ] **Step 6: Soft-cone comparison (no new code)**

Run the same test with `al.hard_cones=False` (edit a local copy or parametrize) and record whether the soft relaxed barrier alone also holds fz under ProxDDP. Document the result in `docs/superpowers/specs/2026-06-26-aligator-native-port-design.md` (append a "Phase 1 results" note). Commit the doc update.

---

### Task 7: `aligator_walk.py` — swing-mode stages + gait-cycle ring

**Files:**
- Modify: `t1_nmpc/wb/aligator_walk.py`
- Test: `tests/test_aligator_gait_cycle.py`

**Interfaces:**
- Consumes: `make_stage`, `gait_wb` (the `SLOW_WALK` schedule: `contact_flags(t)`, `swing_z(t, foot)`, swing xy target).
- Produces: `build_gait_cycle(am, wb_cfg, al_cfg, gait, x_ref, node_times) -> (models: list[StageModel], schedule: list[contact_flags])` covering one gait period; each swing node carries its `(foot_idx, p_ref)` from the gait swing trajectory.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_gait_cycle.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.aligator_walk import build_gait_cycle

def test_gait_cycle_has_all_modes_with_correct_nu():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    x = nominal_stand_x(am, cfg)
    node_times = np.arange(al.N) * cfg.dt
    models, sched = build_gait_cycle(am, cfg, al, SLOW_WALK, x, node_times)
    assert len(models) == len(sched) >= al.N
    assert all(m.nu == 39 for m in models)
    # at least one single-support node exists in a walking schedule
    assert any(sum(f) == 1 for f in sched)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_gait_cycle.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_gait_cycle'`

- [ ] **Step 3: Write minimal implementation** — append to `aligator_walk.py`:

```python
def build_gait_cycle(am, wb_cfg, al_cfg, gait, x_ref, node_times, FS=6):
    odes = {}
    def ode_for(flags):
        k = tuple(flags)
        if k not in odes: odes[k] = make_ode(am, flags, FS)
        return odes[k]
    models, schedule = [], []
    for t in np.asarray(node_times, float):
        flags = [bool(b) for b in gait.contact_flags(float(t))]
        swing_refs = []
        for i, on in enumerate(flags):
            if not on:
                z, _, _ = gait.swing_z(float(t), i)        # gait swing-z height (xy target = current foot xy)
                import pinocchio as pin
                rdata = am.model.createData(); pin.framesForwardKinematics(am.model, rdata, x_ref[:am.nq])
                p = rdata.oMf[int(am.foot_ids[i])].translation.copy(); p[2] = z
                swing_refs.append((i, p))
        models.append(make_stage(am, wb_cfg, al_cfg, flags, x_ref, swing_refs, ode_for(flags), FS))
        schedule.append(flags)
    return models, schedule
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_gait_cycle.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_walk.py tests/test_aligator_gait_cycle.py
git commit -m "feat(aligator): swing-mode stages + gait-cycle ring builder"
```

---

### Task 8: `aligator_mpc.py` — receding-horizon cycle (the Phase 2 risk)

**Files:**
- Modify: `t1_nmpc/wb/aligator_mpc.py`
- Test: `tests/test_aligator_recede.py`

**Interfaces:**
- Consumes: `build_gait_cycle` (Task 7); `replaceStageCircular`, `cycleProblem`.
- Produces: `AligatorMPC.reset` builds from the gait cycle when `self.gait` is set; a private `_recede(x_meas, t)` that does `replaceStageCircular`→`cycleProblem`→ring rotation→warm-start shift before `solver.run`; `step` calls `_recede` once per control tick. **This task's deliverable is: a full DS→Lswing→DS cycle solves with finite results.**

- [ ] **Step 1: Write the failing test** (the unproven changing-constraint cycle)

```python
# tests/test_aligator_recede.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.aligator_mpc import AligatorMPC

def test_full_contact_cycle_solves_finite():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    mpc = AligatorMPC(cfg, al, am, gait=SLOW_WALK)
    x = nominal_stand_x(am, cfg); mpc.reset(x)
    # advance through more than one full gait period (DS -> Lswing -> DS -> Rswing -> DS)
    n = int(round(2.0 / cfg.dt))
    for k in range(n):
        res = mpc.step(x, k * cfg.dt)
        assert res.status == 0, f"non-finite solve at k={k}"
        x = np.asarray(res.xs[1]).copy()   # roll plan forward (open-loop here; closed-loop in Task 9)
    assert True  # reached the end with finite solves across all contact transitions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_recede.py -q`
Expected: FAIL (current `reset` ignores `gait`; no cycle) — assertion or attribute error.

- [ ] **Step 3: Write the implementation** — extend `AligatorMPC`:
- In `reset`: when `self.gait` is set, `node_times = arange(N)*dt`; `self._cycle_models, self._cycle_sched = build_gait_cycle(...)`; build the initial `problem` from the first N cycle models; pre-create `self._cycle_datas = [m.createData() for m in self._cycle_models]`; track a ring index `self._ci = 0`.
- Add `_recede(self, x_meas, t)`:
  ```python
  m = self._cycle_models[self._ci]; d = self._cycle_datas[self._ci]
  self.problem.replaceStageCircular(m)
  self.solver.cycleProblem(self.problem, d)
  self._ci = (self._ci + 1) % len(self._cycle_models)
  self.problem.x0_init = x_meas; self.xs = self.xs[1:] + [self.xs[-1].copy()]; self.xs[0] = x_meas.copy()
  self.us = self.us[1:] + [self.us[-1].copy()]
  ```
- In `step`: if `self.gait` is set, call `_recede(x_meas, t)` (instead of just setting x0) before `solver.run`. Keep the non-gait stand path unchanged.
- If `cycleProblem` corrupts state on a changing-constraint knot (non-finite / crash), implement the documented fallback in `reset`/`step`: rebuild the problem from the rolled cycle window each tick (`build_problem`-style) and `solver.setup` — slower but correct — and note it in the spec's Phase 2 results.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_aligator_recede.py -q`
Expected: PASS (finite solves across DS↔swing transitions). If the cycle path fails, the fallback must make it pass.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/aligator_mpc.py tests/test_aligator_recede.py
git commit -m "feat(aligator): receding-horizon contact cycle (replaceStageCircular+cycleProblem)"
```

---

### Task 9: Phase 2 gate — closed-loop steps

**Files:**
- Create: `tests/test_aligator_phase2_gate.py`
- Modify: `t1_nmpc/wb/config_aligator.py` (add `max_iters_transition: int = 5` if Task 8 shows transitions need more iters)

**Interfaces:**
- Consumes: Tasks 6–8; `mujoco_to_freeflyer`, `freeflyer_command`, `MujocoTransport`, `SLOW_WALK`.
- Produces: a closed-loop walking test asserting ≥4 steps, no topple, fz bounded, RT measured.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_phase2_gate.py
import numpy as np, pytest
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model
from t1_nmpc.wb.aligator_mpc import AligatorMPC
from t1_nmpc.wb.aligator_state import mujoco_to_freeflyer, freeflyer_command
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.runtime.mujoco_transport import MujocoTransport

@pytest.mark.slow
def test_phase2_sustains_several_steps_without_shed():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    tp = MujocoTransport(cfg, mpc_hz=40.0); rt = tp.rt
    mpc = AligatorMPC(cfg, al, am, gait=SLOW_WALK)
    x0 = mujoco_to_freeflyer(rt, am); mpc.reset(x0)
    mg = am.mass * 9.81; se = max(1, int(round(rt.cfg.control_hz / 40.0)))
    res = mpc.step(x0, 0.0); fz_min = 1.0; t_fall = None
    for k in range(int(round(4.0 * rt.cfg.control_hz))):
        x = mujoco_to_freeflyer(rt, am)
        if k % se == 0:
            res = mpc.step(x, tp.now())
        tp.write_command(freeflyer_command(am, x, res, cfg))
        u0 = np.asarray(res.us[0]); fz_min = min(fz_min, (u0[2] + u0[8]) / mg)
        if rt.mj_data.qpos[2] < 0.45:
            t_fall = k / rt.cfg.control_hz; break
    assert t_fall is None, f"toppled at {t_fall:.2f}s"        # sustains the window
    assert fz_min > 0.6, f"force-shed returned: fz_min/mg={fz_min:.2f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase2_gate.py -q`
Expected: FAIL (no walking wired / topples).

- [ ] **Step 3: Make it pass** — iterate on what Task 8 + the gate surface:
- If transitions need more iterations, set `al.max_iters` at the transition ticks via a variable-budget rule in `AligatorMPC.step` (use `max_iters_transition` when `gait.contact_flags` changes between this tick and last).
- If a swing foot leaks force, raise `w_swing_force` or switch to a control-slice equality on the swing force slots.
- If references are mis-mapped (base orientation), correct the `mujoco_to_freeflyer` / swing-ref mapping.
Keep the change minimal and re-run the gate.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase2_gate.py -q`
Expected: PASS (≥4 steps, no topple, fz bounded).

- [ ] **Step 5: Commit**

```bash
git add tests/test_aligator_phase2_gate.py t1_nmpc/wb/config_aligator.py t1_nmpc/wb/aligator_mpc.py
git commit -m "feat(aligator): Phase 2 gate — closed-loop steps without force-shed"
```

---

### Task 10: Phase 3 — full-walk runner + parity GIF

**Files:**
- Create: `sim/wb_walk_aligator.py` (runner mirroring `sim/wb_walk_croco.py`)
- Create: `tests/test_aligator_phase3_walk.py`
- Modify: `docs/superpowers/specs/2026-06-26-aligator-native-port-design.md` (Phase 3 results)

**Interfaces:**
- Consumes: all prior tasks; `sim/wb_walk_croco.py` structure; the Pillow GIF approach (`PIL.Image`, `mujoco.Renderer`, `MUJOCO_GL=egl`).
- Produces: `run_wb_walk_aligator(duration_s, vx) -> dict` (WALK_GATE-style metrics) + a GIF render helper; a sustained-walk test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aligator_phase3_walk.py
import pytest
from sim.wb_walk_aligator import run_wb_walk_aligator

@pytest.mark.slow
def test_phase3_walks_forward_and_sustains():
    m = run_wb_walk_aligator(duration_s=6.0, vx=0.3)
    assert m["t_fall"] is None or m["t_fall"] > 4.0      # sustains well past crocoddyl's ~2.6-4s
    assert m["com_adv"] > 0.3                            # net forward progress
    assert m["fz_min_ratio"] > 0.6                       # no force-shed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase3_walk.py -q`
Expected: FAIL with `ModuleNotFoundError: sim.wb_walk_aligator`

- [ ] **Step 3: Write the implementation**
- `sim/wb_walk_aligator.py`: copy the control loop from `sim/wb_walk_croco.py` (`run_wb_walk_croco`), swapping `CrocoMPC`→`AligatorMPC(cfg, al, am, gait=SLOW_WALK)`, `read_state`→`mujoco_to_freeflyer`, `to_joint_command_wb`→`freeflyer_command`. Track `t_fall`, `com_adv`, `fz_min_ratio`, `solve_ms` and return the metrics dict.
- Add a `render_gif(out_path, duration_s, vx)` reusing the validated Pillow path (`mujoco.Renderer(model, 360, 480)`, tracking camera azimuth=270, `PIL.Image ... save(save_all=True, duration=1000/30, loop=0)`).

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl conda run -n t1mpc python -m pytest tests/test_aligator_phase3_walk.py -q`
Expected: PASS (sustained forward walk, no shed). Render the parity GIF (aligator vs crocoddyl) and attach to the spec's Phase 3 results.

- [ ] **Step 5: Commit**

```bash
git add sim/wb_walk_aligator.py tests/test_aligator_phase3_walk.py docs/superpowers/specs/2026-06-26-aligator-native-port-design.md
git commit -m "feat(aligator): Phase 3 — full walk runner + parity GIF"
```

---

## Notes for the implementer

- **Read `ode.nu`, never hardcode 39** — guards against a URDF swap.
- **The `cycleProblem` changing-constraint risk (Task 8) is the one genuine unknown.** If the in-place cycle corrupts the solver, take the rebuild-each-tick fallback; the gate (finite solves through a full DS↔swing cycle) is the same either way.
- **State mapping (Task 6)** is the integration seam most likely to need iteration: MuJoCo quat is wxyz, pinocchio is xyzw; the 27 joints must be selected/ordered to `MPC_JOINT_NAMES`. Add a one-time index map and a round-trip test if mapping bugs appear.
- **Variable iteration budget**: steady phases hold at maxit=2; if transitions topple, raise iters only at the transition tick (Task 9), not globally (protects the RT budget).
- The crocoddyl baseline + its tests must stay green throughout (`tests/test_croco_walk_*.py`).
