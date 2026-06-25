# Crocoddyl Whole-Body MPC Port — M0 Implementation Plan (Foundation + Closed-Loop Stand)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the acados backend of `t1_nmpc` with a Crocoddyl inverse-dynamics (Option A) backend that holds a closed-loop stand in MuJoCo, then delete the acados code.

**Architecture:** A `T1ProblemBuilder` builds per-node `DifferentialActionModelContactInvDynamics` action models; a `CrocoMPC` driver runs single-RTI (`maxiter=1`) `SolverIntro` and emits a 68-dim `MPCResult` so the *existing* control loop / transport / execution / PD layer is reused verbatim. Spec: `docs/superpowers/specs/2026-06-25-crocoddyl-port-design.md`.

**Tech Stack:** Python 3.12, crocoddyl 3.2.1, pinocchio 4.0.0, casadi 3.7.2, numpy 2.x, mujoco 3.10, pytest. Conda env `t1mpc`.

## Global Constraints

- **Run commands:** all python/pytest runs use `env -u PYTHONPATH conda run -n t1mpc <cmd>` (the ROS Humble pinocchio shadows the conda one unless `PYTHONPATH` is cleared). From repo root `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc`.
- **State convention:** crocoddyl state is **66-dim** `[q(33: xyz, euler-ZYX, 27 joints); v(33)]`. The rest of the stack (transport, `execution_wb`, `MPCResult`, `joint_torque`) speaks the acados **68-dim** state `[q(33); v(33); s; v_s]`. `CrocoMPC` is the ONLY 66↔68 boundary: it slices `x_meas[:66]` on input and pads `s=v_s=0` on output. The first 66 dims are identical, so `execution_wb` slices (`q_joints=6:33`, `v_joints=39:66`) are unchanged.
- **Control layout:** crocoddyl control `u = [a(nv=33); contact_forces(6·n_stance)]`; double-support `nu=45`, single-support `nu=39`. The acados-layout `u_traj` row that `MPCResult` carries is `[W_l(6); W_r(6); a_joints(27); vdot_s=0]` (40-dim), built from the crocoddyl solution.
- **Single-RTI:** `solver.solve(xs, us, 1, False, reg)` — `maxiter=1` per control cycle (= OCS2 `sqpIteration=1`).
- **All weights/params come from `config_wb`** (`Q`, `R`, `friction_mu`, `foot_rect_x/y`, `torque_limit`, `joint_lower/upper`, `kp/kd`, `nominal_*`). Never hardcode a weight that exists in `config_wb`.
- **Reused unchanged:** `wb/model_wb.py`, `wb/config_wb.py`, `wb/gait_wb.py`, `execution.py`, `wb/execution_wb.py`, `mpc_result.py`, `sim/mujoco_runtime.py`, `runtime/mujoco_transport.py` (after Task 5 extraction), `runtime/transport.py`, `sim/_sim_util.py`.
- **Validated baseline:** the spikes `spikes/croco_stand_spike.py` (stand, ‖h‖=1e-7, foot force=body weight) and `spikes/croco_walk_spike.py` (walk) are the working reference; their construction code is the source of truth for the builder.

## File Structure

| File | Disposition |
|---|---|
| `t1_nmpc/wb/croco_costs.py` | **create** — cost/constraint residual builders |
| `t1_nmpc/wb/croco_problem.py` | **create** — `T1ProblemBuilder` |
| `t1_nmpc/wb/croco_mpc.py` | **create** — `CrocoMPC` (emits `MPCResult`) |
| `sim/wb_state.py` | **create** — `wb_state_estimate`, `wb_reset` (extracted from `wb_stand_gate`) |
| `sim/wb_stand_croco.py` | **create** — closed-loop M0 stand gate |
| `tests/test_croco_costs.py`, `tests/test_croco_problem.py`, `tests/test_croco_mpc.py`, `tests/test_wb_stand_croco.py` | **create** |
| `t1_nmpc/runtime/control_loop.py` | **modify** — read `mpc.last_solve_s`; status check `!= 0` |
| `t1_nmpc/runtime/mujoco_transport.py` | **modify** — import from `sim.wb_state` |
| `mpc_result.py`, `execution_wb.py` | **keep unchanged** (generic) |
| acados modules (Task 6) | **delete** |

---

## Task 1: Branch, tag the acados oracle, commit the spec

**Files:** none (git only)

- [ ] **Step 1: Verify clean tree and current branch**

Run: `cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc && git status && git branch --show-current`
Expected: shows the working branch; note any uncommitted changes (the spec/plan docs are expected new files).

- [ ] **Step 2: Tag the acados port as a recoverable oracle**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git tag -f acados-port-final
```

- [ ] **Step 3: Create and switch to the port branch**

```bash
git checkout -b crocoddyl-port
```

- [ ] **Step 4: Commit the spec + this plan**

```bash
git add docs/superpowers/specs/2026-06-25-crocoddyl-port-design.md docs/superpowers/plans/2026-06-25-crocoddyl-port-m0.md
git commit -m "docs: crocoddyl port M0 design + plan"
```
Expected: commit succeeds on branch `crocoddyl-port`; `git tag` lists `acados-port-final`.

---

## Task 2: `croco_costs.py` — cost & constraint residual builders

**Files:**
- Create: `t1_nmpc/wb/croco_costs.py`
- Test: `tests/test_croco_costs.py`

**Interfaces:**
- Consumes: `crocoddyl.StateMultibody`, `crocoddyl.ActuationModelFloatingBase`, `WBConfig` (`config_wb`).
- Produces: `build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg) -> crocoddyl.CostModelSum`. `nu:int`, `x_ref: np.ndarray(66)`, `com_ref: np.ndarray(3)`, `stance_fids: list[int]`, `cfg: WBConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_costs.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb import croco_costs

def _ctx():
    cfg = make_wb_config(); wb = WBModel(cfg)
    state = crocoddyl.StateMultibody(wb.model)
    act = crocoddyl.ActuationModelFloatingBase(state)
    return cfg, wb, state, act

def test_build_costs_double_support_dims():
    cfg, wb, state, act = _ctx()
    nv = wb.model.nv
    nu = nv + 12                                  # double support
    x_ref = np.zeros(state.nx); x_ref[2] = cfg.nominal_base_height
    x_ref[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    assert costs.nu == nu
    names = set(costs.costs.todict().keys())
    assert {"xreg", "ureg", "tau_lim", "joint_lim"} <= names
    assert any(n.startswith("wrenchcone_") for n in names)   # one per stance foot

def test_state_weight_comes_from_config_Q():
    cfg, wb, state, act = _ctx()
    nu = wb.model.nv + 12
    x_ref = np.zeros(state.nx)
    costs = croco_costs.build_costs(state, act, nu, x_ref, np.zeros(3),
                                    list(wb.contact_fids), cfg)
    act_model = costs.costs["xreg"].cost.activation
    assert np.allclose(np.asarray(act_model.weights), cfg.Q[:66])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_costs.py -v`
Expected: FAIL — `ModuleNotFoundError: t1_nmpc.wb.croco_costs`.

- [ ] **Step 3: Implement `croco_costs.py`**

```python
# t1_nmpc/wb/croco_costs.py
"""Crocoddyl cost/constraint residual builders for the T1 whole-body MPC (M0).

All terms map t1_controller costs/constraints to NATIVE crocoddyl residuals reading
config_wb. t1_controller's inequalities are soft relaxed-barriers, so penalty costs are
faithful; M0 approximates the barrier SHAPE with QuadraticBarrier (exact relaxed barrier
is M1). Swing/arm-swing/foot-collision are M1 (skipped here)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import crocoddyl

_BIG = 1e3  # effective +inf for one-sided state bounds


def _control_weights(nv: int, nc: int, R: np.ndarray) -> np.ndarray:
    """Map config_wb.R [W_l(6),W_r(6),qdd(27),vdot_s(1)] to crocoddyl control
    [a(nv); forces(nc)] weights. a[0:6]=base accel (constrained) -> tiny; a[6:33]=qdd
    -> R[12:39]; forces ordered [left, right] -> R[0:6], R[6:12]."""
    w = np.empty(nv + nc)
    w[0:6] = 1e-6
    w[6:nv] = R[12:39]
    if nc >= 6:
        w[nv:nv + 6] = R[0:6]            # left foot wrench
    if nc >= 12:
        w[nv + 6:nv + 12] = R[6:12]      # right foot wrench
    return w


def build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg):
    nv = state.pinocchio.nv
    nc = 6 * len(stance_fids)
    costs = crocoddyl.CostModelSum(state, nu)

    # 1. state tracking/regularization (weights = config_wb.Q diagonal, 68->66)
    xreg = crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)
    xact = crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q[:66], float))
    costs.addCost("xreg", crocoddyl.CostModelResidual(state, xact, xreg), 1.0)

    # 2. CoM tracking (M0: com_ref = com0, low weight; forward drive is M1)
    creg = crocoddyl.ResidualModelCoMPosition(state, np.asarray(com_ref, float), nu)
    costs.addCost("com", crocoddyl.CostModelResidual(state, creg), 1.0)

    # 3. input regularization (weights from config_wb.R)
    ureg = crocoddyl.ResidualModelControl(state, nu)
    uact = crocoddyl.ActivationModelWeightedQuad(_control_weights(nv, nc, np.asarray(cfg.R, float)))
    costs.addCost("ureg", crocoddyl.CostModelResidual(state, uact, ureg), 1.0)

    # 4. torque-limit soft barrier on recovered tau (JointEffort)
    tau_lim = np.asarray(cfg.torque_limit, float)
    teff = crocoddyl.ResidualModelJointEffort(state, actuation, np.zeros(actuation.nu), nu, False)
    tbar = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-tau_lim, tau_lim))
    costs.addCost("tau_lim", crocoddyl.CostModelResidual(state, tbar, teff),
                  float(cfg.jointtorque_weight))

    # 5. joint-position-limit soft barrier (bounds relative to neutral on the joint block)
    lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
    lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float)
    ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
    jres = crocoddyl.ResidualModelState(state, np.zeros(state.nx), nu)
    jbar = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
    costs.addCost("joint_lim", crocoddyl.CostModelResidual(state, jbar, jres),
                  float(cfg.joint_limit_barrier_mu))

    # 6. friction-cone + CoP-in-rectangle + unilateral, per stance foot (WrenchCone)
    box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)   # half-extents
    R_foot = np.eye(3)                                                # feet flat on flat ground (M0)
    for fid in stance_fids:
        cone = crocoddyl.WrenchCone(R_foot, float(cfg.friction_mu), box)
        wres = crocoddyl.ResidualModelContactWrenchCone(state, fid, cone, nu, False)
        wbar = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub))
        costs.addCost(f"wrenchcone_{fid}",
                      crocoddyl.CostModelResidual(state, wbar, wres),
                      float(cfg.friction_cone_reg))
    return costs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_costs.py -v`
Expected: PASS (2 tests). If `ResidualModelContactWrenchCone`/`ResidualModelJointEffort` reject the `fwddyn=False` positional, adjust per the error (the 4-arg form `(state, id, cone, nu)` then `, False`).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/croco_costs.py tests/test_croco_costs.py
git commit -m "feat(croco): cost/constraint residual builders from config_wb"
```

---

## Task 3: `croco_problem.py` — `T1ProblemBuilder`

**Files:**
- Create: `t1_nmpc/wb/croco_problem.py`
- Test: `tests/test_croco_problem.py`

**Interfaces:**
- Consumes: `croco_costs.build_costs`, `WBModel`, `WBConfig`.
- Produces:
  - `class T1ProblemBuilder(cfg, wb)` with `.state`, `.actuation`, `.foot_fids` (`[L,R]`), `.dt`, `.N`, `.nv`.
  - `make_node(stance_fids, x_ref, com_ref, planted, terminal=False) -> IntegratedActionModelEuler` where `planted: dict[int, pin.SE3]`.
  - `build_stand_problem(x0_66) -> crocoddyl.ShootingProblem`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_problem.py
import numpy as np, pinocchio as pin, crocoddyl
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_problem import T1ProblemBuilder

def _builder_x0():
    cfg = make_wb_config(); wb = WBModel(cfg)
    b = T1ProblemBuilder(cfg, wb)
    q0 = pin.neutral(wb.model); q0[2] = cfg.nominal_base_height
    q0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    x0 = np.concatenate([q0, np.zeros(wb.model.nv)])
    return cfg, wb, b, x0

def test_stand_problem_shape_and_nu_nh():
    cfg, wb, b, x0 = _builder_x0()
    prob = b.build_stand_problem(x0)
    assert len(prob.runningModels) == cfg.N
    d = prob.runningModels[0]
    assert d.nu == wb.model.nv + 12                  # double support
    assert d.differential.nh == 18                   # 6 underactuated + 12 contact

def test_stand_problem_solves_holds_contact():
    cfg, wb, b, x0 = _builder_x0()
    prob = b.build_stand_problem(x0)
    solver = crocoddyl.SolverIntro(prob)
    xs = [x0.copy() for _ in range(cfg.N + 1)]
    us = prob.quasiStatic([x0.copy() for _ in range(cfg.N)])
    solver.solve(xs, us, 80, False, 1e-9)
    assert np.all(np.isfinite(np.asarray(solver.xs)))
    drift = max(np.linalg.norm(np.asarray(solver.xs)[k][:3] - x0[:3]) for k in range(cfg.N + 1))
    assert drift < 0.05                              # stand holds (<5 cm)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_problem.py -v`
Expected: FAIL — `ModuleNotFoundError: t1_nmpc.wb.croco_problem`.

- [ ] **Step 3: Implement `croco_problem.py`**

```python
# t1_nmpc/wb/croco_problem.py
"""T1ProblemBuilder: per-node ContactInvDynamics action-model factory + stand problem.
Construction only (no solver, no mutable state). Bodies derive from the validated
spikes/croco_stand_spike.py."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import crocoddyl

from .croco_costs import build_costs

_LWA = pin.LOCAL_WORLD_ALIGNED


class T1ProblemBuilder:
    def __init__(self, cfg, wb):
        self.cfg = cfg
        self.wb = wb
        self.model = wb.model
        self.nv = wb.model.nv
        self.state = crocoddyl.StateMultibody(wb.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.foot_fids = list(wb.contact_fids)            # [L, R]
        self.dt = float(cfg.dt)
        self.N = int(cfg.N)
        # foot-constraint Baumgarte gains = t1_controller foot_constraint feedback
        self._gains = np.array([cfg.foot_pos_err_gain_z, cfg.foot_linvel_err_gain_xy], float)

    def _planted(self, x0_66):
        """SE3 placement of each foot at x0 (held by the contact)."""
        q = np.asarray(x0_66[:self.model.nq], float)
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        return {fid: data.oMf[fid].copy() for fid in self.foot_fids}

    def make_node(self, stance_fids, x_ref, com_ref, planted, terminal=False):
        nu = self.nv + 6 * len(stance_fids)
        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        for fid in stance_fids:
            c6 = crocoddyl.ContactModel6D(self.state, fid, planted[fid], _LWA, nu, self._gains)
            contacts.addContact("c%d" % fid, c6)
        costs = build_costs(self.state, self.actuation, nu, x_ref, com_ref, stance_fids, self.cfg)
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(
            self.state, self.actuation, contacts, costs)
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else self.dt)

    def build_stand_problem(self, x0_66):
        x0 = np.asarray(x0_66, float)
        planted = self._planted(x0)
        data = self.model.createData()
        pin.centerOfMass(self.model, data, x0[:self.model.nq])
        com0 = data.com[0].copy()
        stance = self.foot_fids
        running = [self.make_node(stance, x0, com0, planted) for _ in range(self.N)]
        terminal = self.make_node(stance, x0, com0, planted, terminal=True)
        return crocoddyl.ShootingProblem(x0, running, terminal)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_problem.py -v`
Expected: PASS (2 tests). `nh==18` confirms 6 underactuated + 12 contact rows (matches the spike).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/croco_problem.py tests/test_croco_problem.py
git commit -m "feat(croco): T1ProblemBuilder ContactInvDynamics stand problem"
```

---

## Task 4: `croco_mpc.py` — `CrocoMPC` driver (emits 68-dim `MPCResult`)

**Files:**
- Create: `t1_nmpc/wb/croco_mpc.py`
- Test: `tests/test_croco_mpc.py`

**Interfaces:**
- Consumes: `T1ProblemBuilder`, `crocoddyl.SolverIntro`, `MPCResult` (`t1_nmpc.mpc_result`).
- Produces: `class CrocoMPC(cfg, wb)` with attributes `.cfg`, `.model` (= `wb`), `.last_solve_s: float`, `.solver`; methods `reset(x0_68)`, `step(x_meas_68, t) -> MPCResult`. `MPCResult.x_traj` is `(N+1, 68)`, `u_traj` is `(N, 40)` acados-layout, `status` is `0` on success.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_croco_mpc.py
import numpy as np, pinocchio as pin
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC

def _mpc_x0():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height
    x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    return cfg, wb, mpc, x0

def test_step_returns_compatible_result():
    cfg, wb, mpc, x0 = _mpc_x0()
    mpc.reset(x0)
    res = mpc.step(x0, 0.0)
    assert res.x_traj.shape == (cfg.N + 1, 68)
    assert res.u_traj.shape == (cfg.N, 40)
    assert res.status == 0
    assert np.all(np.isfinite(res.x_traj)) and np.all(np.isfinite(res.u_traj))
    assert mpc.last_solve_s > 0.0

def test_single_rti_holds_stand_over_a_few_steps():
    cfg, wb, mpc, x0 = _mpc_x0()
    mpc.reset(x0)
    x = x0.copy()
    for _ in range(5):
        res = mpc.step(x, 0.0)
        x = res.x_traj[1].copy()                 # advance along the plan (no sim)
    assert np.linalg.norm(x[:3] - x0[:3]) < 0.05  # didn't run away
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_mpc.py -v`
Expected: FAIL — `ModuleNotFoundError: t1_nmpc.wb.croco_mpc`.

- [ ] **Step 3: Implement `croco_mpc.py`**

```python
# t1_nmpc/wb/croco_mpc.py
"""CrocoMPC: receding-horizon single-RTI driver. Drop-in for the acados WholeBodyMPC at
the loop interface (.cfg/.model/.reset/.step -> MPCResult/.last_solve_s). Speaks 66-dim
crocoddyl state internally, 68-dim at the boundary; emits an acados-layout u_traj so the
existing execution_wb/joint_torque path is reused."""
from __future__ import annotations

import time
import numpy as np
import crocoddyl

from ..mpc_result import MPCResult
from .croco_problem import T1ProblemBuilder

_REG = 1e-9


class CrocoMPC:
    def __init__(self, cfg, wb, max_iter: int = 1):
        self.cfg = cfg
        self.model = wb                                  # has .joint_torque (used by execution_wb)
        self.max_iter = int(max_iter)
        self.builder = T1ProblemBuilder(cfg, wb)
        self.nv = wb.model.nv
        self.N = int(cfg.N)
        self._x0 = self._nominal66()
        self.problem = self.builder.build_stand_problem(self._x0)
        self.solver = crocoddyl.SolverIntro(self.problem)
        self.solver.setCallbacks([])
        self._xs = [self._x0.copy() for _ in range(self.N + 1)]
        self._us = self.problem.quasiStatic([self._x0.copy() for _ in range(self.N)])
        self._node_times = np.arange(self.N + 1) * float(cfg.dt)
        self.last_solve_s = 0.0
        self._foot_l, self._foot_r = self.builder.foot_fids

    def _nominal66(self):
        import pinocchio as pin
        q0 = pin.neutral(self.builder.model); q0[2] = self.cfg.nominal_base_height
        q0[6:6 + self.cfg.n_joints] = self.cfg.nominal_joint_pos
        return np.concatenate([q0, np.zeros(self.nv)])

    def reset(self, x0_68):
        x66 = np.asarray(x0_68, float)[:66]
        self._x0 = x66.copy()
        self.problem.x0 = x66
        self._xs = [x66.copy() for _ in range(self.N + 1)]
        self._us = self.problem.quasiStatic([x66.copy() for _ in range(self.N)])

    def step(self, x_meas_68, t) -> MPCResult:
        x66 = np.asarray(x_meas_68, float)[:66]
        self.problem.x0 = x66
        # warm-start shift
        xs = self._xs[1:] + [self._xs[-1]]; xs[0] = x66.copy()
        us = self._us[1:] + [self._us[-1]]
        t0 = time.perf_counter()
        self.solver.solve(xs, us, self.max_iter, False, _REG)
        self.last_solve_s = time.perf_counter() - t0
        self._xs = list(self.solver.xs); self._us = list(self.solver.us)

        xs_arr = np.asarray(self.solver.xs)               # (N+1, 66)
        ok = bool(np.all(np.isfinite(xs_arr)) and self.solver.isFeasible)
        x_traj = np.zeros((self.N + 1, 68)); x_traj[:, :66] = xs_arr
        u_traj = self._acados_layout(self.solver.us)
        if not ok:                                        # degrade safely (ZOH-able): flat plan
            x_traj[:] = x_traj[0]
        return MPCResult(
            x_traj=x_traj, u_traj=u_traj, feasible=ok, solve_time=self.last_solve_s,
            mode_schedule=None, status=0 if ok else 1,
            node_times=t + self._node_times, u_phys_traj=None)

    def _acados_layout(self, us):
        """crocoddyl us[k]=[a(nv); forces...] -> [W_l(6); W_r(6); a_joints(27); vdot_s=0]."""
        out = np.zeros((self.N, 40))
        for k, u in enumerate(us):
            u = np.asarray(u, float)
            a = u[:self.nv]
            forces = u[self.nv:]
            out[k, 12:39] = a[6:33]                        # qdd joints
            # double-support contact order in make_node = [L, R]
            if forces.size >= 6:
                out[k, 0:6] = forces[0:6]                  # W_l
            if forces.size >= 12:
                out[k, 6:12] = forces[6:12]                # W_r
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_croco_mpc.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/croco_mpc.py tests/test_croco_mpc.py
git commit -m "feat(croco): CrocoMPC single-RTI driver emitting 68-dim MPCResult"
```

---

## Task 5: State-estimate extraction + control-loop refine + closed-loop M0 stand gate

**Files:**
- Create: `sim/wb_state.py`, `sim/wb_stand_croco.py`, `tests/test_wb_stand_croco.py`
- Modify: `t1_nmpc/runtime/mujoco_transport.py:10-11`, `t1_nmpc/runtime/control_loop.py:52,55`

**Interfaces:**
- Consumes: `MujocoTransport`, `CrocoMPC`, `run_loop`, `MujocoRuntime`.
- Produces: `sim/wb_state.py` with `wb_state_estimate(rt) -> np.ndarray(68)` and `wb_reset(rt, wb_cfg) -> None`; `sim/wb_stand_croco.py` with `run_wb_stand_croco(duration_s=5.0, control_hz=60.0) -> dict`.

- [ ] **Step 1: Extract the reusable state functions from the acados gate into `sim/wb_state.py`**

Create `sim/wb_state.py` with the exact code below (lifted verbatim from `wb_stand_gate.py:30-60`, `_wb_reset` renamed `wb_reset`):

```python
# sim/wb_state.py
"""Whole-body MuJoCo state estimate + reset (extracted from the deleted acados stand gate;
reused by mujoco_transport and the crocoddyl stand gate)."""
from __future__ import annotations

import numpy as np
import mujoco

from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def wb_state_estimate(rt: MujocoRuntime) -> np.ndarray:
    """68-d WB state from sim: [q_base(6), q_joints(27), v_base(6), v_joints(27), s, v_s].
    q_pin/v_pin are euler-zyx (35,); the 27 MPC joints = the 29 minus the 2 head joints (idx 6:8)."""
    q_pin, v_pin = rt._pin_q_v()
    x = np.zeros(68, dtype=np.float64)
    x[0:6] = q_pin[0:6]
    x[6:33] = q_pin[8:35]
    x[33:39] = v_pin[0:6]
    x[39:66] = v_pin[8:35]
    return x


def wb_reset(rt: MujocoRuntime, wb_cfg) -> None:
    """Spawn at the WB nominal posture (head=0 + 27 MPC joints) above the floor and PD-settle the
    feet onto it."""
    q0 = MJ_JOINT_QPOS0
    njp29 = np.zeros(29); njp29[2:29] = np.asarray(wb_cfg.nominal_joint_pos, dtype=np.float64)
    kp = np.asarray(rt.cfg.kp, dtype=np.float64); kd = np.asarray(rt.cfg.kd, dtype=np.float64)
    d = rt.mj_data
    d.qpos[:] = 0.0; d.qvel[:] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[q0:q0 + 29] = njp29
    d.qpos[2] = wb_cfg.nominal_base_height + 0.10
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    for _ in range(int(round(0.6 * rt.cfg.physics_hz))):
        q = np.array(d.qpos[q0:q0 + 29]); qd = np.array(d.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
        rt._apply_torque(kp * (njp29 - q) - kd * qd)
        rt.step_physics()
    d.qvel[:] = 0.0
    mujoco.mj_forward(rt.mj_model, rt.mj_data)
    rt.t = 0.0
```

**Note:** `MujocoTransport.__init__` currently calls `_wb_reset(self.rt, wb_cfg)` (via the `wb_stand_gate` import). After Step 2's import change (`wb_reset as _wb_reset`), that call resolves to the extracted function unchanged.

- [ ] **Step 2: Repoint `mujoco_transport.py` to the extracted module**

Modify `t1_nmpc/runtime/mujoco_transport.py:11` from
`from sim.wb_stand_gate import wb_state_estimate, _wb_reset`
to
`from sim.wb_state import wb_state_estimate, wb_reset as _wb_reset`

- [ ] **Step 3: Refine `control_loop.py` for the crocoddyl driver**

Modify `t1_nmpc/runtime/control_loop.py`:
- line 52: `if res.status not in (0, 2):` → `if res.status != 0:`
- line 55: `tot_ms.append(float(mpc.solver.get_stats("time_tot")) * 1e3)` → `tot_ms.append(float(mpc.last_solve_s) * 1e3)`

- [ ] **Step 4: Write the failing M0 stand-gate test**

```python
# tests/test_wb_stand_croco.py
from sim.wb_stand_croco import run_wb_stand_croco

def test_m0_stand_holds():
    m = run_wb_stand_croco(duration_s=3.0, control_hz=60.0)
    assert m["n_fail"] == 0
    assert m["peak_tilt_rad"] is not None and m["peak_tilt_rad"] < 0.05
    assert m["final_base_z"] > 0.85 * 0.6734
    assert m["held"] is True
```

- [ ] **Step 5: Run test to verify it fails**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_wb_stand_croco.py -v`
Expected: FAIL — `ModuleNotFoundError: sim.wb_stand_croco`.

- [ ] **Step 6: Implement `sim/wb_stand_croco.py`**

```python
# sim/wb_stand_croco.py
"""Closed-loop M0 stand: CrocoMPC drives MuJoCo via the reused transport + control loop.
Acceptance gate equivalent to the acados M0 (peak_tilt, base_z, no failures)."""
from __future__ import annotations
import argparse, json
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.croco_mpc import CrocoMPC
from t1_nmpc.runtime.mujoco_transport import MujocoTransport
from t1_nmpc.runtime.control_loop import run_loop


def run_wb_stand_croco(duration_s: float = 5.0, control_hz: float = 60.0) -> dict:
    cfg = make_wb_config()
    wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=control_hz)
    mpc = CrocoMPC(cfg, wb)
    return run_loop(transport, mpc, duration_s=duration_s, control_hz=control_hz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--control-hz", type=float, default=60.0)
    args = ap.parse_args()
    m = run_wb_stand_croco(args.duration, args.control_hz)
    print("STAND_GATE=" + json.dumps(m))
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_wb_stand_croco.py -v`
Expected: PASS. If the stand drifts/falls: first verify `tau_ff` sign (Task 4 `_acados_layout` feeds `joint_torque`; if the robot is pushed *down*/explodes, the crocoddyl contact-force sign is opposite `joint_torque`'s wrench convention — negate the `forces` block in `_acados_layout` and re-run). This is the one convention check flagged in the spec.

- [ ] **Step 8: Manual sanity run (optional but recommended)**

Run: `env -u PYTHONPATH conda run -n t1mpc python sim/wb_stand_croco.py --duration 3`
Expected: prints `STAND_GATE={... "held": true ...}` with `peak_tilt_rad < 0.05`.

- [ ] **Step 9: Commit**

```bash
git add sim/wb_state.py sim/wb_stand_croco.py tests/test_wb_stand_croco.py \
        t1_nmpc/runtime/mujoco_transport.py t1_nmpc/runtime/control_loop.py
git commit -m "feat(croco): closed-loop M0 stand gate + state-estimate extraction"
```

---

## Task 6: Acados teardown

**Files:**
- Delete: `t1_nmpc/wb/{ocp_wb,projection_wb,mpc_wb,grid_wb,constraints_wb,cost_wb}.py`, `t1_nmpc/runtime/measure_deploy.py`, `sim/{wb_stand_gate,wb_walk_gate,wb_walk_view}.py`, `.acados_wb/`, and acados-only tests.
- Keep: `mpc_result.py`, `execution_wb.py`, `reference_wb.py` (dormant; M1 refines it).

- [ ] **Step 1: Confirm nothing live imports the acados modules**

Run:
```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
grep -rEl "ocp_wb|projection_wb|mpc_wb|grid_wb|constraints_wb|cost_wb|wb_stand_gate|wb_walk_gate|wb_walk_view|measure_deploy" t1_nmpc sim tests | grep -vE "test_wb_(ocp|projection|mpc_walk|cost|constraints|default_discrete|warmstart|flow|grid|torque)|wb_walk_gate|wb_walk_view|wb_stand_gate"
```
Expected: EMPTY (no live importer outside the files being deleted and their tests). If `reference_wb.py` appears, open it — it must only import `gait_wb`/`config`/`model`; if it imports a to-be-deleted module, comment that import (M1 will rewire it).

- [ ] **Step 2: Delete acados modules, cache, and acados-only tests**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git rm t1_nmpc/wb/ocp_wb.py t1_nmpc/wb/projection_wb.py t1_nmpc/wb/mpc_wb.py \
       t1_nmpc/wb/grid_wb.py t1_nmpc/wb/constraints_wb.py t1_nmpc/wb/cost_wb.py \
       t1_nmpc/runtime/measure_deploy.py \
       sim/wb_stand_gate.py sim/wb_walk_gate.py sim/wb_walk_view.py \
       tests/test_wb_ocp.py tests/test_wb_projection.py tests/test_wb_cost.py \
       tests/test_wb_cost_walk.py tests/test_wb_constraints.py tests/test_wb_constraints_walk.py \
       tests/test_wb_mpc_walk.py tests/test_wb_default_discrete.py tests/test_wb_warmstart.py \
       tests/test_wb_flow.py tests/test_wb_grid.py tests/test_wb_torque.py
rm -rf .acados_wb
```

- [ ] **Step 3: Run the full remaining suite**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/ -q`
Expected: all PASS — the crocoddyl tests (Tasks 2–5) plus the kept infra tests (`test_model`, `test_wb_config*`, `test_wb_gait`, `test_wb_swing`, `test_wb_reference`, `test_mujoco_runtime`, `test_runtime_*`, `test_execution`, `test_env`, `test_sim_util`, `test_sysid_friction`, `test_wb_model_rbd`). If a kept test imports a deleted module, it was misclassified — fix its import or move it to the delete list (note in the commit).

- [ ] **Step 4: Re-run the M0 stand gate to confirm teardown didn't break it**

Run: `env -u PYTHONPATH conda run -n t1mpc python -m pytest tests/test_wb_stand_croco.py -v`
Expected: PASS (still holds).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(croco): remove acados backend (recoverable at tag acados-port-final)"
```

---

## Self-Review

**Spec coverage:** §2 layout (Tasks 1,6) ✓; §3 builder (Task 3) ✓; §4 cost/constraint mapping (Task 2 — all native residuals) ✓; §5 driver + closure (Tasks 4,5; single-RTI `maxiter=1`) ✓; §6 tests + M0 gate vs acados baseline (Tasks 3,4,5) ✓; teardown + tag (Tasks 1,6) ✓.

**Deviations from spec (simplifications, noted):** (1) `execution_wb.py` and `mpc_result.py` are **kept unchanged** rather than refined/deleted — the interface read showed `MPCResult` is solver-agnostic and `execution_wb`'s slices are 66/68-compatible, so `CrocoMPC` emitting a 68-dim `MPCResult` reuses them verbatim (DRY win). (2) `control_loop.py` is **kept+refined** (2 lines) rather than rewritten. (3) Joint-position-limit barrier IS included (spec §4) though barely active at a stand. (4) `wb_state_estimate`/`wb_reset` extraction (Task 5) is an interface dependency the spec didn't surface — `mujoco_transport` imported them from the acados gate.

**Type consistency:** `build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg)` signature identical in Task 2 def and Task 3 call ✓. `MPCResult` fields (`x_traj`,`u_traj`,`feasible`,`solve_time`,`mode_schedule`,`status`,`node_times`,`u_phys_traj`) match `mpc_result.py` ✓. `CrocoMPC` exposes `.cfg/.model/.reset/.step/.last_solve_s/.solver` as `run_loop` requires ✓.

**Known risk flagged in-plan:** crocoddyl contact-force sign vs `joint_torque` wrench convention (Task 5 Step 7) — validated by the closed-loop stand, with the fix (negate forces block) called out.
