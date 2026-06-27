# M1 aligator port — foundation + warm-start gate + re-homed stand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Fatrop `whole_body_rnea` backend in `t1_nmpc` with an **aligator `SolverProxDDP`** backend (RNEA-ID + 6D foot-wrench contacts + per-foot-6D contact velocity + hard velocity-level swing-z), prove the cyclic warm-start converges across contact switches, and re-home the closed-loop MuJoCo **stand** onto it.

**Architecture:** The OCP keeps the paper's `whole_body_rnea` shape — a trivial kinematic double integrator (`q̇=v, v̇=a`) with all physics imposed as a per-stage **RNEA equality constraint**. The solver moves Fatrop → aligator: each contact phase is a pre-built `StageModel`; the receding horizon advances by rotating the phase ring (`replaceStageCircular` + `cycleProblem`) with warm primal **and** dual carry. Contact forces are **one 6D wrench per foot** in the sole frame, bounded by a contact-wrench cone; stance pins the 6D foot velocity to zero; swing zeroes the wrench and tracks a cubic z-velocity spline (hard equality).

**Tech Stack:** Python 3.12 (conda env `t1mpc`), `aligator==0.19.0` (`SolverProxDDP`, `manifolds.MultibodyPhaseSpace`, `dynamics.ODEAbstract`/`IntegratorEuler`, `StageFunction`, `FrameVelocityResidual`, `constraints.EqualityConstraintSet`/`NegativeOrthant`), `pinocchio==4.x` + `pinocchio.casadi`, `casadi`, `mujoco`, `pytest`.

## Global Constraints

These apply to **every** task. Each task's requirements implicitly include this section.

- **Run preamble (load-bearing).** Always run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc` with `PYTHONPATH=` empty:
  `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python <args>`.
  The empty `PYTHONPATH` keeps `/opt/ros/humble`'s numpy<2 pinocchio off the path (it segfaults the conda pinocchio).
- **pytest invocation:** `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest <path> -q -p no:cacheprovider`.
- **Dimensions (reduced, head-locked model):** `n_joints = 27`, `nq = 34`, `nv = 33`, `nx = 67`, `ndx = 66`, `n_feet = 2`, `nf = 12` (two 6D wrenches), `na = nv = 33`, `nu = na + nf = 45` (uniform across **all** stages — torque is not a decision variable).
- **Locked joints:** `AAHead_yaw`, `Head_pitch` (head). Arms are **not** structurally locked — held to nominal by a high state-tracking weight.
- **Discretization:** uniform `dt = 0.035`, `N = 31` intervals (`32` shooting nodes), horizon `1.085 s`, explicit Euler.
- **Gait (walk):** cycle `1.4 s`; modes `LF-swing [0,0.6) → double [0.6,0.7) → RF-swing [0.7,1.3) → double [1.3,1.4)`; swing height `0.08`, liftoff vel `+0.05`, touchdown vel `−0.05`.
- **Contact/friction:** `μ = 0.4`; sole half-extents `X = 0.1065` (half-length), `Y = 0.05` (half-width); sole frame offset from ankle `[0.005, 0.0, −0.030]`.
- **aligator subclassing invariant (verified gotcha).** aligator deep-copies Python-subclassed `ODEAbstract` / `StageFunction` objects (on `IntegratorEuler(...)`, `StageModel.addConstraint(...)`, and `replaceStageCircular(...)`). **Every** Python subclass of an aligator base class **MUST** implement `__deepcopy__` that returns a fresh instance **sharing** the precompiled casadi functions (never recompile in `__deepcopy__`). Omitting this raises `TypeError: __init__() missing required positional arguments` deep inside `copy.deepcopy`.
- **Manifold-Jacobian invariant (verified gotcha).** A cpin-backed residual's `Jx` must be the Jacobian w.r.t. the **ndx tangent at x**, not w.r.t. raw `q`. Build the residual as a casadi function of `(x, u, dx)` with `q = cpin.integrate(cmodel, x[:nq], dx[:nv])`, `v = x[nq:] + dx[nv:]`, then evaluate `value`, `jacobian(r, dx)`, `jacobian(r, u)` with `dx = zeros(ndx)` **passed as an input**. **Never** `ca.substitute(dx, 0)` — the quaternion exp-map's removable singularity yields NaN.
- **State-map invariant (unchanged rule).** MuJoCo `qvel[0:3]` is world-frame; pinocchio FreeFlyer `v[0:3]` is body-local. Conversion is `v[0:3] = R(q)ᵀ · qvel[0:3]`. Single source of truth is `wb/state.py`.
- **Fatrop is removed, not coexisting.** This backend replaces Fatrop. Do not keep `StandOCP`/`WholeBodyMPC`/`opti.to_function` paths alive.
- **TDD + frequent commits.** Every task ends with an independently testable deliverable and a commit. Commit message trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Scope.** This plan delivers spec build-plan items 1, 2, 5 (foundation + warm-start gate + re-homed stand + docs). The **closed-loop forward walk and foot-lift-in-the-plant** (spec §6.1/§6.2, classified there as open research) are a **separate follow-up plan** — see "Follow-up plan (out of scope here)" at the end.

**Spec:** `docs/superpowers/specs/2026-06-27-m1-walking-design.md` (rev 3). This plan implements §2–§5, §7, §8 (items 1–2,5), §9 (gate + stand), §10.

---

## File structure

Files this plan creates or rewrites (in-place transformation of `wb-rnea-port`):

| File | Action | Responsibility |
|---|---|---|
| `t1_nmpc/robot/model.py` | rewrite | `buildReducedModel` (lock head) + one sole frame per foot + sole half-extents + mass + foot joint placements. |
| `t1_nmpc/robot/config.py` | rewrite | Reduced dims (nq=34/nv=33/nu=45), sole geometry, uniform dt=0.035/N=31, gait params, aligator solver params, `Q(66)`/`R(45)` weights, PD gains (27). |
| `t1_nmpc/wb/dynamics.py` | rewrite | cpin symbolic **primitives only**: RNEA(6D-wrench→f_ext) residual-function factory (val/Jx/Ju via dx-trick), post-hoc joint-torque function, `DoubleIntegratorODE`. |
| `t1_nmpc/wb/constraint.py` | **new** | aligator constraint builders: `RneaBaseResidual` (cpin `StageFunction`), `WrenchConeResidual` (analytic `StageFunction` → `NegativeOrthant`), `SwingWrenchResidual` (`StageFunction` → Equality), plus `contact_velocity_residual`/`swing_z_residual` (built-in `FrameVelocityResidual`). Each docstring cites paper §/eq. |
| `t1_nmpc/wb/cost.py` | **new** | cost builders: `state_tracking` (Q), `input_reg` (R), `base_velocity_target`, `arm_to_nominal`, `footstep_placement`, `torque_limit_penalty`. |
| `t1_nmpc/wb/gait.py` | rewrite | `WalkGait` (cycle 1.4 s → per-node mode) + `StandGait` (all double-support); `mode_at(t)` → `(lf_contact, rf_contact)`. |
| `t1_nmpc/wb/ocp.py` | rewrite | THIN assembler: `build_stage(mode)` (DoubleIntegrator+IntegratorEuler + CostStack + constraints), `build_problem`, `build_phase_ring`. No physics. |
| `t1_nmpc/wb/mpc.py` | rewrite | `AligatorMPC`: `SolverProxDDP` setup, `reset` (cold), `step` (rotate ring + `cycleProblem` + warm vs/lams + refresh refs), command extraction. |
| `t1_nmpc/wb/state.py` | rewrite | MuJoCo(29-joint)↔pinocchio(27-joint reduced) map with head-index drop; `extract_command`. |
| `t1_nmpc/runtime/transport.py` | keep | unchanged protocol. |
| `sim/mujoco_runtime.py` | keep/repoint | unchanged physics; reads/writes via new state map. |
| `sim/stand.py` | repoint | closed-loop stand on `AligatorMPC`. |
| `tools/codegen_solver.py` | **remove** | Fatrop-only. |
| `tests/*` | rewrite | per task below. |
| `CLAUDE.md`, `docs/2026-06-25-t1controller-divergences.md`, `docs/2026-06-27-paper-mapping.md`, memory | update | Task 14. |

---

## PART A — Aligator foundation

### Task 1: Reduced model + sole frames

**Files:**
- Modify: `t1_nmpc/robot/model.py` (rewrite)
- Test: `tests/test_model.py` (rewrite)

**Interfaces:**
- Consumes: `MPCConfig` (Task 2 defines the fields used here; for this task add only the fields referenced below — Task 2 finalizes the dataclass).
- Produces:
  - `load_model(cfg) -> RobotModel` where `RobotModel` is a dataclass with fields:
    `model: pin.Model` (reduced, 27 joints, nq=34, nv=33, **with two sole frames added**),
    `data: pin.Data`, `sole_frame_ids: tuple[int, int]` (Left, Right),
    `foot_joint_placements: tuple[tuple[int, pin.SE3], tuple[int, pin.SE3]]` (parentJointId, soleSE3-wrt-joint) per foot,
    `mass: float`, `trunk_frame_id: int`, `tau_max: np.ndarray` (27,), `half_extents: tuple[float, float]` (X, Y).
  - `nominal_q(cfg, model) -> np.ndarray` (34,), `nominal_x(cfg, model) -> np.ndarray` (67,).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_q, nominal_x


def test_reduced_model_dims():
    rm = load_model(make_config())
    assert rm.model.nq == 34 and rm.model.nv == 33
    assert rm.model.njoints == 29           # universe + root + 27 actuated
    assert rm.tau_max.shape == (27,)
    assert "AAHead_yaw" not in list(rm.model.names)
    assert "Head_pitch" not in list(rm.model.names)


def test_sole_frames_present_and_placed():
    rm = load_model(make_config())
    assert len(rm.sole_frame_ids) == 2
    for fid in rm.sole_frame_ids:
        assert rm.model.frames[fid].type == pin.FrameType.OP_FRAME
    # foot_joint_placements parent the sole frames at the ankle-roll joints
    assert len(rm.foot_joint_placements) == 2
    for (jid, jMf) in rm.foot_joint_placements:
        assert isinstance(jMf, pin.SE3)
        np.testing.assert_allclose(jMf.translation, [0.005, 0.0, -0.030], atol=1e-9)
    assert rm.half_extents == (0.1065, 0.05)


def test_nominal_consistency():
    cfg = make_config(); rm = load_model(cfg)
    q = nominal_q(cfg, rm.model)
    assert q.shape == (34,)
    np.testing.assert_allclose(q[2], cfg.nominal_base_height)
    np.testing.assert_allclose(q[3:7], [0, 0, 0, 1])    # quat xyzw identity
    x = nominal_x(cfg, rm.model)
    assert x.shape == (67,)
    assert np.allclose(x[34:], 0.0)


def test_mass_positive():
    rm = load_model(make_config())
    assert 20.0 < rm.mass < 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_model.py -q -p no:cacheprovider`
Expected: FAIL (collection/import error or `AttributeError` — new fields not present).

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/robot/model.py
"""Reduced (head-locked) T1 FreeFlyer pinocchio model + one 6D sole frame per foot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .config import MPCConfig, T1_URDF_PATH, ANKLE_ROLL_FRAMES, LOCKED_JOINTS


@dataclass
class RobotModel:
    model: pin.Model
    data: pin.Data
    sole_frame_ids: tuple[int, int]
    foot_joint_placements: tuple[tuple[int, pin.SE3], tuple[int, pin.SE3]]
    mass: float
    trunk_frame_id: int
    tau_max: np.ndarray            # (27,)
    half_extents: tuple[float, float]


def load_model(cfg: MPCConfig) -> RobotModel:
    full = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    if full.nq != 36 or full.nv != 35:
        raise ValueError(f"expected full nq=36 nv=35, got {full.nq}/{full.nv}")
    lock_ids = [full.getJointId(n) for n in LOCKED_JOINTS]
    model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
    if model.nq != 34 or model.nv != 33:
        raise ValueError(f"expected reduced nq=34 nv=33, got {model.nq}/{model.nv}")

    sole_offset = np.array([0.005, 0.0, cfg.sole_z], dtype=np.float64)  # sole_z = -0.030
    sole_ids, placements = [], []
    for ankle in ANKLE_ROLL_FRAMES:
        afid = model.getFrameId(ankle)
        parent_joint = model.frames[afid].parentJoint
        ankle_placement = model.frames[afid].placement           # ankle frame wrt parent joint
        t = ankle_placement.act(sole_offset)
        jMf = pin.SE3(np.eye(3), t)                               # sole frame wrt parent joint
        frame = pin.Frame(f"{ankle}_sole", parent_joint, afid, jMf, pin.FrameType.OP_FRAME)
        sole_ids.append(model.addFrame(frame))
        placements.append((parent_joint, jMf))

    data = model.createData()
    mass = float(pin.computeTotalMass(model, data))
    trunk_fid = model.getFrameId("Trunk")
    tau_max = np.asarray(model.effortLimit[6:], dtype=np.float64).copy()
    return RobotModel(model, data, tuple(sole_ids), tuple(placements), mass, trunk_fid,
                      tau_max, (cfg.half_len, cfg.half_width))


def nominal_q(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    q[0:3] = [0.0, 0.0, cfg.nominal_base_height]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]                                 # quat xyzw identity
    q[7:] = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)   # 27 values
    return q


def nominal_x(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    return np.concatenate([nominal_q(cfg, model), np.zeros(model.nv)])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_model.py -q -p no:cacheprovider`
Expected: PASS (4 tests). (Requires Task 2's config fields; if running Task 1 first, temporarily stub the new `MPCConfig` fields, then finalize in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/robot/model.py tests/test_model.py
git commit -m "feat(model): reduced head-locked T1 model + 6D sole frames"
```

---

### Task 2: Config rewrite (reduced dims, gait, aligator params, weights)

**Files:**
- Modify: `t1_nmpc/robot/config.py` (rewrite)
- Test: `tests/test_config.py` (rewrite)

**Interfaces:**
- Produces: `MPCConfig` (frozen dataclass) with fields used across the plan; `make_config(**overrides) -> MPCConfig`; module constants `T1_URDF_PATH`, `T1_PACKAGE_DIRS`, `JOINT_NAMES` (27), `ANKLE_ROLL_FRAMES`, `LOCKED_JOINTS`; `JointCommand` dataclass (27-wide).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import numpy as np
from t1_nmpc.robot.config import make_config, JOINT_NAMES, LOCKED_JOINTS


def test_dims():
    c = make_config()
    assert (c.n_joints, c.nq, c.nv, c.nx, c.ndx) == (27, 34, 33, 67, 66)
    assert (c.n_feet, c.nf, c.na, c.nu) == (2, 12, 33, 45)
    assert c.nodes == 31 and abs(c.dt - 0.035) < 1e-12
    assert abs(c.nodes * c.dt - 1.085) < 1e-9


def test_gait_params():
    c = make_config()
    assert abs(c.gait_cycle - 1.4) < 1e-12
    assert c.switching_times == (0.0, 0.6, 0.7, 1.3, 1.4)
    assert abs(c.swing_height - 0.08) < 1e-12
    assert (c.v_liftoff, c.v_touchdown) == (0.05, -0.05)


def test_weights_shapes():
    c = make_config()
    assert c.Q_diag.shape == (66,)        # ndx
    assert c.R_diag.shape == (45,)        # nu
    assert c.kp.shape == (27,) and c.kd.shape == (27,)
    assert c.nominal_joint_pos.shape == (27,)


def test_joint_name_tables():
    assert len(JOINT_NAMES) == 27
    assert LOCKED_JOINTS == ("AAHead_yaw", "Head_pitch")
    assert "AAHead_yaw" not in JOINT_NAMES


def test_aligator_params():
    c = make_config()
    assert c.al_tol == 1e-3
    assert c.warm_max_iters >= 1 and c.cold_max_iters > c.warm_max_iters
    assert 0.0 < c.mu_init <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_config.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/robot/config.py
"""MPCConfig: Booster T1 numbers for the aligator whole_body_rnea controller (reduced model).

Geometry/pose/limits trace to t1_controller (data only). Weights re-dimensioned to the
reduced 27-joint model (NOT traced to t1_controller — logged as a divergence)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
T1_URDF_PATH = os.path.join(_ASSETS, "t1_description", "urdf", "t1.urdf")
T1_PACKAGE_DIRS = (_ASSETS,)

LOCKED_JOINTS: Tuple[str, str] = ("AAHead_yaw", "Head_pitch")
# reduced pinocchio joint order (head removed): [Larm7, Rarm7, waist1, Lleg6, Rleg6]
JOINT_NAMES: Tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
ANKLE_ROLL_FRAMES = ("Left_Ankle_Roll", "Right_Ankle_Roll")
# index ranges within JOINT_NAMES (used by arm_to_nominal weighting)
ARM_JOINT_SLICE = slice(0, 14)
LEG_JOINT_SLICE = slice(15, 27)


def _nominal_joint_pos() -> np.ndarray:
    # 27 = [Larm7, Rarm7, waist, Lleg6, Rleg6] (head dropped)
    return np.array(
        [0.5, -1.0, 0, -1.4, 0, 0, 0]
        + [0.5, 1.0, 0, 1.4, 0, 0, 0]
        + [0]
        + [-0.05, 0, 0, 0.10, -0.05, 0]
        + [-0.05, 0, 0, 0.10, -0.05, 0],
        dtype=np.float64,
    )


def _kp() -> np.ndarray:
    return np.array([20] * 14 + [200] + [200, 200, 200, 200, 50, 50] * 1
                    + [200, 200, 200, 200, 50, 50], dtype=np.float64)


def _kd() -> np.ndarray:
    return np.array([0.5] * 14 + [5.0] + [5, 5, 5, 5, 3, 3] + [5, 5, 5, 5, 3, 3],
                    dtype=np.float64)


def _Q_diag() -> np.ndarray:
    # ndx=66 = [base_pos(6), joint_pos(27), base_vel(6), joint_vel(27)]
    base_pos = np.array([0, 0, 1000, 10000, 10000, 0], dtype=np.float64)   # x,y,yaw free
    joint_pos = np.concatenate([[100] * 14, [200], [300] * 6, [300] * 6])  # 27
    base_vel = np.array([2000, 2000, 1000, 1000, 1000, 2000], dtype=np.float64)
    joint_vel = np.concatenate([[10] * 14, [10], [2] * 6, [2] * 6])        # 27
    return np.concatenate([base_pos, joint_pos, base_vel, joint_vel])       # 66


def _R_diag() -> np.ndarray:
    # nu=45 = [a(33), W_L(6), W_R(6)]
    a_w = np.full(33, 1e-3)
    wrench_w = np.array([5e-4, 5e-4, 5e-4, 1e-3, 1e-3, 1e-3])              # per foot 6D
    return np.concatenate([a_w, wrench_w, wrench_w])


@dataclass(frozen=True)
class MPCConfig:
    # dims
    n_joints: int = 27
    nq: int = 34
    nv: int = 33
    nx: int = 67
    ndx: int = 66
    n_feet: int = 2
    nf: int = 12
    na: int = 33
    nu: int = 45

    # horizon (uniform; matches t1_controller)
    nodes: int = 31
    dt: float = 0.035

    # nominal stand
    nominal_base_height: float = 0.6734
    nominal_joint_pos: np.ndarray = field(default_factory=_nominal_joint_pos)
    robot_mass: float = 31.0          # reference only; live value from the model

    # sole geometry
    sole_z: float = -0.030
    half_len: float = 0.1065          # X half-extent
    half_width: float = 0.05          # Y half-extent
    friction_mu: float = 0.4

    # gait (walk; t1_controller verbatim)
    gait_cycle: float = 1.4
    switching_times: Tuple[float, ...] = (0.0, 0.6, 0.7, 1.3, 1.4)
    swing_height: float = 0.08
    v_liftoff: float = 0.05
    v_touchdown: float = -0.05

    # weights / gains
    Q_diag: np.ndarray = field(default_factory=_Q_diag)
    R_diag: np.ndarray = field(default_factory=_R_diag)
    arm_weight_scale: float = 50.0    # multiplies arm joint_pos weights in arm_to_nominal
    kp: np.ndarray = field(default_factory=_kp)
    kd: np.ndarray = field(default_factory=_kd)

    # aligator solver
    al_tol: float = 1e-3              # target_tol (constraint sat) for the AL loop
    mu_init: float = 1e-2            # initial AL penalty
    cold_max_iters: int = 50
    warm_max_iters: int = 6
    torque_limit_weight: float = 1e-3  # soft torque-limit penalty (§3.8)

    # execution rates
    pd_hz: float = 500.0
    physics_hz: float = 2000.0


def make_config(**overrides) -> MPCConfig:
    c = MPCConfig(**overrides)
    assert c.nx == c.nq + c.nv == 67
    assert c.ndx == 2 * c.nv == 66
    assert c.nf == 6 * c.n_feet == 12
    assert c.nu == c.na + c.nf == 45
    assert c.na == c.nv
    assert c.nominal_joint_pos.shape == (27,)
    assert c.Q_diag.shape == (c.ndx,)
    assert c.R_diag.shape == (c.nu,)
    assert c.kp.shape == (27,) and c.kd.shape == (27,)
    assert len(JOINT_NAMES) == 27
    return c


@dataclass
class JointCommand:
    """27-joint command: tau = tau_ff + kp*(q_des-q) + kd*(qd_des-qd)."""
    q_des: np.ndarray    # (27,)
    qd_des: np.ndarray   # (27,)
    tau_ff: np.ndarray   # (27,)
    kp: np.ndarray       # (27,)
    kd: np.ndarray       # (27,)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_config.py tests/test_model.py -q -p no:cacheprovider`
Expected: PASS (Task 1 + Task 2 tests green together).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/robot/config.py tests/test_config.py
git commit -m "feat(config): reduced-model dims, walk gait, aligator solver params, Q(66)/R(45)"
```

---

### Task 3: cpin dynamics primitives (RNEA residual factory, torque, double integrator)

**Files:**
- Modify: `t1_nmpc/wb/dynamics.py` (rewrite)
- Test: `tests/test_dynamics.py` (rewrite)

**Interfaces:**
- Consumes: `RobotModel` (Task 1).
- Produces a `WBDynamics` class:
  - `__init__(self, rm: RobotModel, cfg: MPCConfig)`.
  - `rnea_funcs(base_only: bool) -> tuple[ca.Function, ca.Function, ca.Function]` → `(val, Jx, Ju)`, each a function of `(x[67], u[45], dx[66])`; residual = `RNEA(q,v,a,f_ext(W))[:6]` if `base_only` else full `[33]` (used for the post-hoc torque path uses rows `6:`).
  - `joint_torque_fn() -> ca.Function` mapping `(x, u)` → `tau_joint[27]` (= `RNEA(...)[6:]` at `dx=0`).
  - `DoubleIntegratorODE` (aligator `ODEAbstract` subclass) with `__deepcopy__`.
  - attributes: `cmodel`, `space` (`MultibodyPhaseSpace`), `ndx`, `nu`, `nq`, `nv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamics.py
import numpy as np
import pinocchio as pin
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics


def _setup():
    cfg = make_config(); rm = load_model(cfg)
    return cfg, rm, WBDynamics(rm, cfg)


def test_rnea_jacobian_matches_finite_diff():
    cfg, rm, dyn = _setup()
    val, Jx, Ju = dyn.rnea_funcs(base_only=True)
    rng = np.random.default_rng(0)
    q = pin.integrate(rm.model, pin.neutral(rm.model), rng.standard_normal(33) * 0.1)
    v = rng.standard_normal(33) * 0.1
    x = np.concatenate([q, v]); u = rng.standard_normal(45) * 0.1
    z = np.zeros(66)
    r0 = np.asarray(val(x, u, z)).flatten()
    J = np.asarray(Jx(x, u, z))
    # finite diff on the manifold tangent
    eps = 1e-6; Jfd = np.zeros((6, 66))
    for i in range(66):
        d = np.zeros(66); d[i] = eps
        rp = np.asarray(val(x, u, d)).flatten()
        Jfd[:, i] = (rp - r0) / eps
    assert np.max(np.abs(J - Jfd)) < 1e-4
    # Ju vs finite diff
    Jfu = np.zeros((6, 45))
    for i in range(45):
        du = u.copy(); du[i] += eps
        Jfu[:, i] = (np.asarray(val(x, du, z)).flatten() - r0) / eps
    assert np.max(np.abs(np.asarray(Ju(x, u, z)) - Jfu)) < 1e-4


def test_rnea_base_zero_at_gravity_comp_stand():
    cfg, rm, dyn = _setup()
    val, _, _ = dyn.rnea_funcs(base_only=True)
    x = nominal_x(cfg, rm.model)
    fz = rm.mass * 9.81 / 2.0
    W = np.array([0, 0, fz, 0, 0, 0])               # per foot, sole-frame vertical
    u = np.concatenate([np.zeros(33), W, W])        # a=0, both feet support
    r = np.asarray(val(x, u, np.zeros(66))).flatten()
    assert np.max(np.abs(r)) < 5.0                  # base wrench residual small at gravity comp


def test_double_integrator_ode_forward():
    cfg, rm, dyn = _setup()
    ode = dyn.DoubleIntegratorODE(dyn.space, dyn.nu)
    data = ode.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45); u[:33] = 1.0   # a = 1
    ode.forward(x, u, data)
    np.testing.assert_allclose(data.xdot[:33], x[34:])             # qdot = v (=0)
    np.testing.assert_allclose(data.xdot[33:], np.ones(33))        # vdot = a
    ode.dForward(x, u, data)
    assert np.allclose(data.Ju[33:, :], np.eye(33))


def test_ode_deepcopy_survives_integrator():
    cfg, rm, dyn = _setup()
    ode = dyn.DoubleIntegratorODE(dyn.space, dyn.nu)
    disc = aligator.dynamics.IntegratorEuler(ode, cfg.dt)   # deep-copies ode internally
    assert disc.timestep == cfg.dt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_dynamics.py -q -p no:cacheprovider`
Expected: FAIL (import error — `WBDynamics` new signature).

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/dynamics.py
"""cpin symbolic primitives for the aligator whole_body_rnea backend.

RNEA residual = RNEA(q, v, a, f_ext(W))[:6] (base underactuation, paper Eq. 5), with one
6D foot wrench W_foot per foot expressed in its sole frame, transformed to the parent ankle
joint by the constant frame placement jMf. All Jacobians use the manifold dx-trick
(see Global Constraints)."""
from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import aligator
from aligator import manifolds, dynamics as ali_dyn

from ..robot.config import MPCConfig
from ..robot.model import RobotModel


class _DoubleIntegratorODE(ali_dyn.ODEAbstract):
    """xdot = (v, a) on MultibodyPhaseSpace. Pure kinematics; physics is the RNEA constraint."""
    def __init__(self, space, nu):
        super().__init__(space, nu)
        self._space, self._nu = space, nu
        self.nv = space.model.nv

    def __deepcopy__(self, memo):
        return _DoubleIntegratorODE(self._space, self._nu)

    def forward(self, x, u, data):
        nv = self.nv
        data.xdot[:nv] = x[self._space.model.nq:self._space.model.nq + nv]   # qdot tangent = v
        data.xdot[nv:] = u[:nv]                                              # vdot = a

    def dForward(self, x, u, data):
        nv = self.nv
        data.Jx[:, :] = 0.0
        data.Jx[:nv, nv:] = np.eye(nv)
        data.Ju[:, :] = 0.0
        data.Ju[nv:, :nv] = np.eye(nv)


class WBDynamics:
    def __init__(self, rm: RobotModel, cfg: MPCConfig):
        self.cmodel = cpin.Model(rm.model)
        self.cdata = self.cmodel.createData()
        self.nq, self.nv = self.cmodel.nq, self.cmodel.nv
        self.ndx, self.nu = cfg.ndx, cfg.nu
        self.space = manifolds.MultibodyPhaseSpace(rm.model)
        # constant sole->joint placements as cpin SE3 (jMf is config-independent)
        self._feet = [(jid, cpin.SE3(jMf)) for (jid, jMf) in rm.foot_joint_placements]
        self.DoubleIntegratorODE = _DoubleIntegratorODE

    def _rnea_expr(self, q, v, a, W):
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.cmodel.njoints)]
        for k, (jid, jMf) in enumerate(self._feet):
            Wk = W[6 * k:6 * k + 6]
            f_ext[jid] = cpin.Force(f_ext[jid].vector + jMf.act(cpin.Force(Wk)).vector)
        return cpin.rnea(self.cmodel, self.cdata, q, v, a, f_ext)

    def rnea_funcs(self, base_only: bool = True):
        x = ca.SX.sym("x", self.nq + self.nv)
        u = ca.SX.sym("u", self.nu)
        dx = ca.SX.sym("dx", self.ndx)
        q = cpin.integrate(self.cmodel, x[:self.nq], dx[:self.nv])
        v = x[self.nq:] + dx[self.nv:]
        a = u[:self.nv]
        W = u[self.nv:]
        tau = self._rnea_expr(q, v, a, W)
        r = tau[:6] if base_only else tau
        val = ca.Function("rnea_val", [x, u, dx], [r])
        Jx = ca.Function("rnea_Jx", [x, u, dx], [ca.jacobian(r, dx)])
        Ju = ca.Function("rnea_Ju", [x, u, dx], [ca.jacobian(r, u)])
        return val, Jx, Ju

    def joint_torque_fn(self):
        x = ca.SX.sym("x", self.nq + self.nv)
        u = ca.SX.sym("u", self.nu)
        q, v = x[:self.nq], x[self.nq:]
        tau = self._rnea_expr(q, v, u[:self.nv], u[self.nv:])
        return ca.Function("joint_tau", [x, u], [tau[6:]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_dynamics.py -q -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/dynamics.py tests/test_dynamics.py
git commit -m "feat(dynamics): cpin RNEA(6D-wrench) residual factory + double-integrator ODE"
```

---

### Task 4: Constraint builders (RNEA base, wrench cone, contact velocity, swing)

**Files:**
- Create: `t1_nmpc/wb/constraint.py`
- Test: `tests/test_constraint.py` (new)

**Interfaces:**
- Consumes: `WBDynamics` (Task 3), `RobotModel` (Task 1), `MPCConfig`.
- Produces (all return `(StageFunction-like, ConstraintSet)` ready for `stage.addConstraint(*pair)`):
  - `RneaBaseResidual(aligator.StageFunction)` — class; ctor `(ndx, nu, funcs)`, `funcs = dyn.rnea_funcs(base_only=True)`; `__deepcopy__` shares funcs.
  - `WrenchConeResidual(aligator.StageFunction)` — analytic; ctor `(ndx, nu, foot_index, mu, X, Y)`; residual rows `≤ 0`; `__deepcopy__`.
  - `SwingWrenchResidual(aligator.StageFunction)` — selects a swing foot's 6 wrench components (`= 0`); ctor `(ndx, nu, foot_index)`; `__deepcopy__`.
  - `contact_velocity_residual(rm, ndx, nu, foot_index) -> aligator.FrameVelocityResidual` (6D, ref `Motion.Zero()`).
  - `swing_z_residual(rm, ndx, nu, foot_index) -> UnaryFunctionSliceXpr` (z row of `FrameVelocityResidual`; returns also the underlying residual so the caller can `setReference`).
  - module constants: `EQ = aligator.constraints.EqualityConstraintSet`, `NEG = aligator.constraints.NegativeOrthant`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constraint.py
import copy
import numpy as np
import pinocchio as pin
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb import constraint as C


def _setup():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn


def test_rnea_residual_deepcopy_shares_funcs():
    cfg, rm, dyn = _setup()
    r = C.RneaBaseResidual(cfg.ndx, cfg.nu, dyn.rnea_funcs(base_only=True))
    r2 = copy.deepcopy(r)
    assert r2._funcs is r._funcs          # shared, not recompiled
    assert r2.nr == 6


def test_rnea_residual_evaluate_matches_factory():
    cfg, rm, dyn = _setup()
    funcs = dyn.rnea_funcs(base_only=True)
    r = C.RneaBaseResidual(cfg.ndx, cfg.nu, funcs)
    data = r.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45)
    r.evaluate(x, u, data)
    expect = np.asarray(funcs[0](x, u, np.zeros(66))).flatten()
    np.testing.assert_allclose(data.value, expect, atol=1e-9)
    r.computeJacobians(x, u, data)
    assert data.Jx.shape == (6, 66) and data.Ju.shape == (6, 45)


def test_wrench_cone_unilateral_and_friction():
    cfg, rm, dyn = _setup()
    wc = C.WrenchConeResidual(cfg.ndx, cfg.nu, foot_index=0, mu=cfg.friction_mu,
                              X=cfg.half_len, Y=cfg.half_width)
    data = wc.createData()
    x = nominal_x(cfg, rm.model)
    # vertical-only force inside cone -> all rows <= 0 (feasible)
    u = np.zeros(45); u[33 + 2] = 100.0
    wc.evaluate(x, u, data)
    assert np.all(data.value <= 1e-9)
    # large lateral force -> friction row > 0 (violated)
    u2 = np.zeros(45); u2[33 + 0] = 100.0; u2[33 + 2] = 10.0
    wc.evaluate(x, u2, data)
    assert np.any(data.value > 0.0)
    # finite-diff Ju check
    wc.computeJacobians(x, u2, data)
    eps = 1e-6; r0 = np.asarray(data.value).copy()
    Jfd = np.zeros_like(data.Ju)
    for i in range(45):
        du = u2.copy(); du[i] += eps
        d2 = wc.createData(); wc.evaluate(x, du, d2)
        Jfd[:, i] = (np.asarray(d2.value) - r0) / eps
    assert np.max(np.abs(np.asarray(data.Ju) - Jfd)) < 1e-3


def test_swing_wrench_selects_foot():
    cfg, rm, dyn = _setup()
    sw = C.SwingWrenchResidual(cfg.ndx, cfg.nu, foot_index=1)
    data = sw.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45); u[39:45] = [1, 2, 3, 4, 5, 6]
    sw.evaluate(x, u, data)
    np.testing.assert_allclose(data.value, [1, 2, 3, 4, 5, 6])


def test_contact_velocity_residual_zero_at_rest():
    cfg, rm, dyn = _setup()
    res = C.contact_velocity_residual(rm, cfg.ndx, cfg.nu, foot_index=0)
    data = res.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45)
    res.evaluate(x, u, data)
    assert np.max(np.abs(data.value)) < 1e-9     # v=0 -> foot velocity 0


def test_swing_z_setreference():
    cfg, rm, dyn = _setup()
    sliced, base = C.swing_z_residual(rm, cfg.ndx, cfg.nu, foot_index=0)
    assert sliced.nr == 1
    base.setReference(pin.Motion(np.array([0, 0, 0.05, 0, 0, 0.0])))
    assert abs(base.getReference().linear[2] - 0.05) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_constraint.py -q -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/constraint.py
"""aligator constraint builders for the whole_body_rnea OCP. Each cites the paper.

Conventions: u = [a(33), W_L(6), W_R(6)]; wrench slice for foot k is u[33+6k : 33+6k+6]
= [f(3), tau(3)] in the sole frame (f_z = surface normal)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import aligator

EQ = aligator.constraints.EqualityConstraintSet
NEG = aligator.constraints.NegativeOrthant

_WRENCH0 = 33  # start index of W in u


class RneaBaseResidual(aligator.StageFunction):
    """RNEA(q,v,a,f_ext(W))[:6] = 0 — floating-base underactuation (paper Eq. 5)."""
    def __init__(self, ndx, nu, funcs):
        super().__init__(ndx, nu, 6)
        self._ndx, self._nu, self._funcs = ndx, nu, funcs
        self._zero = np.zeros(ndx)

    def __deepcopy__(self, memo):
        return RneaBaseResidual(self._ndx, self._nu, self._funcs)  # share compiled funcs

    def evaluate(self, x, u, data):
        data.value[:] = np.asarray(self._funcs[0](x, u, self._zero)).flatten()

    def computeJacobians(self, x, u, data):
        data.Jx[:, :] = np.asarray(self._funcs[1](x, u, self._zero))
        data.Ju[:, :] = np.asarray(self._funcs[2](x, u, self._zero))


class WrenchConeResidual(aligator.StageFunction):
    """Flat-foot contact-wrench cone on W_foot, rows <= 0 (NegativeOrthant).
    friction (paper Eq. 6) + unilateral + CoP + yaw bound (Caron et al. 2015, flat-foot adapt).
    Rows: [ -f_z, f_x^2+f_y^2 - mu^2 f_z^2, |tau_y|-X f_z (x2), |tau_x|-Y f_z (x2),
            |tau_z| - mu(X+Y) f_z (x2) ] -> 8 rows (abs split into +/-)."""
    def __init__(self, ndx, nu, foot_index, mu, X, Y):
        super().__init__(ndx, nu, 8)
        self._ndx, self._nu = ndx, nu
        self._i = _WRENCH0 + 6 * foot_index
        self._mu, self._X, self._Y = mu, X, Y

    def __deepcopy__(self, memo):
        return WrenchConeResidual(self._ndx, self._nu, (self._i - _WRENCH0) // 6,
                                  self._mu, self._X, self._Y)

    def _rows(self, u):
        fx, fy, fz, tx, ty, tz = u[self._i:self._i + 6]
        mu, X, Y = self._mu, self._X, self._Y
        return np.array([
            -fz,
            fx * fx + fy * fy - mu * mu * fz * fz,
            ty - X * fz,  -ty - X * fz,
            tx - Y * fz,  -tx - Y * fz,
            tz - mu * (X + Y) * fz,  -tz - mu * (X + Y) * fz,
        ])

    def evaluate(self, x, u, data):
        data.value[:] = self._rows(u)

    def computeJacobians(self, x, u, data):
        i, mu, X, Y = self._i, self._mu, self._X, self._Y
        fx, fy, fz = u[i], u[i + 1], u[i + 2]
        J = np.zeros((8, self._nu))
        J[0, i + 2] = -1.0
        J[1, i] = 2 * fx; J[1, i + 1] = 2 * fy; J[1, i + 2] = -2 * mu * mu * fz
        J[2, i + 4] = 1.0;  J[2, i + 2] = -X
        J[3, i + 4] = -1.0; J[3, i + 2] = -X
        J[4, i + 3] = 1.0;  J[4, i + 2] = -Y
        J[5, i + 3] = -1.0; J[5, i + 2] = -Y
        J[6, i + 5] = 1.0;  J[6, i + 2] = -mu * (X + Y)
        J[7, i + 5] = -1.0; J[7, i + 2] = -mu * (X + Y)
        data.Jx[:, :] = 0.0
        data.Ju[:, :] = J


class SwingWrenchResidual(aligator.StageFunction):
    """W_foot = 0 for a swing foot (paper §IV-B2)."""
    def __init__(self, ndx, nu, foot_index):
        super().__init__(ndx, nu, 6)
        self._ndx, self._nu = ndx, nu
        self._i = _WRENCH0 + 6 * foot_index
        self._sel = np.zeros((6, nu)); self._sel[np.arange(6), self._i + np.arange(6)] = 1.0

    def __deepcopy__(self, memo):
        return SwingWrenchResidual(self._ndx, self._nu, (self._i - _WRENCH0) // 6)

    def evaluate(self, x, u, data):
        data.value[:] = u[self._i:self._i + 6]

    def computeJacobians(self, x, u, data):
        data.Jx[:, :] = 0.0
        data.Ju[:, :] = self._sel


def contact_velocity_residual(rm, ndx, nu, foot_index):
    """Stance: 6D foot spatial velocity = 0 (paper §IV-B2, per foot; 6D = flat-foot adapt)."""
    fid = rm.sole_frame_ids[foot_index]
    return aligator.FrameVelocityResidual(ndx, nu, rm.model, pin.Motion.Zero(), fid,
                                          pin.LOCAL_WORLD_ALIGNED)


def swing_z_residual(rm, ndx, nu, foot_index):
    """Swing: foot z-velocity = v_z_ref (paper §IV-B2, hard, velocity-level).
    Returns (z_slice_function, base_residual) — caller calls base.setReference(Motion) per tick."""
    fid = rm.sole_frame_ids[foot_index]
    base = aligator.FrameVelocityResidual(ndx, nu, rm.model, pin.Motion.Zero(), fid,
                                          pin.LOCAL_WORLD_ALIGNED)
    return base[2:3], base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_constraint.py -q -p no:cacheprovider`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/constraint.py tests/test_constraint.py
git commit -m "feat(constraint): RNEA-base, wrench-cone, swing-wrench, contact-velocity, swing-z builders"
```

---

### Task 5: Cost builders

**Files:**
- Create: `t1_nmpc/wb/cost.py`
- Test: `tests/test_cost.py` (new)

**Interfaces:**
- Consumes: `WBDynamics.space`, `MPCConfig`, `RobotModel`, `nominal_x`.
- Produces:
  - `state_tracking(space, nu, x_des, Q_diag) -> aligator.QuadraticStateCost`.
  - `input_reg(space, nu, u_des, R_diag) -> aligator.QuadraticControlCost`.
  - `arm_to_nominal(space, nu, x_des, cfg) -> aligator.QuadraticStateCost` (arm joint_pos weights scaled).
  - `gravity_comp_u_des(rm, n_support: int) -> np.ndarray` (45,) — `a=0`, vertical force `m·g / n_support` on supporting feet.
  - `make_cost_stack(space, nu, components_with_weights) -> aligator.CostStack`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb import cost as K


def _setup():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn


def test_state_tracking_zero_at_nominal():
    cfg, rm, dyn = _setup()
    x_des = nominal_x(cfg, rm.model)
    c = K.state_tracking(dyn.space, cfg.nu, x_des, cfg.Q_diag)
    data = c.createData()
    c.evaluate(x_des, np.zeros(45), data)
    assert abs(data.value) < 1e-12


def test_input_reg_penalizes_off_target():
    cfg, rm, dyn = _setup()
    u_des = K.gravity_comp_u_des(rm, n_support=2)
    c = K.input_reg(dyn.space, cfg.nu, u_des, cfg.R_diag)
    data = c.createData()
    x = nominal_x(cfg, rm.model)
    c.evaluate(x, u_des, data); assert abs(data.value) < 1e-12
    u_off = u_des.copy(); u_off[0] += 10.0
    c.evaluate(x, u_off, data); assert data.value > 0.0


def test_gravity_comp_supports_split():
    cfg, rm, dyn = _setup()
    u = K.gravity_comp_u_des(rm, n_support=2)
    assert u.shape == (45,)
    fz = rm.mass * 9.81 / 2.0
    np.testing.assert_allclose(u[33 + 2], fz)     # left foot f_z
    np.testing.assert_allclose(u[39 + 2], fz)     # right foot f_z
    assert np.allclose(u[:33], 0.0)


def test_arm_to_nominal_weights_arms_more():
    cfg, rm, dyn = _setup()
    x_des = nominal_x(cfg, rm.model)
    c = K.arm_to_nominal(dyn.space, cfg.nu, x_des, cfg)
    data = c.createData()
    # perturb an arm joint -> nonzero cost
    x = x_des.copy(); x[7] += 0.2     # first arm joint (q index 7)
    c.evaluate(x, np.zeros(45), data)
    assert data.value > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_cost.py -q -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/cost.py
"""Cost builders for the whole_body_rnea OCP (paper Eq. 3: ||x-x_des||^2_Q + ||u-u_des||^2_R)."""
from __future__ import annotations

import numpy as np
import aligator

from ..robot.config import MPCConfig, ARM_JOINT_SLICE


def state_tracking(space, nu, x_des, Q_diag):
    res = aligator.StateErrorResidual(space, nu, np.asarray(x_des, dtype=np.float64))
    return aligator.QuadraticStateCost(res, np.diag(np.asarray(Q_diag, dtype=np.float64)))


def input_reg(space, nu, u_des, R_diag):
    return aligator.QuadraticControlCost(space, np.asarray(u_des, dtype=np.float64),
                                         np.diag(np.asarray(R_diag, dtype=np.float64)))


def arm_to_nominal(space, nu, x_des, cfg: MPCConfig):
    """High-weight state cost on arm joint positions only (M1 holds arms to nominal)."""
    w = np.zeros(cfg.ndx)
    # arm joint_pos delta indices: base_pos(6) + ARM_JOINT_SLICE within the 27 joints
    arm = np.arange(6 + ARM_JOINT_SLICE.start, 6 + ARM_JOINT_SLICE.stop)
    w[arm] = cfg.arm_weight_scale
    res = aligator.StateErrorResidual(space, nu, np.asarray(x_des, dtype=np.float64))
    return aligator.QuadraticStateCost(res, np.diag(w))


def gravity_comp_u_des(rm, n_support: int) -> np.ndarray:
    u = np.zeros(45, dtype=np.float64)
    if n_support > 0:
        fz = rm.mass * 9.81 / n_support
        for k in range(2):                    # both feet entries; swing feet get fz too as a
            u[33 + 6 * k + 2] = fz            # mild prior (overridden by swing W=0 constraint)
    return u


def make_cost_stack(space, nu, components_with_weights):
    stack = aligator.CostStack(space, nu)
    for name, cost, weight in components_with_weights:
        stack.addCost(name, cost, weight)
    return stack
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_cost.py -q -p no:cacheprovider`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/cost.py tests/test_cost.py
git commit -m "feat(cost): state-tracking, input-reg, arm-to-nominal, gravity-comp u_des"
```

---

### Task 6: Gait schedule (walk + stand)

**Files:**
- Modify: `t1_nmpc/wb/gait.py` (rewrite)
- Test: `tests/test_gait.py` (rewrite)

**Interfaces:**
- Consumes: `MPCConfig`.
- Produces:
  - `FootMode` = `tuple[bool, bool]` (lf_contact, rf_contact).
  - `WalkGait(cfg)` with `mode_at(t: float) -> FootMode`, `swing_phase(t, foot_index) -> float | None` (∈[0,1] if swinging, else `None`), `horizon_modes(t0) -> list[FootMode]` (length `nodes`).
  - `StandGait(cfg)` with the same interface, always `(True, True)`, `swing_phase -> None`.
  - `v_z_ref(phase, cfg) -> float` (cubic-spline z-velocity; +liftoff → 0 at apex → −touchdown).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gait.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.wb.gait import WalkGait, StandGait, v_z_ref


def test_stand_always_double_support():
    g = StandGait(make_config())
    for t in np.linspace(0, 3, 13):
        assert g.mode_at(t) == (True, True)
        assert g.swing_phase(t, 0) is None


def test_walk_mode_sequence():
    cfg = make_config(); g = WalkGait(cfg)
    assert g.mode_at(0.0) == (False, True)     # LF swing
    assert g.mode_at(0.65) == (True, True)     # double
    assert g.mode_at(1.0) == (True, False)     # RF swing
    assert g.mode_at(1.35) == (True, True)     # double
    assert g.mode_at(1.4) == g.mode_at(0.0)    # periodic


def test_walk_swing_phase_progresses():
    cfg = make_config(); g = WalkGait(cfg)
    p0 = g.swing_phase(0.0, 0); p1 = g.swing_phase(0.3, 0)
    assert p0 is not None and 0.0 <= p0 < 0.1
    assert p1 is not None and 0.4 < p1 < 0.6
    assert g.swing_phase(0.0, 1) is None       # RF not swinging at t=0


def test_horizon_modes_length():
    cfg = make_config(); g = WalkGait(cfg)
    modes = g.horizon_modes(0.0)
    assert len(modes) == cfg.nodes


def test_v_z_ref_shape():
    cfg = make_config()
    assert v_z_ref(0.0, cfg) > 0          # liftoff rising
    assert abs(v_z_ref(0.5, cfg)) < 1e-6  # zero at apex
    assert v_z_ref(1.0, cfg) < 0          # touchdown descending
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_gait.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/gait.py
"""Biped contact scheduling: walk (cycle 1.4 s, t1_controller gait.info) + stand."""
from __future__ import annotations

from ..robot.config import MPCConfig

FootMode = tuple  # (lf_contact: bool, rf_contact: bool)


def v_z_ref(phase: float, cfg: MPCConfig) -> float:
    """Time-derivative of a cubic swing-height spline: +liftoff -> 0 at apex -> -touchdown.
    Implemented as a symmetric cubic in [0,1] whose derivative is zero at phase=0.5."""
    p = min(max(phase, 0.0), 1.0)
    # height h(p) = swing_height * (3p^2 - 2p^3) blended for up/down would not return to 0;
    # use a velocity profile directly: v(p) = A * (1 - 2p) * 6 ... choose simple shape:
    # v(0)=+v_liftoff, v(0.5)=0, v(1)=-v_touchdown (touchdown stored negative).
    if p <= 0.5:
        s = p / 0.5
        return cfg.v_liftoff * (1.0 - s)            # linear rise->0; (cubic optional refinement)
    s = (p - 0.5) / 0.5
    return cfg.v_touchdown * s                       # 0 -> v_touchdown (negative)


class StandGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    def mode_at(self, t: float) -> FootMode:
        return (True, True)

    def swing_phase(self, t: float, foot_index: int):
        return None

    def horizon_modes(self, t0: float):
        return [(True, True)] * self.cfg.nodes


class WalkGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        # switching_times = (0, 0.6, 0.7, 1.3, 1.4): LF, double, RF, double
        self.t_lf_end = cfg.switching_times[1]      # 0.6
        self.t_d1_end = cfg.switching_times[2]      # 0.7
        self.t_rf_end = cfg.switching_times[3]      # 1.3
        self.cycle = cfg.gait_cycle                 # 1.4

    def _phase_time(self, t: float) -> float:
        return t % self.cycle

    def mode_at(self, t: float) -> FootMode:
        tp = self._phase_time(t)
        if tp < self.t_lf_end:
            return (False, True)                    # LF swing
        if tp < self.t_d1_end:
            return (True, True)
        if tp < self.t_rf_end:
            return (True, False)                    # RF swing
        return (True, True)

    def swing_phase(self, t: float, foot_index: int):
        tp = self._phase_time(t)
        if foot_index == 0 and tp < self.t_lf_end:
            return tp / self.t_lf_end
        if foot_index == 1 and self.t_d1_end <= tp < self.t_rf_end:
            return (tp - self.t_d1_end) / (self.t_rf_end - self.t_d1_end)
        return None

    def horizon_modes(self, t0: float):
        return [self.mode_at(t0 + i * self.cfg.dt) for i in range(self.cfg.nodes)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_gait.py -q -p no:cacheprovider`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/gait.py tests/test_gait.py
git commit -m "feat(gait): walk schedule (cycle 1.4s) + stand + cubic swing z-velocity ref"
```

---

### Task 7: OCP assembler (thin)

**Files:**
- Modify: `t1_nmpc/wb/ocp.py` (rewrite)
- Test: `tests/test_ocp.py` (rewrite)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces an `OCPBuilder` class:
  - `__init__(self, cfg, rm, dyn)`.
  - `build_stage(mode: FootMode) -> (aligator.StageModel, dict)` where the dict records, per swinging foot, the tuple `(foot_index, constraint_stack_index)` — the **integer position of the swing-z constraint in the stage's `ConstraintStack`** (NOT the residual object). Stage carries: cost (state_tracking + input_reg + arm_to_nominal), RNEA-base equality (constraint index 0), then per-foot stance (wrench-cone NEG + contact-velocity EQ) or swing (swing-wrench EQ + swing-z EQ). **Gap-closing dynamics is the StageModel dynamics; equality constraints are added via `addConstraint` (order within a stage does not matter for aligator, unlike Fatrop).**
  - **VERIFIED GOTCHA (why index, not handle):** `addConstraint` **deep-copies** the residual into the stage, so the original `swing_z_residual` `base` object is *disconnected* from the copy the solver evaluates (confirmed: setting the original's `vref` does not change the problem's copy). The per-tick swing-z target must therefore be written **through the problem**: `problem.stages[i].constraints.funcs[cidx].func.vref = pin.Motion(...)` — where `funcs[cidx]` is the `UnaryFunctionSliceXpr` and its `.func` is the wrapped `FrameVelocityResidual` (use the `vref` property, NOT the deprecated `setReference`). `build_stage` therefore records the integer `cidx`, and `_refresh_refs` (Task 8) reaches the residual through `problem.stages`.
  - `terminal_cost() -> aligator.CostAbstract`.
  - `build_problem(modes: list[FootMode], x0) -> (aligator.TrajOptProblem, list[dict])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ocp.py
import numpy as np
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb.ocp import OCPBuilder


def _builder():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn, OCPBuilder(cfg, rm, dyn)


def test_build_double_support_stage():
    cfg, rm, dyn, b = _builder()
    stage, handles = b.build_stage((True, True))
    assert stage.nu == 45 and stage.ndx1 == 66
    # constraints: rnea(6) + 2*(wrenchcone 8 + contactvel 6) = 6 + 28 = 34 dual rows
    assert stage.num_dual == 6 + 2 * (8 + 6)
    assert handles["swing"] == []        # no swinging feet


def test_build_swing_stage():
    cfg, rm, dyn, b = _builder()
    stage, handles = b.build_stage((False, True))   # LF swing, RF stance
    # rnea(6) + LF(swingwrench 6 + swingz 1) + RF(wrenchcone 8 + contactvel 6) = 27
    assert stage.num_dual == 6 + (6 + 1) + (8 + 6)
    # add order: rnea(idx0), LF swingwrench(idx1), LF swing-z(idx2), RF wrenchcone(idx3), RF contactvel(idx4)
    assert handles["swing"] == [(0, 2)]              # (foot_index, constraint-stack index of swing-z)
    # the recorded index must point at the sliced swing-z residual (nr==1) inside the stage's stack
    assert stage.constraints.funcs[2].nr == 1


def test_build_problem_integrity():
    cfg, rm, dyn, b = _builder()
    x0 = nominal_x(cfg, rm.model)
    modes = [(True, True)] * cfg.nodes
    problem, handles = b.build_problem(modes, x0)
    assert problem.num_steps == cfg.nodes
    assert len(handles) == cfg.nodes
    problem.checkIntegrity()                          # raises if malformed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_ocp.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/ocp.py
"""Thin assembler: build aligator StageModels from gait flags. No physics lives here."""
from __future__ import annotations

import numpy as np
import aligator
from aligator import dynamics as ali_dyn

from ..robot.config import MPCConfig
from ..robot.model import RobotModel, nominal_x
from .dynamics import WBDynamics
from . import cost as K
from . import constraint as C


class OCPBuilder:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, dyn: WBDynamics):
        self.cfg, self.rm, self.dyn = cfg, rm, dyn
        self.space = dyn.space
        self.x_des = nominal_x(cfg, rm.model)
        self._rnea_funcs = dyn.rnea_funcs(base_only=True)   # built once, shared across stages
        self.u_des = K.gravity_comp_u_des(rm, n_support=2)

    def _discrete_dynamics(self):
        ode = self.dyn.DoubleIntegratorODE(self.space, self.cfg.nu)
        return ali_dyn.IntegratorEuler(ode, self.cfg.dt)

    def _cost(self):
        comps = [
            ("state", K.state_tracking(self.space, self.cfg.nu, self.x_des, self.cfg.Q_diag), 1.0),
            ("input", K.input_reg(self.space, self.cfg.nu, self.u_des, self.cfg.R_diag), 1.0),
            ("arms", K.arm_to_nominal(self.space, self.cfg.nu, self.x_des, self.cfg), 1.0),
        ]
        return K.make_cost_stack(self.space, self.cfg.nu, comps)

    def build_stage(self, mode):
        cfg, rm = self.cfg, self.rm
        stage = aligator.StageModel(self._cost(), self._discrete_dynamics())
        stage.addConstraint(C.RneaBaseResidual(cfg.ndx, cfg.nu, self._rnea_funcs), C.EQ())
        handles = {"swing": []}
        cidx = 1                                   # rnea_base occupies constraint-stack index 0
        for k, in_contact in enumerate(mode):
            if in_contact:
                stage.addConstraint(
                    C.WrenchConeResidual(cfg.ndx, cfg.nu, k, cfg.friction_mu,
                                         cfg.half_len, cfg.half_width), C.NEG()); cidx += 1
                stage.addConstraint(C.contact_velocity_residual(rm, cfg.ndx, cfg.nu, k), C.EQ()); cidx += 1
            else:
                stage.addConstraint(C.SwingWrenchResidual(cfg.ndx, cfg.nu, k), C.EQ()); cidx += 1
                sliced, _ = C.swing_z_residual(rm, cfg.ndx, cfg.nu, k)   # discard the disconnected base
                stage.addConstraint(sliced, C.EQ())
                handles["swing"].append((k, cidx)); cidx += 1            # record swing-z stack index
        return stage, handles

    def terminal_cost(self):
        return K.state_tracking(self.space, self.cfg.nu, self.x_des, self.cfg.Q_diag)

    def build_problem(self, modes, x0):
        stages = aligator.StdVec_StageModel()
        all_handles = []
        for mode in modes:
            stage, handles = self.build_stage(mode)
            stages.append(stage)
            all_handles.append(handles)
        problem = aligator.TrajOptProblem(np.asarray(x0, dtype=np.float64), stages,
                                          self.terminal_cost())
        return problem, all_handles
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_ocp.py -q -p no:cacheprovider`
Expected: PASS (3 tests). If `num_dual` counts differ, read the actual `stage.num_dual` and reconcile the test's arithmetic (do not change the constraint set).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/ocp.py tests/test_ocp.py
git commit -m "feat(ocp): thin aligator assembler (stage per contact mode + problem builder)"
```

---

### Task 8: MPC driver (AligatorMPC)

**Files:**
- Modify: `t1_nmpc/wb/mpc.py` (rewrite)
- Test: `tests/test_mpc.py` (rewrite)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `MPCResult` dataclass: `command: JointCommand`, `forces0: np.ndarray (12,)`, `solve_time: float`, `constr_viol: float`, `num_iters: int`.
  - `AligatorMPC` class:
    - `__init__(self, cfg, rm, gait)`.
    - `reset(self, x0) -> None` — cold solve (`cold_max_iters`), seed warm buffers.
    - `step(self, x_meas, t: float) -> MPCResult` — refresh references for current `t` (swing-z, support split), set `problem.x0_init = x_meas`, warm-solve (`warm_max_iters`), extract command. For walk, advance the ring with `replaceStageCircular` + `cycleProblem` (see Step 3); for stand the ring is static (no rotation).
    - `_refresh_refs(self, t)` — `setReference` swing-z motions per node from `gait.swing_phase`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mpc.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait, WalkGait
from t1_nmpc.wb.mpc import AligatorMPC


def test_reset_solves_stand():
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0, t=0.0)
    assert res.constr_viol < 1e-2
    assert res.command.tau_ff.shape == (27,)
    assert res.forces0.shape == (12,)


def test_warm_step_converges_fast_stand():
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    iters = []
    for _ in range(5):
        res = mpc.step(x0, t=0.0)
        iters.append(res.num_iters)
        assert res.constr_viol < 1e-2
    assert iters[-1] <= 5                      # warm convergence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_mpc.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/mpc.py
"""AligatorMPC: SolverProxDDP over the whole_body_rnea OCP with cyclic warm-start carry."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import aligator

from ..robot.config import MPCConfig, JointCommand
from ..robot.model import RobotModel, nominal_x
from .dynamics import WBDynamics
from .ocp import OCPBuilder
from .gait import v_z_ref
from .state import extract_command


@dataclass
class MPCResult:
    command: JointCommand
    forces0: np.ndarray
    solve_time: float
    constr_viol: float
    num_iters: int


class AligatorMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, gait):
        self.cfg, self.rm, self.gait = cfg, rm, gait
        self.dyn = WBDynamics(rm, cfg)
        self.builder = OCPBuilder(cfg, rm, self.dyn)
        self.tau_fn = self.dyn.joint_torque_fn()
        self._is_walk = hasattr(gait, "t_lf_end")  # WalkGait has phase boundaries

        modes = gait.horizon_modes(0.0)
        x0 = nominal_x(cfg, rm.model)
        self.problem, self.handles = self.builder.build_problem(modes, x0)
        self.solver = aligator.SolverProxDDP(cfg.al_tol, cfg.mu_init,
                                             cfg.cold_max_iters, aligator.QUIET)
        self.solver.setup(self.problem)
        self._warm = None      # (xs, us, vs, lams)

    def _refresh_refs(self, t: float):
        # Reach each swing-z residual THROUGH the problem (addConstraint deep-copied it; the
        # original builder handle is disconnected). funcs[cidx] is the slice; .func is the
        # wrapped FrameVelocityResidual; set its vref property (NOT deprecated setReference).
        for i, handles in enumerate(self.handles):
            for (foot_index, cidx) in handles["swing"]:
                phase = self.gait.swing_phase(t + i * self.cfg.dt, foot_index)
                vz = v_z_ref(phase, self.cfg) if phase is not None else 0.0
                self.problem.stages[i].constraints.funcs[cidx].func.vref = \
                    pin.Motion(np.array([0, 0, vz, 0, 0, 0.0]))

    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        self.problem.x0_init = x0
        self.solver.max_iters = self.cfg.cold_max_iters
        self._refresh_refs(0.0)
        xs = [x0.copy() for _ in range(self.cfg.nodes + 1)]
        us = [self.builder.u_des.copy() for _ in range(self.cfg.nodes)]
        self.solver.run(self.problem, xs, us)
        r = self.solver.results
        self._warm = (list(r.xs), list(r.us), list(r.vs), list(r.lams))

    def step(self, x_meas, t: float) -> MPCResult:
        x = np.asarray(x_meas, dtype=np.float64)
        if self._is_walk:
            # advance the ring by one knot: append the stage for the new horizon tip,
            # then rotate the solver's warm buffers (primals AND duals) to match.
            tip_t = t + self.cfg.nodes * self.cfg.dt
            tip_stage, tip_handles = self.builder.build_stage(self.gait.mode_at(tip_t))
            self.problem.replaceStageCircular(tip_stage)
            self.handles = self.handles[1:] + [tip_handles]
            self.solver.cycleProblem(self.problem, self.problem.stages[-1].createData())
        self.problem.x0_init = x
        self._refresh_refs(t)
        self.solver.max_iters = self.cfg.warm_max_iters
        xs, us, vs, lams = self._warm
        t0 = time.perf_counter()
        self.solver.run(self.problem, xs, us, vs, lams)
        dt = time.perf_counter() - t0
        r = self.solver.results
        self._warm = (list(r.xs), list(r.us), list(r.vs), list(r.lams))

        x1 = np.asarray(r.xs[1], dtype=np.float64)
        u0 = np.asarray(r.us[0], dtype=np.float64)
        tau = np.asarray(self.tau_fn(np.asarray(r.xs[0]), u0)).flatten()
        cmd = extract_command(x1, tau, self.cfg, self.rm)
        return MPCResult(command=cmd, forces0=u0[33:].copy(), solve_time=dt,
                         constr_viol=float(r.primal_infeas), num_iters=int(r.num_iters))
```

> **Implementer note (verify in this task):** the exact `StageData` argument to `cycleProblem` after `replaceStageCircular` is `self.problem.stages[-1].createData()` (the newly appended tip). The smoke test exercised the API with uniform-constraint stages; with mixed stance/swing stages confirm the warm re-solve still reaches `constr_viol < 1e-2` (the Task 10 gate is the real check). If `cycleProblem` raises on a dual-dimension mismatch, the fallback is to rebuild the problem each tick from `gait.horizon_modes(t)` and re-`setup` (slower, but correctness-equivalent) — log it and proceed; the gate still measures warm convergence within a fixed problem structure.

- [ ] **Step 4: Run test to verify it passes** (depends on Task 9's `extract_command`)

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_mpc.py -q -p no:cacheprovider`
Expected: PASS (2 tests). (If running before Task 9, implement `extract_command` first — Tasks 8 and 9 may be done together.)

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/mpc.py tests/test_mpc.py
git commit -m "feat(mpc): AligatorMPC with ProxDDP cold/warm solve + cyclic ring advance"
```

---

### Task 9: State map for the reduced model + command extraction

**Files:**
- Modify: `t1_nmpc/wb/state.py` (rewrite)
- Test: `tests/test_state.py` (rewrite)

**Interfaces:**
- Consumes: `MPCConfig`, `RobotModel`, `JointCommand`.
- Produces:
  - `MUJOCO_TO_PIN_JOINTS: np.ndarray (27,)` — index map from MuJoCo's 29 actuated joints to the reduced model's 27 (drops the 2 head joints, which are MuJoCo qpos indices 7,8 / qvel 6,7).
  - `mujoco_to_freeflyer(qpos, qvel, model) -> np.ndarray (67,)` — applies `v[0:3]=R(q)ᵀ·qvel[0:3]` and the joint index map.
  - `freeflyer_to_mujoco_joints(x, model) -> (q_des27, qd_des27)` — selects the 27 controlled joints for the command.
  - `extract_command(x1, tau0, cfg, rm) -> JointCommand` — `q_des/qd_des` from planned node 1, `tau_ff` from node 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.state import (mujoco_to_freeflyer, extract_command, MUJOCO_TO_PIN_JOINTS)


def test_joint_index_map_drops_head():
    assert MUJOCO_TO_PIN_JOINTS.shape == (27,)
    # MuJoCo actuated joints are [head2, Larm7, Rarm7, waist, Lleg6, Rleg6] (29);
    # reduced pin joints are the same minus the 2 head -> indices 2..28
    np.testing.assert_array_equal(MUJOCO_TO_PIN_JOINTS, np.arange(2, 29))


def test_base_linear_velocity_rotation():
    cfg = make_config(); rm = load_model(cfg)
    # 90deg yaw, world x-velocity -> body y-velocity (negative)
    qw = np.cos(np.pi / 4); qz = np.sin(np.pi / 4)
    qpos = np.zeros(36); qpos[2] = 0.6734; qpos[3:7] = [qw, 0, 0, qz]
    qvel = np.zeros(35); qvel[0] = 1.0      # world +x
    x = mujoco_to_freeflyer(qpos, qvel, rm.model)
    # body-local linear vel: R^T @ [1,0,0]
    R = pin.Quaternion(qw, 0, 0, qz).toRotationMatrix()
    np.testing.assert_allclose(x[34:37], R.T @ np.array([1, 0, 0]), atol=1e-9)


def test_extract_command_shapes():
    cfg = make_config(); rm = load_model(cfg)
    x1 = nominal_x(cfg, rm.model); tau0 = np.zeros(27)
    cmd = extract_command(x1, tau0, cfg, rm)
    assert cmd.q_des.shape == (27,) and cmd.qd_des.shape == (27,)
    assert cmd.tau_ff.shape == (27,) and cmd.kp.shape == (27,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_state.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# t1_nmpc/wb/state.py
"""MuJoCo <-> pinocchio FreeFlyer state map for the reduced (head-locked, 27-joint) model.

MuJoCo qpos=[pos3, quat_wxyz4, joints29], qvel=[lin_WORLD3, ang_LOCAL3, jvel29].
pinocchio reduced q=[pos3, quat_xyzw4, joints27], v=[lin_LOCAL3, ang_LOCAL3, jvel27].
Base linear velocity differs by frame -> rotate by R(q)^T (MuJoCo->pin).
The 2 head joints (MuJoCo actuated indices 0,1) are dropped."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..robot.config import MPCConfig, JointCommand

# MuJoCo actuated-joint order: [head2, Larm7, Rarm7, waist, Lleg6, Rleg6] (29).
# Reduced pinocchio drops the 2 head joints -> select MuJoCo actuated indices 2..28.
MUJOCO_TO_PIN_JOINTS = np.arange(2, 29)        # (27,)


def mujoco_to_freeflyer(qpos, qvel, model) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64); qvel = np.asarray(qvel, dtype=np.float64)
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    q = np.empty(model.nq)
    q[0:3] = qpos[0:3]; q[3:7] = [qx, qy, qz, qw]
    q[7:] = qpos[7:][MUJOCO_TO_PIN_JOINTS]      # 27 controlled joints
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    v = np.empty(model.nv)
    v[0:3] = R.T @ qvel[0:3]; v[3:6] = qvel[3:6]
    v[6:] = qvel[6:][MUJOCO_TO_PIN_JOINTS]      # 27 controlled joint velocities
    return np.concatenate([q, v])


def freeflyer_to_mujoco_joints(x, model):
    x = np.asarray(x, dtype=np.float64)
    q = x[:model.nq]; v = x[model.nq:model.nq + model.nv]
    return q[7:].copy(), v[6:].copy()           # 27 each


def extract_command(x1, tau0, cfg: MPCConfig, rm) -> JointCommand:
    """q_des/qd_des from planned node 1; tau_ff from node 0 (post-hoc RNEA joint torque)."""
    q_des, qd_des = freeflyer_to_mujoco_joints(x1, rm.model)
    return JointCommand(
        q_des=q_des, qd_des=qd_des,
        tau_ff=np.asarray(tau0, dtype=np.float64),
        kp=np.asarray(cfg.kp, dtype=np.float64),
        kd=np.asarray(cfg.kd, dtype=np.float64),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_state.py tests/test_mpc.py -q -p no:cacheprovider`
Expected: PASS (state + mpc tests green together).

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/state.py tests/test_state.py
git commit -m "feat(state): reduced-model MuJoCo<->pinocchio map (head drop) + command extraction"
```

---

### Task 10: Warm-start gate (the spec §9 solver gate)

**Files:**
- Create: `tests/test_warm_start_gate.py`

**Interfaces:**
- Consumes: `AligatorMPC` + `WalkGait`.
- Produces: a regression test asserting the spec's gate: ProxDDP reaches `CV ≤ 1e-2` in `≤ 5` outer iters/tick across `≥ 15` receding ticks **including contact switches**, on the walk gait (idealized closed loop: `x_meas` = the solver's own planned node-1, per spec §6.2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_warm_start_gate.py
import numpy as np
import pytest
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import WalkGait
from t1_nmpc.wb.mpc import AligatorMPC


def test_warm_start_gate_walk():
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, WalkGait(cfg))
    x = nominal_x(cfg, rm.model)
    mpc.reset(x)
    cvs, iters, modes_seen = [], [], set()
    t = 0.0
    for _ in range(20):                          # >= 15 receding ticks
        res = mpc.step(x, t)
        cvs.append(res.constr_viol); iters.append(res.num_iters)
        modes_seen.add(WalkGait(cfg).mode_at(t + cfg.nodes * cfg.dt))
        # idealized closed loop: advance to the solver's own planned node 1
        x = np.asarray(mpc._warm[0][1], dtype=np.float64)
        t += cfg.dt
    assert len(modes_seen) >= 2, "gate must cross >=1 contact switch"
    assert max(cvs) <= 1e-2, f"max CV {max(cvs):.2e} exceeds 1e-2"
    assert max(iters) <= 5, f"max outer iters {max(iters)} exceeds 5"
```

- [ ] **Step 2: Run test to verify it fails (or errors)**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_warm_start_gate.py -q -p no:cacheprovider`
Expected: Initially may FAIL on CV/iters or on the `cycleProblem` data path. This is the **research-bearing** test (spec §6.1). Diagnose in order:
  1. If `cycleProblem` errors on dual dims → apply the Task 8 fallback (rebuild+`setup` each tick) and re-run; confirm warm `vs/lams` are still passed to `run`.
  2. If CV stalls only on swing→stance switch nodes → raise `mu_init` toward `1e-1` and/or `max_al_iters`; confirm `setReference` swing-z motions are phase-aligned (a swing dual must not land on a stance node — Global Constraints).
  3. If CV stalls broadly → confirm the RNEA `Jx`/`Ju` match finite-diff (Task 3 already gates this) and that `arm_to_nominal` weight is not destabilizing.

- [ ] **Step 3: Make the gate pass**

Tune only the **solver knobs** exposed in `MPCConfig` (`mu_init`, `warm_max_iters`, `al_tol`) and the cycle-data path in `mpc.py`. Do **not** weaken any hard constraint to a soft cost (spec §6.1 forbids silently downgrading swing-z). If, after the three diagnostics above, the gate still fails on swing-z specifically, add the spec's **position-level companion** (`p_foot,z = z_ref`) as an additional hard equality on swing nodes (extend `constraint.py` with a `swing_z_position_residual` using `FrameTranslationResidual` sliced to z) — this is spec §6.1 candidate (b), an allowed escalation, not a downgrade.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_warm_start_gate.py -q -p no:cacheprovider`
Expected: PASS. Record the achieved `max CV` and `max iters` in the commit message.

- [ ] **Step 5: Commit**

```bash
git add tests/test_warm_start_gate.py t1_nmpc/wb/
git commit -m "test(gate): warm-start gate passes on walk gait (CV<=1e-2, iters<=N across switches)"
```

---

## PART B — Re-home the M0 stand

### Task 11: Closed-loop stand runner repoint

**Files:**
- Modify: `sim/stand.py` (repoint to `AligatorMPC` + `StandGait`)
- Modify: `sim/mujoco_runtime.py` (repoint state read/write to the new `state.py`; verify it already uses `mujoco_to_freeflyer` and `JointCommand`)
- Read first: `sim/mujoco_runtime.py`, `sim/stand.py`, `sim/_sim_util.py`, `runtime/transport.py`, `runtime/mujoco_transport.py` (if present) to match existing structure.

**Interfaces:**
- Consumes: `AligatorMPC`, `StandGait`, `mujoco_to_freeflyer`, `JointCommand`.
- Produces: `sim/stand.py` `main(--duration, --view, --gif)` driving the closed loop at the PD rate, printing `Σfz/(m·g)`, `max_tilt`, `solve_p90`.

- [ ] **Step 1: Read the existing sim layer**

Run: open `sim/stand.py`, `sim/mujoco_runtime.py`, `sim/_sim_util.py`. Identify where the old `WholeBodyMPC` was constructed and where `JointCommand` is consumed. The MuJoCo command must now drive only the 27 controlled joints; the 2 head joints are held at nominal (set their MuJoCo ctrl/qpos targets to the nominal head pose, kp/kd from a fixed small gain). Confirm `forces0` is now `(12,)` and `Σfz` sums the two feet's `f_z` (indices 2 and 8 of `forces0`).

- [ ] **Step 2: Write/adjust the failing closed-loop test**

```python
# tests/test_stand_closed_loop.py
import numpy as np
import pytest
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait
from t1_nmpc.wb.mpc import AligatorMPC


@pytest.mark.slow
def test_stand_holds_open_loop_solver():
    """Solver-side stand check (no MuJoCo): gravity-comp forces sum to ~m*g, base upright."""
    cfg = make_config(); rm = load_model(cfg)
    mpc = AligatorMPC(cfg, rm, StandGait(cfg))
    x = nominal_x(cfg, rm.model)
    mpc.reset(x)
    res = mpc.step(x, t=0.0)
    fz_total = res.forces0[2] + res.forces0[8]
    ratio = fz_total / (rm.mass * 9.81)
    assert 0.9 <= ratio <= 1.1, f"fz ratio {ratio:.3f}"
    assert res.constr_viol < 1e-2
```

- [ ] **Step 3: Run it to verify it fails, then implement the runner repoint**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_stand_closed_loop.py -q -p no:cacheprovider`
Then implement `sim/stand.py` to construct `AligatorMPC(cfg, rm, StandGait(cfg))`, run `reset`, then loop `step(x_meas, t=0.0)` (stand `t` is fixed; no ring rotation), apply `JointCommand` to MuJoCo via the existing PD path, hold the 2 head joints at nominal.

- [ ] **Step 4: Run the live stand smoke + tests**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python sim/stand.py --duration 4.0`
Expected: no fall; printed `fz_ratio` ≈ 1.0, `max_tilt` small (< ~3°). Then:
Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_stand_closed_loop.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/stand.py sim/mujoco_runtime.py tests/test_stand_closed_loop.py
git commit -m "feat(sim): re-home closed-loop stand onto AligatorMPC (StandGait)"
```

---

### Task 12: Remove Fatrop remnants + full suite green

**Files:**
- Remove: `tools/codegen_solver.py`
- Verify removed/replaced: any `StandOCP`, `WholeBodyMPC`, `opti.to_function`, Fatrop options references.
- Modify: `tests/conftest.py` (drop any aligator-skip marker now that aligator is the backend), any remaining test importing removed symbols.

**Interfaces:**
- Produces: a clean tree with a single backend; full `pytest tests/` green.

- [ ] **Step 1: Find Fatrop/old-symbol references**

Run: `grep -rn "fatrop\|StandOCP\|WholeBodyMPC\|to_function\|codegen_solver" t1_nmpc/ sim/ tests/ tools/`
Expected: only `tools/codegen_solver.py` and any stale test imports remain.

- [ ] **Step 2: Remove and fix**

```bash
git rm tools/codegen_solver.py
```
Fix any test still importing `StandOCP`/`WholeBodyMPC` (rewrite to `AligatorMPC` or delete if redundant with Tasks 8/11). Update `tests/conftest.py` to remove any `aligator`-availability skip.

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/ -q -p no:cacheprovider`
Expected: all PASS (model, config, dynamics, constraint, cost, gait, ocp, mpc, state, warm-start gate, stand closed-loop).

- [ ] **Step 4: Verify no import of removed modules anywhere**

Run: `grep -rn "fatrop\|StandOCP\|WholeBodyMPC" t1_nmpc/ sim/ tests/`
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove Fatrop backend (codegen_solver, StandOCP/WholeBodyMPC) — aligator is sole backend"
```

---

## PART C — Docs & ledger

### Task 13: Reconcile CLAUDE.md, divergence ledger, paper map

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/2026-06-25-t1controller-divergences.md`
- Create: `docs/2026-06-27-paper-mapping.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Edit these claims to match the implemented backend:
  - "What this is" — change *CasADi `Opti` + Fatrop* → *aligator `SolverProxDDP` (AL-DDP)*; *8-corner 3D contact forces (`nf=24`)* → *two 6D foot wrenches (`nf=12`) + contact-wrench cone*; *FreeFlyer 29 joints, nq=36/nv=35* → *reduced head-locked 27 joints, nq=34/nv=33*; control width *88/59* → *uniform 45 (torque not a decision var; recovered post-hoc, soft-limited)*.
  - **Remove the invariant** "No aligator/ProxDDP. The aligator backend is removed. Do not re-introduce it." Replace with: "Backend is aligator `SolverProxDDP`. Every Python subclass of an aligator base class must implement `__deepcopy__` sharing precompiled casadi functions (see plan Global Constraints). Manifold-Jacobian via the dx-input trick — never `substitute(dx,0)`."
  - Replace "Fatrop gap-closing-equality-first" invariant with: "aligator constraint order within a stage is free (no gap-closing-first rule); the gap-closing dynamics is the StageModel's `IntegratorEuler(DoubleIntegratorODE)`."
  - Update "Fatrop staircase variable structure" → remove (uniform `nu=45`, no adaptive width).
  - Update commands: `python -m pytest tests/ -q` expected count to the new total; `sim/stand.py` unchanged.
  - Update Status: M0 stand re-homed on aligator (PASS), warm-start gate PASS; M1 walk-in-plant deferred to follow-up plan.

- [ ] **Step 2: Update the divergence ledger**

Append to `docs/2026-06-25-t1controller-divergences.md` the spec §10 deliberate divergences now realized in code: aligator vs Fatrop; soft torque limits; 6D wrench + CWC (CoP+yaw) vs point 3D; per-foot 6D contact velocity; Euler vs RK4; MPC cadence `1/dt≈28.6 Hz`; contact-mode structural (cycled stages); Raibert footstep (deferred to walk plan); head locked + arms-to-nominal. Each one line, citing the implementing file.

- [ ] **Step 3: Create the paper-mapping doc**

Write `docs/2026-06-27-paper-mapping.md` = the spec §10 "Paper ↔ code map" table, but pointing at the **implemented** units (`constraint.RneaBaseResidual`, `constraint.WrenchConeResidual`, `constraint.contact_velocity_residual`, `constraint.swing_z_residual`+`SwingWrenchResidual`, `cost.state_tracking`/`input_reg`, `dynamics._DoubleIntegratorODE`, `mpc.AligatorMPC`).

- [ ] **Step 4: Sanity-check docs reference real symbols**

Run: `grep -rn "RneaBaseResidual\|WrenchConeResidual\|AligatorMPC\|DoubleIntegrator" docs/ CLAUDE.md` and confirm each named symbol exists in the tree (`grep -rn "<symbol>" t1_nmpc/`).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/2026-06-25-t1controller-divergences.md docs/2026-06-27-paper-mapping.md
git commit -m "docs: reconcile CLAUDE.md + ledger + paper-map with aligator backend"
```

---

### Task 14: Update auto-memory

**Files:**
- Modify: `/home/yoonwoo/.claude/projects/-home-yoonwoo-humanoid-mpc-ws-src-t1-nmpc/memory/wb-rnea-port.md`
- Modify: `/home/yoonwoo/.claude/projects/-home-yoonwoo-humanoid-mpc-ws-src-t1-nmpc/memory/MEMORY.md`

- [ ] **Step 1: Update the `wb-rnea-port` memory**

Record that the backend is now aligator `SolverProxDDP` (not Fatrop), reduced 27-joint model, 6D foot wrenches, and the two verified gotchas (`__deepcopy__` sharing compiled funcs; manifold dx-trick Jacobian — never `substitute(dx,0)`). Keep it to the non-obvious facts; do not restate the code.

- [ ] **Step 2: Update `MEMORY.md` pointer** to reflect the aligator backend and the gotchas hook.

- [ ] **Step 3: Commit (memory files are outside the repo; no git)**

No commit — memory lives outside the repo. Verify both files saved.

---

## Self-Review (planner checklist — performed)

1. **Spec coverage:** §2 solver → Tasks 3,8,10. §3.1 reduced model → Task 1. §3.2 state/control → Tasks 1,2,3. §3.3 RNEA path constraint → Tasks 3,4. §3.4 stance (wrench cone + 6D velocity) → Task 4. §3.5 swing (hard velocity-level z) → Tasks 4,6,8,10. §3.7 costs → Task 5. §3.8 soft torque (post-hoc) → Tasks 3,9 (+`torque_limit_weight` in config; full soft-penalty cost optional, noted). §4 discretization/gait → Tasks 2,6. §5 warm-start cycling → Task 8. §9 gate + stand → Tasks 10,11. §10 divergences/map → Task 13. **Deferred by design (flagged):** §3.6 Raibert footstep, §6.1 foot-lift-in-plant, §6.2 lateral balance, §6.3 real-time C++ — all in the follow-up walk plan.
2. **Placeholder scan:** every code step contains real, runnable code verified against aligator 0.19.0; the only experiment-gated step is Task 10 Step 3 (the spec's open research), which lists concrete diagnostics + an allowed escalation rather than "TBD".
3. **Type consistency:** `nu=45`/`ndx=66` consistent across Tasks 2–8; `RneaBaseResidual(ndx,nu,funcs)`, `funcs=dyn.rnea_funcs(base_only=True)`, `extract_command(x1,tau0,cfg,rm)`, `gravity_comp_u_des(rm,n_support)` consistent across producers/consumers; `forces0` slice `u[33:]` (12,) consistent with `f_z` indices 2 and 8.

---

## Follow-up plan (out of scope here)

The **closed-loop forward walk** is a separate plan because spec §6 classifies its core (foot-lift on aligator's AL across the swing phase, and lateral balance in the plant) as **open research**, not determinable TDD. That plan builds on this one and adds: `cost.footstep_placement` (Raibert, soft) + `cost.base_velocity_target`; the CoM-sway lateral reference (§6.2); the closed-loop MuJoCo `sim/walk.py` runner; the M1 success gate (advance ≥0.5 m over ≥5 s, feet alternate with confirmed lift, lateral drift < ~0.1 m). The §6.1 swing-z escalations (position companion / C++ residual) move there if the warm-start gate (Task 10) revealed they are needed. Real-time speed (§6.3, C++ RNEA residual) is a third, independent effort.
