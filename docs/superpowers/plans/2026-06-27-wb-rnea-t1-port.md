# wb-mpc-locoman → T1 port — Implementation Plan (iteration 1: closed-loop MuJoCo stand)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the aligator ProxDDP controller with wb-mpc-locoman's `whole_body_rnea` formulation (CasADi `Opti` + Fatrop, symbolic RNEA) on the Booster T1 humanoid, and hold a closed-loop MuJoCo stand.

**Architecture:** Direct-transcription NLP over `x=[q(36),v(35)]` with adaptive input `u=[a(35), forces(24), τ_j(29-on-first-nodes)]`; floating base = `JointModelFreeFlyer`; flat feet = 8 unilateral 3D corner contacts (4/foot); RNEA enters as a per-node path constraint (`τ_rnea[:6]=0`, `τ_rnea[6:]=τ_j` on the first `tau_nodes`). The OCP is solved each MPC tick by Fatrop (warm-started). A MuJoCo closed loop reads state → solves → commands joint-torque feedforward + joint PD.

**Tech Stack:** Python 3.10 (conda env `t1mpc`), casadi 3.7.2 (bundled Fatrop), pinocchio 4.0 + `pinocchio.casadi`, mujoco 3.10, numpy. **No new dependencies.**

**Reference spec:** `docs/superpowers/specs/2026-06-27-wb-rnea-t1-port-design.md` (read it first).
**Proven scratch prototype (every code block below is adapted from working, convergent code):**
`/tmp/claude-1000/-home-yoonwoo-humanoid-mpc-ws-src-t1-nmpc/4dd30544-6cab-4be8-b582-9ef5880db991/scratchpad/spike/` (`model.py`, `dynamics.py`, `ocp.py`, `state_freeflyer.py`, `run_stand.py`).

## Global Constraints

- **Run preamble (load-bearing):** `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python <args>`. Empty `PYTHONPATH` keeps ROS numpy<2 pinocchio off the path (else segfault).
- **Structure-only reuse.** Keep the `robot/ wb/ runtime/ sim/` layout; rewrite all module contents. No faithfulness-to-t1_controller constraint for formulation (retired for this work); cite t1_controller only for T1 *data*.
- **Dimensions (exact, verified):** `nq=36, nv=35, nx=71, ndx=70, n_joints=29, n_corners=8, nf=24, na=35`. Adaptive input width = `88` (`a35+f24+τ29`) for `i<tau_nodes`, `59` (`a35+f24`) after. `f_idx=35, tau_idx=59`.
- **Nominal stand (FK-verified soles-flat):** base height **0.6734 m**, identity orientation; `nominal_joint_pos` (29, pinocchio order `[head2, Larm7, Rarm7, waist1, Lleg6, Rleg6]`) = head `[0,0]`, L-arm `[0.5,-1.0,0,-1.4,0,0,0]`, R-arm `[0.5,1.0,0,1.4,0,0,0]`, waist `[0]`, L-leg `[-0.05,0,0,0.10,-0.05,0]`, R-leg `[-0.05,0,0,0.10,-0.05,0]`.
- **Corner frames (8):** added on `Left_Ankle_Roll`/`Right_Ankle_Roll`; offset in the ankle frame `x∈{-0.1015, 0.1115}, y∈{-0.05, 0.05}, z=-0.030`. Map into the parent-joint frame via `parent_placement.act(offset)` before `addFrame`.
- **Mass:** `34.5135 kg` via `pin.computeTotalMass` (`data.mass[0]` is `-1.0` until computed). `m·g = 338.58 N`. Even per-corner `m·g/8 ≈ 42.3 N` is a **reference/warm-start only**, not the equilibrium (the solver redistributes front/rear).
- **Friction μ = 0.4.**
- **Fatrop invariants:** (1) variables created **staircase** `x0,u0,x1,u1,…,xN`; (2) each stage emits the **gap-closing state-transition equality FIRST**, then RNEA/torque/friction/velocity; (3) `structure_detection='auto'`; (4) **adaptive width works** (preferred) — uniform width (τ on all nodes, zero-weighted after `tau_nodes`) is the fallback.
- **CasADi MX/SX:** at the `Opti` graph level use `ca.MX.zeros` / `ca.DM` for padding/targets — **never `ca.SX.zeros`** (mixing MX+SX in `vertcat` raises). The cpin `Function`s are built internally with SX and are fine.
- **Base/trunk frame is `'Trunk'`** — `'base_link'` does not exist on T1.
- **Stand balance comes from the MPC** (corner-force redistribution driving `τ_rnea[:6]=0`), not passive ankle PD. Do **not** raise shipped ankle kp to force a passive hold.
- **Validation gate (final):** `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/ -q -p no:cacheprovider` is green, and the closed-loop stand runner holds upright with `Σ corner f_z/(m·g) ∈ [0.9,1.1]`.

---

### Task 1: Vendor URDF + slim config + clean slate

**Files:**
- Create: `t1_nmpc/robot/assets/t1_description/urdf/t1.urdf`, `t1_nmpc/robot/assets/t1_description/meshes/*.STL` (30 files)
- Rewrite: `t1_nmpc/robot/config.py`
- Delete: `t1_nmpc/wb/ode.py`, `t1_nmpc/wb/swing.py`, `t1_nmpc/wb/execution.py`, `t1_nmpc/wb/dynamics.py`, `t1_nmpc/wb/gait.py`, `t1_nmpc/wb/ocp.py`, `t1_nmpc/wb/mpc.py`, `t1_nmpc/wb/state.py`, `t1_nmpc/robot/model.py`, `t1_nmpc/robot/execution.py`, `sim/walk.py`, `sim/state.py`, and all `tests/test_*.py` (every existing test targets the deleted aligator code)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `t1_nmpc.robot.config.MPCConfig` (frozen dataclass), `make_config(**overrides) -> MPCConfig`, `JointCommand`. `T1_URDF_PATH`, `T1_PACKAGE_DIRS` constants.

- [ ] **Step 1: Vendor the URDF + meshes**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
SRC=/home/yoonwoo/humanoid_mpc_ws/src/t1_controller/robot_models/booster_t1/t1_description
mkdir -p t1_nmpc/robot/assets/t1_description/urdf t1_nmpc/robot/assets/t1_description/meshes
cp "$SRC/urdf/t1.urdf" t1_nmpc/robot/assets/t1_description/urdf/t1.urdf
cp "$SRC"/meshes/*.STL t1_nmpc/robot/assets/t1_description/meshes/
ls t1_nmpc/robot/assets/t1_description/meshes/ | wc -l   # expect 30
```

- [ ] **Step 2: Rewrite `t1_nmpc/robot/config.py`** (slim T1 config; keeps the fields `MujocoRuntime` already imports — `physics_hz, control_hz, mpc_hz, nominal_joint_pos, kp, kd, nominal_base_height` — plus new OCP fields)

```python
"""MPCConfig: Booster T1 numbers for the whole_body_rnea (CasADi+Fatrop) controller.

Geometry/pose/limits trace to t1_controller (data only, not formulation). Weights are
re-dimensioned to T1's 29 joints (NOT traced to t1_controller — logged as a divergence)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
T1_URDF_PATH = os.path.join(_ASSETS, "t1_description", "urdf", "t1.urdf")
T1_PACKAGE_DIRS = (_ASSETS,)  # so 'package://t1_description/...' resolves

# §A.5 / pinocchio joint order: [head2, Larm7, Rarm7, waist1, Lleg6, Rleg6]
JOINT_NAMES: Tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
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


def _nominal_joint_pos() -> np.ndarray:
    return np.array(
        [0, 0]
        + [0.5, -1.0, 0, -1.4, 0, 0, 0]
        + [0.5, 1.0, 0, 1.4, 0, 0, 0]
        + [0]
        + [-0.05, 0, 0, 0.10, -0.05, 0]
        + [-0.05, 0, 0, 0.10, -0.05, 0],
        dtype=np.float64,
    )


def _kp() -> np.ndarray:
    return np.array(
        [20, 20] + [20] * 14 + [200]
        + [200, 200, 200, 200, 50, 50]
        + [200, 200, 200, 200, 50, 50], dtype=np.float64)


def _kd() -> np.ndarray:
    return np.array(
        [0.2, 0.2] + [0.5] * 14 + [5.0]
        + [5, 5, 5, 5, 3, 3] + [5, 5, 5, 5, 3, 3], dtype=np.float64)


def _Q_diag() -> np.ndarray:
    # state delta weights, ndx=70 = [base_pos(6), joint_pos(29), base_vel(6), joint_vel(29)]
    base_pos = np.array([0, 0, 1000, 10000, 10000, 0], dtype=np.float64)   # x,y,yaw free
    joint_pos = np.concatenate([
        [50, 50], [100] * 14, [200],
        [300] * 6, [300] * 6,
    ])
    base_vel = np.array([2000, 2000, 1000, 1000, 1000, 2000], dtype=np.float64)
    joint_vel = np.concatenate([[10, 10], [10] * 14, [10], [2] * 6, [2] * 6])
    return np.concatenate([base_pos, joint_pos, base_vel, joint_vel])


def _R_diag() -> np.ndarray:
    # input weights for the FULL width (na+nf+nj = 88): [a(35), forces(24), tau_j(29)]
    return np.concatenate([
        [1e-3] * 35,
        [5e-4] * 24,
        [1e-4] * 2, [1e-2] * 14, [1e-3], [1e-4] * 12,
    ])


@dataclass(frozen=True)
class MPCConfig:
    # dimensions / horizon
    nodes: int = 14
    tau_nodes: int = 3
    dt_min: float = 0.02
    dt_max: float = 0.06
    n_joints: int = 29
    nq: int = 36
    nv: int = 35
    nx: int = 71
    ndx: int = 70
    n_corners: int = 8
    nf: int = 24
    na: int = 35

    # nominal stand
    nominal_base_height: float = 0.6734
    nominal_joint_pos: np.ndarray = field(default_factory=_nominal_joint_pos)
    robot_mass: float = 34.5135   # reference; the live value comes from the model

    # corner geometry (ankle-frame offsets)
    corner_x: Tuple[float, float] = (-0.1015, 0.1115)
    corner_y: Tuple[float, float] = (-0.05, 0.05)
    corner_z: float = -0.030

    # friction
    friction_mu: float = 0.4

    # weights
    Q_diag: np.ndarray = field(default_factory=_Q_diag)
    R_diag: np.ndarray = field(default_factory=_R_diag)

    # per-joint PD
    kp: np.ndarray = field(default_factory=_kp)
    kd: np.ndarray = field(default_factory=_kd)

    # Fatrop options (single max_iter cap; warm-started ticks converge well under it)
    fatrop_max_iter: int = 50
    fatrop_tol: float = 1e-3
    fatrop_mu_init: float = 1e-4

    # execution rates
    mpc_hz: float = 50.0
    control_hz: float = 500.0
    physics_hz: float = 2000.0


def make_config(**overrides) -> MPCConfig:
    cfg = MPCConfig(**overrides)
    assert cfg.nx == cfg.nq + cfg.nv == 71
    assert cfg.ndx == 2 * cfg.nv == 70
    assert cfg.nf == 3 * cfg.n_corners == 24
    assert cfg.na == cfg.nv
    assert cfg.nominal_joint_pos.shape == (29,)
    assert cfg.Q_diag.shape == (cfg.ndx,)
    assert cfg.R_diag.shape == (cfg.na + cfg.nf + cfg.n_joints,)
    assert cfg.kp.shape == (29,) and cfg.kd.shape == (29,)
    return cfg


@dataclass
class JointCommand:
    """29-joint command to the control layer: tau = tau_ff + kp*(q_des-q) - kd*(qd_des-qd)."""
    q_des: np.ndarray    # (29,)
    qd_des: np.ndarray   # (29,)
    tau_ff: np.ndarray   # (29,)
    kp: np.ndarray       # (29,)
    kd: np.ndarray       # (29,)
```

- [ ] **Step 3: Delete dead aligator modules + obsolete tests**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git rm t1_nmpc/wb/ode.py t1_nmpc/wb/swing.py t1_nmpc/wb/execution.py \
       t1_nmpc/wb/dynamics.py t1_nmpc/wb/gait.py t1_nmpc/wb/ocp.py \
       t1_nmpc/wb/mpc.py t1_nmpc/wb/state.py \
       t1_nmpc/robot/model.py t1_nmpc/robot/execution.py \
       sim/walk.py sim/state.py
git rm tests/test_*.py
```
Then blank `t1_nmpc/wb/__init__.py` to an empty file (`: > t1_nmpc/wb/__init__.py`) and confirm `runtime/mujoco_transport.py` / `runtime/sdk_transport.py` do not import any deleted symbol (if they do, comment those imports with `# rewritten in wb-rnea port`).

- [ ] **Step 4: Write the failing config test** — `tests/test_config.py`

```python
import numpy as np
from t1_nmpc.robot.config import make_config, T1_URDF_PATH, T1_PACKAGE_DIRS, JOINT_NAMES
import os

def test_config_dims_and_pose():
    cfg = make_config()
    assert cfg.nx == 71 and cfg.ndx == 70 and cfg.nf == 24 and cfg.na == 35
    assert cfg.nodes == 14 and cfg.tau_nodes == 3
    assert abs(cfg.nominal_base_height - 0.6734) < 1e-12
    assert cfg.nominal_joint_pos.shape == (29,)
    # shallow-crouch legs (knee 0.10)
    assert abs(cfg.nominal_joint_pos[20] - 0.10) < 1e-12  # Left_Knee_Pitch idx in 29-order
    assert cfg.Q_diag.shape == (70,)
    assert cfg.R_diag.shape == (35 + 24 + 29,)

def test_vendored_urdf_present():
    assert os.path.isfile(T1_URDF_PATH)
    mesh_dir = os.path.join(T1_PACKAGE_DIRS[0], "t1_description", "meshes")
    assert len([f for f in os.listdir(mesh_dir) if f.endswith(".STL")]) == 30
    assert len(JOINT_NAMES) == 29
```

- [ ] **Step 5: Run tests, verify pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_config.py -q`
Expected: 2 passed. Also confirm `python -c "import t1_nmpc.robot.config, sim.mujoco_runtime"` imports clean.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(wb-rnea): vendor T1 URDF + slim MPCConfig; remove aligator code"
```

---

### Task 2: `robot/model.py` — FreeFlyer T1 model + 8 corner frames + mass

**Files:**
- Create: `t1_nmpc/robot/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `MPCConfig` (`T1_URDF_PATH`, `corner_x/y/z`, `ANKLE_ROLL_FRAMES`, `nominal_joint_pos`, `nominal_base_height`).
- Produces:
  - `RobotModel` dataclass: `.model (pin.Model)`, `.data`, `.corner_frame_ids (tuple[int]*8)`, `.mass (float)`, `.trunk_frame_id (int)`, `.tau_max (np.ndarray (29,))`.
  - `load_model(cfg) -> RobotModel`.
  - `nominal_q(cfg, model) -> np.ndarray (36,)`, `nominal_x(cfg, model) -> np.ndarray (71,)`.

- [ ] **Step 1: Write the failing test** — `tests/test_model.py`

```python
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config, JOINT_NAMES
from t1_nmpc.robot.model import load_model, nominal_q

def test_model_build_corners_mass():
    cfg = make_config()
    rm = load_model(cfg)
    assert rm.model.nq == 36 and rm.model.nv == 35
    assert tuple(rm.model.names[2:]) == JOINT_NAMES          # joint order
    assert len(rm.corner_frame_ids) == 8
    assert abs(rm.mass - 34.5135) < 1e-3
    assert rm.tau_max.shape == (29,) and np.all(rm.tau_max > 0)
    # 8 corners coplanar at the ground at nominal stand
    q = nominal_q(cfg, rm.model)
    pin.forwardKinematics(rm.model, rm.data, q)
    pin.updateFramePlacements(rm.model, rm.data)
    zs = np.array([rm.data.oMf[c].translation[2] for c in rm.corner_frame_ids])
    assert zs.max() - zs.min() < 1e-6                         # coplanar
    assert abs(zs.mean()) < 2e-3                              # ~on the ground
    # corners share 2 parent joints, 4 each
    parents = [rm.model.frames[c].parentJoint for c in rm.corner_frame_ids]
    assert len(set(parents)) == 2 and all(parents.count(p) == 4 for p in set(parents))
```

- [ ] **Step 2: Run test, verify it fails** (`ModuleNotFoundError: t1_nmpc.robot.model`).

- [ ] **Step 3: Implement `t1_nmpc/robot/model.py`** (adapted from proven `spike/model.py`)

```python
"""T1 FreeFlyer pinocchio model + 8 foot-corner contact frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .config import MPCConfig, T1_URDF_PATH, ANKLE_ROLL_FRAMES


@dataclass
class RobotModel:
    model: pin.Model
    data: pin.Data
    corner_frame_ids: tuple
    mass: float
    trunk_frame_id: int
    tau_max: np.ndarray   # (29,)


def load_model(cfg: MPCConfig) -> RobotModel:
    model = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"expected nq=36 nv=35, got {model.nq}/{model.nv}")

    corner_ids = []
    for ankle in ANKLE_ROLL_FRAMES:
        fid = model.getFrameId(ankle)
        parent_joint = model.frames[fid].parentJoint
        parent_placement = model.frames[fid].placement   # ankle frame wrt its parent joint
        for cx in cfg.corner_x:
            for cy in cfg.corner_y:
                t = parent_placement.act(np.array([cx, cy, cfg.corner_z], dtype=np.float64))
                placement = pin.SE3(np.eye(3), t)
                name = f"{ankle}_corner_{cx:+.4f}_{cy:+.4f}"
                frame = pin.Frame(name, parent_joint, fid, placement, pin.FrameType.OP_FRAME)
                corner_ids.append(model.addFrame(frame))

    data = model.createData()
    mass = float(pin.computeTotalMass(model, data))
    trunk_fid = model.getFrameId("Trunk")
    tau_max = np.asarray(model.effortLimit[6:], dtype=np.float64).copy()
    return RobotModel(model, data, tuple(corner_ids), mass, trunk_fid, tau_max)


def nominal_q(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    q[0:3] = [0.0, 0.0, cfg.nominal_base_height]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]            # quat xyzw identity
    q[7:] = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)
    return q


def nominal_x(cfg: MPCConfig, model: pin.Model) -> np.ndarray:
    return np.concatenate([nominal_q(cfg, model), np.zeros(model.nv)])
```

- [ ] **Step 4: Run test, verify pass.** Run: `PYTHONPATH= conda run -n t1mpc python -m pytest tests/test_model.py -q` → 1 passed.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wb-rnea): T1 FreeFlyer model + 8 corner contact frames"`

---

### Task 3: `wb/dynamics.py` — cpin state ops + 8-corner RNEA

**Files:**
- Create: `t1_nmpc/wb/dynamics.py`
- Test: `tests/test_dynamics.py`

**Interfaces:**
- Consumes: `RobotModel.model`, `RobotModel.corner_frame_ids`.
- Produces: `WBDynamics(model, ee_frames)` with attrs `nq, nv, nj, nf, cmodel, cdata` and methods returning `ca.Function`:
  - `state_integrate() : (x[71], dx[70]) -> x_next[71]`
  - `state_difference() : (x0[71], x1[71]) -> dx[70]`
  - `rnea_dynamics() : (q[36], v[35], a[35], forces[24]) -> tau_rnea[35]`
  - `frame_velocity(fid) : (q[36], v[35]) -> vel6` (LOCAL_WORLD_ALIGNED)

- [ ] **Step 1: Write the failing test** — `tests/test_dynamics.py` (RNEA must match numpy `pin.rnea` with accumulated `f_ext`)

```python
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_q
from t1_nmpc.wb.dynamics import WBDynamics

def _numeric_rnea(model, data, ee, q, v, a, forces):
    pin.framesForwardKinematics(model, data, q)
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force(np.zeros(6)))
    for idx, fid in enumerate(ee):
        jid = model.frames[fid].parentJoint
        trans = model.frames[fid].placement.translation
        R = data.oMi[jid].rotation.T
        fl = R @ forces[idx*3:idx*3+3]
        fext[jid] = pin.Force(fext[jid].vector + np.concatenate([fl, np.cross(trans, fl)]))
    return pin.rnea(model, data, q, v, a, fext)

def test_rnea_matches_pinocchio():
    cfg = make_config(); rm = load_model(cfg)
    dyn = WBDynamics(rm.model, rm.corner_frame_ids)
    assert dyn.nf == 24 and dyn.nj == 29
    q = nominal_q(cfg, rm.model)
    rng = np.random.default_rng(0)
    v = 0.05*rng.standard_normal(rm.model.nv); a = 0.1*rng.standard_normal(rm.model.nv)
    forces = rng.uniform(-10, 10, 24); forces[2::3] += 42.0
    tau_sym = np.array(dyn.rnea_dynamics()(q, v, a, forces)).flatten()
    tau_ref = _numeric_rnea(rm.model, rm.data, rm.corner_frame_ids, q, v, a, forces)
    assert np.max(np.abs(tau_sym - tau_ref)) < 1e-8

def test_state_roundtrip():
    cfg = make_config(); rm = load_model(cfg)
    dyn = WBDynamics(rm.model, rm.corner_frame_ids)
    x = np.concatenate([nominal_q(cfg, rm.model), np.zeros(rm.model.nv)])
    dx = np.zeros(70)
    x2 = np.array(dyn.state_integrate()(x, dx)).flatten()
    assert np.allclose(x2, x, atol=1e-12)
    d = np.array(dyn.state_difference()(x, x)).flatten()
    assert np.allclose(d, 0.0, atol=1e-12)
```

- [ ] **Step 2: Run test, verify it fails** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `t1_nmpc/wb/dynamics.py`** (verbatim-adapted from proven `spike/dynamics.py`; the f_ext **accumulation** is mandatory)

```python
"""cpin symbolic dynamics: FreeFlyer state ops + RNEA over the 8 corner frames."""
from __future__ import annotations

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


class WBDynamics:
    def __init__(self, model: pin.Model, ee_frames):
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()
        self.nq = self.cmodel.nq
        self.nv = self.cmodel.nv
        self.nj = self.nq - 7
        self.ee_frames = tuple(ee_frames)
        self.nf = 3 * len(self.ee_frames)

    def state_integrate(self) -> ca.Function:
        x = ca.SX.sym("x", self.nq + self.nv)
        dx = ca.SX.sym("dx", self.nv + self.nv)
        q_next = cpin.integrate(self.cmodel, x[:self.nq], dx[:self.nv])
        v_next = x[self.nq:] + dx[self.nv:]
        return ca.Function("integrate", [x, dx], [ca.vertcat(q_next, v_next)])

    def state_difference(self) -> ca.Function:
        x0 = ca.SX.sym("x0", self.nq + self.nv)
        x1 = ca.SX.sym("x1", self.nq + self.nv)
        dq = cpin.difference(self.cmodel, x0[:self.nq], x1[:self.nq])
        dv = x1[self.nq:] - x0[self.nq:]
        return ca.Function("difference", [x0, x1], [ca.vertcat(dq, dv)])

    def rnea_dynamics(self) -> ca.Function:
        q = ca.SX.sym("q", self.nq); v = ca.SX.sym("v", self.nv)
        a = ca.SX.sym("a", self.nv); forces = ca.SX.sym("forces", self.nf)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.cmodel.njoints)]
        for idx, fid in enumerate(self.ee_frames):
            jid = self.cmodel.frames[fid].parentJoint
            trans = self.cmodel.frames[fid].placement.translation
            R_w2j = self.cdata.oMi[jid].rotation.T
            f_lin = R_w2j @ forces[idx*3:(idx+1)*3]
            f_ang = ca.cross(trans, f_lin)
            f_ext[jid] = cpin.Force(f_ext[jid].vector + ca.vertcat(f_lin, f_ang))  # ACCUMULATE
        tau = cpin.rnea(self.cmodel, self.cdata, q, v, a, f_ext)
        return ca.Function("rnea_dyn", [q, v, a, forces], [tau])

    def frame_velocity(self, fid: int) -> ca.Function:
        q = ca.SX.sym("q", self.nq); v = ca.SX.sym("v", self.nv)
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        vel = cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).vector
        return ca.Function(f"vel_{fid}", [q, v], [vel])
```

- [ ] **Step 4: Run test, verify pass** → 2 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat(wb-rnea): cpin dynamics + 8-corner RNEA (accumulated f_ext)"`

---

### Task 4: `wb/gait.py` — biped stand contact schedule

**Files:**
- Create: `t1_nmpc/wb/gait.py`
- Test: `tests/test_gait.py`

**Interfaces:**
- Produces: `StandGait(n_corners=8)` with `contact_schedule(t, dts, nodes) -> np.ndarray (8, nodes)` (all ones for stand) and `swing_schedule(...) -> np.ndarray (8, nodes)` (all zeros). Corners 0-3 = left foot, 4-7 = right foot (one flag per corner, but a foot's 4 corners always share a value — trivially all-1 at stand).

- [ ] **Step 1: Write the failing test** — `tests/test_gait.py`

```python
import numpy as np
from t1_nmpc.wb.gait import StandGait

def test_stand_schedule_all_contact():
    g = StandGait(n_corners=8)
    dts = [0.02]*14
    cs = g.contact_schedule(0.0, dts, 14)
    sw = g.swing_schedule(0.0, dts, 14)
    assert cs.shape == (8, 14) and sw.shape == (8, 14)
    assert np.all(cs == 1.0) and np.all(sw == 0.0)
```

- [ ] **Step 2: Run test, verify it fails.**
- [ ] **Step 3: Implement `t1_nmpc/wb/gait.py`**

```python
"""Biped contact scheduling. Iteration 1: STAND only (all 8 corners always in contact).
Walking (2 swing groups -> 8 corner flags) is deferred; the interface is shaped for it."""
from __future__ import annotations

import numpy as np


class StandGait:
    def __init__(self, n_corners: int = 8):
        self.n_corners = n_corners

    def contact_schedule(self, t_current: float, dts, nodes: int) -> np.ndarray:
        return np.ones((self.n_corners, nodes), dtype=np.float64)

    def swing_schedule(self, t_current: float, dts, nodes: int) -> np.ndarray:
        return np.zeros((self.n_corners, nodes), dtype=np.float64)
```

- [ ] **Step 4: Run test, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(wb-rnea): stand contact schedule (8 corners in contact)"`

---

### Task 5: `wb/ocp.py` — CasADi Opti transcription of whole_body_rnea (stand)

**Files:**
- Create: `t1_nmpc/wb/ocp.py`
- Test: `tests/test_ocp.py`

**Interfaces:**
- Consumes: `RobotModel`, `WBDynamics`, `MPCConfig`, `StandGait`.
- Produces: `StandOCP(cfg, robot_model)` with:
  - attrs `opti, DX (list[nodes+1]), U (list[nodes]), x_init (param 71), Q_diag/R_diag (params), nu(i), ndx, dts`
  - `set_weights()`, `set_x_init(x71)`
  - `solve_function()` -> `ca.Function('solver_fn', [x_init, Q_diag, R_diag, opti.x], [opti.x])`
  - `g_data()` -> `ca.Function([opti.x, opti.p], [g, lbg, ubg])`, plus `constr_viol_inf(g,lbg,ubg)` static helper.
  - `retract(sol_x) -> dict{q_sol, v_sol, a_sol, forces_sol, tau_sol}` lists (only node 0 + 1 needed for the command, but retract all for metrics).

**Constraint emission order (Fatrop — DO NOT REORDER):** initial `DX[0]==0`; then per node `i`: (1) `dq_next==dq+v·dt`, `dv_next==dv+a·dt`; (2) `tau_rnea[:6]==0`; (3) if `i<tau_nodes`: `tau_rnea[6:]==tau_j` + box; (4) per-corner friction cone; (5) if `i>0`: per-corner zero contact velocity.

**Deliberate scope decision — joint pos/vel limits omitted (spec §6.5).** The proven spike converged without joint position/velocity box constraints, and they are non-binding at the nominal stand. They are **deferred to the walking milestone** (where the swing leg approaches limits). This is a conscious divergence from the wb-mpc source, logged in Task 9. Do not add them in iteration 1.

- [ ] **Step 1: Write the failing test** — `tests/test_ocp.py` (one-shot Fatrop solve at nominal stand must converge)

```python
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.ocp import StandOCP

def test_stand_ocp_converges():
    cfg = make_config()
    rm = load_model(cfg)
    ocp = StandOCP(cfg, rm)
    ocp.set_weights()
    x0 = nominal_x(cfg, rm.model)
    ocp.set_x_init(x0)
    sol_fn = ocp.solve_function(max_iter=cfg.fatrop_max_iter)
    # cold start: warm param = current opti.x initial (zeros for DX, gravity-comp handled inside)
    sol_x = np.array(sol_fn(x0, cfg.Q_diag, cfg.R_diag, ocp.x_initial())).flatten()
    g, lbg, ubg = ocp.g_data()(sol_x, ocp.opti.value(ocp.opti.p))
    cv = StandOCP.constr_viol_inf(np.array(g).flatten(), np.array(lbg).flatten(), np.array(ubg).flatten())
    assert cv < 1e-4, f"constraint violation too high: {cv}"
    out = ocp.retract(sol_x)
    fz = np.array(out["forces_sol"][0]).reshape(8, 3)[:, 2]
    assert abs(fz.sum() - rm.mass * 9.81) / (rm.mass * 9.81) < 0.05   # vertical balance
    assert np.all(fz > -1e-6)                                          # unilateral
```

- [ ] **Step 2: Run test, verify it fails.**
- [ ] **Step 3: Implement `t1_nmpc/wb/ocp.py`** (adapted from proven `spike/ocp.py`; use `ca.MX.zeros`/`ca.DM` at the Opti graph level)

```python
"""CasADi Opti transcription of whole_body_rnea for T1 stand (8 corners all in contact)."""
from __future__ import annotations

import numpy as np
import casadi as ca

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .dynamics import WBDynamics


class StandOCP:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, uniform_width: bool = False):
        self.cfg = cfg
        self.rm = rm
        self.dyn = WBDynamics(rm.model, rm.corner_frame_ids)
        self.mass = rm.mass
        self.nq, self.nv, self.nj, self.nf = self.dyn.nq, self.dyn.nv, self.dyn.nj, self.dyn.nf
        self.na = self.nv
        self.nodes, self.tau_nodes = cfg.nodes, cfg.tau_nodes
        self.mu, self.uniform_width = cfg.friction_mu, uniform_width
        self.nx = self.nq + self.nv
        self.ndx = 2 * self.nv
        self.f_idx, self.tau_idx = self.na, self.na + self.nf
        self.tau_max = rm.tau_max
        ratio = cfg.dt_max / cfg.dt_min
        gamma = ratio ** (1.0 / (self.nodes - 1))
        self.dts = [cfg.dt_min * gamma ** i for i in range(self.nodes)]
        self.opti = ca.Opti()
        self._build()

    def _nu(self, i):
        return self.na + self.nf + (self.nj if (self.uniform_width or i < self.tau_nodes) else 0)

    def _has_tau(self, i):
        return self.uniform_width or (i < self.tau_nodes)

    def _build(self):
        opti = self.opti
        self.DX, self.U = [], []
        for i in range(self.nodes):                       # staircase order
            self.DX.append(opti.variable(self.ndx))
            self.U.append(opti.variable(self._nu(i)))
        self.DX.append(opti.variable(self.ndx))

        self.x_init = opti.parameter(self.nx)
        self.Q_diag = opti.parameter(self.ndx)
        self.R_diag = opti.parameter(self.na + self.nf + self.nj)

        q0 = self.x_init[:self.nq]
        x_des = ca.vertcat(q0, ca.MX.zeros(self.nv))      # track current q, zero velocity
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_grav = self.mass * 9.81 / self.cfg.n_corners
        self.f_des = ca.vertcat(*[ca.DM([0, 0, f_grav]) for _ in range(self.cfg.n_corners)])
        self.u_des_full = ca.vertcat(ca.DM.zeros(self.na), self.f_des, ca.DM.zeros(self.nj))

        self._constraints()
        opti.minimize(self._objective())

    # accessors
    def _x(self, i): return self.dyn.state_integrate()(self.x_init, self.DX[i])
    def _q(self, i): return self._x(i)[:self.nq]
    def _v(self, i): return self._x(i)[self.nq:]
    def _a(self, i): return self.U[i][:self.na]
    def _f(self, i): return self.U[i][self.f_idx:self.tau_idx]
    def _tau(self, i): return self.U[i][self.tau_idx:]

    def _constraints(self):
        opti = self.opti
        opti.subject_to(self.DX[0] == np.zeros(self.ndx))
        rnea = self.dyn.rnea_dynamics()
        velfn = {fid: self.dyn.frame_velocity(fid) for fid in self.dyn.ee_frames}
        for i in range(self.nodes):
            dq, dv = self.DX[i][:self.nv], self.DX[i][self.nv:]
            dq_n, dv_n = self.DX[i + 1][:self.nv], self.DX[i + 1][self.nv:]
            q, v, a, forces, dt = self._q(i), self._v(i), self._a(i), self._f(i), self.dts[i]
            opti.subject_to(dq_n == dq + v * dt)                    # (1) gap-closing FIRST
            opti.subject_to(dv_n == dv + a * dt)
            tau_rnea = rnea(q, v, a, forces)
            opti.subject_to(tau_rnea[:6] == np.zeros(6))            # (2) base underactuation
            if self._has_tau(i):                                    # (3) torque eq + box
                tau_j = self._tau(i)
                opti.subject_to(tau_rnea[6:] == tau_j)
                opti.subject_to(opti.bounded(-self.tau_max, tau_j, self.tau_max))
            for c in range(self.cfg.n_corners):                    # (4) friction cone
                fe = forces[c*3:(c+1)*3]
                opti.subject_to(fe[2] >= 0)
                opti.subject_to(self.mu**2 * fe[2]**2 >= fe[0]**2 + fe[1]**2)
            if i == 0:
                continue
            for fid in self.dyn.ee_frames:                         # (5) zero contact velocity
                opti.subject_to(velfn[fid](q, v)[:3] == np.zeros(3))

    def _objective(self):
        Q, R = ca.diag(self.Q_diag), ca.diag(self.R_diag)
        obj = 0
        for i in range(self.nodes):
            u = self.U[i]
            if not self._has_tau(i):
                u = ca.vertcat(u, ca.MX.zeros(self.nj))
            e_dx = self.DX[i] - self.dx_des
            obj += e_dx.T @ Q @ e_dx + (u - self.u_des_full).T @ R @ (u - self.u_des_full)
        e_dx = self.DX[self.nodes] - self.dx_des
        return obj + e_dx.T @ Q @ e_dx

    # --- API ---
    def set_weights(self):
        self.opti.set_value(self.Q_diag, self.cfg.Q_diag)
        self.opti.set_value(self.R_diag, self.cfg.R_diag)

    def set_x_init(self, x71):
        self.opti.set_value(self.x_init, np.asarray(x71, dtype=np.float64))

    def x_initial(self):
        return self.opti.value(self.opti.x, self.opti.initial())

    def _fatrop_opts(self, max_iter):
        return {"expand": True, "structure_detection": "auto", "debug": True,
                "fatrop": {"print_level": 0, "max_iter": int(max_iter),
                           "tol": self.cfg.fatrop_tol, "mu_init": self.cfg.fatrop_mu_init,
                           "warm_start_init_point": True,
                           "warm_start_mult_bound_push": 1e-7, "bound_push": 1e-7}}

    def solve_function(self, max_iter):
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "solver_fn", [self.x_init, self.Q_diag, self.R_diag, self.opti.x], [self.opti.x])

    def g_data(self):
        return ca.Function("g_data", [self.opti.x, self.opti.p],
                           [self.opti.g, self.opti.lbg, self.opti.ubg])

    @staticmethod
    def constr_viol_inf(g, lbg, ubg):
        viol = np.concatenate([np.maximum(0, lbg - g), np.maximum(0, g - ubg)])
        return float(np.max(np.abs(viol))) if viol.size else 0.0

    def retract(self, sol_x):
        sol_x = np.asarray(sol_x).flatten()
        out = {"q_sol": [], "v_sol": [], "a_sol": [], "forces_sol": [], "tau_sol": []}
        x_init = self.opti.value(self.x_init)
        integ = self.dyn.state_integrate()
        idx = 0
        for i in range(self.nodes):
            nu = self._nu(i)
            dx = sol_x[idx:idx + self.ndx]
            u = sol_x[idx + self.ndx: idx + self.ndx + nu]
            idx += self.ndx + nu
            x = np.array(integ(x_init, dx)).flatten()
            out["q_sol"].append(x[:self.nq]); out["v_sol"].append(x[self.nq:])
            out["a_sol"].append(u[:self.na])
            out["forces_sol"].append(u[self.f_idx:self.tau_idx])
            out["tau_sol"].append(u[self.tau_idx:] if self._has_tau(i) else np.zeros(self.nj))
        return out
```

- [ ] **Step 4: Run test, verify pass.** Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_ocp.py -q` → 1 passed (expect CV ~1e-7..1e-4 and `Σfz/mg ≈ 1.0`). If Fatrop rejects the adaptive width (it should NOT — verified), set `StandOCP(..., uniform_width=True)` as the documented fallback and re-run.
- [ ] **Step 5: Commit** — `git commit -am "feat(wb-rnea): Opti+Fatrop whole_body_rnea OCP (stand) — converges"`

---

### Task 6: `wb/state.py` — MuJoCo↔pinocchio FreeFlyer map + command extraction

**Files:**
- Create: `t1_nmpc/wb/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `pin.Model` (for nq/nv).
- Produces:
  - `mujoco_to_freeflyer(qpos36, qvel35, model) -> x71`
  - `freeflyer_to_mujoco(x71, model) -> (qpos36, qvel35)`
  - `extract_command(retracted, cfg) -> JointCommand` — `tau_ff = tau_sol[0]`, `q_des = q_sol[1][7:]`, `qd_des = v_sol[1][6:]`, `kp/kd = cfg.kp/cfg.kd`.

- [ ] **Step 1: Write the failing test** — `tests/test_state.py` (exact round-trip under non-identity orientation guards the world/local linear-velocity bug)

```python
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config, T1_URDF_PATH
from t1_nmpc.wb.state import mujoco_to_freeflyer, freeflyer_to_mujoco

def test_state_roundtrip_under_yaw():
    model = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    rng = np.random.default_rng(0); nj = model.nq - 7
    yaw = np.pi/2
    qpos = np.empty(36)
    qpos[0:3] = [0.31, -0.22, 0.6734]
    qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]   # (w,x,y,z) about z
    qpos[7:] = rng.uniform(-0.3, 0.3, nj)
    qvel = np.empty(35)
    qvel[0:3] = [0.7, -0.4, 0.15]    # WORLD linear
    qvel[3:6] = [0.2, -0.13, 0.5]    # LOCAL angular
    qvel[6:] = rng.uniform(-0.5, 0.5, nj)
    x = mujoco_to_freeflyer(qpos, qvel, model)
    assert x.shape == (71,)
    # the rotation must actually be applied: body-linear differs from world-linear under yaw
    assert np.max(np.abs(x[36:39] - qvel[0:3])) > 1e-3
    qpos2, qvel2 = freeflyer_to_mujoco(x, model)
    assert np.max(np.abs(qpos2 - qpos)) < 1e-12
    assert np.max(np.abs(qvel2 - qvel)) < 1e-12
```

- [ ] **Step 2: Run test, verify it fails.**
- [ ] **Step 3: Implement `t1_nmpc/wb/state.py`** (verbatim-adapted from proven `spike/state_freeflyer.py`)

```python
"""MuJoCo <-> pinocchio FreeFlyer state map (single source of truth) + command extraction.

MuJoCo: qpos=[pos3, quat_wxyz4, joints29], qvel=[lin_WORLD3, ang_LOCAL3, jvel29].
pinocchio FreeFlyer: q=[pos3, quat_xyzw4, joints29], v=[lin_LOCAL3, ang_LOCAL3, jvel29].
Base linear velocity differs by frame -> rotate by R(q)^T (MuJoCo->pin) / R(q) (pin->MuJoCo)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ..robot.config import MPCConfig, JointCommand


def mujoco_to_freeflyer(qpos, qvel, model) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64); qvel = np.asarray(qvel, dtype=np.float64)
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    q = np.empty(model.nq); q[0:3] = qpos[0:3]; q[3:7] = [qx, qy, qz, qw]; q[7:] = qpos[7:]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    v = np.empty(model.nv); v[0:3] = R.T @ qvel[0:3]; v[3:6] = qvel[3:6]; v[6:] = qvel[6:]
    return np.concatenate([q, v])


def freeflyer_to_mujoco(x, model):
    x = np.asarray(x, dtype=np.float64)
    q = x[:model.nq]; v = x[model.nq:model.nq + model.nv]
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    qpos = np.empty(model.nq); qpos[0:3] = q[0:3]; qpos[3:7] = [qw, qx, qy, qz]; qpos[7:] = q[7:]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    qvel = np.empty(model.nv); qvel[0:3] = R @ v[0:3]; qvel[3:6] = v[3:6]; qvel[6:] = v[6:]
    return qpos, qvel


def extract_command(retracted: dict, cfg: MPCConfig) -> JointCommand:
    """tau_ff from node 0; q_des/qd_des from the planned next node (1)."""
    return JointCommand(
        q_des=np.asarray(retracted["q_sol"][1][7:], dtype=np.float64),
        qd_des=np.asarray(retracted["v_sol"][1][6:], dtype=np.float64),
        tau_ff=np.asarray(retracted["tau_sol"][0], dtype=np.float64),
        kp=np.asarray(cfg.kp, dtype=np.float64),
        kd=np.asarray(cfg.kd, dtype=np.float64),
    )
```

- [ ] **Step 4: Run test, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(wb-rnea): MuJoCo<->pinocchio FreeFlyer state map + command extraction"`

---

### Task 7: `wb/mpc.py` — WholeBodyMPC wrapper (build, solver_function, reset, step, warm-start)

**Files:**
- Create: `t1_nmpc/wb/mpc.py`
- Test: `tests/test_mpc.py`

**Interfaces:**
- Consumes: `MPCConfig`, `RobotModel`, `StandOCP`, `extract_command`.
- Produces:
  - `WBResult` dataclass: `command (JointCommand)`, `forces0 (np.ndarray (24,))`, `solve_time (float)`, `constr_viol (float)`.
  - `WholeBodyMPC(cfg, robot_model)`: builds the OCP + Fatrop `solver_function` once; `reset(x0)` (cold solve, store warm); `step(x_meas) -> WBResult` (warm-started solve, retract, extract command, advance warm state).

- [ ] **Step 1: Write the failing test** — `tests/test_mpc.py`

```python
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.mpc import WholeBodyMPC

def test_mpc_reset_and_step():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0)
    assert res.command.tau_ff.shape == (29,)
    assert res.command.q_des.shape == (29,) and res.command.qd_des.shape == (29,)
    assert res.constr_viol < 1e-3
    fz = res.forces0.reshape(8, 3)[:, 2]
    assert abs(fz.sum() - rm.mass * 9.81) / (rm.mass * 9.81) < 0.05
    # a second warm-started step also converges
    res2 = mpc.step(x0)
    assert res2.constr_viol < 1e-3
```

- [ ] **Step 2: Run test, verify it fails.**
- [ ] **Step 3: Implement `t1_nmpc/wb/mpc.py`**

```python
"""WholeBodyMPC: build the whole_body_rnea OCP once, run warm-started Fatrop each tick."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..robot.config import MPCConfig, JointCommand
from ..robot.model import RobotModel
from .ocp import StandOCP
from .state import extract_command


@dataclass
class WBResult:
    command: JointCommand
    forces0: np.ndarray
    solve_time: float
    constr_viol: float


class WholeBodyMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, uniform_width: bool = False):
        self.cfg = cfg
        self.rm = rm
        self.ocp = StandOCP(cfg, rm, uniform_width=uniform_width)
        self.ocp.set_weights()
        # ONE solver_function built once, reused for reset + every tick (warm param = opti.x).
        # A single max_iter cap is correct: warm-started ticks converge early and stop well under it.
        self._solve = self.ocp.solve_function(max_iter=cfg.fatrop_max_iter)
        self._gdata = self.ocp.g_data()
        self._warm = None        # last opti.x solution vector (warm start)

    def reset(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        self.ocp.set_x_init(x0)
        sol = np.array(self._solve(x0, self.cfg.Q_diag, self.cfg.R_diag,
                                   self.ocp.x_initial())).flatten()
        self._warm = sol

    def step(self, x_meas) -> WBResult:
        x = np.asarray(x_meas, dtype=np.float64)
        self.ocp.set_x_init(x)
        warm = self._warm if self._warm is not None else self.ocp.x_initial()
        t0 = time.perf_counter()
        sol = np.array(self._solve(x, self.cfg.Q_diag, self.cfg.R_diag, warm)).flatten()
        dt = time.perf_counter() - t0
        self._warm = sol
        g, lbg, ubg = self._gdata(sol, self.ocp.opti.value(self.ocp.opti.p))
        cv = StandOCP.constr_viol_inf(np.array(g).flatten(), np.array(lbg).flatten(),
                                      np.array(ubg).flatten())
        out = self.ocp.retract(sol)
        return WBResult(command=extract_command(out, self.cfg),
                        forces0=np.asarray(out["forces_sol"][0], dtype=np.float64),
                        solve_time=dt, constr_viol=cv)
```

- [ ] **Step 4: Run test, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(wb-rnea): WholeBodyMPC wrapper (warm-started Fatrop, command extraction)"`

---

### Task 8: `sim/stand.py` — closed-loop MuJoCo stand + metrics (SUCCESS GATE)

**Files:**
- Create: `sim/stand.py`
- Modify: `sim/mujoco_runtime.py` — set `MujocoRuntime` to read FreeFlyer state (add `freeflyer_state()` using `wb.state.mujoco_to_freeflyer`), and ensure `reset_to_nominal` uses `cfg.nominal_base_height` (now 0.6734).
- Test: `tests/test_stand_closed_loop.py`

**Interfaces:**
- Consumes: `MujocoRuntime` (`reset_to_nominal`, `step_physics`, `_apply_torque`, `mj_data`, `control_decim`, `mpc_decim`), `WholeBodyMPC`, `mujoco_to_freeflyer`, `load_model`.
- Produces: `run_stand(cfg, duration, view=False, gif=None) -> dict{fz_ratio_p50, fz_ratio_min, fz_ratio_max, max_tilt_deg, fell, solve_p90_ms, t_end}`.

- [ ] **Step 1: Add `freeflyer_state()` to `sim/mujoco_runtime.py`** (after `_pin_q_v`)

```python
    def freeflyer_state(self, pin_model):
        """Measured FreeFlyer x[71] for the whole_body_rnea MPC (NOT the euler _pin_q_v)."""
        from t1_nmpc.wb.state import mujoco_to_freeflyer
        return mujoco_to_freeflyer(self.mj_data.qpos, self.mj_data.qvel, pin_model)
```

- [ ] **Step 2: Write the failing test** — `tests/test_stand_closed_loop.py` (the success gate; short duration for CI)

```python
import numpy as np
from t1_nmpc.robot.config import make_config
from sim.stand import run_stand

def test_closed_loop_stand_holds():
    cfg = make_config()
    m = run_stand(cfg, duration=2.0, view=False)
    assert not m["fell"], "robot fell during stand"
    assert 0.9 <= m["fz_ratio_p50"] <= 1.1, m["fz_ratio_p50"]
    assert m["max_tilt_deg"] < 10.0, m["max_tilt_deg"]
    assert m["solve_p90_ms"] < 60.0, m["solve_p90_ms"]
```

- [ ] **Step 3: Run test, verify it fails** (`ModuleNotFoundError: sim.stand`).
- [ ] **Step 4: Implement `sim/stand.py`**

```python
"""Closed-loop MuJoCo stand under the whole_body_rnea (Fatrop) MPC.

Control: at mpc_hz solve -> JointCommand (q_des, qd_des, tau_ff); at control_hz apply
tau = tau_ff + kp*(q_des - q) - kd*(qd_des - qd); physics at physics_hz."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.mpc import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def _tilt_deg(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    R = pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))   # angle of body-z from world-z


def run_stand(cfg: MPCConfig, duration: float = 2.0, view: bool = False, gif: str = None) -> dict:
    rm = load_model(cfg)
    rt = MujocoRuntime(cfg, rm)
    rt.reset_to_nominal()
    mpc = WholeBodyMPC(cfg, rm)
    mpc.reset(nominal_x(cfg, rm.model))

    cmd = None
    solve_ms, fz_ratios, tilts = [], [], []
    mg = rm.mass * 9.81
    n_steps = int(round(duration * cfg.physics_hz))
    fell = False
    for k in range(n_steps):
        if k % rt.mpc_decim == 0:                                   # MPC tick (ZOH)
            x = rt.freeflyer_state(rm.model)
            res = mpc.step(x)
            cmd = res.command
            solve_ms.append(res.solve_time * 1e3)
            fz_ratios.append(res.forces0.reshape(8, 3)[:, 2].sum() / mg)
        if k % rt.control_decim == 0 and cmd is not None:           # control tick
            q = np.array(rt.mj_data.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
            qd = np.array(rt.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
            tau = cmd.tau_ff + cmd.kp * (cmd.q_des - q) - cmd.kd * (cmd.qd_des - qd)
            rt._apply_torque(tau)
        rt.step_physics()
        tilts.append(_tilt_deg(rt.mj_data.qpos))
        if rt.mj_data.qpos[2] < 0.3 or tilts[-1] > 45.0:            # fell
            fell = True
            break
    return {
        "fz_ratio_p50": float(np.median(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_min": float(np.min(fz_ratios)) if fz_ratios else 0.0,
        "fz_ratio_max": float(np.max(fz_ratios)) if fz_ratios else 0.0,
        "max_tilt_deg": float(np.max(tilts)) if tilts else 0.0,
        "fell": fell,
        "solve_p90_ms": float(np.percentile(solve_ms, 90)) if solve_ms else 0.0,
        "t_end": float(rt.t),
    }


if __name__ == "__main__":
    import argparse
    from t1_nmpc.robot.config import make_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    a = ap.parse_args()
    print(run_stand(make_config(), duration=a.duration))
```

- [ ] **Step 5: Run the test, verify pass.** Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_stand_closed_loop.py -q`. Expected: 1 passed (robot holds, `fz_ratio_p50≈1.0`, tilt small). **If it falls:** this is the real integration risk — debug in this order (use `systematic-debugging`): (a) confirm `freeflyer_state` matches the MPC's frame (round-trip test already guards it); (b) raise `fatrop_max_iter_warm` (more re-convergence per tick); (c) raise `mpc_hz` toward 100; (d) verify `tau_ff` sign/order maps through `_apply_torque`; (e) only as a diagnostic, confirm the OCP-free hold passes with elevated ankle kp (a crutch, not a fix — the MPC must hold it). Do not ship elevated ankle kp.
- [ ] **Step 6: Commit** — `git commit -am "feat(wb-rnea): closed-loop MuJoCo stand runner + success-gate test"`

---

### Task 9: Docs — CLAUDE.md rewrite, divergence ledger, pyproject

**Files:**
- Modify: `CLAUDE.md`, `pyproject.toml`, `docs/2026-06-25-t1controller-divergences.md`

- [ ] **Step 1: Rewrite `CLAUDE.md`** to describe the new controller: CasADi `Opti` + Fatrop `whole_body_rnea` on T1; `x=[q(36),v(35)]`, adaptive input `[a, forces(24), τ_j(29 first-nodes)]`; FreeFlyer base; 8-corner flat-foot contacts; the §9 MuJoCo↔pinocchio conversion (with the `R(q)ᵀ` base-linear rotation as an invariant); Fatrop gap-closing-first + staircase + f_ext-accumulation invariants; reuse-t1mpc-env (no new deps); status M0-stand. Drop all aligator/ProxDDP/serial-walk text.

- [ ] **Step 2: Append to `docs/2026-06-25-t1controller-divergences.md`** the wb-rnea divergences: 8 corner 3D forces vs OCS2 6D-wrench+CoP (`nf=24` vs 12); μ=0.4 retained; Fatrop NLP vs OCS2 SQP/HPIPM and vs aligator ProxDDP; all 29 joints kept (no head fixing); **Q/R weights re-dimensioned to T1 and NOT traced to t1_controller** (must be re-cited or kept as an explicit divergence).

- [ ] **Step 3: Update `pyproject.toml`** description `aligator ProxDDP + pinocchio.casadi` → `CasADi Opti + Fatrop whole_body_rnea + pinocchio.casadi`. Remove the stale `acados_template`/`adam-robotics` comment lines.

- [ ] **Step 4: Full suite green.** Run: `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/ -q -p no:cacheprovider`. Expected: all green (config, model, dynamics, gait, ocp, state, mpc, stand).

- [ ] **Step 5: Commit** — `git commit -am "docs(wb-rnea): rewrite CLAUDE.md + divergence ledger + pyproject for the Fatrop port"`

---

## Self-review notes (for the executor)

- **Branch:** start on a fresh branch (`git checkout -b wb-rnea-port`) before Task 1 — this abandons the aligator backend.
- **Highest risk = Task 8** (closed-loop integration). Everything up to Task 7 is unit-verified against the proven spike; Task 8 is where the MPC must actually balance the plant. Budget debug time there.
- **Deferred (out of scope):** walking gait (the `StandGait` interface is shaped for it but swing/contact-schedule + Fatrop structure detection under changing contact rank must be re-validated), arm-EE loco-manipulation, Ipopt/OSQP, codegen/hardware, MJCF vendoring.
- **Weights** are plausible but untuned/uncited — fine for stand; flag before any walking work.
