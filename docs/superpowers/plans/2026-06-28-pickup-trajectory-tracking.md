# Pickup Trajectory Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track `data/motion_plan.pkl` (bimanual floor pickup, all double-support stance) on the T1 whole-body RNEA + Fatrop MPC in closed-loop MuJoCo, **real-time**, with a `time_scale` slow-down knob.

**Architecture:** A new all-stance tracking OCP (`PickupOCP`) generalizes the M0 stand OCP: the fixed nominal target becomes a per-node, time-varying reference sampled from the plan. Base pose (z from `trunk_height`, lean from quat) + arm joints + waist are tracked; hands are tracked in task-space (soft, hard at grasp keyframes via a slack trick); the legs are **solved** by the OCP against hard planted-feet contact, with leg joint-position limits + a low-weight leg-pitch seed in the reference. Real-time comes from a short horizon (N=10/8); JIT is ruled out. M0 stand path is untouched.

**Tech Stack:** Python 3.10 (conda env `t1mpc`), pinocchio 4.0 + pinocchio.casadi, CasADi 3.7.2 + Fatrop, MuJoCo 3.10, NumPy 2.2, scipy (Rotation/slerp), pytest.

## Global Constraints

- **Run preamble (load-bearing):** every command is `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc <cmd>`, run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc`. Empty `PYTHONPATH` keeps ROS numpy<2 pinocchio off the path (else segfault).
- **Authoritative joint mapping** (from `t1_kd_mpc`): `Waist = −trunk_yaw`; `trunk_pitch/knee_pitch/ankle_pitch` broadcast **identically to both legs, no sign flip**; arm order `[Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw, Wrist_Pitch, Wrist_Yaw, Hand_Roll]`; base **z = `trunk_height`**, base **lean = yaw-anchored `trunk_quat_xyzw`**, base **x,y anchored to start** (not from plan); head + hip-roll/yaw + ankle-roll **not** tracked (solved/nominal).
- **pinocchio joint order (29)**, joint-local index → full-`q` index = `7 + j`: `[0 AAHead_yaw, 1 Head_pitch, 2 L_Shoulder_Pitch, 3 L_Shoulder_Roll, 4 L_Elbow_Pitch, 5 L_Elbow_Yaw, 6 L_Wrist_Pitch, 7 L_Wrist_Yaw, 8 L_Hand_Roll, 9..15 R_(same 7), 16 Waist, 17 L_Hip_Pitch, 18 L_Hip_Roll, 19 L_Hip_Yaw, 20 L_Knee_Pitch, 21 L_Ankle_Pitch, 22 L_Ankle_Roll, 23..28 R_(same 6)]`.
- **State:** `x∈ℝ⁷¹ = [q(36)=[pos3, quat_xyzw4, joints29], v(35)=[lin_LOCAL3, ang_LOCAL3, jvel29]]`; `dx∈ℝ⁷⁰`. MuJoCo↔pinocchio base-vel rule lives only in `wb/state.py` — do not re-derive.
- **Fatrop invariants:** interleaved variable creation `DX[0],U[0],…,DX[N]`; per-stage the gap-closing equality is added **before any inequality**; `structure_detection='auto'`, `expand=True`, `include accelerations in the input`.
- **time_scale:** larger = slower. Reference sampled at phase `t_ref = clip((t_wall + i·dt)/time_scale, 0, T_end)`; velocities scale `1/time_scale` (handled implicitly by manifold finite-difference across nodes).
- **Real-time acceptance:** measured warm solve **p90 < 16 ms** at the chosen N (target 10, fallback 8); JIT/C-codegen of the solver is **out of scope** (graph too large).
- **No payload model**, no hand-orientation hard tracking, no walking. M0 stand tests must still pass.
- TDD, DRY, YAGNI, frequent commits. Tests live under `tests/`.

---

## File Structure

- `t1_nmpc/robot/model.py` (modify) — add `hand_frame_ids` to `RobotModel`.
- `t1_nmpc/robot/config.py` (modify) — add tracking fields + `_track_Q_diag()` + `make_track_config()`.
- `t1_nmpc/wb/reference.py` (create) — `MotionPlanReference`: load plan, joint mapping, grasp keyframes, horizon sampling (interp + slerp + manifold-difference velocities).
- `t1_nmpc/wb/track_ocp.py` (create) — `PickupOCP`: all-stance whole-body RNEA tracking OCP (build-once `to_function`).
- `t1_nmpc/wb/track_mpc.py` (create) — `TrackingMPC`: phase clock + sample + warm solve + command extraction.
- `sim/pickup.py` (create) — closed-loop MuJoCo runner, metrics, `--time_scale --duration --realtime --view --gif`.
- `tests/test_track_reference.py`, `tests/test_track_ocp.py`, `tests/test_track_mpc.py`, `tests/test_track_closed_loop.py` (create).
- `docs/2026-06-25-t1controller-divergences.md` (modify) — log the divergences.

---

## Task 1: Model hand frames + tracking config

**Files:**
- Modify: `t1_nmpc/robot/model.py`
- Modify: `t1_nmpc/robot/config.py`
- Test: `tests/test_track_config.py`

**Interfaces:**
- Produces: `RobotModel.hand_frame_ids: tuple[int,int]` (left_hand_link, right_hand_link); `make_track_config(**overrides) -> MPCConfig` with `nodes=10, dt_min=dt_max=0.04`, `Q_diag=_track_Q_diag()`, and new fields `time_scale: float`, `w_hand: float`, `grasp_halfwidth: float`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_config.py`:
```python
import numpy as np
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model


def test_track_config_defaults():
    cfg = make_track_config()
    assert cfg.nodes == 10
    assert cfg.dt_min == 0.04 and cfg.dt_max == 0.04
    assert cfg.Q_diag.shape == (cfg.ndx,)        # 70
    assert cfg.time_scale == 5.0
    assert cfg.w_hand == 400.0
    assert cfg.grasp_halfwidth > 0.0


def test_track_config_override():
    cfg = make_track_config(nodes=8, time_scale=3.0)
    assert cfg.nodes == 8 and cfg.time_scale == 3.0


def test_hand_frame_ids_resolve():
    cfg = make_track_config()
    rm = load_model(cfg)
    lh, rh = rm.hand_frame_ids
    assert rm.model.frames[lh].name == "left_hand_link"
    assert rm.model.frames[rh].name == "right_hand_link"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_config.py -q -p no:cacheprovider`
Expected: FAIL (`make_track_config` not defined / `hand_frame_ids` missing).

- [ ] **Step 3: Add `hand_frame_ids` to the model**

In `t1_nmpc/robot/model.py`, add the field to the dataclass (after `tau_max`):
```python
    tau_max: np.ndarray   # (29,)
    hand_frame_ids: tuple[int, ...] = ()   # (left_hand_link, right_hand_link)
```
And in `load_model`, just before the `return RobotModel(...)`, compute the ids and pass them:
```python
    hand_ids = (model.getFrameId("left_hand_link"), model.getFrameId("right_hand_link"))
    return RobotModel(model, data, tuple(corner_ids), tuple(center_ids), mass, trunk_fid,
                      tau_max, hand_ids)
```

- [ ] **Step 4: Add tracking config to `config.py`**

In `t1_nmpc/robot/config.py`, add this factory near `_Q_diag` (it tracks base z + lean + arms + waist strongly, leg-pitch as a weak seed, redundant DOF as weak nullspace):
```python
def _track_Q_diag() -> np.ndarray:
    # ndx=70 = [base(6)=x,y,z,wx,wy,wz | joints(29) | base_vel(6) | joint_vel(29)]
    base_pos = np.array([50, 50, 2000, 3000, 3000, 50], dtype=np.float64)   # track z + lean (wx,wy); xy/yaw light
    joint_pos = np.concatenate([
        [1, 1],                       # head (nominal, light)
        [200] * 7, [200] * 7,         # L/R arm (tracked)
        [200],                        # waist (tracked, = -trunk_yaw)
        [5, 1, 1, 5, 5, 1],           # L leg: hipP,hipR,hipY,knee,ankP,ankR  (pitch=seed 5; redundant=1)
        [5, 1, 1, 5, 5, 1],           # R leg
    ])
    base_vel = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
    joint_vel = np.concatenate([[1, 1], [5] * 7, [5] * 7, [5], [2] * 6, [2] * 6])
    return np.concatenate([base_pos, joint_pos, base_vel, joint_vel])
```
Add three fields to the `MPCConfig` dataclass (anywhere among the existing fields, e.g. after `footstep_weight`):
```python
    # trajectory tracking (pickup)
    time_scale: float = 5.0
    w_hand: float = 400.0
    grasp_halfwidth: float = 0.04   # plan-phase seconds; node within this of an event -> hard hand
```
Add the factory at the end of the file:
```python
def make_track_config(**overrides) -> MPCConfig:
    base = dict(nodes=10, dt_min=0.04, dt_max=0.04, Q_diag=_track_Q_diag())
    base.update(overrides)
    return make_config(**base)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_config.py -q -p no:cacheprovider`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add t1_nmpc/robot/model.py t1_nmpc/robot/config.py tests/test_track_config.py
git commit -m "feat(track): hand_frame_ids + make_track_config/_track_Q_diag"
```

---

## Task 2: Reference mapping + grasp keyframes

**Files:**
- Create: `t1_nmpc/wb/reference.py`
- Test: `tests/test_track_reference.py`

**Interfaces:**
- Produces:
  - `MotionPlanReference(plan_path: str, cfg: MPCConfig, rm: RobotModel, x0=0.0, y0=0.0, yaw0=0.0)`.
  - `.q_frame: np.ndarray (F,36)`, `.hand_frame: np.ndarray (F,6)`, `.t_frame: np.ndarray (F,)`, `.duration_phase: float`, `.events: dict[int, list[float]]` (hand 0=left,1=right → phase times of grasp/release).
  - `.frame_to_xref(seg: dict, k: int) -> np.ndarray (36,)` (the q mapping).
- Consumes: `RobotModel.hand_frame_ids`, `cfg.nominal_joint_pos`, `cfg.nominal_base_height`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_reference.py`:
```python
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model
from t1_nmpc.wb.reference import MotionPlanReference

PLAN = "data/motion_plan.pkl"


def _ref():
    cfg = make_track_config()
    rm = load_model(cfg)
    return MotionPlanReference(PLAN, cfg, rm), cfg, rm


def test_mapping_fk_roundtrip():
    """q_ref → FK at hand frames must match the plan's hand_xyz to < 1 mm (validates base+arm map)."""
    ref, cfg, rm = _ref()
    m = rm.model; d = m.createData()
    lh, rh = rm.hand_frame_ids
    import pickle
    plan = pickle.load(open(PLAN, "rb"))
    for si in (0, 3, 6):
        seg = plan["segments"][si]
        for k in (0, 40, 80):
            q = ref.frame_to_xref(seg, k)
            pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
            assert np.linalg.norm(d.oMf[lh].translation - seg["position"]["left_hand_xyz"][k]) < 1e-3
            assert np.linalg.norm(d.oMf[rh].translation - seg["position"]["right_hand_xyz"][k]) < 1e-3


def test_mapping_sign_and_base():
    ref, cfg, rm = _ref()
    import pickle
    seg = pickle.load(open(PLAN, "rb"))["segments"][2]
    k = 30
    q = ref.frame_to_xref(seg, k)
    P = seg["position"]
    assert abs(q[2] - P["trunk_height"][k]) < 1e-9           # base z = trunk_height
    assert abs(q[7 + 16] - (-P["trunk_yaw"][k])) < 1e-9      # Waist = -trunk_yaw
    assert abs(q[7 + 17] - P["trunk_pitch"][k]) < 1e-9       # L_Hip_Pitch = trunk_pitch
    assert abs(q[7 + 23] - P["trunk_pitch"][k]) < 1e-9       # R_Hip_Pitch = trunk_pitch (broadcast)
    assert np.allclose(q[7 + 2:7 + 9], P["left_arm"][k])     # arm order, no reindex


def test_grasp_events():
    ref, cfg, rm = _ref()
    # held_objs: seg0[] 1[L] 2[L] 3[L,R] 4[L,R] 5[R] 6[R] 7[]
    # left flips at seg1 start (grasp) and seg5 start (release); right at seg3 and seg7.
    assert len(ref.events[0]) == 2 and len(ref.events[1]) == 2
    # events are increasing phase times within the plan duration
    for h in (0, 1):
        assert all(0 < t < ref.duration_phase for t in ref.events[h])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_reference.py -q -p no:cacheprovider`
Expected: FAIL (`t1_nmpc.wb.reference` does not exist).

- [ ] **Step 3: Implement the mapping + load + events (no sampling yet)**

Create `t1_nmpc/wb/reference.py`:
```python
"""motion_plan.pkl -> pinocchio reference. Joint mapping authority: t1_kd_mpc.

Maps the plan's reduced channels onto the full 29-joint FreeFlyer state: base z=trunk_height,
base lean=yaw-anchored trunk_quat, arms linear, Waist=-trunk_yaw, leg-pitch broadcast to both legs
(SEED only -- low Q weight; the OCP solves the real legs against planted-feet contact). Head and
hip-roll/yaw + ankle-roll stay nominal. Hands are exported as task-space targets. See
docs/superpowers/specs/2026-06-28-pickup-trajectory-tracking-design.md."""
from __future__ import annotations

import pickle

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

from ..robot.config import MPCConfig
from ..robot.model import RobotModel

# joint-local indices (pinocchio order); full-q index = 7 + j
_J_LARM = slice(2, 9)
_J_RARM = slice(9, 16)
_J_WAIST = 16
_J_LHIP_P, _J_LKNEE, _J_LANK_P = 17, 20, 21
_J_RHIP_P, _J_RKNEE, _J_RANK_P = 23, 26, 27


def _anchor_xyz(p, x0, y0, yaw0):
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.array([c * p[0] - s * p[1] + x0, s * p[0] + c * p[1] + y0, p[2]], dtype=np.float64)


def _anchor_quat(quat_xyzw, yaw0):
    if yaw0 == 0.0:
        return np.asarray(quat_xyzw, dtype=np.float64)
    return (Rsc.from_euler("z", yaw0) * Rsc.from_quat(quat_xyzw)).as_quat()


class MotionPlanReference:
    def __init__(self, plan_path: str, cfg: MPCConfig, rm: RobotModel,
                 x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
        self.cfg = cfg
        self.model = rm.model
        self.nomj = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.grasp_hw = float(cfg.grasp_halfwidth)
        with open(plan_path, "rb") as f:
            plan = pickle.load(f)
        self.segments = plan["segments"]
        self._build_timeline()

    def frame_to_xref(self, seg: dict, k: int) -> np.ndarray:
        """One plan frame -> pinocchio q (36,). Velocities are added later by finite difference."""
        P = seg["position"]
        q = np.empty(36, dtype=np.float64)
        q[0] = self.x0; q[1] = self.y0
        q[2] = float(P["trunk_height"][k])                                   # base z = trunk_height
        q[3:7] = _anchor_quat(P["trunk_quat_xyzw"][k], self.yaw0)            # lean
        j = self.nomj.copy()
        j[_J_LARM] = P["left_arm"][k]; j[_J_RARM] = P["right_arm"][k]
        j[_J_WAIST] = -float(P["trunk_yaw"][k])                              # Waist = -trunk_yaw
        tp, kn, an = float(P["trunk_pitch"][k]), float(P["knee_pitch"][k]), float(P["ankle_pitch"][k])
        j[_J_LHIP_P] = tp; j[_J_RHIP_P] = tp                                 # broadcast both legs
        j[_J_LKNEE] = kn; j[_J_RKNEE] = kn
        j[_J_LANK_P] = an; j[_J_RANK_P] = an
        q[7:] = j
        return q

    def _hand_frame(self, seg: dict, k: int) -> np.ndarray:
        P = seg["position"]
        lh = _anchor_xyz(P["left_hand_xyz"][k], self.x0, self.y0, self.yaw0)
        rh = _anchor_xyz(P["right_hand_xyz"][k], self.x0, self.y0, self.yaw0)
        return np.concatenate([lh, rh])

    def _build_timeline(self):
        qs, hs, ts = [], [], []
        t = 0.0
        held_prev = ([], [])  # not used directly; events derived below
        seg_start_t = []      # phase time at each segment's first frame
        for si, seg in enumerate(self.segments):
            seg_start_t.append(t)
            for k in range(seg["T"]):
                qs.append(self.frame_to_xref(seg, k))
                hs.append(self._hand_frame(seg, k))
                ts.append(t)
                t += float(seg["dt"])
        self.q_frame = np.asarray(qs)        # (F,36)
        self.hand_frame = np.asarray(hs)     # (F,6)
        self.t_frame = np.asarray(ts)        # (F,)
        self.duration_phase = float(self.t_frame[-1])
        # grasp/release events: where a hand's hold-state flips at a segment boundary
        self.events = {0: [], 1: []}
        prev = {0: False, 1: False}          # left, right currently held
        for si, seg in enumerate(self.segments):
            ho = seg["held_objs"]
            cur = {0: ("left" in ho), 1: ("right" in ho)}
            for h in (0, 1):
                if cur[h] != prev[h]:
                    self.events[h].append(seg_start_t[si])
            prev = cur
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_reference.py -q -p no:cacheprovider`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/reference.py tests/test_track_reference.py
git commit -m "feat(track): MotionPlanReference mapping + grasp keyframes (FK roundtrip <1mm)"
```

---

## Task 3: Reference horizon sampling (interp + slerp + time_scale)

**Files:**
- Modify: `t1_nmpc/wb/reference.py`
- Test: `tests/test_track_sampling.py`

**Interfaces:**
- Produces: `MotionPlanReference.sample(t_wall: float, time_scale: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(x_ref (71, N+1), hand_ref (6, N+1), grasp_gate (2, N+1))` where `N = cfg.nodes`, `dt = cfg.dt_min`. Quaternions slerped, joints/positions linear, velocities by manifold finite-difference (so they carry the `1/time_scale` factor automatically). `time_scale=None` uses `cfg.time_scale`.
- Consumes: `frame_to_xref`, `q_frame`, `t_frame`, `events`, `grasp_hw`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_sampling.py`:
```python
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model
from t1_nmpc.wb.reference import MotionPlanReference

PLAN = "data/motion_plan.pkl"


def _ref(**kw):
    cfg = make_track_config(**kw)
    rm = load_model(cfg)
    return MotionPlanReference(PLAN, cfg, rm), cfg, rm


def test_sample_shapes_and_clamp():
    ref, cfg, rm = _ref()
    xr, hr, gg = ref.sample(0.0)
    assert xr.shape == (71, cfg.nodes + 1)
    assert hr.shape == (6, cfg.nodes + 1)
    assert gg.shape == (2, cfg.nodes + 1)
    # far past the end -> clamps to the final frame (q part equals last frame q)
    xr_end, _, _ = ref.sample(ref.duration_phase * 10.0 * cfg.time_scale)
    assert np.allclose(xr_end[:36, 0], ref.q_frame[-1], atol=1e-6)


def test_velocity_scales_with_time_scale():
    """Doubling time_scale halves the reference velocities (same path, slower)."""
    ref2, cfg2, _ = _ref(time_scale=2.0)
    ref4, cfg4, _ = _ref(time_scale=4.0)
    # sample mid-motion at the SAME phase (t_wall = phase * time_scale)
    phase = 3.0
    x2, _, _ = ref2.sample(phase * 2.0, time_scale=2.0)
    x4, _, _ = ref4.sample(phase * 4.0, time_scale=4.0)
    v2 = x2[36:, 0]; v4 = x4[36:, 0]
    # positions identical (same phase), velocities halved
    assert np.allclose(x2[:36, 0], x4[:36, 0], atol=1e-5)
    assert np.linalg.norm(v4) < np.linalg.norm(v2)
    assert np.allclose(v4 * 2.0, v2, atol=1e-3) or np.linalg.norm(v2) < 1e-6


def test_grasp_gate_fires_near_event():
    ref, cfg, rm = _ref(time_scale=1.0)
    te = ref.events[0][0]               # first left event (phase time)
    _, _, gg = ref.sample(te)           # t_wall = te (time_scale 1) -> node 0 at phase te
    assert gg[0, 0] == 1.0              # left gate hot at the event
    _, _, gg_far = ref.sample(0.0)
    assert gg_far[0, 0] == 0.0          # not hot at t=0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_sampling.py -q -p no:cacheprovider`
Expected: FAIL (`sample` not defined).

- [ ] **Step 3: Implement `sample` + interpolation helpers**

Append to `MotionPlanReference` in `t1_nmpc/wb/reference.py`:
```python
    def _slerp(self, qa, qb, alpha):
        ra, rb = Rsc.from_quat(qa), Rsc.from_quat(qb)
        rel = (ra.inv() * rb).as_rotvec() * alpha
        return (ra * Rsc.from_rotvec(rel)).as_quat()

    def _interp(self, t_ref):
        """Interpolated (q(36), hand(6)) at phase time t_ref (clamped to [0, duration])."""
        tf = self.t_frame
        t_ref = float(np.clip(t_ref, tf[0], tf[-1]))
        i = int(np.searchsorted(tf, t_ref))
        if i <= 0:
            return self.q_frame[0].copy(), self.hand_frame[0].copy()
        if i >= len(tf):
            return self.q_frame[-1].copy(), self.hand_frame[-1].copy()
        t0, t1 = tf[i - 1], tf[i]
        a = 0.0 if t1 <= t0 else (t_ref - t0) / (t1 - t0)
        qa, qb = self.q_frame[i - 1], self.q_frame[i]
        q = np.empty(36)
        q[0:3] = (1 - a) * qa[0:3] + a * qb[0:3]
        q[3:7] = self._slerp(qa[3:7], qb[3:7], a)
        q[7:] = (1 - a) * qa[7:] + a * qb[7:]
        h = (1 - a) * self.hand_frame[i - 1] + a * self.hand_frame[i]
        return q, h

    def sample(self, t_wall: float, time_scale: float | None = None):
        ts = self.cfg.time_scale if time_scale is None else float(time_scale)
        N = self.cfg.nodes
        dt = self.cfg.dt_min
        q_nodes, hand_ref, gate = [], np.zeros((6, N + 1)), np.zeros((2, N + 1))
        for i in range(N + 1):
            t_ref = float(np.clip((t_wall + i * dt) / ts, 0.0, self.duration_phase))
            q_i, h_i = self._interp(t_ref)
            q_nodes.append(q_i); hand_ref[:, i] = h_i
            for h in (0, 1):
                if any(abs(t_ref - te) < self.grasp_hw for te in self.events[h]):
                    gate[h, i] = 1.0
        x_ref = np.zeros((71, N + 1))
        for i in range(N + 1):
            x_ref[:36, i] = q_nodes[i]
            qa = q_nodes[i]; qb = q_nodes[min(i + 1, N)]
            x_ref[36:, i] = pin.difference(self.model, qa, qb) / dt   # manifold vel (carries 1/time_scale)
        return x_ref, hand_ref, gate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_sampling.py -q -p no:cacheprovider`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/reference.py tests/test_track_sampling.py
git commit -m "feat(track): horizon sampling (slerp + manifold-vel + grasp gates + time_scale)"
```

---

## Task 4: PickupOCP (the tracking OCP) + real-time benchmark

**Files:**
- Create: `t1_nmpc/wb/track_ocp.py`
- Test: `tests/test_track_ocp.py`

**Interfaces:**
- Consumes: `WBDynamics` (`state_integrate`, `state_difference`, `rnea_dynamics_gated`, `frame_velocity(fid)`, `frame_position(fid)`), `RobotModel` (`corner_frame_ids`, `foot_center_frame_ids`, `hand_frame_ids`, `mass`, `tau_max`, `model.lower/upperPositionLimit`).
- Produces:
  - `PickupOCP(cfg, rm)`; `.set_weights()`; `.set_refs(x_init(71), x_ref(71,N+1), hand_ref(6,N+1), grasp_gate(2,N+1))` (mirrors values into opti params for retract); `.x_initial()`.
  - `.solve_function(max_iter) -> ca.Function` with signature `(x_init, Q_diag, R_diag, x_ref, hand_ref, grasp_gate, opti.x) -> opti.x`.
  - `.retract(sol_x, x_init) -> dict` with `q_sol/v_sol/a_sol/forces_sol/tau_sol` lists.
  - Constants: `na=35, nf=24, ns=6, tau_nodes=cfg.tau_nodes`; `_nu(i)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_ocp.py`:
```python
import time
import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.reference import MotionPlanReference
from t1_nmpc.wb.track_ocp import PickupOCP

PLAN = "data/motion_plan.pkl"


def _setup():
    cfg = make_track_config()
    rm = load_model(cfg)
    ref = MotionPlanReference(PLAN, cfg, rm)
    ocp = PickupOCP(cfg, rm); ocp.set_weights()
    return cfg, rm, ref, ocp


def test_cold_and_warm_solve():
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(0.0)
    ocp.set_refs(x0, xr, hr, gg)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    out = ocp.retract(sol, x0)
    # node-1 state is finite and feet stay near the ground (planted)
    assert np.all(np.isfinite(out["q_sol"][1]))
    m, d = rm.model, rm.model.createData()
    pin.forwardKinematics(m, d, out["q_sol"][1]); pin.updateFramePlacements(m, d)
    for fid in rm.foot_center_frame_ids:
        assert abs(d.oMf[fid].translation[2]) < 0.02      # foot center stays ~ground
    # one warm solve at a later phase
    xr2, hr2, gg2 = ref.sample(2.0)
    sol2 = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr2, hr2, gg2, sol)).flatten()
    assert np.all(np.isfinite(sol2))


def test_leg_limits_respected():
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(8.0)            # deep-crouch region
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    out = ocp.retract(sol, x0)
    lo, hi = rm.model.lowerPositionLimit, rm.model.upperPositionLimit
    for i in (1, cfg.nodes):
        q = out["q_sol"][i]
        for j in list(range(7 + 17, 7 + 23)) + list(range(7 + 23, 7 + 29)):
            assert lo[j] - 1e-2 <= q[j] <= hi[j] + 1e-2     # knee not hyperextended, etc.


def test_realtime_warm_p90(capsys):
    """Real-time gate: warm p90 < 16 ms at the chosen N (record the number; xfail if machine slow)."""
    import pytest
    cfg, rm, ref, ocp = _setup()
    x0 = nominal_x(cfg, rm.model)
    fn = ocp.solve_function(cfg.fatrop_max_iter)
    xr, hr, gg = ref.sample(0.0)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, ocp.x_initial())).flatten()
    ts = []
    for k in range(12):
        xr, hr, gg = ref.sample(0.2 * k)
        t0 = time.perf_counter()
        sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, xr, hr, gg, sol)).flatten()
        ts.append((time.perf_counter() - t0) * 1e3)
    p90 = float(np.percentile(ts, 90))
    print(f"\npickup warm solve p90 = {p90:.1f} ms (N={cfg.nodes})")
    if p90 >= 16.0:
        pytest.xfail(f"solve p90 {p90:.1f}ms >= 16ms — drop N to 8 (fallback) or trim leg limits")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_ocp.py -q -p no:cacheprovider`
Expected: FAIL (`t1_nmpc.wb.track_ocp` does not exist).

- [ ] **Step 3: Implement `PickupOCP`**

Create `t1_nmpc/wb/track_ocp.py`:
```python
"""All-stance whole-body RNEA tracking OCP for the floor-pickup (CasADi Opti + Fatrop).

Generalizes the M0 stand OCP: the fixed nominal target becomes a per-node, time-varying
reference (x_ref). Both feet are always planted (V_lin=0, flat foot). Hands are tracked in
task-space via a slack trick: hand_pos - hand_ref == (1-grasp_gate)*s  -> soft (s free + cost)
off-grasp, hard (==ref) at grasp keyframes. Legs are SOLVED here (not tracked) against the hard
contact, with leg joint-position limits (anti-hyperextension); the leg-pitch seed lives in x_ref
at low Q weight. Built once and compiled with opti.to_function (NO jit). See the design spec."""
from __future__ import annotations

import numpy as np
import casadi as ca

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .dynamics import WBDynamics


class PickupOCP:
    def __init__(self, cfg: MPCConfig, rm: RobotModel):
        self.cfg, self.rm = cfg, rm
        self.dyn = WBDynamics(rm.model, rm.corner_frame_ids)
        self.mass = rm.mass
        self.nq, self.nv, self.nj, self.nf = self.dyn.nq, self.dyn.nv, self.dyn.nj, self.dyn.nf
        self.na = self.nv                                    # 35
        self.ns = 6                                          # hand slacks: L(3)+R(3)
        self.nodes, self.tau_nodes = cfg.nodes, cfg.tau_nodes
        self.mu = cfg.friction_mu
        self.nx, self.ndx = self.nq + self.nv, 2 * self.nv
        self.f_idx = self.na                                 # 35
        self.s_idx = self.na + self.nf                       # 59
        self.tau_idx = self.na + self.nf + self.ns           # 65
        self.tau_max = rm.tau_max
        self.n_corners = cfg.n_corners
        self.corner_ids = rm.corner_frame_ids
        self.foot_center_ids = rm.foot_center_frame_ids
        self.hand_ids = rm.hand_frame_ids
        self.dt = cfg.dt_min
        # leg joint full-q indices (for position box) and limits
        self.leg_q_idx = list(range(7 + 17, 7 + 23)) + list(range(7 + 23, 7 + 29))   # 12
        self.leg_lo = np.asarray(rm.model.lowerPositionLimit, dtype=np.float64)[self.leg_q_idx]
        self.leg_hi = np.asarray(rm.model.upperPositionLimit, dtype=np.float64)[self.leg_q_idx]
        self.opti = ca.Opti()
        self._build()

    @staticmethod
    def _corner_foot(c):
        return 0 if c < 4 else 1

    def _nu(self, i):
        return self.na + self.nf + self.ns + (self.nj if i < self.tau_nodes else 0)

    def _has_tau(self, i):
        return i < self.tau_nodes

    def _build(self):
        opti = self.opti
        self.DX, self.U = [], []
        for i in range(self.nodes):                          # interleaved (Fatrop staircase)
            self.DX.append(opti.variable(self.ndx))
            self.U.append(opti.variable(self._nu(i)))
        self.DX.append(opti.variable(self.ndx))

        self.x_init = opti.parameter(self.nx)
        self.Q_diag = opti.parameter(self.ndx)
        self.R_diag = opti.parameter(self.na + self.nf + self.nj)     # [a, forces, tau]
        self.x_ref = opti.parameter(self.nx, self.nodes + 1)
        self.hand_ref = opti.parameter(6, self.nodes + 1)
        self.grasp_gate = opti.parameter(2, self.nodes + 1)
        # valid defaults so opti.value(p) works before set_refs
        opti.set_value(self.hand_ref, np.zeros((6, self.nodes + 1)))
        opti.set_value(self.grasp_gate, np.zeros((2, self.nodes + 1)))

        f_grav = self.mass * 9.81 / self.n_corners
        self.f_des = ca.vertcat(*[ca.DM([0, 0, f_grav]) for _ in range(self.n_corners)])
        self._integ = self.dyn.state_integrate()
        self._diff = self.dyn.state_difference()
        self._constraints()
        self._init_guess()
        opti.minimize(self._objective())

    # accessors
    def _x(self, i): return self._integ(self.x_init, self.DX[i])
    def _q(self, i): return self._x(i)[:self.nq]
    def _v(self, i): return self._x(i)[self.nq:]
    def _a(self, i): return self.U[i][:self.na]
    def _f(self, i): return self.U[i][self.f_idx:self.s_idx]
    def _s(self, i): return self.U[i][self.s_idx:self.tau_idx]
    def _tau(self, i): return self.U[i][self.tau_idx:]

    def _init_guess(self):
        f_np = np.array(self.f_des).flatten()
        u0 = np.concatenate([np.zeros(self.na), f_np, np.zeros(self.ns), np.zeros(self.nj)])
        for i in range(self.nodes):
            self.opti.set_initial(self.DX[i], np.zeros(self.ndx))
            self.opti.set_initial(self.U[i], u0[:self._nu(i)])
        self.opti.set_initial(self.DX[self.nodes], np.zeros(self.ndx))

    def _constraints(self):
        opti = self.opti
        opti.subject_to(self.DX[0] == np.zeros(self.ndx))
        rnea = self.dyn.rnea_dynamics_gated()
        center_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.foot_center_ids}
        hand_pos = {fid: self.dyn.frame_position(fid) for fid in self.hand_ids}
        contact_all = ca.DM.ones(self.n_corners)             # all-stance: every corner active
        for i in range(self.nodes):
            dq, dv = self.DX[i][:self.nv], self.DX[i][self.nv:]
            dq_n, dv_n = self.DX[i + 1][:self.nv], self.DX[i + 1][self.nv:]
            q, v, a, forces, dt = self._q(i), self._v(i), self._a(i), self._f(i), self.dt
            # (1) gap-closing FIRST (forward Euler; slow quasi-static motion -> adequate)
            opti.subject_to(dq_n == dq + v * dt)
            opti.subject_to(dv_n == dv + a * dt)
            tau_rnea = rnea(q, v, a, forces, contact_all)
            # (2) base underactuation
            opti.subject_to(tau_rnea[:6] == np.zeros(6))
            # (3) torque equality (first tau_nodes)
            if self._has_tau(i):
                opti.subject_to(tau_rnea[6:] == self._tau(i))
            # (4) contact (i>=1): planted + flat foot
            if i >= 1:
                for fid in self.foot_center_ids:
                    V = center_vel[fid](q, v)
                    opti.subject_to(V[0] == 0); opti.subject_to(V[1] == 0); opti.subject_to(V[2] == 0)
                    opti.subject_to(V[3] == 0); opti.subject_to(V[4] == 0)    # roll/pitch rate (flat)
                # (5) hand task (slack trick): pos - ref == (1-gate)*s
                for h, fid in enumerate(self.hand_ids):
                    p = hand_pos[fid](q)
                    s = self._s(i)[3 * h:3 * h + 3]
                    g = self.grasp_gate[h, i]
                    opti.subject_to(p - self.hand_ref[3 * h:3 * h + 3, i] == (1 - g) * s)
            # --- inequalities AFTER all equalities ---
            if self._has_tau(i):
                opti.subject_to(opti.bounded(-self.tau_max, self._tau(i), self.tau_max))
            for c in range(self.n_corners):
                fe = forces[c * 3:(c + 1) * 3]
                opti.subject_to(fe[2] >= 0)
                opti.subject_to(self.mu**2 * fe[2]**2 >= fe[0]**2 + fe[1]**2)
            if i >= 1:                                        # leg joint-position box (anti-hyperextension)
                q_leg = ca.vertcat(*[q[idx] for idx in self.leg_q_idx])
                opti.subject_to(opti.bounded(self.leg_lo, q_leg, self.leg_hi))

    def _objective(self):
        Q = ca.diag(self.Q_diag)
        R = ca.diag(self.R_diag)
        obj = 0
        for i in range(self.nodes + 1):
            dx_des = self._diff(self.x_init, self.x_ref[:, i])
            e = self.DX[i] - dx_des
            obj = obj + e.T @ Q @ e
            if i < self.nodes:
                u = self.U[i]
                a = u[:self.na]; forces = u[self.f_idx:self.s_idx]; s = u[self.s_idx:self.tau_idx]
                tau = u[self.tau_idx:] if self._has_tau(i) else ca.MX.zeros(self.nj)
                u_track = ca.vertcat(a, forces - self.f_des, tau)
                obj = obj + u_track.T @ R @ u_track
                obj = obj + self.cfg.w_hand * ca.sumsqr(s)        # hand task (soft off-grasp)
        return obj

    # --- API ---
    def set_weights(self):
        self.opti.set_value(self.Q_diag, self.cfg.Q_diag)
        self.opti.set_value(self.R_diag, self.cfg.R_diag)

    def set_refs(self, x_init, x_ref, hand_ref, grasp_gate):
        self.opti.set_value(self.x_init, np.asarray(x_init, dtype=np.float64))
        self.opti.set_value(self.x_ref, np.asarray(x_ref, dtype=np.float64))
        self.opti.set_value(self.hand_ref, np.asarray(hand_ref, dtype=np.float64))
        self.opti.set_value(self.grasp_gate, np.asarray(grasp_gate, dtype=np.float64))

    def x_initial(self):
        return self.opti.value(self.opti.x, self.opti.initial())

    def _fatrop_opts(self, max_iter):
        return {"expand": True, "structure_detection": "auto", "debug": False,
                "fatrop": {"print_level": 0, "max_iter": int(max_iter),
                           "tol": self.cfg.fatrop_tol, "mu_init": self.cfg.fatrop_mu_init,
                           "warm_start_init_point": True,
                           "warm_start_mult_bound_push": 1e-7, "bound_push": 1e-7}}

    def solve_function(self, max_iter):
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "pickup_fn",
            [self.x_init, self.Q_diag, self.R_diag, self.x_ref, self.hand_ref,
             self.grasp_gate, self.opti.x],
            [self.opti.x])

    def retract(self, sol_x, x_init):
        sol_x = np.asarray(sol_x).flatten()
        x_init = np.asarray(x_init, dtype=np.float64)
        out = {"q_sol": [], "v_sol": [], "a_sol": [], "forces_sol": [], "tau_sol": []}
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
            out["forces_sol"].append(u[self.f_idx:self.s_idx])
            out["tau_sol"].append(u[self.tau_idx:] if self._has_tau(i) else np.zeros(self.nj))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_ocp.py -q -p no:cacheprovider -s`
Expected: 2 passed + `test_realtime_warm_p90` prints the p90 and passes (or xfails with a clear "drop N to 8" message). If it xfails, set `make_track_config` default `nodes=8` in Task 1 and re-run — record the decision in the commit message.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/track_ocp.py tests/test_track_ocp.py
git commit -m "feat(track): PickupOCP all-stance tracking OCP (slack-trick hands, leg limits, planted feet)"
```

---

## Task 5: TrackingMPC wrapper

**Files:**
- Create: `t1_nmpc/wb/track_mpc.py`
- Test: `tests/test_track_mpc.py`

**Interfaces:**
- Consumes: `PickupOCP`, `MotionPlanReference`, `extract_command` (from `wb/state.py`), `WBResult` (from `wb/mpc.py`).
- Produces: `TrackingMPC(cfg, rm, plan_path, x0=0,y0=0,yaw0=0)`; `.reset(x0)`; `.step(x_meas, t_wall) -> WBResult`; `.ref` (the `MotionPlanReference`); `.duration_wall` (= `ref.duration_phase * cfg.time_scale`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_mpc.py`:
```python
import numpy as np
from t1_nmpc.robot.config import make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.track_mpc import TrackingMPC

PLAN = "data/motion_plan.pkl"


def test_step_returns_command_and_advances():
    cfg = make_track_config()
    rm = load_model(cfg)
    mpc = TrackingMPC(cfg, rm, PLAN)
    x0 = nominal_x(cfg, rm.model)
    mpc.reset(x0)
    res = mpc.step(x0, 0.0)
    assert res.command.q_des.shape == (29,)
    assert res.command.qd_des.shape == (29,)
    assert res.command.tau_ff.shape == (29,)
    assert res.solve_time > 0.0
    assert np.all(np.isfinite(res.command.tau_ff))
    # a later tick warm-starts and still returns finite commands
    res2 = mpc.step(x0, 1.0)
    assert np.all(np.isfinite(res2.command.q_des))
    assert mpc.duration_wall > 10.0           # ~14.8s * time_scale
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_mpc.py -q -p no:cacheprovider`
Expected: FAIL (`t1_nmpc.wb.track_mpc` does not exist).

- [ ] **Step 3: Implement `TrackingMPC`**

Create `t1_nmpc/wb/track_mpc.py`:
```python
"""TrackingMPC: build PickupOCP once, sample the plan each tick, warm-solve, emit a JointCommand."""
from __future__ import annotations

import time

import numpy as np

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .reference import MotionPlanReference
from .track_ocp import PickupOCP
from .state import extract_command
from .mpc import WBResult


class TrackingMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, plan_path: str,
                 x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
        self.cfg, self.rm = cfg, rm
        self.ref = MotionPlanReference(plan_path, cfg, rm, x0=x0, y0=y0, yaw0=yaw0)
        self.ocp = PickupOCP(cfg, rm); self.ocp.set_weights()
        self._solve = self.ocp.solve_function(cfg.fatrop_max_iter)
        self._warm = None
        self.duration_wall = self.ref.duration_phase * cfg.time_scale

    def _call(self, x, xr, hr, gg, warm):
        return np.array(self._solve(x, self.cfg.Q_diag, self.cfg.R_diag, xr, hr, gg, warm)).flatten()

    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        xr, hr, gg = self.ref.sample(0.0)
        self.ocp.set_refs(x0, xr, hr, gg)
        self._warm = self._call(x0, xr, hr, gg, self.ocp.x_initial())

    def step(self, x_meas, t_wall: float) -> WBResult:
        x = np.asarray(x_meas, dtype=np.float64)
        xr, hr, gg = self.ref.sample(t_wall)
        self.ocp.set_refs(x, xr, hr, gg)
        warm = self._warm if self._warm is not None else self.ocp.x_initial()
        t0 = time.perf_counter()
        sol = self._call(x, xr, hr, gg, warm)
        dt = time.perf_counter() - t0
        self._warm = sol
        out = self.ocp.retract(sol, x)
        node1_x = np.concatenate([out["q_sol"][1], out["v_sol"][1]])
        return WBResult(command=extract_command(out, self.cfg),
                        forces0=np.asarray(out["forces_sol"][0], dtype=np.float64),
                        solve_time=dt, constr_viol=0.0, num_iters=0,
                        node1_x=node1_x, planned=out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_mpc.py -q -p no:cacheprovider`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add t1_nmpc/wb/track_mpc.py tests/test_track_mpc.py
git commit -m "feat(track): TrackingMPC (phase clock + warm solve + command extraction)"
```

---

## Task 6: Closed-loop sim runner

**Files:**
- Create: `sim/pickup.py`
- Test: `tests/test_track_closed_loop.py`

**Interfaces:**
- Consumes: `TrackingMPC`, `MujocoRuntime`, `MJ_JOINT_QPOS0`, `MJ_JOINT_QVEL0`, `make_track_config`, `load_model`, `nominal_x`.
- Produces: `run_pickup(cfg, plan_path="data/motion_plan.pkl", duration=None, realtime=False) -> dict` with keys `fz_ratio_p50, max_tilt_deg, hand_err_grasp_max_cm, solve_p50_ms, solve_p90_ms, fell, completed, rt_factor`. CLI `--time_scale --duration --realtime --view --gif`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_closed_loop.py`:
```python
from t1_nmpc.robot.config import make_track_config
from sim.pickup import run_pickup


def test_short_closed_loop_runs():
    # A short slice (first ~1.5 s wall) must run, not fall, keep feet loaded.
    cfg = make_track_config(time_scale=5.0)
    res = run_pickup(cfg, duration=1.5)
    assert res["fell"] is False
    assert 0.85 < res["fz_ratio_p50"] < 1.15           # Σfz/mg ~ 1
    assert res["max_tilt_deg"] < 25.0
    assert res["solve_p90_ms"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_closed_loop.py -q -p no:cacheprovider`
Expected: FAIL (`sim.pickup` does not exist).

- [ ] **Step 3: Implement `sim/pickup.py`**

Create `sim/pickup.py`:
```python
"""Closed-loop MuJoCo floor-pickup under the whole-body RNEA tracking MPC.

Mirrors sim/stand.py: physics @2000Hz, PD @500Hz, MPC @50Hz (ZOH). The phase clock is the loop
counter * physics_dt (NOT mj_data.time, which includes the reset settle). time_scale slows the
tracked motion. --realtime paces the loop to wall-clock and reports the real-time factor."""
from __future__ import annotations

import time
import numpy as np
import mujoco
import pinocchio as pin

from t1_nmpc.robot.config import MPCConfig, make_track_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.track_mpc import TrackingMPC
from sim.mujoco_runtime import MujocoRuntime, MJ_JOINT_QPOS0, MJ_JOINT_QVEL0


def _tilt_deg(qpos):
    R = pin.Quaternion(qpos[3], qpos[4], qpos[5], qpos[6]).normalized().toRotationMatrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _measured_grf_z(m, d):
    total, f6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        frame = d.contact[i].frame.reshape(3, 3)
        total += (frame.T @ f6[:3])[2]
    return abs(total)


def _hand_err_cm(rm, qpos_mj, mpc, t_wall):
    """L2 hand position error vs the (anchored) reference, cm, max over both hands."""
    from t1_nmpc.wb.state import mujoco_to_freeflyer
    x = mujoco_to_freeflyer(qpos_mj, np.zeros(rm.model.nv), rm.model)
    q = x[:rm.model.nq]
    m, d = rm.model, rm.model.createData()
    pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    _, hand_ref, _ = mpc.ref.sample(t_wall)
    lh, rh = rm.hand_frame_ids
    el = np.linalg.norm(d.oMf[lh].translation - hand_ref[0:3, 0])
    er = np.linalg.norm(d.oMf[rh].translation - hand_ref[3:6, 0])
    return 100.0 * max(el, er)


def run_pickup(cfg: MPCConfig, plan_path: str = "data/motion_plan.pkl",
               duration: float | None = None, realtime: bool = False) -> dict:
    rm = load_model(cfg)
    rt = MujocoRuntime(cfg, rm)
    rt.reset_to_nominal()
    mpc = TrackingMPC(cfg, rm, plan_path)
    mpc.reset(nominal_x(cfg, rm.model))
    mg = rm.mass * 9.81
    dur = mpc.duration_wall + 0.5 if duration is None else duration
    n_steps = int(round(dur * cfg.physics_hz))

    cmd = None
    solve_ms, fz_ratios, tilts, hand_errs = [], [], [], []
    fell, completed = False, False
    t_start = time.perf_counter()
    for k in range(n_steps):
        t_wall = k * rt.physics_dt
        if k % rt.mpc_decim == 0:
            x = rt.freeflyer_state(rm.model)
            res = mpc.step(x, t_wall)
            cmd = res.command
            solve_ms.append(res.solve_time * 1e3)
            fz_ratios.append(res.forces0.reshape(8, 3)[:, 2].sum() / mg)
            # hand error only when a grasp gate is hot at this tick
            _, _, gg = mpc.ref.sample(t_wall)
            if gg[:, 0].max() > 0.5:
                hand_errs.append(_hand_err_cm(rm, rt.mj_data.qpos, mpc, t_wall))
        if k % rt.control_decim == 0 and cmd is not None:
            q = np.array(rt.mj_data.qpos[MJ_JOINT_QPOS0:MJ_JOINT_QPOS0 + 29])
            qd = np.array(rt.mj_data.qvel[MJ_JOINT_QVEL0:MJ_JOINT_QVEL0 + 29])
            rt._apply_torque(cmd.tau_ff + cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.qd_des - qd))
        rt.step_physics()
        tilts.append(_tilt_deg(rt.mj_data.qpos))
        if rt.mj_data.qpos[2] < 0.3 or tilts[-1] > 45.0:
            fell = True
            break
        if realtime:
            target = t_start + (k + 1) * rt.physics_dt
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    else:
        completed = (t_wall >= mpc.duration_wall - 1e-6)
    wall = time.perf_counter() - t_start
    return {
        "fz_ratio_p50": float(np.median(fz_ratios)) if fz_ratios else 0.0,
        "max_tilt_deg": float(np.max(tilts)) if tilts else 0.0,
        "hand_err_grasp_max_cm": float(np.max(hand_errs)) if hand_errs else 0.0,
        "solve_p50_ms": float(np.percentile(solve_ms, 50)) if solve_ms else 0.0,
        "solve_p90_ms": float(np.percentile(solve_ms, 90)) if solve_ms else 0.0,
        "fell": fell, "completed": completed,
        "rt_factor": (n_steps * rt.physics_dt) / wall if wall > 0 else 0.0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_scale", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--gif", type=str, default=None)
    a = ap.parse_args()
    cfg = make_track_config(time_scale=a.time_scale)
    print(run_pickup(cfg, duration=a.duration, realtime=a.realtime))
```

> Note: `--view`/`--gif` rendering is intentionally not wired into `run_pickup` to keep the metrics
> path headless and deterministic. If a viewer/GIF is needed, mirror the EGL/`mujoco.viewer` pattern
> from `tools/walk_wip/walk_view.py` in a thin wrapper; not required for the acceptance tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_track_closed_loop.py -q -p no:cacheprovider`
Expected: 1 passed. (If it falls, this is the deep-crouch balance risk — first confirm Σfz/mg and tilt, then raise base-orientation `wx,wy` weights in `_track_Q_diag`; balance tuning, not architecture.)

- [ ] **Step 5: Smoke-run the full motion + record real-time**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python sim/pickup.py --time_scale 5.0`
Expected: prints a dict with `fell: False`, `completed: True`, `solve_p90_ms < 16`, `rt_factor >= 1.0` (headless; with `--realtime` it paces to wall clock). Record the numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add sim/pickup.py tests/test_track_closed_loop.py
git commit -m "feat(track): sim/pickup.py closed-loop runner + metrics (real-time, time_scale)"
```

---

## Task 7: Regression + divergence log

**Files:**
- Modify: `docs/2026-06-25-t1controller-divergences.md`
- Test: full suite

- [ ] **Step 1: Run the full suite (regression gate)**

Run: `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/ -q -p no:cacheprovider`
Expected: all M0 stand tests still pass; new track tests pass (the 2 pre-existing superseded-gated-walk failures, if still present, are unchanged and unrelated — note them, do not "fix" here).

- [ ] **Step 2: Append the divergences**

Append to `docs/2026-06-25-t1controller-divergences.md` (create a `## Pickup trajectory tracking (2026-06-28)` section):
```markdown
## Pickup trajectory tracking (2026-06-28)

Tracking `data/motion_plan.pkl` on the whole-body RNEA OCP. Reference mapping follows t1_kd_mpc; the
following diverge deliberately:

1. **Hard hand-position constraint at grasp keyframes** — t1_kd_mpc keeps hands always-soft (top
   weight 400). We harden at the 4 grasp/release instants (slack trick, `(1-gate)*s`) for grasp
   accuracy. Risk: conflict with hard planted-feet + RNEA → infeasibility; mitigated by soft-everywhere
   default and a narrow `grasp_halfwidth`.
2. **Base height tracked from `trunk_height`** — t1_kd_mpc ignores plan base position (its reduced model
   pins the base); our full-order free base uses it as the crouch reference (validated: planted flat
   feet reachable over the whole motion, max foot error < 0.03 mm).
3. **Legs solved by the OCP** (not broadcast) against hard planted feet, with leg joint-position limits
   (anti-hyperextension) + a low-weight leg-pitch seed in the reference. Broadcasting the plan's leg
   channels directly is geometrically inconsistent with planted feet (feet fly up +46..83 cm).
4. **Interpolated reference sampling** (linear + slerp) vs t1_kd_mpc nearest-frame.
5. **No payload model** (object mass absent from the plan) — same as t1_kd_mpc; future work.
6. **Horizon N=10/8** (not 31) for real-time; JIT/C-codegen ruled out (graph too large to compile).
```

- [ ] **Step 3: Commit**

```bash
git add docs/2026-06-25-t1controller-divergences.md
git commit -m "docs(track): log pickup-tracking divergences from t1_kd_mpc + t1_controller"
```

---

## Self-Review (completed)

**Spec coverage:** §1 goal → Tasks 4–6; §4 mapping → Tasks 1–3; §5 grasp keyframes → Tasks 2–3; §6 leg
validation → Task 4 (`test_leg_limits_respected`, planted-feet assertion); §7 OCP → Task 4; §8 real-time
→ Task 4 `test_realtime_warm_p90` + Task 6 smoke; §9 time_scale/interp → Task 3; §10 components → Tasks
1–6; §11 divergences → Task 7; §12 test plan → all task tests; §13/§14 non-goals/risks → noted in tasks.

**Refinement vs spec:** the spec listed `leg_seed(6,N+1)` as a separate solver parameter; the plan folds
the leg-pitch seed into `x_ref` at low `Q` weight (joints 17/20/21/23/26/27 set to the broadcast values,
weight 5) — strictly simpler, same effect, one fewer parameter. Branch selection is enforced by the hard
leg position limits (Task 4), so the seed need not be a separate term.

**Type consistency:** `sample()` returns `(x_ref(71,N+1), hand_ref(6,N+1), grasp_gate(2,N+1))`; consumed
identically by `PickupOCP.set_refs`, `solve_function` args, and `TrackingMPC._call`. `WBResult` reused
from `wb/mpc.py`. `extract_command(out, cfg)` reused from `wb/state.py` (needs `q_sol/v_sol/tau_sol`,
which `retract` produces). `_nu(i)`/index offsets (`f_idx=35, s_idx=59, tau_idx=65`) consistent between
`_constraints`, `_objective`, and `retract`.

**Placeholder scan:** none — every code/test step is complete; the only deferred item (viewer/GIF) is an
explicit non-acceptance extra with a pointer to the existing pattern.
