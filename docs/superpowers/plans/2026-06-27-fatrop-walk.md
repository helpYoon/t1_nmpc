# Fatrop forward-walk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the proven Fatrop `whole_body_rnea` M0 stand to a **stable closed-loop forward biped walk** for Booster T1 (≥0.5 m over ≥5 s, feet alternating with confirmed lift, bounded lateral drift, watchable in `--view`).

**Architecture:** Keep the M0 backend unchanged (CasADi `Opti` → Fatrop, whole_body_rnea inverse-dynamics OCP, 8-corner 3D contact, full 29-joint model). Add walking by making the **contact/swing schedule `opti.parameter`s** and writing every node's contact constraints as **flag-gated residuals** (one fixed-structure NLP, warm-started, schedule set per tick) — exactly the `wb-mpc-locoman` reference pattern. New: gait schedule, swing-z velocity spline, per-foot swing-z frame, Raibert footstep cost, walk runner.

**Tech Stack:** Python 3.10 conda env `t1mpc` (pinocchio 4.0, casadi 3.7.2 + Fatrop, numpy 2.2, mujoco 3.10); `pinocchio.casadi`; pytest.

## Global Constraints

Apply to **every** task; each task's requirements implicitly include this section.

- **Run preamble (load-bearing):** run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc` with `PYTHONPATH=` empty:
  `PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python <args>`.
- **pytest:** `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest <path> -q -p no:cacheprovider`.
- **Branch:** `wb-fatrop-walk` (already created from the M0 stand point `7135a0f`). Do NOT work on `main` or `wb-rnea-port`.
- **Dimensions (unchanged from M0):** 29 joints, `nq=36`, `nv=35`, `nx=71`, `ndx=70`, `n_corners=8`, `nf=24`, `na=35`, `tau_nodes=3`. Input `u_i=[a(35),forces(24),τ_j(29)]` (88) for `i<tau_nodes` else `[a(35),forces(24)]` (59).
- **Discretization (this plan, follows t1_controller):** uniform `dt=0.035`, `nodes=31` (horizon 1.085 s). Set `dt_min=dt_max=0.035` so the existing geometric-grid formula yields a uniform grid (γ=1). Verified: stand solves at this grid (CV 1.4e-7, warm ~49 ms).
- **Gait (t1_controller `gait.info`):** cycle 1.4 s; `LF-swing [0,0.6) → double [0.6,0.7) → RF-swing [0.7,1.3) → double [1.3,1.4)`; swing 0.6 s; `swing_height=0.08`, `v_liftoff=+0.05`, `v_touchdown=−0.05`.
- **Contact (8-corner, gated):** per corner `c` of foot `f`: stance friction `in_contact_f·f_{c,z}≥0`, `in_contact_f·μ²f_{c,z}² ≥ in_contact_f·(f_{c,x}²+f_{c,y}²)`; swing `(1−in_contact_f)·f_c==0`. μ=0.4.
- **Velocity constraints (gated, node-0 SKIPPED):** for `i≥1`: stance = each of the foot's 4 corner frames `in_contact_f·V_corner(q,v)[:3]==0` (M0's pin, gated); swing = the foot-center frame `(1−in_contact_f)·(V_footcenter(q,v)[2] − v_z^ref)==0`. **Node 0 carries NO velocity constraint** (initial velocity is fixed by `x_init`; matches reference `ocp.py:162-165`).
- **Fatrop gap-closing-first invariant:** at each stage the state-transition (gap-closing) equality `dq_n==dq+v·dt`, `dv_n==dv+a·dt` must be added BEFORE any inequality. Maintain this ordering (M0 already does).
- **Keep M0 green:** after every task, the stand path must still pass. Stand = all contact flags 1, all swing phases 0.
- **TDD + frequent commits.** Each task ends with an independently testable deliverable + a commit. Trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Spec:** `docs/superpowers/specs/2026-06-27-m1-fatrop-walk-design.md`.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `t1_nmpc/robot/config.py` | modify | uniform `dt=0.035`/`nodes=31`; add walk gait params + footstep gains + `base_vel_des` default. |
| `t1_nmpc/wb/spline.py` | **new** | `CubicSpline` + `get_spline_vel_z` (ported from `wb-mpc-locoman/utils/gait_sequence.py`). |
| `t1_nmpc/robot/model.py` | modify | add one **foot-center frame per foot** (keep the 8 corner frames). |
| `t1_nmpc/wb/gait.py` | modify | `WalkGait`: `schedules(t0) -> (contact 2×N, swing 2×N)` sliding/periodic; keep `StandGait`. |
| `t1_nmpc/wb/ocp.py` | modify | generalize `StandOCP`→`WalkOCP`: schedules as `opti.parameter`; gated force+velocity (node-0 skip); swing-z spline; base-vel + footstep costs. Stand = all-stance case. |
| `t1_nmpc/wb/mpc.py` | modify | `WholeBodyMPC` sets schedules + `base_vel_des` + footstep targets each tick; warm-start unchanged. |
| `sim/walk.py` | **new** | closed-loop forward-walk runner + metrics (distance, feet alternation, lateral drift, solve p90). |
| `tests/*` | modify/new | per task. |

---

### Task 1: Config — uniform grid + walk gait params

**Files:** Modify `t1_nmpc/robot/config.py`; Modify `tests/test_config.py`.

**Interfaces — Produces:** `MPCConfig` gains fields: `nodes=31`, `dt_min=dt_max=0.035`; `gait_cycle=1.4`, `switching_times=(0.0,0.6,0.7,1.3,1.4)`, `swing_height=0.08`, `v_liftoff=0.05`, `v_touchdown=-0.05`, `n_feet=2`; `base_vx_des=0.0` (default; set per-run), `footstep_k=0.1`, `footstep_weight=50.0`. All M0 fields/dims unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_config.py
def test_walk_grid_and_gait():
    from t1_nmpc.robot.config import make_config
    c = make_config()
    assert c.nodes == 31 and c.dt_min == 0.035 and c.dt_max == 0.035
    assert abs(c.nodes * c.dt_min - 1.085) < 1e-9
    assert c.n_feet == 2
    assert c.gait_cycle == 1.4
    assert c.switching_times == (0.0, 0.6, 0.7, 1.3, 1.4)
    assert (c.swing_height, c.v_liftoff, c.v_touchdown) == (0.08, 0.05, -0.05)
    assert hasattr(c, "footstep_k") and hasattr(c, "footstep_weight")
    # M0 dims unchanged
    assert (c.nq, c.nv, c.n_corners, c.nf, c.na, c.tau_nodes) == (36, 35, 8, 24, 35, 3)
```

- [ ] **Step 2: Run it — Expected FAIL** (`AttributeError`/assert): `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python -m pytest tests/test_config.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement** — edit `MPCConfig` defaults in `t1_nmpc/robot/config.py`:

```python
    # dimensions / horizon   (uniform grid: dt_min==dt_max -> gamma=1)
    nodes: int = 31
    tau_nodes: int = 3
    dt_min: float = 0.035
    dt_max: float = 0.035
```
and add after the friction field:
```python
    # gait (walk; t1_controller gait.info)
    n_feet: int = 2
    gait_cycle: float = 1.4
    switching_times: Tuple[float, ...] = (0.0, 0.6, 0.7, 1.3, 1.4)
    swing_height: float = 0.08
    v_liftoff: float = 0.05
    v_touchdown: float = -0.05
    # forward command + Raibert footstep
    base_vx_des: float = 0.0
    footstep_k: float = 0.1
    footstep_weight: float = 50.0
```
Leave `make_config` asserts as-is (they don't pin `nodes`). If any existing test pinned `nodes==14`, update it to 31.

- [ ] **Step 4: Run — Expected PASS:** `tests/test_config.py`.
- [ ] **Step 5: Commit:** `git add t1_nmpc/robot/config.py tests/test_config.py && git commit -m "feat(config): t1_controller uniform dt=0.035/N=31 + walk gait params"`

---

### Task 2: Swing-z velocity spline

**Files:** Create `t1_nmpc/wb/spline.py`; Create `tests/test_spline.py`.

**Interfaces — Produces:** `get_spline_vel_z(swing_phase, swing_period, h_max, v_liftoff, v_touchdown) -> casadi expr` and `class CubicSpline`. casadi-compatible (uses `ca.if_else`), accepts symbolic `swing_phase`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spline.py
import casadi as ca
from t1_nmpc.wb.spline import get_spline_vel_z


def _vz(phase):
    e = get_spline_vel_z(phase, swing_period=0.6, h_max=0.08, v_liftoff=0.05, v_touchdown=-0.05)
    return float(ca.DM(ca.substitute(e, ca.MX(), ca.MX())) if not isinstance(e, (int, float)) else e)


def test_spline_shape():
    # liftoff rising, ~0 at apex, descending at touchdown
    assert _vz(0.0) > 0.0
    assert abs(_vz(0.5)) < 1e-6
    assert _vz(1.0) < 0.0


def test_spline_symbolic():
    p = ca.MX.sym("p")
    e = get_spline_vel_z(p, 0.6, 0.08, 0.05, -0.05)
    f = ca.Function("f", [p], [e])
    assert float(f(0.0)) > 0 and float(f(1.0)) < 0
```

- [ ] **Step 2: Run it — Expected FAIL** (module missing).

- [ ] **Step 3: Implement** — port verbatim from `wb-mpc-locoman/utils/gait_sequence.py:96-134`:

```python
# t1_nmpc/wb/spline.py
"""Cubic swing-height spline (z-velocity reference). Ported from wb-mpc-locoman (OCS2 form)."""
from __future__ import annotations
import casadi as ca


class CubicSpline:
    def __init__(self, t0, t1, pos0, vel0, pos1, vel1):
        self.t0, self.t1, self.dt = t0, t1, t1 - t0
        dpos = pos1 - pos0
        dvel = vel1 - vel0
        self.c0 = pos0
        self.c1 = vel0 * self.dt
        self.c2 = -(3.0 * vel0 + dvel) * self.dt + 3.0 * dpos
        self.c3 = (2.0 * vel0 + dvel) * self.dt - 2.0 * dpos

    def velocity(self, t):
        tn = (t - self.t0) / self.dt
        return (3.0 * self.c3 * tn**2 + 2.0 * self.c2 * tn + self.c1) / self.dt


def get_spline_vel_z(swing_phase, swing_period, h_max=0.08, v_liftoff=0.05, v_touchdown=-0.05):
    mid = swing_period / 2.0
    s1 = CubicSpline(0.0, mid, 0.0, v_liftoff, h_max, 0.0)
    s2 = CubicSpline(mid, swing_period, h_max, 0.0, 0.0, v_touchdown)
    return ca.if_else(swing_phase < 0.5,
                      s1.velocity(swing_phase * swing_period),
                      s2.velocity(swing_phase * swing_period))
```

- [ ] **Step 4: Run — Expected PASS.** If `_vz` helper is awkward, evaluate via a `ca.Function` of a constant; keep the sign assertions.
- [ ] **Step 5: Commit:** `git add t1_nmpc/wb/spline.py tests/test_spline.py && git commit -m "feat(spline): port cubic swing-z velocity reference from wb-mpc-locoman"`

---

### Task 3: Foot-center frames

**Files:** Modify `t1_nmpc/robot/model.py`; Modify `tests/test_model.py`.

**Interfaces — Consumes:** `MPCConfig` (`corner_x`, `corner_y`, `corner_z`, `ANKLE_ROLL_FRAMES`). **Produces:** `RobotModel` gains `foot_center_frame_ids: tuple[int, int]` (Left, Right) — one frame per foot at the sole-rectangle center (`x=(corner_x[0]+corner_x[1])/2`, `y=0`, `z=corner_z`) parented to the ankle-roll joint, added alongside the existing 8 corner frames. All M0 fields unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_model.py
def test_foot_center_frames():
    import numpy as np
    from t1_nmpc.robot.config import make_config
    from t1_nmpc.robot.model import load_model
    rm = load_model(make_config())
    assert len(rm.foot_center_frame_ids) == 2
    assert len(rm.corner_frame_ids) == 8          # corners unchanged
    cx = (make_config().corner_x[0] + make_config().corner_x[1]) / 2.0
    for fid in rm.foot_center_frame_ids:
        t = rm.model.frames[fid].placement.translation
        np.testing.assert_allclose([t[0], t[1], t[2]], [cx, 0.0, make_config().corner_z], atol=1e-9)
```

- [ ] **Step 2: Run it — Expected FAIL** (no `foot_center_frame_ids`).

- [ ] **Step 3: Implement** — in `load_model`, after the corner-frame loop, add foot-center frames (mirror the corner construction), and add the field to `RobotModel`:

```python
# RobotModel dataclass: add
    foot_center_frame_ids: tuple[int, ...]

# in load_model, after corner_ids built, before data = model.createData():
    center_ids = []
    cx = (cfg.corner_x[0] + cfg.corner_x[1]) / 2.0
    for ankle in ANKLE_ROLL_FRAMES:
        fid = model.getFrameId(ankle)
        parent_joint = model.frames[fid].parentJoint
        parent_placement = model.frames[fid].placement
        t = parent_placement.act(np.array([cx, 0.0, cfg.corner_z], dtype=np.float64))
        frame = pin.Frame(f"{ankle}_center", parent_joint, fid, pin.SE3(np.eye(3), t),
                          pin.FrameType.OP_FRAME)
        center_ids.append(model.addFrame(frame))
    # ... then createData() AFTER all addFrame calls; pass tuple(center_ids) into RobotModel(...)
```
Add `foot_center_frame_ids=tuple(center_ids)` to the `RobotModel(...)` constructor call.

- [ ] **Step 4: Run — Expected PASS:** `tests/test_model.py` (incl. existing M0 model tests).
- [ ] **Step 5: Commit:** `git add t1_nmpc/robot/model.py tests/test_model.py && git commit -m "feat(model): add per-foot center frame (for swing-z) alongside 8 corners"`

---

### Task 4: Walk gait schedule

**Files:** Modify `t1_nmpc/wb/gait.py`; Modify `tests/test_gait.py`.

**Interfaces — Consumes:** `MPCConfig`. **Produces:**
- `WalkGait(cfg)` with `schedules(t0: float) -> (contact: np.ndarray (2,N), swing: np.ndarray (2,N))` — for nodes `i=0..N-1` at gait time `t0+i·dt`: `contact[f,i]=1.0` if foot `f` in contact else 0; `swing[f,i]=` swing phase ∈[0,1) if foot `f` swinging else 0. Foot index 0=Left, 1=Right. Mode sequence per Global Constraints (half-open, periodic with cycle 1.4).
- `StandGait(cfg)` updated to expose `schedules(t0)` returning all-ones contact, all-zeros swing (so the OCP API is uniform).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gait.py  (replace the old corner-flag tests)
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.wb.gait import WalkGait, StandGait


def test_stand_schedules_all_contact():
    c = make_config(); ct, sw = StandGait(c).schedules(0.0)
    assert ct.shape == (2, c.nodes) and sw.shape == (2, c.nodes)
    assert np.all(ct == 1.0) and np.all(sw == 0.0)


def test_walk_mode_sequence():
    c = make_config(); g = WalkGait(c)
    ct, sw = g.schedules(0.0)
    # node 0 at t=0 -> LF swing (foot0 out), RF stance (foot1 in)
    assert ct[0, 0] == 0.0 and ct[1, 0] == 1.0
    assert 0.0 <= sw[0, 0] < 0.1 and sw[1, 0] == 0.0
    # a node near t=1.0 (RF swing): find node index for ~1.0s
    i = int(round((1.0 - 0.0) / c.dt_min))
    if i < c.nodes:
        assert ct[1, i] == 0.0 and ct[0, i] == 1.0


def test_walk_boundaries_and_periodicity():
    c = make_config(); g = WalkGait(c)
    # mode_at helper boundaries (half-open)
    assert g.mode_at(0.6) == (True, True)
    assert g.mode_at(0.7) == (True, False)
    assert g.mode_at(1.3) == (True, True)
    assert g.mode_at(1.4) == g.mode_at(0.0)
```

- [ ] **Step 2: Run it — Expected FAIL.**

- [ ] **Step 3: Implement** — rewrite `t1_nmpc/wb/gait.py`:

```python
# t1_nmpc/wb/gait.py
"""Biped contact scheduling for the Fatrop whole_body_rnea walk (cycle 1.4s, t1_controller)."""
from __future__ import annotations
import numpy as np
from ..robot.config import MPCConfig


class WalkGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        self.cycle = cfg.gait_cycle
        self.t_lf_end, self.t_d1_end, self.t_rf_end = cfg.switching_times[1:4]  # 0.6,0.7,1.3
        self.swing_period = self.t_lf_end                                        # 0.6

    def mode_at(self, t: float):
        tp = t % self.cycle
        if tp < self.t_lf_end:   return (False, True)    # LF swing
        if tp < self.t_d1_end:   return (True, True)
        if tp < self.t_rf_end:   return (True, False)    # RF swing
        return (True, True)

    def _swing_phase(self, t: float, foot: int):
        tp = t % self.cycle
        if foot == 0 and tp < self.t_lf_end:
            return tp / self.swing_period
        if foot == 1 and self.t_d1_end <= tp < self.t_rf_end:
            return (tp - self.t_d1_end) / self.swing_period
        return None

    def schedules(self, t0: float):
        N = self.cfg.nodes
        contact = np.zeros((2, N)); swing = np.zeros((2, N))
        for i in range(N):
            t = t0 + i * self.cfg.dt_min
            m = self.mode_at(t)
            for f in (0, 1):
                contact[f, i] = 1.0 if m[f] else 0.0
                ph = self._swing_phase(t, f)
                if ph is not None:
                    swing[f, i] = ph
        return contact, swing


class StandGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    def schedules(self, t0: float = 0.0):
        N = self.cfg.nodes
        return np.ones((2, N)), np.zeros((2, N))
```

- [ ] **Step 4: Run — Expected PASS.**
- [ ] **Step 5: Commit:** `git add t1_nmpc/wb/gait.py tests/test_gait.py && git commit -m "feat(gait): WalkGait contact/swing schedules (cycle 1.4s) + StandGait schedules API"`

---

### Task 5: WalkOCP — gated contact/swing constraints (the core)

**Files:** Modify `t1_nmpc/wb/ocp.py`; Modify `tests/test_ocp.py`.

**Interfaces — Consumes:** Tasks 1–4 + `WBDynamics` (`rnea_dynamics`, `frame_velocity(fid)`, `state_integrate/difference`). **Produces:** `WalkOCP` (generalizes `StandOCP`): adds `opti.parameter`s `contact_sched (2,N)`, `swing_sched (2,N)`; gated per-corner force constraints; gated velocity constraints (node-0 skipped) — stance via the foot's 4 corner frames, swing-z via the foot-center frame using `get_spline_vel_z`; base-velocity + Raibert footstep soft costs. Methods preserved from `StandOCP`: `set_weights`, `set_x_init`, `x_initial`, `solve_function`, `g_data`, `retract`, `constr_viol_inf`. New: `set_schedules(contact, swing)`, `set_base_vx(vx)`, `set_footstep_targets(targets)`. Keep `StandOCP` as a thin subclass/alias building `WalkOCP` with all-stance schedules so M0 tests pass.

**Corner→foot map:** corners 0–3 belong to foot 0 (Left), 4–7 to foot 1 (Right) — matches `model.py` corner-frame creation order (Left ankle first, 4 corners; then Right). The first 4 `corner_frame_ids` are Left, last 4 Right.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ocp.py  (replace StandOCP-only tests)
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.ocp import WalkOCP
from t1_nmpc.wb.gait import WalkGait, StandGait


def _solve(ocp, cfg, rm, x0, contact, swing, max_iter=200):
    ocp.set_weights(); ocp.set_x_init(x0)
    ocp.set_schedules(contact, swing); ocp.set_base_vx(0.0)
    fn = ocp.solve_function(max_iter)
    sol = np.array(fn(x0, cfg.Q_diag, cfg.R_diag, ocp.x_initial())).flatten()
    g, lbg, ubg = ocp.g_data()(sol, ocp.opti.value(ocp.opti.p))
    return ocp.retract(sol), WalkOCP.constr_viol_inf(np.array(g).flatten(),
                                                     np.array(lbg).flatten(), np.array(ubg).flatten())


def test_walkocp_stand_case_solves():
    cfg = make_config(); rm = load_model(cfg); ocp = WalkOCP(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    ct, sw = StandGait(cfg).schedules(0.0)
    out, cv = _solve(ocp, cfg, rm, x0, ct, sw)
    assert cv < 1e-2
    fz = sum(out["forces_sol"][0][3*c+2] for c in range(8))
    assert 0.9 <= fz / (rm.mass * 9.81) <= 1.1          # gravity supported


def test_walkocp_walk_case_solves_and_lifts():
    cfg = make_config(); rm = load_model(cfg); ocp = WalkOCP(cfg, rm)
    x0 = nominal_x(cfg, rm.model)
    ct, sw = WalkGait(cfg).schedules(0.0)                # LF swing at the front of the horizon
    out, cv = _solve(ocp, cfg, rm, x0, ct, sw, max_iter=300)
    assert cv < 1e-2, f"walk OCP CV {cv:.2e}"
    # swing foot (Left = corners 0-3) carries ~zero force at an interior swing node
    swing_node = 5
    f_left = sum(abs(out["forces_sol"][swing_node][3*c+k]) for c in range(4) for k in range(3))
    assert f_left < 5.0, f"swing-foot force not ~0: {f_left}"
```

- [ ] **Step 2: Run it — Expected FAIL** (`WalkOCP` missing).

- [ ] **Step 3: Implement** — generalize `StandOCP` into `WalkOCP` in `t1_nmpc/wb/ocp.py`. Key changes vs the current `_constraints`/`_build` (full file edit — preserve the gap-closing-first order and all M0 methods):

```python
# imports: add
from .spline import get_spline_vel_z

# in __init__: store foot-center frame ids + corner->foot map
        self.foot_center_ids = rm.foot_center_frame_ids
        self.corner_ids = rm.corner_frame_ids            # 8, first 4 Left, last 4 Right
        self.n_feet = cfg.n_feet

# in _build(): add schedule + command parameters (after Q/R params)
        self.contact_sched = opti.parameter(self.n_feet, self.nodes)
        self.swing_sched = opti.parameter(self.n_feet, self.nodes)
        self.base_vx = opti.parameter(1)
        # footstep targets: xy per foot per node (soft); default 0, set per tick
        self.footstep_tgt = opti.parameter(2 * self.n_feet, self.nodes)

# desired base velocity injected into dx_des: keep M0 dx_des, the base-vx cost is added in _objective.

    def _corner_foot(self, c):       # corner index -> foot index (0 Left:0-3, 1 Right:4-7)
        return 0 if c < 4 else 1

    def _constraints(self):
        opti = self.opti
        opti.subject_to(self.DX[0] == np.zeros(self.ndx))
        rnea = self.dyn.rnea_dynamics()
        corner_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.corner_ids}
        center_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.foot_center_ids}
        for i in range(self.nodes):
            dq, dv = self.DX[i][:self.nv], self.DX[i][self.nv:]
            dq_n, dv_n = self.DX[i + 1][:self.nv], self.DX[i + 1][self.nv:]
            q, v, a, forces, dt = self._q(i), self._v(i), self._a(i), self._f(i), self.dts[i]
            opti.subject_to(dq_n == dq + v * dt)                 # (1) gap-closing FIRST
            opti.subject_to(dv_n == dv + a * dt)
            tau_rnea = rnea(q, v, a, forces)
            opti.subject_to(tau_rnea[:6] == np.zeros(6))         # (2) base underactuation
            if self._has_tau(i):                                 # (3) torque eq + box
                tau_j = self._tau(i)
                opti.subject_to(tau_rnea[6:] == tau_j)
                opti.subject_to(opti.bounded(-self.tau_max, tau_j, self.tau_max))
            for c in range(self.cfg.n_corners):                  # (4) gated friction / swing zero force
                fe = forces[c*3:(c+1)*3]
                ic = self.contact_sched[self._corner_foot(c), i]
                opti.subject_to(ic * fe[2] >= 0)
                opti.subject_to(ic * self.mu**2 * fe[2]**2 >= ic * (fe[0]**2 + fe[1]**2))
                opti.subject_to((1 - ic) * fe == np.zeros(3))
            if i == 0:
                continue                                          # (5) node-0: NO velocity constraints
            for c, fid in enumerate(self.corner_ids):             # stance: gated corner velocity
                ic = self.contact_sched[self._corner_foot(c), i]
                opti.subject_to(ic * corner_vel[fid](q, v)[:3] == np.zeros(3))
            for f, fid in enumerate(self.foot_center_ids):        # swing: gated foot-center z-velocity
                ic = self.contact_sched[f, i]
                vz = center_vel[fid](q, v)[2]
                vz_ref = get_spline_vel_z(self.swing_sched[f, i], self.cfg.swing_period_s,
                                          self.cfg.swing_height, self.cfg.v_liftoff, self.cfg.v_touchdown)
                opti.subject_to((1 - ic) * (vz - vz_ref) == 0)
```

Add `swing_period_s` to config (= `switching_times[1]` = 0.6) or compute in `__init__`. Add the base-velocity + footstep soft costs in `_objective` (append to `obj`):

```python
        # base forward-velocity tracking: pin base local x-velocity (v[0]) to base_vx across nodes
        w_bvx = 200.0
        for i in range(self.nodes):
            vx = self._v(i)[0]
            obj += w_bvx * (vx - self.base_vx)**2
        # Raibert footstep (soft): swing foot-center xy -> target, only where swinging
        getpos = {fid: self.dyn.frame_position(fid) for fid in self.foot_center_ids} \
                 if hasattr(self.dyn, "frame_position") else None
        if getpos is not None:
            for i in range(self.nodes):
                for f, fid in enumerate(self.foot_center_ids):
                    sw = self.swing_sched[f, i]
                    pxy = getpos[fid](self._q(i))[:2]
                    tgt = self.footstep_tgt[2*f:2*f+2, i]
                    obj += self.cfg.footstep_weight * sw * ca.sumsqr(pxy - tgt)
```

If `WBDynamics` lacks `frame_position`, add it (cheap: `cpin.framesForwardKinematics`, return `cdata.oMf[fid].translation`) — small dynamics helper. New API methods:

```python
    def set_schedules(self, contact, swing):
        self.opti.set_value(self.contact_sched, np.asarray(contact, float))
        self.opti.set_value(self.swing_sched, np.asarray(swing, float))
    def set_base_vx(self, vx):
        self.opti.set_value(self.base_vx, float(vx))
    def set_footstep_targets(self, targets):       # (2*n_feet, N)
        self.opti.set_value(self.footstep_tgt, np.asarray(targets, float))
```
`solve_function` must add the new parameters to the function signature OR (simpler) keep them as `set_value`-only parameters NOT in the function args — but `opti.to_function` requires all parameters be either function inputs or fixed by `set_value` before `to_function`. **Set the schedule/command parameters via `set_value` is not allowed for to_function inputs that vary per call.** Therefore add `contact_sched, swing_sched, base_vx, footstep_tgt` as **function arguments** to `solve_function`/the compiled `solver_fn`, and have `WholeBodyMPC` pass them each tick (Task 6). Update `solve_function`:

```python
    def solve_function(self, max_iter):
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "solver_fn",
            [self.x_init, self.Q_diag, self.R_diag, self.contact_sched, self.swing_sched,
             self.base_vx, self.footstep_tgt, self.opti.x],
            [self.opti.x])
```

Keep `StandOCP = WalkOCP` (alias) or a subclass; update the M0 stand test harness to pass all-stance schedules through the new signature.

- [ ] **Step 4: Run — Expected PASS:** `tests/test_ocp.py` (stand case CV<1e-2 + gravity supported; walk case CV<1e-2 + swing foot force ~0). If the walk case needs more iters, raise `max_iter` (reference converges); do NOT relax the CV gate. Then run the full suite and confirm stand-related tests still pass.
- [ ] **Step 5: Commit:** `git add t1_nmpc/wb/ocp.py t1_nmpc/wb/dynamics.py tests/test_ocp.py && git commit -m "feat(ocp): WalkOCP — gated 8-corner contact + swing-z spline (node-0 skip) + base-vel/footstep costs"`

---

### Task 6: MPC driver — schedules per tick

**Files:** Modify `t1_nmpc/wb/mpc.py`; Modify `tests/test_mpc.py`.

**Interfaces — Consumes:** `WalkOCP`, `WalkGait`/`StandGait`. **Produces:** `WholeBodyMPC(cfg, rm, gait=None)` — builds `WalkOCP`, compiles the solver once; `reset(x0)` cold-solves with the gait's schedules at `t=0`; `step(x_meas, t)` computes `gait.schedules(t)`, footstep targets (Raibert from `x_meas` + `base_vx_des`), passes them + `base_vx_des` to the compiled `solver_fn`, warm-starts with the previous solution, returns `WBResult` (command, forces0, solve_time, constr_viol, num_iters). Default `gait=StandGait` (preserves M0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mpc.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import StandGait, WalkGait
from t1_nmpc.wb.mpc import WholeBodyMPC


def test_stand_still_holds():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=StandGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    r = mpc.step(x0, t=0.0)
    assert r.constr_viol < 1e-2
    assert r.command.tau_ff.shape == (29,)
    fz = r.forces0[2::3].sum()
    assert 0.9 <= fz / (rm.mass * 9.81) <= 1.1


def test_walk_tick_solves():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))
    x0 = nominal_x(cfg, rm.model); mpc.reset(x0)
    r = mpc.step(x0, t=0.0)
    assert r.constr_viol < 1e-2
```

- [ ] **Step 2: Run it — Expected FAIL** (signature/gait).

- [ ] **Step 3: Implement** — update `WholeBodyMPC`: accept `gait`; build `WalkOCP`; in `reset`/`step` compute `contact, swing = gait.schedules(t)`, build footstep targets, and call `self._solve(x, Q, R, contact, swing, base_vx, footstep, warm)`. Raibert target per swing foot: `stance_xy + 0.5*swing_period*v_des + k*(v_meas - v_des)` (approx; for in-place first set targets = current foot-center xy so v_des=0). Keep warm-start (`self._warm = sol`). Compute `num_iters` if available from Fatrop stats or omit (set 0). Keep `extract_command` (state.py) unchanged.

- [ ] **Step 4: Run — Expected PASS:** `tests/test_mpc.py`; then full suite green.
- [ ] **Step 5: Commit:** `git add t1_nmpc/wb/mpc.py tests/test_mpc.py && git commit -m "feat(mpc): WholeBodyMPC sets gait schedules + base-vel + footstep per tick (stand+walk)"`

---

### Task 7: Closed-loop forward-walk runner

**Files:** Create `sim/walk.py`; Create `tests/test_walk_closed_loop.py`. Read `sim/stand.py`, `sim/mujoco_runtime.py`, `sim/_sim_util.py` first to match structure.

**Interfaces — Produces:** `sim/walk.py` `main(--duration, --vx, --view, --gif)` driving the closed loop (~60 Hz MPC, 500 Hz PD, 2000 Hz physics) with `WalkGait`, advancing gait time `t` each MPC tick, commanding forward `base_vx_des=vx`; prints distance advanced, feet-alternation count, max lateral drift, solve p90.

- [ ] **Step 1: Read the sim layer** and identify where `WholeBodyMPC` is constructed in `sim/stand.py`, how `JointCommand` is applied, and how gait time should advance (one MPC tick = `1/mpc_hz`).

- [ ] **Step 2: Write the failing test** (solver-side, no MuJoCo — fast):

```python
# tests/test_walk_closed_loop.py
import numpy as np, pytest
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.gait import WalkGait
from t1_nmpc.wb.mpc import WholeBodyMPC


@pytest.mark.slow
def test_walk_receding_solves_across_switch():
    cfg = make_config(); rm = load_model(cfg)
    mpc = WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))
    x = nominal_x(cfg, rm.model); mpc.reset(x)
    cvs = []
    t = 0.0
    for _ in range(20):                      # crosses a contact switch (>=0.6s span at dt steps)
        r = mpc.step(x, t)
        cvs.append(r.constr_viol)
        x = np.asarray(r.next_x if hasattr(r, "next_x") else x)   # idealized: hold x (open-loop-ish)
        t += 1.0 / cfg.mpc_hz
    assert max(cvs) < 5e-2, f"max CV {max(cvs):.2e}"
```

(If `WBResult` lacks a predicted next state, have `step` also return `next_x = retract(...)["..."]` node-1 state, or keep `x` fixed for this solver-convergence check.)

- [ ] **Step 3: Run it — Expected FAIL/needs `next_x`; then implement `sim/walk.py`** mirroring `sim/stand.py` but: construct `WholeBodyMPC(cfg, rm, gait=WalkGait(cfg))`, advance `t` per MPC tick, set `base_vx_des=vx`, and compute the walk metrics. Keep the head joints commanded to nominal.

- [ ] **Step 4: Run the live walk + tests:** `PYTHONPATH= OMP_NUM_THREADS=1 conda run -n t1mpc python sim/walk.py --duration 6.0 --vx 0.1` — expect forward progress, feet alternating, no fall; then `pytest tests/test_walk_closed_loop.py -q`. Tune footstep/base-vel weights as needed (this is the M1 gate; iterate here).
- [ ] **Step 5: Commit:** `git add sim/walk.py tests/test_walk_closed_loop.py && git commit -m "feat(sim): closed-loop forward-walk runner + receding-solve test"`

---

### Task 8: Docs

**Files:** Modify `CLAUDE.md`; Modify `docs/2026-06-25-t1controller-divergences.md`.

- [ ] **Step 1:** Update `CLAUDE.md`: status → M1 walk on Fatrop (in progress/PASS); note the aligator pivot was abandoned for walk (link the memory `aligator-wholebodyrnea-walk-mismatch`); describe the gait-schedule-as-parameter mechanism, the node-0 velocity skip, foot-center frames, uniform `dt=0.035/N=31`.
- [ ] **Step 2:** Append to the divergence ledger: uniform grid (t1_controller) vs reference geometric; per-foot center frame for swing-z; Raibert footstep; and the recorded aligator-walk mismatch (why Fatrop).
- [ ] **Step 3:** `grep` the docs for any symbol they name and confirm it exists in the tree.
- [ ] **Step 4:** Commit: `git add CLAUDE.md docs/2026-06-25-t1controller-divergences.md && git commit -m "docs: M1 Fatrop walk status + divergences (uniform grid, foot-center, footstep, aligator-walk mismatch)"`

---

## Self-Review (planner checklist — performed)

1. **Spec coverage:** §2 backend reuse → Tasks reuse M0 (no change). §3.1 schedule-as-parameter → Task 5 (+ to_function args). §3.2 gated forces → Task 5. §3.3 gated velocity + node-0 skip → Task 5. §3.4 swing-z spline → Tasks 2,5. §3.5 footstep → Tasks 1,5,6. §4 discretization/gait → Tasks 1,4. §5 receding protocol → Task 6. §6 module changes → all. §7 build plan → Tasks 1–8. §8 success criteria → Tasks 5 (open-loop solve + lift), 6/7 (stand preserved, walk). **Deferred:** arm/hand (M2), turning/variable-speed, real-time C++.
2. **Placeholder scan:** code is grounded in the actual current `StandOCP`/`config`/`dynamics`; the one runtime caveat flagged explicitly is that varying parameters must be `to_function` arguments (not `set_value`) — Task 5 handles it. Task 7's footstep/weight tuning is the genuine closed-loop iteration (the M1 gate), not a placeholder.
3. **Type consistency:** `schedules(t0)->(2,N),(2,N)`, corner→foot map (0–3 Left / 4–7 Right), `solver_fn(x_init,Q,R,contact,swing,base_vx,footstep,x)` consistent across Tasks 4/5/6; `forces0[2::3].sum()` = Σf_z over 8 corners consistent with `nf=24`.

## Notes / risks
- **Heaviest assumption verified:** stand solves at uniform N=31 (CV 1.4e-7, ~49 ms warm). Walk OCP convergence is **proven by the reference** (same formulation+solver); Task 5 is the first place it's confirmed on T1.
- If `opti.to_function` rejects the per-call parameters, the fallback is to rebuild/`set_value`+`opti.solve()` per tick (slower, no codegen) — but the function-argument route (Task 5 Step 3) is the M0 pattern extended and should work.
- Footstep/base-vel weights (Task 1 defaults) are starting points; Task 7 tunes them for the closed-loop gate.
