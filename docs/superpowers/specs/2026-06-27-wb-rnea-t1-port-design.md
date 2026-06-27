# wb-mpc-locoman → T1 port — design (iteration 1: closed-loop MuJoCo stand)

**Date:** 2026-06-27
**Branch:** aligator-port (implementation should branch fresh — see §15)
**Status:** design (approved in brainstorming); ground-truth-verified; plan to follow.

## 1. Problem & goal

The `aligator-port` branch carries a whole-body NMPC built on **aligator (ProxDDP)**. We are
replacing that controller backend with the formulation from **wb-mpc-locoman** (Molnar et al.,
RA-L 2025, *Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation*,
[arXiv:2511.19709](https://arxiv.org/abs/2511.19709)) — specifically its **`whole_body_rnea`**
model: a **CasADi `Opti` direct-transcription NLP** solved by **Fatrop**, with symbolic RNEA
from `pinocchio.casadi`.

**Goal (iteration 1):** port `whole_body_rnea` onto the **Booster T1 humanoid** and demonstrate a
**closed-loop MuJoCo stand** — the robot holds an upright, soles-flat standing pose under the new
controller.

**Governing constraints for this work:**
- **Reuse only the project *structure*** (`robot/ wb/ runtime/ sim/` layout). Every module's
  *contents* are rewritten from scratch. No file inside those packages is sacred.
- The CLAUDE.md "faithfulness to `t1_controller`" rule is **retired** for this port. The new
  formulation authority is wb-mpc-locoman applied to T1. `t1_controller` remains a useful *data*
  source for T1 numbers (limits, standing pose, foot geometry) — not a formulation authority.
- **aligator is no longer a dependency of the new code.** Fatrop (via CasADi) is the solver.

## 2. Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| First milestone | **Closed-loop MuJoCo stand** |
| Contact model | **4 corner point-contacts per foot** (8 × 3D forces, `nf=24`) |
| Environment | **Reuse the existing `t1mpc` conda env** — zero new dependencies |
| Reuse scope | **Structure only**; all module contents rewritten |
| URDF | **Vendor `t1.urdf` + meshes into `t1_nmpc`** (self-contained) |
| Solver | **Fatrop** (via `casadi.Opti.solver('fatrop')`) — Ipopt/OSQP out of scope |
| Base joint | **`JointModelFreeFlyer`** (quaternion), matching wb-mpc + MuJoCo |
| Joints | **All 29 actuated joints** (MuJoCo parity; no head-joint reduction) |
| MuJoCo command | **Joint-torque feedforward `τ_j` + joint PD** around planned `(q, v)` |

## 3. Verified ground-truth facts

All numbers below were verified by **running real code** on this env (pinocchio 4.0.0, casadi
3.7.2, mujoco 3.10, the T1 URDF/MJCF) during a 4-agent verification sweep. Corrections to the
original brainstorming assumptions are flagged **⚠**.

### Dimensions
- pinocchio FreeFlyer T1 (full 29 joints): **`nq=36, nv=35`**, `njoints=31`, `nframes=67`.
- MuJoCo `t1.xml`: **`nq=36, nv=35, nu=29`**; free joint at `qpos[0:7]`/`qvel[0:6]`; the 29
  actuated joints at `qpos[7:36]`/`qvel[6:35]`, in §A.5 order — matching pinocchio `names[2:]`.
- Port state/control sizes: `nx = nq+nv = 71`; `ndx_opt = 2·nv = 70`; `na_opt = nv = 35`;
  `nf = 24` (8 corners × 3); `nj = 29`. Adaptive input width: **88** on the first `tau_nodes`
  nodes (`a35 + f24 + τ29`), **59** thereafter (`a35 + f24`).

### Mass & forces
- Total mass **34.5135 kg** (pinocchio `computeTotalMass`); `m·g = 338.58 N`.
  ⚠ `data.mass[0]` reads `-1.0` until `computeAllTerms`/`crba`/`computeTotalMass` is called — the
  port must compute mass explicitly before building the OCP.
- Nominal per-corner vertical force: double support `338.58/8 ≈ 42.3 N`; single support `≈ 84.6 N`.

### Standing pose & base height (⚠ original 0.62 m was wrong)
- ⚠ **Base height 0.62 m penetrates the floor by ~4.3 cm** with the old deep-crouch pose. Base
  height and joint pose are **coupled**; both must come from one self-consistent FK.
- **Adopt the hardware-proven reference stand** (`t1_controller .../reference.info`), FK-verified
  to put soles exactly on the ground (foot-box bottom at world `z = 0`):
  - **`nominal_base_height = 0.6734 m`**.
  - **`nominal_joint_pos`** = old arms/waist/head **unchanged**, legs replaced with the
    **shallow crouch**: per leg `Hip_Pitch=-0.05, Hip_Roll=0, Hip_Yaw=0, Knee_Pitch=0.10,
    Ankle_Pitch=-0.05, Ankle_Roll=0`.
    Full 29-vector (§A.5 order): head `[0,0]`; L-arm `[0.5,-1.0,0,-1.4,0,0,0]`;
    R-arm `[0.5,1.0,0,1.4,0,0,0]`; waist `[0]`; L-leg `[-0.05,0,0,0.10,-0.05,0]`;
    R-leg `[-0.05,0,0,0.10,-0.05,0]`.
- ⚠ The MJCF spawns `Trunk` at `z=0.7` (a third, distinct height). The sim runner must
  **initialize/settle to the FK-consistent stand height (0.6734)**, not the MJCF spawn height,
  or the first MPC ticks fight a large ground-penetration transient.

### Foot contact geometry (the 8 corners)
- Real foot collision is an explicit **box** on `left_foot_link`/`right_foot_link`
  (children of `Left_Ankle_Roll`/`Right_Ankle_Roll`, whose frames **coincide** with the foot link
  frame in pinocchio): origin `[0.01, 0, -0.015]`, size `[0.223, 0.1, 0.03]` → half-extents
  `[0.1115, 0.05, 0.015]`, **box bottom at `z = -0.030`** in the foot frame.
- ⚠ Use sole offset **`z = -0.030`** (true box bottom), not the old `-0.027`.
- **8 corner frames** (4 per foot), added via `pin.Model.addFrame` on the ankle-roll joints, each
  with placement translation **relative to the ankle-roll/foot frame**:
  `x ∈ {0.01 − 0.1115, 0.01 + 0.1115} = {−0.1015, 0.1115}`, `y ∈ {−0.05, 0.05}`, `z = −0.030`.
  At the nominal pose all 8 corners are coplanar (numerically identical world `z`).
  (A conservative inset `x∈[−0.10,0.10], y∈[−0.045,0.045]` is an option for stability margin; the
  default is the **true box corners** above.)

### URDF vendoring
- URDF references **30 unique `.STL` meshes** via `package://t1_description/meshes/<name>.STL`
  (9.3 MB total). **Keep the `package://` text verbatim** — no rewrite. Resolution is purely the
  `package_dirs` argument: pinocchio resolves when `package_dirs = <parent of t1_description>`.
- **Vendored layout:** `t1_nmpc/t1_nmpc/robot/assets/t1_description/{urdf/t1.urdf, meshes/*.STL}`,
  load with `package_dirs = [".../robot/assets"]`.
- **No SRDF exists** for T1 — set `q0` directly from `nominal_joint_pos`, no SRDF load.
- Meshes are **not** needed to build the pinocchio kinematic/dynamic Model (MPC correctness is
  mesh-independent); they are needed for viz/MuJoCo only — vendor them anyway for self-containment.
- ⚠ The MJCF `t1.xml` in `t1_controller` is **modified** (a Waist motor was added). If vendored,
  vendor **that** copy; it references meshes by bare filename with `<compiler meshdir="../meshes"/>`.
  (Iteration 1 may keep loading `t1.xml` from its current path and defer MJCF vendoring; the URDF
  is what must move.)

## 4. Architecture

### Solver stack
CasADi `Opti` builds a sparse NLP; **Fatrop** (compiled into casadi 3.7.2, confirmed loadable)
solves it via `opti.solver('fatrop', opts)`. Each MPC tick evaluates a `solver_function` produced
by `opti.to_function(...)` whose parameters include `opti.x` for warm-starting. Symbolic RNEA and
the manifold maps come from `pinocchio.casadi` (`cpin`), confirmed working on pinocchio 4.0.

### State / control
- **State** `x = [q(36), v(35)]`, `nx = 71`. `q = [pos(3), quat_xyzw(4), joints(29)]`,
  `v = [v_lin_local(3), v_ang_local(3), jvel(29)]` — **both base-velocity parts in the body-local
  frame** (pinocchio FreeFlyer convention; see §9).
- **Decision delta-state** `dx = [dq(35), dv(35)]`, `ndx_opt = 70`, with `q` reconstructed per node
  as `state_integrate(x_init, dx)` using `cpin.integrate` on the manifold.
- **Input** (adaptive per node, wb-mpc scheme): `u_i = [a(35), forces(24), τ_j(29)]` for
  `i < tau_nodes`, then `[a(35), forces(24)]`. `include_acc = True` (accelerations in the input,
  **required** for Fatrop structure detection). `f_idx = 35`, `τ_idx = 59`.

## 5. Module layout (structure kept, contents rewritten)

```
t1_nmpc/
  robot/
    assets/t1_description/   NEW — vendored t1.urdf + 30 meshes (package_dirs root = robot/assets)
    model.py    FreeFlyer T1 load (package_dirs); add 8 corner contact frames; mass via
                computeTotalMass; q0 from nominal_joint_pos; base frame = 'Trunk'
    config.py   slim T1 dataclass: horizon (nodes, dt_min, dt_max, tau_nodes), Q/R diagonals,
                corner geometry, μ, nominal_joint_pos (shallow crouch), base height 0.6734,
                joint pos/vel/torque limits.  (old OCS2/centroidal fields discarded)
  wb/                         the casadi/fatrop whole-body-RNEA controller
    dynamics.py  cpin: state_integrate/difference (FreeFlyer manifold), rnea_dynamics (8-corner
                 force application), frame velocity (LOCAL_WORLD_ALIGNED)
    ocp.py       Opti transcription: adaptive vars, params, RNEA path constraint, contact/
                 friction/velocity constraints, tracking objective, gap-closing-first ordering
    gait.py      biped contact schedule: stand = all 8 corners in contact (walk deferred); 2
                 physical-foot swing groups, each expanded to its 4 corner flags
    mpc.py       WholeBodyMPC: build Opti, init Fatrop solver_function, update_params, solve,
                 warm-start (prev solution incl. lam_g), retract first-node command
    state.py     MuJoCo <-> pinocchio FreeFlyer map (the §9 conversion — single source of truth);
                 first-node command extraction (q_des, v_des, τ_j)
  runtime/        transport protocol kept; repointed to new mpc/state
sim/
  mujoco_runtime.py  2000 Hz physics (kept), FreeFlyer state read; settle to 0.6734 stand height
  stand.py    NEW runner: closed-loop stand, metrics (Σfz/mg, tilt, solve p90) + viewer/GIF
```

## 6. The `whole_body_rnea` formulation — port checklist

Reproduce exactly (verified against wb-mpc-locoman source):

1. **State transition** (per node `i`): `dq_next == dq + v·dt`; and (since `include_acc`)
   `dv_next == dv + a·dt`. `a = u_i[:35]`. `dt` from the geometric series
   `dt_i = dt_min·γ^i`, `γ = (dt_max/dt_min)^{1/(nodes-1)}`.
2. **RNEA path constraint**: `τ_rnea = rnea_dynamics(q, v, a, forces)` (whole-body, length `nv`).
   - **Base rows**: `τ_rnea[:6] == 0` (6-DOF floating-base underactuation) — **all nodes**.
   - **Joint rows** (only `i < tau_nodes`): `τ_rnea[6:] == τ_j`, with box bounds
     `−τ_max ≤ τ_j ≤ τ_max` (`τ_max = effortLimit[6:]`, finite for T1).
3. **Force application** (`rnea_dynamics`, reused verbatim, generic over `ee_frames`):
   `framesForwardKinematics(q)`; build `f_ext` (length `njoints`, `cpin.Force` zeros); per corner
   frame: `joint_id = frames[fid].parentJoint`, `trans = frames[fid].placement.translation`,
   `R = data.oMi[joint_id].rotation.T`, `f_lin = R·f_world`, `f_ang = cross(trans, f_lin)`,
   `f_ext[joint_id] += Force([f_lin; f_ang])`. **4 corners share one ankle-roll parent joint; their
   wrenches accumulate into the same `f_ext` slot — this is correct** (verified). `τ = cpin.rnea(q,
   v, a, f_ext)`.
4. **Contact / swing constraints** (per corner frame, stand → all in contact):
   - Friction cone: `in_contact·f_z ≥ 0` and `in_contact·μ²·f_z² ≥ in_contact·(f_x²+f_y²)`.
   - Swing zero-force: `(1−in_contact)·f_e == 0` (no swing at stand).
   - Contact zero xy-velocity: `in_contact·vel_xy == 0`.
   - Combined z: `in_contact·vel_z + (1−in_contact)·(vel_z − vel_z_des) == 0`
     (`vel_z_des` from the swing spline; irrelevant at stand).
   - **First node (`i==0`) skips all velocity constraints** (avoids over-constraining `x0`).
   - Frame velocity uses `pin.LOCAL_WORLD_ALIGNED`.
5. **Joint limits**: box on `q[7:]` (pos) and `v[6:]` (vel).
6. **Objective**: `Σ ‖dx − dx_des‖²_Q + ‖u − u_des‖²_R` over nodes + terminal `‖dx − dx_des‖²_Q`.
   `dx_des = state_difference(x_init, x_des)`; `x_des` = nominal stand `[q0_base@0.6734, upright,
   nominal_joint_pos, v=0]`. `u_des` force block = gravity comp (see §10). Torque rows in `u` for
   `i ≥ tau_nodes` are zero-padded in the objective.
7. **Warm-start**: `solver_params = [x_init, dt_min, dt_max, contact_schedule, swing_schedule,
   n_contacts, swing_period, swing_height, swing_vel_limits, Q_diag, R_diag, base_vel_des]`,
   then append `opti.x`; `solver_function = opti.to_function(...)`. Reuse previous solution
   (`DX_prev, U_prev, lam_g`) each tick. Fatrop opts: `expand=True,
   structure_detection='auto', fatrop.max_iter=10, fatrop.tol=1e-3, fatrop.mu_init=1e-4,
   warm_start_init_point=True, warm_start_mult_bound_push=1e-7, bound_push=1e-7`.

### Quadruped → biped adaptations (mandatory; all verified necessary)
- ⚠ Base frame `'base_link'` **does not exist** in T1 → use **`'Trunk'`** (frame id 2). The old
  `getFrameId('base_link')` returns an invalid id silently.
- ⚠ Quadruped foot frames `FR/FL/RR/RL_foot` **do not exist** → add the **8 corner frames** (§3).
- ⚠ `robot.nf` hardcoded `12` → **24**. (`Dynamics.nf = 3·len(ee_frames)` self-corrects.)
- ⚠ Reference forces: drop `front_force_ratio` front/rear split → **even split** `m·g` over the
  in-contact corners (double support: `m·g/8` per corner).
- ⚠ `GaitSequence` is a hardcoded 4-foot quadruped → biped with **2 physical-foot swing groups**
  (left, right) but **8 contact frames**. All 4 corners of a foot **share one contact flag and one
  swing phase**. Implementation: schedule per physical foot (2-wide), expand each flag ×4 to the 8
  corner frames. (Stand: all 8 always in contact.)
- ⚠ `set_weights` is quadruped-shaped (`tile([1000,500,500], 4)`, `[2]*12` leg block) → re-dimension
  for the T1 kinematic tree (29 joints: head 2 + arms 14 + waist 1 + legs 12) and retune
  (left/right symmetric; legs stiffer than arms).
- ⚠ Compute total mass before the OCP (see §3).

## 7. Cost / references (stand)

Track the nominal stand: base at `z=0.6734`, upright (identity orientation); 29 joints →
`nominal_joint_pos`; all velocities zero. Reuse wb-mpc's **diagonal Q/R** structure, retuned for
T1: base z + roll/pitch weighted heavily, base xy/yaw soft; joint-position weights stiffer on legs
than arms; velocity and input (accel/force/torque) regularization small. Force reference =
`m·g` distributed evenly over the 8 in-contact corners (`≈42.3 N` z each), zero tangential.

## 8. Contact model — 4 corners per foot (rationale)

T1 has **flat feet**; a single 3D point force per foot cannot resist ankle roll/pitch moments, so a
single-point stand is unstable. Representing each foot as **4 unilateral 3D corner forces**
reconstructs a 6D foot wrench with the center-of-pressure constrained inside the support rectangle
(each corner `f_z ≥ 0`), while **reusing wb-mpc's point-force machinery unchanged**. This is a
deliberate divergence from `t1_controller`'s single 6D-wrench-per-foot + CoP/wrench-cone model
(§14).

## 9. MuJoCo ↔ pinocchio FreeFlyer state map (single source of truth)

This conversion lives in `wb/state.py` and is the **only** place these rules are encoded — do not
re-derive elsewhere (the old euler-base path uses a *different*, non-interchangeable convention).

**MuJoCo → pinocchio** (state estimate):
- Position: `q[0:3] = qpos[0:3]` (both world).
- Quaternion: **reorder** `qpos[3:7]=(w,x,y,z)` → `q[3:7]=(x,y,z,w)`.
- Joints: `q[7:36] = qpos[7:36]`.
- ⚠ **Base linear velocity**: `v[0:3] = R(q)ᵀ · qvel[0:3]` — MuJoCo `qvel[0:3]` is **world**-frame;
  pinocchio FreeFlyer `v[0:3]` is **body-local**. The rotation is **required** (omitting it is a
  latent bug, harmless only when `R ≈ I`, i.e. exactly at stand).
- Base angular velocity: `v[3:6] = qvel[3:6]` (both body-local — no rotation).
- Joint velocity: `v[6:35] = qvel[6:35]`.

**pinocchio → MuJoCo** (command/back-conversion): invert the above —
quaternion reorder `(x,y,z,w)→(w,x,y,z)`, base linear `qvel[0:3] = R(q)·v[0:3]`, angular and joint
parts straight through.

**MuJoCo command (stand):** at `mpc_hz` (start ~50 Hz), read state → solve → take the node-0/1
solution; send to the 29 actuators **joint-torque feedforward `τ_j` (RNEA solution) + joint PD
around the planned `(q_des, v_des)`** using `cfg.kp/kd`. Physics runs at 2000 Hz.

## 10. Solver configuration & the Fatrop ordering invariant

⚠ **Fatrop requires the per-stage constraint Jacobian to *start* with the gap-closing
(state-transition/initial) equality** — otherwise it aborts with
`fatrop_interface.cpp: Assertion equality_[0] failed: Constraint Jacobian must start with
gap-closing constraint`. The OCP must **emit the state-transition/initial-state equality before any
inequality** at each stage. `structure_detection='auto'` must also correctly segment the
**adaptive input width** (`τ_j` present only on the first `tau_nodes` nodes changes stage width
mid-horizon). A structured multi-stage OCP + `to_function` + Fatrop solve was verified to work on
this env; the **full T1 OCP must be re-verified** for structure detection on the adaptive stages.

## 11. Success gate (iteration 1)

Closed-loop MuJoCo stand for **≥ a few seconds** with:
- **Σ contact `f_z` / (m·g) ∈ [0.9, 1.1]** (vertical force balance).
- Trunk **upright** (small tilt from vertical), no fall.
- Reported **p90 solve time** (informational target < one MPC period).

Plus light unit checks:
- Model builds with the 8 corner frames; `q0` puts soles coplanar at the ground.
- One-shot OCP **solves** (Fatrop) with **low constraint violation** (inf-norm small).
- Symbolic `rnea_dynamics` matches pinocchio `rnea` numerically at a sample point.
- The §9 conversion round-trips (`mujoco→pin→mujoco` is identity) including a non-identity base
  orientation (guards against the world/local linear-velocity bug).

## 12. Scope — in / out

**In (iteration 1):** stand only; Fatrop only; FreeFlyer base; all 29 joints; 8-corner 3D contacts;
closed-loop MuJoCo; vendored URDF + meshes; the §9 conversion.

**Out / deferred:** walking gait (biped swing schedule beyond the 2-group scaffold); arm-EE
loco-manipulation (force/velocity) tracking; Ipopt/OSQP backends; Fatrop codegen / hardware
deployment; MJCF vendoring (optional). The old aligator `wb/` modules are deleted.

**Dependencies:** **none new** — casadi(+Fatrop), `pinocchio.casadi`, mujoco all present in `t1mpc`.
aligator may remain installed but unused.

## 13. Validation commands (project preamble)

```bash
# unit checks
PYTHONPATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
  python -m pytest tests/ -q -p no:cacheprovider
# closed-loop stand runner
PYTHONPATH= conda run -n t1mpc python sim/stand.py [--duration 4.0] [--view] [--gif out.gif]
```

## 14. Divergences from t1_controller to log (`docs/2026-06-25-t1controller-divergences.md`)

- **Contact model:** 8 unilateral 3D corner forces (`nf=24`) vs OCS2's single 6D wrench + CoP /
  wrench-cone per foot (`nf=12`). Deliberate — reuses wb-mpc's point-force formulation.
- **Friction μ:** wb-mpc hardcodes `μ=0.9`; OCS2 uses `μ=0.4` (and `contactMomentXY μ=0.1`). The
  port uses **`μ=0.4`** (the hardware value). Re-derive any momentum/CoP params from the corner
  geometry rather than copying wb-mpc's value.
- **Solver:** Fatrop interior-point NLP vs OCS2 SQP/HPIPM and vs the prior aligator ProxDDP.
- **Joints:** all 29 kept in the OCP (MuJoCo parity) vs OCS2 fixing the 2 head joints. Optional
  future reduction.

## 15. Risks & open questions

- **Fatrop adaptive-stage structure detection** must be confirmed on the *full* T1 OCP (not just a
  toy), given the `τ_j`-only-on-first-nodes width change. If `structure_detection='auto'` fails,
  fall back to a uniform input layout (keep `τ_j` on all nodes, zero-weighted after `tau_nodes`).
- **Stand initialization transient:** the sim must start at the FK-consistent `0.6734` height, not
  the MJCF `0.7` spawn — otherwise the first ticks fight ground penetration.
- **Base-velocity frame bug** (§9) is the single highest-leverage correctness item for any future
  walking milestone; it is encoded once in `wb/state.py` and covered by the round-trip unit check.
- **Branch:** implementation should start on a fresh branch (e.g. `wb-rnea-port`) rather than
  `aligator-port`, since it abandons the aligator backend.

## 16. Out of scope (explicit)

- Walking, hand/arm manipulation, hardware/SDK bring-up.
- Tuning beyond what stand requires.
- Any change that re-introduces the aligator solver path.
