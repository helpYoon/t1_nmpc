# M1 walking — design (aligator ProxDDP + whole_body_rnea inverse dynamics)

**Date:** 2026-06-27 (rev 3 — clean rewrite)
**Branch:** `wb-rnea-port` (stays — in-place transformation, no new branch)
**Status:** design; supersedes rev 2. Plan to follow.

**What changed from rev 2 (and why):** the formulation is unchanged in intent, but rev 2 mis-described
three things and they are corrected here, verified against the paper (arXiv:2511.19709, fetched) and the
reference code (`wb-mpc-locoman`, on disk):
1. The swing-foot constraint is a **hard velocity-level equality**, *not* a "Baumgarte / acceleration
   residual." There is no Baumgarte anywhere in the paper or the reference (§3.5).
2. `RNEA τ_base = 0` is the **floating-base underactuation** (dynamics-consistency) constraint, *not* the
   contact constraint. RNEA-ID does **not** move contact to the acceleration level; contact is velocity-level
   in both the old and new formulations (§3.3–3.4).
3. Discretization now **follows t1_controller** (uniform `dt = 0.035`, `N = 31`, horizon ≈ 1.1 s). The rev 2
   event-adaptive grid is dropped: it was a speed optimization the doc itself called deferrable, and it
   endangered the cyclic warm-start the aligator pivot depends on (§4).
4. Contact is modeled as **one 6D foot wrench per foot** + a contact-wrench cone (CoP + yaw bounds), replacing
   rev 2's / M0's 8 corner 3D forces. Rationale: halves the force variables (24→12), removes the corner-force
   redundancy, and is the standard flat-foot abstraction; the yaw bound is the one approximate piece (§3.4).

---

## 0. Decisions at a glance

| Axis | Choice | Faithful to | Note |
|---|---|---|---|
| Backend | aligator `SolverProxDDP` (AL-DDP) | — (divergence) | paper uses Fatrop; we pivot for cheaper warm-start carry + hard-equality AL (§2) |
| Dynamics | trivial integrator `ẋ=[v,a]` + **RNEA path constraint** | paper §IV-B (RNEA-ID) | physics lives in the constraint, not the integrator (§3.3) |
| Base underactuation | `RNEA(q,v,a,W)[:6] = 0` hard equality | paper Eq. 5 | the equation of motion as a stagewise constraint |
| Stance contact | per-foot **6D** `FrameVelocity = 0` (velocity-level) | paper §IV-B2 (per-foot) | 6D is a T1 flat-foot adaptation of the paper's 3D point-foot |
| Swing | `W = 0` (zero wrench) **and** `v_{foot,z}(q,v) = v_z^ref` (velocity-level, hard) | paper §IV-B2 | cubic-spline z-velocity; **no Baumgarte** (§3.5) |
| Contact wrench cone | per foot: `f_z≥0`, `μ²f_z²≥f_x²+f_y²`, CoP `|τ_x|≤Y·f_z, |τ_y|≤X·f_z`, yaw bound | adaptation (μ=0.4) | flat-foot CWC; paper has point feet (no CoP) |
| Contact forces | **6D foot wrench** per foot (`nf = 12`) | adaptation | paper uses 1 point 3D force/foot; T1 has flat soles |
| Torque limits | **soft / relaxed** (post-hoc RNEA torque) | — (divergence) | the one genuine divergence (§3.8) |
| Discretization | **uniform** `dt = 0.035`, `N = 31`, horizon 1.085 s | t1_controller | one divergence: Euler not RK4 (§4) |
| Gait | walk, cycle 1.4 s, `[LF,dbl,RF,dbl]` | t1_controller verbatim | swing 0.6 s, double-support 0.1 s ×2 (§4) |
| Footstep | Raibert target, **soft cost** | our addition | flagged; not in the quadruped paper (§3.6) |
| Reduced model | head locked (27 joints); arms held to nominal by cost | standing directive | one model/state-map for M1→M2 (§3.1) |

---

## 1. Goal & scope

**M1 = stable forward biped walk** for Booster T1 in closed-loop MuJoCo: advance ≥ ~0.5 m over ≥ 5 s without
falling, feet alternating with a confirmed lift, lateral drift bounded, watchable in `--view`.

This is an **in-place transformation of `wb-rnea-port`** — the same `robot/ wb/ runtime/ sim/` structure, the
same paper formulation (`whole_body_rnea` inverse dynamics), but the solver moves Fatrop → aligator and the
stand controller re-homes onto it as the all-double-support case of one OCP.

**In scope:** the Fatrop→aligator rewrite; the `cost.py` / `constraint.py` paper-auditable refactor;
RNEA-ID + 6D foot-wrench contacts + per-foot-6D contact velocity + hard velocity-level swing-z; receding warm-start;
the re-homed stand; foot-lift + forward walk + closed-loop balance.
**Out / deferred:** arm swing + hand tracking (arms held to nominal — that is M2); turning / lateral /
variable-speed commands; the C++ RNEA residual for real-time (the iteration gate passes in sim, serial);
hardware.

---

## 2. Solver: aligator ProxDDP (committed)

aligator's `SolverProxDDP` is an **augmented-Lagrangian differential dynamic programming** (AL-DDP) solver:
a Riccati/DDP backward–forward sweep wrapped in an outer AL loop that drives hard stagewise constraints to
satisfaction by updating per-constraint multipliers and penalties.

**Why it fits this formulation:**
- **Hard stagewise equalities, kept hard.** Every physics constraint here is a hard equality —
  `RNEA[:6]=0`, stance `FrameVelocity=0`, swing-z `v_z=v_z^ref`. aligator carries each as an
  `EqualityConstraintSet` enforced by AL, with **exact cpin autodiff Jacobians**. (acados has no hard
  stagewise-equality projection; crocoddyl is penalty-only and segfaults on hard equalities.)
- **Cheap receding warm-start.** AL-DDP **warm-starts the receding walk** by carrying the previous
  solution's primals *and duals*; in the verification spike the warm dual carry let warm ticks converge to
  CV ≤ 1e-2 in 2–4 outer iters across receding ticks crossing contact switches (`al_iter = 0` — the warm
  duals carried it), matching the t1_controller `sqpIteration = 1` real-time-iteration regime.
- **Clean conditioning at contact switches** (no indefinite Hessian) from the Riccati structure.

**Honest note on the alternative we left behind (Fatrop, the paper's solver).** The paper achieves real-time
whole-body MPC with the **interior-point** solver Fatrop and *does* warm-start it (paper §V-A2: "warm-started
using the solution from the previous step … 15–20 % reduction in solve time", real-time at 80 Hz on a
quadruped). On our **T1** setup, IP warm restarts needed ~8–15 iters/tick to reach CV ≤ 1e-2 (interior-point
central-path re-centering warm-starts weakly relative to AL/SQP). We commit to aligator on that empirical
basis. Two caveats kept on the record so this stays honest: (a) this is a *relative* IP limitation observed
on our setup, not a proven law — the paper warm-starts Fatrop successfully; (b) the reference's own
warm-startable OSQP-SQP (`wb-mpc-locoman` `ocp.py`, Armijo line search, `g_max/g_min`) — the nearest
in-family alternative to t1_controller's SQP — was **not** exhaustively benchmarked. If aligator ever
disappoints, that SQP is the first fallback to try.

> This re-introduces the aligator backend that the current `CLAUDE.md` invariant forbids
> ("No aligator/ProxDDP … Do not re-introduce it"). **Implementing this spec means updating that
> invariant first** — deliberately, not silently.

---

## 3. The formulation

The core idea of `whole_body_rnea` (paper §IV-B): the trajectory is integrated with a **trivial kinematic
double integrator**, and **all of the physics is imposed as a per-stage equality constraint** via inverse
dynamics (RNEA). The optimizer chooses accelerations and contact forces; RNEA is the law that ties them to
the robot's mass/Coriolis/gravity and to the floating-base underactuation.

### 3.1 Reduced model & frames

`buildReducedModel(FreeFlyer T1, lock {AAHead_yaw, Head_pitch})` → **27 joints, nq = 34, nv = 33**
(head locked per the standing directive). Keep **one sole frame per foot** — at the sole-rectangle center,
`z = −0.030` below the ankle — used for **both** the 6D contact wrench and the 6D contact-velocity
constraint. From the sole rectangle (corners x ∈ {−0.1015, 0.1115}, y ∈ {−0.05, 0.05}) take the
**half-extents** `X ≈ 0.106 m` (half-length) and `Y = 0.05 m` (half-width) for the CoP bounds (§3.4).

Arms are **not** structurally locked — they are held at the nominal pose by a high state-tracking weight
(§3.7). This keeps **one model and one MuJoCo↔pinocchio state map across M1→M2**: for M2 (hand tracking) you
relax the arm weight and add the hand-tracking cost, with no dimension change.

### 3.2 State & control

- **State** `x = [q, v]` on aligator `MultibodyPhaseSpace`:
  `q ∈ ℝ³⁴ = [p_base(3), quat(4), q_joint(27)]`, `v ∈ ℝ³³ = [v_base(6, body-local), v_joint(27)]`.
- **Control** `u = [a, W] ∈ ℝ⁴⁵` (uniform width, every node):
  `a ∈ ℝ³³` generalized acceleration, `W ∈ ℝ¹²` the two **6D foot wrenches** `W_foot = [f(3), τ(3)]`, each
  expressed in its **sole (contact) frame** so `f_z` is the surface normal. Joint torque is **not** a
  decision variable (recovered post-hoc — §3.8).

### 3.3 Dynamics = trivial integrator + RNEA path constraint

**Integrator** (`DoubleIntegrator` on the Lie group, explicit/forward Euler, `dt = 0.035`):
```
q_{k+1} = q_k ⊕ (v_k · dt)        # manifold integrate (FreeFlyer); v_k = node velocity
v_{k+1} = v_k + a_k · dt
```
This carries **no physics** — it is pure kinematics, and matches the reference
(`dq_next = dq + v·dt`, `dv_next = dv + a·dt`). The physics is the next constraint.

**RNEA path constraint (every stage)** — the inverse dynamics residual
`τ = RNEA(q, v, a, f_ext(W))`, where each foot's 6D wrench `W_foot` is transformed from its sole frame to the
parent ankle joint (via the frame placement) and placed in that joint's `f_ext` slot — one wrench per foot,
no per-corner accumulation:
- **`τ[:6] = 0`** — *floating-base underactuation*. The 6 base DOF are unactuated, so their generalized
  force must vanish. **This is the equation of motion**, enforced as a hard `EqualityConstraintSet`. It makes
  the chosen `(a, F)` dynamically consistent. (It is *not* the contact constraint — contact is §3.4.)
- **`τ[6:]`** — the joint torques. Recovered **post-hoc** for the command; soft-limited (§3.8).

### 3.4 Contact constraints (stance)

For each foot flagged in contact at node `k`:
- **Velocity-level no-slip/no-lift:** the foot's **6D** spatial velocity (LWA) is zero,
  `V_foot(q, v) = 0 ∈ ℝ⁶`. A pure function of `(q, v)` — velocity level, no acceleration term.
  *Paper faithfulness:* the paper enforces one velocity constraint **per foot** (`v_ci = 0`); it uses **3D**
  linear (point feet). We use **6D** to also pin a flat foot's orientation — a T1 adaptation.
- **Contact-wrench cone** on `W_foot = [f, τ]` (sole frame) — the flat-foot generalization of the paper's
  point friction cone, each row a `NegativeOrthant` residual:
  - **unilateral:** `f_z ≥ 0`;
  - **tangential friction:** `μ² f_z² ≥ f_x² + f_y²`, `μ = 0.4` (paper Eq. 6, on the wrench's linear part);
  - **CoP inside the sole:** `|τ_y| ≤ X·f_z` and `|τ_x| ≤ Y·f_z` — the center of pressure
    `CoP = (−τ_y, τ_x)/f_z` must stay in the `±X × ±Y` rectangle (four linear inequalities);
  - **yaw / spin bound:** `|τ_z| ≤ μ(X + Y)·f_z` — a conservative linear bound from the contact-wrench cone
    (Caron et al., 2015). **This is the one approximate piece** (the corner model captured yaw exactly).

  For a rigid sole these together are equivalent to the 8 unilateral corner cones, minus the corner-force
  redundancy. *Divergence:* the paper has point feet, so the CoP and yaw bounds are our flat-foot additions.

### 3.5 Swing constraints — hard, velocity-level (NOT Baumgarte)

For each foot flagged in swing — as in the paper (§IV-B2) and reference (`ocp.py:160, 182`):
- **Zero contact wrench:** `W_foot = 0` (all 6 components) for the swing foot.
- **Vertical-velocity tracking:** `v_{foot,z}(q, v) = v_z^ref(swing_phase)`, a **hard
  `EqualityConstraintSet`** on the foot's z-velocity. Horizontal velocity is left free (optionally a light
  soft cost on foot pitch/roll to keep it flat — our addition).

`v_z^ref` is the **time-derivative of a cubic swing-height spline** (`get_spline_vel_z`): it rises to a
liftoff velocity (+0.05), crosses **zero at the apex** (where height = `swingHeight`), then descends to a
touchdown velocity (−0.05). Parameters from t1_controller (§4): `swingHeight = 0.08 m`,
`v_liftoff = +0.05`, `v_touchdown = −0.05`.

> **Why this is not Baumgarte.** Baumgarte stabilization enforces a constraint at the **acceleration** level
> and adds position+velocity error-feedback gains (`φ̈ + 2αφ̇ + β²φ = 0`) to stop drift. This constraint has
> **no acceleration term and no feedback gains** — it is a plain algebraic equality on `(q, v)`. The paper
> never uses Baumgarte; the reference repo has zero occurrences of it. Calling it a "Baumgarte residual" was
> a rev-2 mislabel of a spike-only reformulation.

**The open M1 question this raises (the crux — §6.1):** Fatrop and the reference's SQP solve this
velocity-level equality directly. Whether **aligator's AL** drives it to satisfaction while *warm-starting*
across the swing phase is genuinely unproven (the reference never used aligator). That validation is the
primary M1 risk — restated correctly: *make the hard velocity-level swing-z warm-start on aligator's AL.*

### 3.6 Footstep placement (our addition — soft)

A Raibert-style target for the swing foot's landing xy, as a **soft cost** through swing:
`p_foot,xy = stance_xy + ½ · T_step · v_des + k · (v_meas − v_des) + nominal_half_width`.
Forward motion is commanded via `base_vel_des` (track `v_x`). Flagged: this is **not** in the quadruped
paper; it is our biped footstep heuristic.

### 3.7 Costs & regularization

Quadratic stage cost (paper Eq. 3): `‖x − x_des‖²_Q + ‖u − u_des‖²_R`, plus terminal `‖x_N − x_des‖²_Q`.
- `x_des` = nominal configuration + commanded base velocity (`v_x` forward, zero joint velocity).
- `u_des` = `a = 0`, wrench = vertical force `m·g` split over the **stance** feet at zero CoP (`τ = 0`).
- **Arms → nominal** via high `Q` weight on the arm joints (M1 only).
- Plus the soft footstep cost (§3.6) and the soft torque-limit penalty (§3.8).

### 3.8 The one genuine divergence — soft torque limits

The paper's formulation carries joint torque as a decision variable — the reference code keeps it on the
first `tau_nodes` nodes — with a **hard** box. We instead recover torque **post-hoc** from
`τ[6:] = RNEA(...)[6:]` and apply torque limits as a **soft / relaxed** penalty (no hard box, torque not in
`u`). Reason: the hard `tau_nodes` box is a warm-start trap *and* cold-infeasible in dynamic phases (spike:
CV ~ 31). Logged in §10. (Bonus: this makes the control width uniform across all stages.)

---

## 4. Discretization & gait — follows t1_controller

**Discretization (uniform, matching t1_controller `task.info`):**

| | t1_controller | this spec | reference (paper) |
|---|---|---|---|
| dt | 0.035 (uniform) | **0.035 (uniform)** | geometric `δt_k = γ^k δt_0` |
| nodes | ≈31 | **31** | 15 |
| horizon | 1.1 s | **1.085 s** | 0.56 s |
| integrator | RK4 | **explicit/forward Euler** | Euler |

We follow t1_controller's **uniform `dt = 0.035`, `N = 31` intervals (32 shooting nodes), horizon
31·0.035 = 1.085 s** (t1_controller's nominal 1.1 s, realized exactly). The rev-2 event-adaptive grid is
dropped: aligator runs the uniform grid directly, the uniform grid is the **cleanest case for the cyclic
receding warm-start** (§5), and the only argument for fewer nodes (solver speed) is deferrable for M1-in-sim
(the iteration-count gate passes regardless of `N`).

The **one discretization divergence** is the integrator: **explicit/forward Euler, not RK4** — legitimate
because this backend integrates a *trivial double integrator* (`ẋ = [v, a]`) with piecewise-constant accel
and all physics imposed **at the nodes** by the RNEA constraint, so RK4's multi-stage evaluations add cost
without changing the constrained solution. This matches the reference, which deliberately uses plain Euler
(it comments out even the ½·a·dt² term). Logged in §10.

**Gait (walk — t1_controller `gait.info` verbatim):** cycle **1.4 s**, mode sequence
`LF [0, 0.6) → dbl [0.6, 0.7) → RF [0.7, 1.3) → dbl [1.3, 1.4)`. Per foot: **swing 0.6 s**, double-support
**0.1 s ×2**, stance 0.8 s, double-support fraction ≈ 14 %. Each physical-foot contact/swing flag selects
that foot's constraint set per node — *in contact:* wrench-cone + `V_foot = 0`; *in swing:* `W_foot = 0` +
z-velocity spline. Horizon (1.085 s) < cycle (1.4 s).

---

## 5. Receding-horizon warm-start protocol

### 5.1 Why contact mode is *structural* here (not just a parameter)

A natural question: *isn't the contact mode just a per-node parameter of the OCP?* It splits into two parts,
and the answer differs for each:

- **Continuous per-node references** — swing phase → `v_z^ref`, desired stance forces, footstep target,
  base-velocity command. These **are** runtime data: in aligator they are *mutable cost/residual targets*
  overwritten each tick (the analogue of `simple-mpc`'s `setReferencePose` / `setReferenceForce` / …).
- **Discrete per-node mode** — *which* constraints exist at a node (stance ⇒ wrench-cone + `V_foot = 0`;
  swing ⇒ `W = 0` + swing-z). This is **not** a parameter in aligator.

This is the crux. In CasADi/`Opti` (the `wb-mpc-locoman` reference) and in OCS2/t1_controller, the *discrete*
mode **is** a parameter: the OCP is built once with fixed structure, every node carries the **union** of both
constraint sets written as **flag-gated residuals** (`in_contact·r` / `(1−in_contact)·r`), and each tick you
just `set_value` the `contact_schedule`/`swing_schedule` arrays and re-solve. The sparsity is bit-identical
every tick, so one compiled Fatrop function serves all ticks and **nothing rotates**. *(Confirmed in the
reference: `ocp.py:63-64` declares the schedules as `opti.parameter`; `ocp.py:146-182` are the flag-gated
constraints; `update_gait_sequence` just `set_value`s them.)* **Your intuition is exactly right — for that
backend.**

**aligator has no such mechanism.** A `StageModel`'s dynamics and **constraint set are frozen at
construction** — there is no symbolic `opti.parameter`, and constraints cannot be switched on/off at runtime
(only their *targets* are mutable). To change which feet are in contact you must hand aligator a **different
stage**. So the discrete mode becomes *structural*. This is a direct, accepted cost of the aligator pivot
(§2): we trade CasADi's "contact-is-a-parameter" convenience for AL hard-equality handling + warm dual carry.
*(You could emulate the gating by giving every stage the union of constraints times a mutable flag, but that
is non-idiomatic and leaves inert, zeroed constraint rows across the whole horizon — not recommended.)*

### 5.2 The mechanism: cycle pre-built per-phase stages

Because the discrete mode is structural, we pre-build **one `StageModel` per distinct contact phase of the
cycle** (LF-swing, double-support, RF-swing, double-support) once, hold them in a ring, and advance the
horizon by *rotating the ring* — exactly aligator's `simple-mpc` `recedeWithCycle()`. Per tick, before the
solve:
1. **Rotate the horizon:** `replaceStageCircular(<pre-built stage for gait phase t + N·dt>)` drops the
   expired front stage and appends the correct new tip stage; then `solver.cycleProblem(...)` rotates the
   solver's warm-start buffers (primals **and** duals) by one knot to match.
2. **Refresh references** on the rotated stages: `v_z^ref(phase)`, footstep target, stance-force and
   base-velocity targets (the mutable data of §5.1).
3. **Set `x₀ = x_meas` and solve** with warm caps `max_iters ≈ 6`, `target_tol = 1e-2`
   (mirrors `g_max = 1e-2`, `g_min = 1e-6`).

**The warm-start is contact-mode-aware by construction.** The stage ring (the per-node contact modes), the
warm-start buffers (`xs/us/vs/lams`), and the contact-state bookkeeping all rotate **together** by one knot
(`simple-mpc` does `rotate_vec_left` on `cycle_horizon_` + `cycle_horizon_data_` + `contact_states_` in
lockstep). On the gait-phase-periodic grid, new node *i* sits at the gait phase old node *i+1* had, so it
inherits old node *i+1*'s mode **and** its primal/dual datum — stance primals/duals land on stance nodes,
swing on swing; a stance solution never seeds a swing node. The only seam is the freshly appended terminal
knot (one step beyond the old horizon), seeded from the rotated-in phase stage's nominal — the far end, where
it matters least. This alignment is a chief reason warm ticks converge in 2–4 outer iters *across* contact
switches.

**Caution (spike):** preserve that alignment — do **not** apply an *additional* manual shift to the duals
`vs`/`lams` beyond the rotation `cycleProblem` already performs; an extra shift offsets the multipliers one
knot out of phase with the stages (a swing dual lands on a stance node) and diverged (CV ~ 33). *(The exact
split of what `cycleProblem`/`cycleAppend` rotate internally vs. what you carry by hand is the one detail to
confirm in Task 1; the invariant — each warm-start datum's gait phase must equal its node's — is what must
hold regardless.)*

### 5.3 Cadence & the uniform-grid requirement

The MPC solves **once per knot**: the cycle advances exactly one knot (`dt`) per solve, so the steady-state
cadence is `1/dt ≈ 28.6 Hz`, with a **500 Hz PD layer** (t1_controller's `mrtDesiredFrequency`) tracking the
first-node command between solves. (Tying the MPC rate to `dt` differs from t1_controller's 60 Hz, which
re-anchors at the current time each solve rather than shifting one knot — logged §10. One-knot-per-solve is
real-time-correct only when a solve fits in `dt = 35 ms`; see §6.3.)

Stage-cycling is **valid only on a uniform, gait-phase-periodic grid**: the pre-built per-phase stages are
reused as the ring rotates, so the rotated-in stage's `dt` must equal the slot it fills — identically true
when all `dt` are equal and the dt-pattern is periodic with the 1.4 s cycle. **This is the concrete reason
§4 uses a uniform grid.** *(Contrast: the reference's parameter approach warm-starts a non-uniform geometric
grid fine, because nothing rotates structurally — so the uniform requirement is a consequence of cycling,
i.e. of the aligator pivot, not of receding-horizon MPC in general.)*

---

## 6. The real open problems (the milestone work)

The solver/formulation foundation is settled; the *locomotion* is not.

### 6.1 Foot-lift — hard velocity-level swing-z on aligator (THE CRUX)
We keep swing-z a hard equality (§3.5). The reference solves it with Fatrop/SQP; whether **aligator's AL**
warm-starts it cleanly across the swing phase is the central unproven M1 work. If the AL stalls, candidates,
in order: (a) AL weight/penalty conditioning on the swing-z set; (b) add a position-level companion
(`p_foot,z = z^ref`) so the constraint is not velocity-only; (c) a C++ swing residual. **Do not** silently
downgrade to a soft cost.

### 6.2 Lateral balance (closed loop)
The warm-start gate was idealized (`x_meas` = the solver's own node-1, no MuJoCo feedback). Real closed-loop
lateral balance needs the MuJoCo test and very likely an **explicit lateral CoM-sway reference**. The OCP now
*solves* single support (velocity-level contact is AL-tractable here) — the prerequisite — but staying
upright in the plant is the open milestone.

### 6.3 Real-time speed (deferrable)
~49–90 ms/tick (Python cpin residuals → serial `LQ_SOLVER_SERIAL` under the GIL) vs the per-knot budget
`dt = 35 ms` (so the serial loop runs **below real time** today — acceptable for M1-in-sim validation; at
real-time speed the cyclic one-knot-per-solve shift of §5 holds). The
**iteration-count gate passes regardless**; speed needs a **C++/pinocchio-bindings RNEA residual** (which
also re-enables `LQ_SOLVER_PARALLEL`). Not an M1-in-sim blocker.

---

## 7. Module layout (in-place)

```
robot/
  config.py   rewrite: head-locked 27-joint dims, per-foot frame geometry, gait (cycle 1.4 s),
              uniform dt=0.035/N=31, aligator solver params, g_max/g_min, Q/R weights. (remove Fatrop opts.)
  model.py    rewrite: buildReducedModel(lock head) + one sole frame per foot (6D wrench + 6D velocity)
              + sole half-extents (X, Y) + mass.
wb/
  dynamics.py   cpin symbolic PRIMITIVES only: RNEA (+ exact Jacobians, per-foot 6D wrench → f_ext,
                post-hoc joint torque), swing-z velocity reference, frame velocities.
  constraint.py NEW (paper-auditable). Each hard constraint = a named StageFunction + ConstraintSet builder,
                docstring citing paper §/eq + flagging divergence:
                  rnea_base        EqualityConstraintSet  [paper Eq. 5: base underactuation τ[:6]=0]
                  contact_velocity per-foot 6D EqualitySet [paper §IV-B2: per-foot v=0; 6D = T1 adaptation]
                  wrench_cone      per foot NegativeOrthant [friction (paper Eq. 6) + CoP + yaw; flat-foot CWC]
                  swing_z          HARD EqualityConstraintSet [paper §IV-B2: velocity-level z-spline]
                  swing_wrench     W_foot = 0 on swing feet   [paper §IV-B2]
                  torque_limit     SOFT/relaxed [DIVERGENCE — §3.8/§10]
                  joint_pos/vel_limit [paper: state/input bounds]
  cost.py       NEW (paper-auditable). Named builders, each cited:
                  state_tracking (Q) [paper Eq. 3] · input_reg (R) [paper Eq. 3] · base_velocity (v_x cmd)
                  · arm_to_nominal (high Q, M1) · footstep_placement (Raibert — our addition, flagged).
  ocp.py        rewrite: THIN assembler — builds the aligator TrajOptProblem by wiring cost.py + constraint.py
                per stage from the gait flags (DoubleIntegrator + IntegratorEuler). No physics here.
                (remove StandOCP / Fatrop / opti.to_function.)
  mpc.py        rewrite: AligatorMPC — SolverProxDDP(serial), reset (cold), step (warm via
                replaceStageCircular + cycleProblem; xs/us shift; vs/lams carry), command extraction.
                (remove WholeBodyMPC / Fatrop.)
  gait.py       rewrite/extend: biped walk schedule (cycle 1.4 s) + stand (all double-support).
  state.py      rewrite: MuJoCo↔pinocchio for the reduced 27-joint model; command (q_des, qd_des, tau_ff).
runtime/      kept; repoint to the new mpc/state.
sim/
  mujoco_runtime.py  kept; state read + command for the reduced model.
  stand.py / walk.py closed-loop runners on the aligator MPC.
tools/codegen_solver.py  REMOVE (Fatrop-only).
```

---

## 8. Build plan (incremental)

1. **Port solver + formulation + the paper-auditable refactor** into `robot/{model,config}.py` and
   `wb/{dynamics,constraint,cost,ocp,mpc}.py` (RNEA-ID + 6D foot-wrench + per-foot-6D velocity + ProxDDP;
   each cost/constraint paper-cited). **Verify the warm-start gate in-tree on the uniform grid**: ProxDDP
   reaches CV ≤ 1e-2 in ≤ 5 (target ≤ 3) outer iters/tick across ≥ 15 receding ticks including contact
   switches.
2. **Re-home M0 stand** on aligator (all-double-support case of the same OCP); closed-loop MuJoCo stand
   holds (Σ f_z/(m·g) ∈ [0.9, 1.1], upright). Retire the Fatrop stand + rewrite its tests.
3. **Foot-lift** — make the hard velocity-level swing-z warm-start on aligator (the §6.1 crux).
4. **Forward walk + lateral balance** (closed-loop MuJoCo; add the CoM-sway reference as needed) — the gate.
5. **Docs:** update `CLAUDE.md` (drop the "no aligator" invariant), the divergence ledger, and the
   paper↔code map. (C++ RNEA residual + real-time is a separate effort.)

---

## 9. Success criteria

- **Warm-start gate (re-confirmed in-tree, uniform grid):** ProxDDP CV ≤ 1e-2 in ≤ 5 (target ≤ 3) outer
  iters/tick across ≥ 15 receding ticks incl. contact switches.
- **M0 stand (re-homed):** closed-loop MuJoCo stand holds, Σ f_z/(m·g) ∈ [0.9, 1.1], upright.
- **M1 walk:** closed-loop advances forward ≥ ~0.5 m over ≥ 5 s without falling; feet alternate with
  confirmed lift; lateral drift bounded (< ~0.1 m); watchable in `--view`.

---

## 10. Divergences + paper↔code map

Logged in `docs/2026-06-25-t1controller-divergences.md`; mirror in `docs/2026-06-27-paper-mapping.md`.

**Divergences (deliberate):**
- **Solver — aligator ProxDDP** replaces the paper's Fatrop IP. Reason: IP warm-restart was slow on our T1
  setup (~8–15 iters/tick); aligator carries warm duals across the receding walk. *Honest caveat:* the
  paper warm-starts Fatrop successfully (80 Hz) and the reference's OSQP-SQP was not benchmarked — so this is
  a preference backed by a T1 spike, not a proof that IP cannot work.
- **Torque limits soft/relaxed** (post-hoc RNEA torque), not the hard `tau_nodes` box — §3.8.
- **6D foot wrench** (`nf=12`) + contact-wrench cone (**CoP + yaw bounds**) vs the paper's point 3D force per
  foot — T1 has flat soles; the paper's point feet have no CoP. The yaw bound is a conservative CWC
  approximation (Caron et al., 2015).
- **Per-foot 6D contact velocity** vs the paper's 3D point-foot velocity — pins a flat foot's orientation.
- **Euler integrator** vs t1_controller's RK4 — trivial double-integrator dynamics, physics in the RNEA
  constraint; matches the reference.
- **MPC cadence `1/dt ≈ 28.6 Hz`** (one cyclic knot-shift per solve) vs t1_controller's 60 Hz (which
  re-anchors at the current time each solve). Forced by aligator's stage-cycling warm-start (§5); the 500 Hz
  PD tracking layer matches t1_controller's `mrtDesiredFrequency`.
- **Contact mode is structural** (cycled per-phase `StageModel`s), not a per-node parameter as in
  CasADi/OCS2 — a consequence of aligator freezing each stage's constraint set at construction (§5.1).
- **Footstep placement (Raibert, soft)** — our biped addition, not in the quadruped paper.
- **Head locked always**; arms held to nominal for M1 (relaxed in M2).

**NOT divergences (paper-faithful — corrections from rev 2):**
- Swing-z is a **hard velocity-level equality** (paper §IV-B2) — no Baumgarte, no acceleration level.
- Contact is **velocity-level** in the RNEA-ID formulation; `RNEA[:6]=0` is base underactuation, separate.
- *(Replacing rev 2's event-adaptive grid with a uniform `dt=0.035`/`N=31` grid is a fix that **follows
  t1_controller**, but it is still a deliberate **divergence from the paper's geometric grid** — counted under
  Divergences above, not here.)*

**Paper ↔ code map (arXiv:2511.19709 → our unit):**

| Paper element | Paper ref | Our unit | Status |
|---|---|---|---|
| RNEA inverse dynamics, `[0₆; τ_j] = f_RNEA(q,v,a,F)` | §IV-B1, Eq. 5 | `constraint.rnea_base` | match |
| Stance contact `v_ci = 0` (per foot) | §IV-B2 | `constraint.contact_velocity` (6D) | match + adaptation (6D) |
| Friction cone `μ²f_z² ≥ f_x²+f_y²`, `f_z ≥ 0` | §IV-B2, Eq. 6 | `constraint.wrench_cone` (friction part) | match; CoP+yaw added (flat foot) |
| Swing `F_c = 0` + `v_{c,z} = v_z^ref` (cubic spline) | §IV-B2 | `constraint.swing_wrench` (W=0) + `swing_z` | match (hard, velocity-level) |
| Stage cost `‖x−x_des‖²_Q + ‖u−u_des‖²_R` | §IV-A, Eq. 3 | `cost.state_tracking` + `input_reg` | match |
| Geometric adaptive grid `δt_k = γ^k δt_0` | §IV-C | uniform dt=0.035 (t1_controller) | divergence (follow t1_controller) |
| Fatrop IP on full NLP, warm-started | §V-A | aligator ProxDDP | divergence (§2) |
| 1 point 3D force/foot (`R¹⁸`) | §III-B, Eq. 2 | 6D foot wrench (`nf=12`) + CWC | adaptation (flat feet) |
| (none) | — | `cost.footstep_placement` (Raibert) | our addition |
