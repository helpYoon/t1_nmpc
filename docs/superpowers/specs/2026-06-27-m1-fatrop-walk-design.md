# M1 forward walk — design (Fatrop `whole_body_rnea`, 8-corner contact)

**Date:** 2026-06-27
**Branch:** `wb-fatrop-walk` (new, from `7135a0f` — the clean Fatrop M0 stand)
**Status:** design. Supersedes the aligator pivot for the walk milestone.

## 0. Why Fatrop (the pivot rationale, settled)

The aligator pivot was attempted and **abandoned for walking**: aligator `SolverProxDDP` (AL-DDP) is built for contact via **constrained forward dynamics**, and the paper's `whole_body_rnea` formulation (contact as a **velocity-level AL equality** + forces as decision variables) is so stiff for it that the cold solve needs ~335 AL iters and a one-knot warm advance diverges. Stand works on aligator; walking does not. (Recorded in memory `aligator-wholebodyrnea-walk-mismatch` and the divergence ledger.)

**Fatrop** is the solver the paper (arXiv:2511.19709) and the reference (`wb-mpc-locoman`) actually use, and it is the matching solver for velocity-equality contacts (interior-point on the full sparse NLP). The reference **walks** on this exact stack — so walking here is a **proven-feasible adaptation, not research**. The T1 Fatrop M0 stand already **PASSES** (`fz_ratio_p50≈1.0`, `max_tilt≈1.95°`, `solve_p90≈28.7 ms`); we extend it to walking.

## 1. Goal & scope

**M1 = stable forward biped walk** for Booster T1 in closed-loop MuJoCo: advance ≥ ~0.5 m over ≥ 5 s without falling, feet alternating with a confirmed lift, lateral drift bounded, watchable in `--view`.

This is an **incremental extension of the proven M0 Fatrop stand** — same `robot/ wb/ runtime/ sim/` structure, same `whole_body_rnea` formulation and Fatrop solver, adding only the walk machinery (gait schedule, gated swing/contact constraints, swing-z spline, footstep target, walk runner). **In scope:** the walk OCP extension; gait schedule; closed-loop forward walk + balance. **Out / deferred:** arm/hand tracking (M2); turning/lateral/variable-speed; hardware; real-time C++ residual.

## 2. Backend & formulation (reused from M0, unchanged)

- **Solver:** CasADi `Opti` → Fatrop (`opti.to_function` → `solver_function`), warm-started each tick (prev solution incl. `lam_g`). One compiled function serves all ticks (fixed NLP structure; the gait is **data**, see §5).
- **Model:** full FreeFlyer T1, **29 joints, nq=36, nv=35** (the M0 model — head kept as a DOF, held to nominal by cost/PD; no reduced model).
- **Dynamics:** trivial double integrator + **RNEA path constraint** (`whole_body_rnea`): `τ = RNEA(q,v,a, f_ext(forces))`; `τ[:6]=0` (base underactuation); `τ[6:]=τ_j` with a hard box on the first `tau_nodes` nodes. **Unchanged from M0** (`wb/dynamics.py`, `wb/ocp.py`).
- **State/control:** `x∈ℝ⁷¹=[q(36),v(35)]`; input `u_i=[a(35), forces(24), τ_j(29)]` (width 88) for `i<tau_nodes`, `[a(35),forces(24)]` (width 59) otherwise. **Unchanged from M0.**
- **Contact:** **8 corner 3D forces** (`nf=24`, 4 per foot), per-corner friction cone (`μ`). **Unchanged from M0** (`dynamics.py` f_ext accumulation is verified).

## 3. The walk additions (delta from M0)

The M0 `StandOCP` hard-codes "all 8 corners always in contact, zero contact velocity for `i≥1`." Walking makes the contact state **per-foot, per-node data** and adds swing constraints — mirroring `wb-mpc-locoman/optimization/ocp.py:56-188`.

### 3.1 Contact schedule as `opti.parameter` (the key mechanism)
Add parameters (set per tick, never recompiled):
- `contact_schedule ∈ ℝ^{2×N}` — per-foot in-contact flag (0/1).
- `swing_schedule ∈ ℝ^{2×N}` — per-foot swing phase (0→1).
- `swing_period, swing_height, swing_vel_limits(2)` — swing spline params.
- `base_vel_des(6)` — commanded base velocity (forward `v_x`).

The OCP is built **once** with fixed structure; every node carries the **union** of stance+swing constraints written as **flag-gated residuals**. Each tick we `set_value` the schedules and re-solve (warm-started). The sparsity is bit-identical every tick, so one Fatrop `solver_function` serves all ticks. *(This is why Fatrop needs no structural cycling — contrast the aligator pivot.)*

### 3.2 Per-corner force constraints (gated)
For each corner `c` of foot `f` (mirror reference `:153-160`):
- **stance friction cone:** `in_contact_f · f_{c,z} ≥ 0`, `in_contact_f · μ² f_{c,z}² ≥ in_contact_f · (f_{c,x}²+f_{c,y}²)`.
- **swing zero force:** `(1 − in_contact_f) · f_c == 0` (all 3, all 4 corners of a swing foot).

### 3.3 Per-foot velocity constraints (gated) — node-0 skipped
Add **one foot-center frame per foot** to `model.py` (sole-rectangle center, like M0's corner-frame construction). For each foot, for `i ≥ 1` (mirror reference `:162-182`):
- **stance:** `in_contact_f · V_foot(q,v)[:3] == 0` (zero linear velocity; xy no-slip + z no-lift). *(6D optional to also pin foot orientation; start with 3D linear to match the reference closely.)*
- **swing:** `(1 − in_contact_f) · (V_foot(q,v)[2] − v_z^ref) == 0` (z-velocity tracks the spline; xy free).
- combined per the reference: `in_contact·v_z + (1−in_contact)·(v_z − v_z^ref) == 0`, and `in_contact·v_xy == 0`.

**Node 0 carries NO velocity constraint** (`if i==0: continue`) — the initial state's velocity is fixed by `x_init` and would be over-constrained. *(This is in the reference at `:162-165`; it is also the bug that broke the aligator attempt.)*

### 3.4 Swing-z velocity reference (cubic spline)
`v_z^ref = get_spline_vel_z(swing_phase, swing_period, swing_height, v_liftoff, v_touchdown)` — the reference's cubic swing-height spline derivative: rises to `v_liftoff` at liftoff, **0 at apex**, descends to `v_touchdown` at touchdown. Port `get_spline_vel_z` from `wb-mpc-locoman/utils`.

### 3.5 Footstep placement (Raibert, soft)
Soft cost on each swing foot's landing xy: `p_foot,xy → stance_xy + ½·T_step·v_des + k·(v_meas − v_des)`. Drives forward motion together with `base_vel_des` (track `v_x`). Flagged as our biped addition (not in the quadruped reference's core).

### 3.6 Costs (reuse M0 + add)
M0 quadratic `‖x−x_des‖²_Q + ‖u−u_des‖²_R` stays. Add: base-velocity tracking (`v_x` forward), the soft footstep cost, and `u_des` force = `m·g` split over the **stance** corners (n_contacts-aware, per the reference `:184-188`).

## 4. Discretization & gait — follows t1_controller

**Discretization:** **uniform `dt = 0.035`, `N = 31`** (32 nodes), horizon `1.085 s` — per the standing directive to follow t1_controller (`task.info`: `dt=0.035`, `N≈31`, `timeHorizon=1.1`). *Divergence from M0's geometric grid (`N=14`, `dt 0.02→0.06`) and from the reference's geometric grid — chosen to match t1_controller; increases solve time vs M0 (revisit a geometric grid if real-time needs it).* Integrator: explicit Euler (M0; matches the reference). RK4 is unnecessary (physics is in the RNEA constraint at the nodes).

**Gait (walk — t1_controller `gait.info`):** cycle **1.4 s**, `LF [0,0.6) → dbl [0.6,0.7) → RF [0.7,1.3) → dbl [1.3,1.4)`. Per foot: swing 0.6 s, double-support 0.1 s ×2, stance 0.8 s. `swing_height = 0.08`, `v_liftoff = +0.05`, `v_touchdown = −0.05`. Horizon (1.085 s) < cycle (1.4 s).

## 5. Receding-horizon protocol (Fatrop, parameter-based — no cycling)

Per tick: (1) read `x_meas`; (2) compute `contact_schedule`/`swing_schedule` for the horizon window starting at the current gait time (a sliding lookup into the periodic gait); (3) `set_value` the schedules + `x_init` + `base_vel_des` + footstep targets; (4) call the warm-started `solver_function` (prev solution incl. duals). Nothing rotates structurally — the schedule **is** a parameter (the natural, idiomatic approach for this backend). Cadence: ~60 Hz MPC (t1_controller `mpcDesiredFrequency`), 500 Hz PD layer tracking the first-node command.

## 6. Module changes (delta from the M0 Fatrop code)

```
robot/config.py   add: uniform dt=0.035/N=31; walk gait params (cycle 1.4s, switching times,
                  swing_height/liftoff/touchdown); base_vel_des default; footstep gains. keep 8-corner.
robot/model.py    add: one foot-center frame per foot (for the per-foot velocity constraint). keep 8 corners.
wb/dynamics.py    add: foot-frame velocity fn (per foot); port get_spline_vel_z. keep RNEA/f_ext (unchanged).
wb/ocp.py         extend StandOCP -> WalkOCP: contact/swing schedules as opti.parameter; flag-gated
                  force + velocity constraints (node-0 skipped); swing-z spline; base-vel + footstep costs.
wb/gait.py        extend: WalkGait producing per-foot contact_schedule + swing_schedule for a horizon
                  window at gait time t (sliding, periodic). keep StandGait.
wb/mpc.py         extend WholeBodyMPC: set_value the schedules + base_vel_des + footstep each tick;
                  warm-start unchanged.
wb/state.py       unchanged (full model).
sim/walk.py       NEW: closed-loop forward-walk runner (advance v_x; metrics: distance, feet alternation,
                  lateral drift, solve p90). sim/stand.py kept.
```

## 7. Build plan (incremental, each step testable)

1. **Discretization + gait schedule:** config (uniform dt/N, walk params) + `WalkGait` (contact/swing schedules for a horizon window). Unit-test the schedule (mode sequence, periodicity, swing phase).
2. **WalkOCP (open-loop):** extend the OCP with gated force+velocity constraints (node-0 skip) + swing-z spline + foot-center frames. Gate: the OCP **solves** a single walk problem to Fatrop tolerance from the stand seed (CV small), and the swing foot lifts (z-velocity tracks the spline) — the reference proves this converges.
3. **Re-confirm stand** as the all-stance special case of WalkOCP (all contact flags = 1) — M0 metrics hold.
4. **Closed-loop forward walk** (`sim/walk.py`): foot-lift + forward progress + balance in MuJoCo. Add the footstep/base-vel costs; tune. The M1 gate.
5. **Docs:** update CLAUDE.md + divergence ledger + paper map.

## 8. Success criteria

- **Walk OCP solves** (open-loop): Fatrop reaches its tolerance on a single walk problem from the stand seed; swing foot z-velocity tracks the spline (lift confirmed).
- **Stand preserved:** all-stance WalkOCP holds the M0 stand metrics in closed loop (`Σf_z/(m·g)∈[0.9,1.1]`, upright).
- **M1 walk:** closed-loop advances ≥ ~0.5 m over ≥ 5 s without falling; feet alternate with confirmed lift; lateral drift < ~0.1 m; watchable in `--view`.

## 9. Divergences (from t1_controller / paper / reference)

- **Discretization uniform `dt=0.035/N=31`** (t1_controller) vs M0/reference/paper geometric grid — follows the standing directive; heavier solve.
- **8-corner 3D contact** (T1 flat foot) vs the reference's point-foot single 3D force — exact CoP/support-polygon; matches M0.
- **Per-foot velocity constraint at a foot-center frame** (added) vs the reference's point-foot frame — flat-foot adaptation.
- **Footstep (Raibert, soft)** — our biped addition.
- **Euler integrator** vs t1_controller RK4 — physics in the RNEA constraint; matches the reference.
- **NOT a divergence:** node-0 velocity-constraint skip is the reference's own behavior (`:162-165`).
