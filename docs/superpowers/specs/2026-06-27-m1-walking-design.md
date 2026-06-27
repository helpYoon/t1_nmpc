# M1 walking — design (stable forward walk on the wb-rnea / Fatrop OCP)

**Date:** 2026-06-27
**Branch:** wb-rnea-port (builds directly on the M0 stand)
**Status:** design (approved in brainstorming); plan to follow.

## 1. Problem & goal

M0 (closed-loop MuJoCo stand) is done on the CasADi `Opti` + **Fatrop** `whole_body_rnea`
controller. M1 extends it to a **stable forward biped walk**: T1 advances forward, alternating
single-support steps, without falling, in closed-loop MuJoCo.

This is the milestone the prior aligator port never closed — it "advanced but toppled laterally
~1.5 s." Two things changed that make it worth re-attempting on this backend:

- **We now solve the OCP to convergence (Fatrop), not 2 iterations of ProxDDP.** In single
  support, `τ_rnea[:6]=0` together with per-corner `f_z ≥ 0` forces the planned center-of-pressure
  *inside* the 4-corner stance polygon, so a **converged plan sways the CoM over the stance foot by
  construction**. The prior 2-iteration solver likely produced *infeasible* (unbalanced) plans; a
  converged plan cannot.
- So balance is primarily a **closed-loop tracking** question, not a "we forgot the sway reference"
  question.

## 2. Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Target | **Stable forward walk** (full M1), not just "first steps" |
| Balance strategy | **Minimal refs, lean on the converged OCP** — no explicit CoM-sway reference (yet) |
| Explicit references added | contact schedule · swing-foot z-trajectory · forward base-velocity command · swing-foot **(x,y) landing target** |
| Build order | **Incremental, spike first** — de-risk Fatrop-under-changing-contact, then stand→step→walk |
| Escalation if it drifts | add the explicit CoM-sway + footstep-cost reference (kept as the clean next iteration) |
| Scope | forward walk only — no turning, no lateral command, no stairs, no arm tasks |

## 3. Carried-over invariants (from the M0 build — still binding)

- Fatrop: **staircase** variable creation; per-stage **gap-closing equality emitted first**;
  `structure_detection='auto'`; RNEA **`f_ext` accumulated** across the 4 corners sharing each ankle.
- **MX/SX:** `ca.MX.zeros` / `ca.DM` at the Opti graph level, never `ca.SX.zeros`.
- **State map** `wb/state.py`: base linear `v[0:3] = R(q)ᵀ·qvel[0:3]` (single source of truth);
  quaternion `(w,x,y,z)↔(x,y,z,w)`.
- Dimensions unchanged: `x=[q(36),v(35)]`, `nf=24` (8 corners), adaptive input width 88→59.
- Base/trunk frame `'Trunk'`; μ=0.4; mass 34.5135 kg.

## 4. Gait — `WalkGait` (new, `wb/gait.py`)

Biped walk schedule, parameterized by a gait clock `t`. Phases over one cycle:
`L-stance/R-swing → double-support → R-stance/L-swing → double-support`. **Each foot's 4 corners
share one contact flag and one swing phase** (the schedule is computed per *physical foot* (2) and
expanded to the 8 corner flags). Returns, for a horizon of `nodes` with timesteps `dts` starting at
`t`:
- `contact_schedule (8, nodes)` ∈ {0,1}
- `swing_schedule (8, nodes)` ∈ [0,1] (swing phase; 0 when in contact)
- `n_contacts` (stance-corner count, for the gravity-comp force reference)

Proposed defaults (tunable in the spec/plan, refined empirically): `cycle_period ≈ 0.9 s`,
swing ≈ 0.35 s/step, double-support ≈ 0.10 s, `swing_height ≈ 0.06 m`,
`lift_off_velocity ≈ 0.1 m/s`, `touch_down_velocity ≈ −0.2 m/s`.

**Invariant (hard):** the gait cycle must stay **longer than the MPC horizon**
(`0.9 s > ~0.45 s`), so the horizon never wraps a full period.

`StandGait` (all-corners-in-contact) is **kept** for the stand path.

## 5. Contact-schedule-parameterized OCP (the core change, `wb/ocp.py`)

Generalize the stand OCP so the schedule is **`opti.parameter`s set every tick** — the OCP is built
**once** and never recompiled (exactly how wb-mpc-locoman does it). This keeps Fatrop's **constraint
structure fixed**; only parameter *values* change across ticks. New parameters:
`contact_schedule (8, nodes)`, `swing_schedule (8, nodes)`, `base_vel_des (6)`, and the per-step
**foot-placement targets**.

Per-node, per-corner constraints become schedule-aware (`in_contact = contact_schedule[c,i]`):
- **Friction cone:** `in_contact·f_z ≥ 0` and `in_contact·μ²·f_z² ≥ in_contact·(f_x²+f_y²)`
- **Swing zero-force:** `(1−in_contact)·f_e = 0`
- **Contact velocity:** `in_contact·vel_xy = 0`; z: `in_contact·vel_z + (1−in_contact)·(vel_z − vel_z_des) = 0`,
  with `vel_z_des` from the swing spline (§6)
- First node (`i=0`) still skips velocity constraints (fixed `x0`)

The **stand is the all-ones special case** of this schedule, so this *replaces* `StandOCP` with one
parameterized OCP. The existing M0 stand tests guard the regression (stand = schedule of all 1s).
RNEA, the adaptive input width, the gap-closing/staircase ordering, and `_init_guess` are unchanged.

## 6. Swing trajectory + foot placement (the only explicit refs)

- **Swing z (`wb/swing.py`, new):** port wb-mpc-locoman's `CubicSpline` / `get_spline_vel_z` — a
  liftoff→apex→touchdown z-velocity profile as a function of swing phase, used in the contact-z
  constraint above.
- **Foot placement (`wb/footstep.py`, new):** a Raibert/velocity heuristic for the swing foot's
  landing target:
  `p_foot_xy = p_stance_xy + ½·T_step·v_des + k·(v_meas − v_des)`, plus a nominal lateral **stance
  width** (~hip width) so the feet don't converge. Fed as a **soft tracking cost** pulling the swing
  foot's center (the swing ankle-roll/foot frame xy) toward its landing target through the swing.
  This is the one piece that prevents a biped from under-stepping/scuffing (a quadruped can leave
  swing-xy free; a biped cannot).

## 7. Forward command + objective

Add `base_vel_des` (forward `vx`, default ~0.2 m/s; `vy=0, wz=0`). The **velocity block** of the
tracking target switches from zero to `base_vel_des`; base x/y **position** weight stays 0 (already),
so forward drift is unpenalized and motion is driven by velocity tracking. Height, upright, and
joint-posture tracking are unchanged from M0. The swing-foot placement cost (§6) is added to the
objective for swing nodes only.

## 8. MPC recede + warm-start (`wb/mpc.py`)

Each tick: advance the gait clock `t` by the control period, recompute the contact/swing schedule +
foot-placement params from `WalkGait`/`footstep`, set `x_init = measured FreeFlyer state`, then
warm-start and solve. **Add the shift-and-resample warm start** (port `warm_start_interpolate`):
shift the previous solution by one tick and resample onto the geometric time grid — it matters once
the trajectory is actually moving (it was unnecessary at the static stand).

## 9. Module layout

```
wb/gait.py     + WalkGait (biped contact/swing schedule + n_contacts)   [StandGait kept]
wb/swing.py    NEW — CubicSpline swing-z velocity (get_spline_vel_z)
wb/footstep.py NEW — Raibert foot-placement target (+ stance width)
wb/ocp.py      generalize -> parameterized contact/swing schedule + base_vel_des + foot-placement cost
wb/mpc.py      recede gait clock, update schedule params, shift-resample warm start
robot/config.py + walk params (gait timing, swing, stance width, vx, Raibert gain, foot-placement weight)
sim/walk.py    NEW — closed-loop walk runner + metrics + viewer (mirrors sim/stand.py)
```

## 10. The key risk + de-risk spike (FIRST task)

**The one genuinely un-validated thing:** the `in_contact` multipliers create **trivial / zero-Jacobian
constraint rows** at contact switches (e.g. `0 ≥ 0`, `0 = 0`). Fatrop's `structure_detection='auto'`
must tolerate these across single↔double-support transitions, on T1's adaptive-width OCP. wb-mpc-locoman
relies on exactly this for its quadruped trot, but **our T1 OCP must be re-validated** — just like the
M0 stand was spiked before its plan.

**Spike (before the plan):** build the parameterized contact-schedule OCP with a real biped walking
schedule and confirm it converges (low CV) across a few receding ticks spanning a contact switch.
**Fallbacks if it doesn't:** (a) a tiny normal-force floor / bound relaxation on swing corners;
(b) drop the trivial rows by emitting only the active-phase constraints per node at the cost of a
small set of per-phase compiled problems. The plan picks the fallback only if the spike needs it.

## 11. Success criteria

Closed-loop MuJoCo walk, watchable via `--view`:
- advances forward **≥ ~0.5 m over ≥ 5 s without falling**
- feet **alternate** with confirmed lift (swing-foot z clears the ground)
- **lateral drift bounded** (< ~0.1 m) and trunk upright
- measured GRF sane; solve **p90 < the control period budget** (informational)

Unit tests: `WalkGait` schedule shapes/timing; the swing spline; the footstep target; and the
parameterized OCP **converges under a walking schedule** (CV low). The **M0 stand tests must still
pass** (stand = all-ones schedule).

## 12. Scope — in / out

**In:** forward walk only; the parameterized contact OCP; swing z + Raibert foot placement; forward
`vx` command; shift-resample warm start; closed-loop MuJoCo walk runner + viewer.

**Out / deferred:** turning / lateral / variable velocity commands; explicit CoM-sway reference
(the escalation if leaning-on-the-OCP drifts); stairs / uneven terrain; arm / hand tasks; Fatrop
codegen for real-time; hardware.

## 13. Divergences to log (`docs/2026-06-25-t1controller-divergences.md`)

- Biped balance via the **converged OCP's support-polygon constraints**, no explicit CoM/ZMP
  reference (a deliberate minimal-reference bet, unlike OCS2/t1_controller's explicit CoM trajectory).
- Foot placement via a **Raibert heuristic**, not an optimized footstep planner.
- Gait timing / swing / stance-width values are **re-dimensioned for T1 and not yet hardware-cited**.

## 14. Incremental build (each step verified before the next)

1. **Spike** the parameterized contact-schedule OCP under a walking schedule (§10).
2. **Generalize the OCP** to the parameterized schedule; **stand still passes** (all-ones).
3. **WalkGait + swing spline + footstep** modules (unit-tested).
4. **In-place stepping** closed loop (no forward command) — validate gait/swing/contact mechanics.
5. **Forward walk** (add `base_vel_des` + foot placement) — the success gate (§11).
6. Docs + divergences + viewer.
