# Crocoddyl Whole-Body MPC Port — M1 Design (Walking)

**Date:** 2026-06-25
**Status:** Approved design (brainstorm), ready for implementation plan
**Builds on:** M0 (`2026-06-25-crocoddyl-port-design.md`, merged at `master`). M1 extends the M0 crocoddyl backend to a closed-loop forward walk; no backend change.

---

## 1. Context & goal

M0 stands closed-loop in MuJoCo (peak_tilt 0.0279 = acados baseline). M1 makes it **walk**, faithful to the OCS2 `t1_controller`, on the same `ContactInvDynamics` + `SolverIntro` backend.

**Acceptance gate — "A" (it walks):** closed-loop forward walk in MuJoCo at a commanded `vx≈0.3 m/s` via the faithful `reference_wb` velocity command, using the faithful `SLOW_WALK` gait, for ≥10 s / several steps without falling. Faithfulness checks (step length, CoP-in-foot, gait shape) are **telemetry, not hard-fail** — get it walking, then tighten.

**The faithful infra already exists** (kept from the acados era, both confirmed faithful ports during M1 exploration):
- `gait_wb.py` — OCS2 `Gait`/`GaitSchedule`: mode enum (FLY/RF/LF/STANCE), `mode_to_stance`, the 2-segment Hermite swing-z (`swing_z`, faithful to `SplineCpg`), `impact_proximity`, `switch_times_in`, the `SLOW_WALK` template (1.7 s cycle > 1.085 s horizon — load-bearing).
- `reference_wb.py` — OCS2 `MpcTargetTrajectoriesCalculator` + `SwitchedModelReferenceManager`: 0.8 command filter, heading rotation, two-phase base-pose blend, gait-phase arm swing, command `[vx, vy, height, wz]`.

So M1 is largely **wiring** these into the per-node builder + the M0-deferred costs.

### Key faithfulness finding (research): swing-foot xy is EMERGENT
`t1_controller` has **no footstep planner** (no Raibert/capture-point/xy-placement cost). `SwingTrajectoryPlanner` produces only a z-trajectory; `EndEffectorDynamicsFootCost` forces position error to **zero** and penalizes only orientation + velocity. xy landing emerges from the SQP minimizing base-velocity tracking. M1 must keep xy emergent — adding a footstep planner would be *less* faithful. The acados port's "missing foot-placement cost" note was a misdiagnosis.

---

## 2. Faithfulness boundary (honest)

**Faithful:**
- Kinodynamic ID model + nullspace projection (M0). Gait schedule, swing-z trajectory, references, arm swing — all faithful ports. Command filter 0.8. Single-RTI (`maxiter=1` = `sqpIteration=1`). Friction-cone + CoP as the **exact OCS2 `RelaxedBarrier`** (spike-validated custom activation) — *more* faithful than M0's `QuadraticBarrier`. Emergent xy foot placement. Terminal `Q_final·4`. Yawed `R_foot`.

**Deviations (de-risked, with reasons + escalation paths):**
1. **Solver:** DDP-family (`SolverIntro`), not OCS2 GN-SQP+HPIPM (accepted at the M0 pivot).
2. **Swing-z: soft cost, not hard constraint.** `t1_controller`'s `SwingLegVerticalConstraint` is a hard equality. **Spike finding:** `SolverIntro` enforces *only* the inverse-dynamics nullspace — it does **not** enforce a user-added `ConstraintModelManager` equality (verified: a swing-z equality stayed at residual 1.43 regardless of analytic vs numerical Jacobian). Hard constraints need mim_solvers CSQP (uninstalled, ADMM-not-HPIPM, no legged-hardware track record, abandons `SolverIntro`). → swing-z as a **strong cost** (crocoddyl loco-3d idiom; the walk spike walked with it). Escalation: CSQP.
3. **Contact stabilization: uniform gains + decomposition, not per-axis hard constraint.** `t1_controller`'s `ZeroAccelerationConstraint` is `accel + Av·vel + Ax·pose_err = 0` with per-axis `Av=diag(linvel_xy 20, linvel_z 10, angvel 20)`, `Ax=diag(pos_z 100, ori 80, xy 0)`. Crocoddyl `ContactModel6D` takes only uniform 2-scalar `[kp,kd]`. **`kp` must be 0** (a foot swinging-this-horizon becomes stance later; a nonzero `kp` would pull it toward its stale swing-placement reference, breaking emergent landing). **Spike finding:** `gains=[0,0]` slips the stance foot **95.78 mm/0.7 s** (no velocity damping); `gains=[0,20]` → **6.85 mm** (14× better). → contact `gains=[0, foot_linvel_err_gain_xy=20]` (zero-accel + velocity damp + xy-free) **plus** a stance-foot stabilization **cost** for the `Ax` position feedback (z→ground ∝ `pos_z=100`, foot-flat ∝ `ori=80`, no xy). Escalation (verified feasible: `ContactModelAbstract` is Python-subclass-able): a custom per-axis contact model.

All three deviations are instrumented as walk-gate telemetry (§7) so we catch any that prove load-bearing.

---

## 3. Module changes (extend M0; no new backend)

**Core choice — rebuild-per-cycle receding gait:** each MPC cycle, recompute the per-node stance/swing schedule from `gait` at the current phase and **rebuild the running action models** (≈µs/node). Rebuild (not in-place mutate) because the schedule slides (nodes flip DS↔SS) and per-node `nu` varies (45↔39) — both make mutation fragile; crocoddyl handles per-node `nu` natively (walk spike proved this).

- `wb/croco_costs.py` — extend `build_costs`: swing-foot block, relaxed-barrier friction+CoP, stance z/foot-flat stabilization cost, terminal `Q_final·4` branch, yawed `R_foot`.
- `wb/croco_activations.py` — **new** `RelaxedBarrier(ActivationModelAbstract)` (spike-validated: `μ·log(h+√(h²+δ²))`, quadratic continuation for `h≤δ`).
- `wb/croco_problem.py` — extend `make_node(stance_fids, swing, x_ref, com_ref, planted, terminal)`; add `build_walk_problem(x0_66, t_gait, comm_filt, gait, x_meas_68)`; `planted` from the measured state.
- `wb/croco_mpc.py` — extend `CrocoMPC` with a walk mode (gait clock, command filter, per-cycle rebuild, stance-aware `u_traj`).
- `wb/reference_wb.py` — refine: 68→66 `x_ref` (drop `s,v_s`); drop the acados-only gravity-split `u_ref`.
- **Reused unchanged:** `gait_wb.py`, `config_wb.py`, `execution_wb.py`, `mujoco_transport.py`, `control_loop.py`/viewer.
- **New gate:** `sim/wb_walk_croco.py` + `tests/test_wb_walk_croco.py` + `tests/test_croco_walk*.py`.

---

## 4. Per-node gait schedule + `build_walk_problem`

For the horizon at gait-phase `t_gait`, node `k` at `t_k = t_gait + k·dt`:
- `stance_fids[k] = [fid for in_contact in gait.contact_flags(t_k)]`.
- `swing[k]` = non-contact foot during SS (or `None` in DS), carrying `gait.swing_z(t_k, side)` → `(z,ż,z̈)` and `gait.impact_proximity(t_k, side)`.

**`build_walk_problem(x0_66, t_gait, comm_filt, gait, x_meas_68) → ShootingProblem`:**
1. `node_times = t_gait + arange(N+1)·dt`.
2. `x_ref_all = reference_wb.build_reference(x_meas_68, comm_filt, gait, t_gait, node_times, cfg, model)` → per-node ref, **adapted 68→66** (drop `s,v_s`); the gravity-split `u_ref` discarded.
3. `planted = {fid: oMf[fid] from FK at x0}` (measured-state anchored).
4. running: `make_node(stance_fids[k], swing[k], x_ref_all[k], com_ref[k], planted, terminal=False)`.
5. terminal: `make_node(stance_fids[N], None, x_ref_all[N], com_ref[N], planted, terminal=True)`.
6. `ShootingProblem(x0, running, terminal)`.

`com_ref[k]` = CoM of the *reference* config (`pin.centerOfMass` at `x_ref_all[k][:nq]`), so the M0 CoM term now tracks the forward-moving reference rather than a fixed `com0`. (Forward progress is driven primarily by the base-pose + base-velocity tracking in `x_ref` via `Q`, faithful to `t1_controller`, which tracks trunk not CoM; the CoM term is a light secondary regularizer carried from M0.)

**Contact in `make_node`:** per stance foot, `ContactModel6D(state, fid, planted[fid], LWA, nu, gains=[0, cfg.foot_linvel_err_gain_xy])` (the §2 decomposition; xy free, velocity damped). Swing→stance transitions are automatic (a later node simply lists the foot in `stance_fids`; no impulse model — matches OCS2's smooth mode switch).

---

## 5. Per-node cost terms (`build_costs`)

**Carried from M0 (always on):** state-tracking `Q[:66]`, input-reg `R`, torque-limit (`JointEffort`+`QuadraticBarrier`).

**Swing-foot block** (only when the foot swings, every term **× `gait.impact_proximity(t_node)`**; faithful to `EndEffectorDynamicsFootCost` — no position error):

| term | residual | weight (`config_wb.swingfoot_cost_weights`) |
|---|---|---|
| foot-flat orientation | `ResidualModelFramePlacement` + `WeightedQuad` (rotation rows; translation 0) | `1e4` |
| lin-vel xy → 0 | `ResidualModelFrameVelocity(ref=0)`, linear-xy rows | `5` |
| ang-vel → 0 | …angular rows | `2` |
| swing-z tracking (soft, replaces `SwingLegVerticalConstraint`) | `ResidualModelFrameTranslation` z → `gait.swing_z(t)[0]` | strong, tunable (≈1e3–1e4) |
| xy position | — none (emergent) — | — |

(`t1_controller` weights `lin_velocity_z=0`, so we don't double-penalize z-velocity.)

**Stance-foot block** (only when the foot is in contact):

| term | residual | params |
|---|---|---|
| friction cone | `ResidualModelContactFrictionCone(FrictionCone(R_foot, μ=0.4))` + `RelaxedBarrier` | `friction_barrier_mu=0.2, delta=5.0` |
| CoP-in-rectangle | `ResidualModelContactCoPPosition(CoPSupport(R_foot, foot_rect))` + `RelaxedBarrier` | `cop_barrier_mu=0.1, delta=0.03`, `foot_rect=(±0.1115, ±0.05)` |
| z→ground stabilization (§2 decomposition) | `ResidualModelFrameTranslation` z → ground | tuning weight, seeded ~`pos_z=100` scale |
| foot-flat stabilization (§2 decomposition) | `ResidualModelFramePlacement` orientation → flat | tuning weight, seeded ~`ori=80` scale |

`R_foot` = `data.oMf[fid].rotation` (yawed). `RelaxedBarrier` = the spike-validated custom activation. (The two stabilization weights are GN cost weights, not Baumgarte gains — different units; the gain values are only the seed for tuning, and stance-foot slip/sink telemetry (§7) drives the final values.)

**Terminal** (`terminal=True`): state-only, `ResidualModelState(x_ref[N])` × `Q_final[:66]·terminal_scale(4.0)`. No input/friction/swing terms.

---

## 6. `CrocoMPC` walk loop

State held: **gait clock `t_gait`** (advances by real elapsed time), **`comm_filt`** (0.8 EMA filter), previous `xs`/`us`.

**`step(x_meas_68, t, command) → MPCResult`:**
1. advance `t_gait`; `comm_filt = filter_command(comm_filt, command)`.
2. `prob = builder.build_walk_problem(x_meas[:66], t_gait, comm_filt, gait, x_meas_68)`.
3. `solver = SolverIntro(prob)`; warm-start (shift `xs`; **re-`quasiStatic` the `us`** — dimension-safe across DS↔SS `nu` changes).
4. `solver.solve(xs, us, 1, …)` — single-RTI.
5. extract → 68-dim `x_traj` + **stance-aware** 40-dim `u_traj` (per node, map each contact force to its `W_l`/`W_r` slot by stance; swing → 0).

Reused unchanged: `execution_wb.to_joint_command_wb` (handles zeroed swing wrench), `mujoco_transport`, `run_loop`/viewer.

**Open uncertainties (flagged, not silently handled):**
- **Warm-start across `nu` changes** — re-`quasiStatic` `us` is the robust default; refine only if convergence suffers.
- **Single-RTI walking** — unverified closed-loop. Keep `maxiter=1` (faithful); if the gate shows it can't track through contact switches, bring options back (better warm-start, or `>1` iteration as a deviation) — do not silently raise `maxiter`.

---

## 7. Testing, gate, telemetry

**Unit tests (TDD, pure crocoddyl):**
- `build_walk_problem`: per-node `stance_fids` == `gait.contact_flags`; swing block only on the swinging foot; DS `nu=45` / SS `nu=39`; terminal uses `Q_final·4`.
- `RelaxedBarrier`: a cone-violating force → positive penalty with correct gradient sign (the §5 honesty item — verified, not assumed).
- `reference_wb` adaptation: 68→66 `x_ref`; gravity-split `u_ref` dropped.
- `CrocoMPC.step` (walk): rebuild + warm-start; **stance-aware `u_traj`** (force in correct `W_l`/`W_r`, swing→0); finite τ at `maxiter=1`.

**M1 acceptance gate (`sim/wb_walk_croco.py`, gate A):** closed-loop forward walk `vx≈0.3`, ≥10 s. `WALK_GATE` pass = `n_solver_failures=0`, `peak_tilt < ~0.2`, `final_base_z > 0.85·nominal`, forward progress (`mean_vx` within a band of commanded), `n_steps ≥` a few. Structured like the acados walk gate for direct comparison to the `acados-port-final` oracle.

**Telemetry (the backend gaps as first-class signals — watched, not hard-failed):**
- **Stance-foot slip / sink / flatness** — the contact-gain decomposition gap (§2); decides whether `[0,20]`+cost suffices or we escalate to the custom per-axis contact model.
- **Swing-z tracking error + foot clearance** — the soft-vs-hard swing-z gap.
- **Single-RTI convergence** (`stop`, `‖h‖` per tick) — the `maxiter=1` uncertainty.
- **CoP-in-foot, per-foot `fz`, step length / cadence** — faithfulness checks vs `t1_controller`.

---

## 8. Out of scope (later milestones)
- M2 contouring/tracking (`s,v_s`, motion-tracking), omnidirectional commands, `mpc-rl` reference interface.
- Multi-process MRT for uncapped-MPC real-time (M0 spec §8).
- Custom per-axis contact model + CSQP hard constraints (escalations only if M1 telemetry demands).
