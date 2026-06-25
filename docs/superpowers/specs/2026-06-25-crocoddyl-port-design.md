# Crocoddyl Whole-Body MPC Port — Design (Milestone M0: Foundation + Closed-Loop Stand)

**Date:** 2026-06-25
**Status:** Approved design (brainstorm), ready for implementation plan
**Scope of this spec:** M0 only — acados teardown + Crocoddyl backend core + closed-loop stand in MuJoCo. Walking (M1) and contouring/tracking (M2) are separate specs.

---

## 1. Context & decision

`t1_nmpc` was an acados/CasADi port of the OCS2 C++ `t1_controller` whole-body kinodynamic NMPC (Booster T1). It passed the M0 stand but the walk **diverged** (`res_eq → 1e80`) because acados has **no native state-input equality projection**; the affine-projector-into-RK4 workaround re-linearizes into a destabilizing `A+BQ` feedback. That blocker is structural, not a tuning gap.

A 19-agent assessment workflow (with adversarial fact-checking) plus two validated feasibility spikes established the pivot to **Crocoddyl**: its inverse-dynamics action models + `SolverIntro` nullspace resolution **are** the projected-ID formulation, using Pinocchio analytic derivatives — dissolving *both* acados blockers (the missing projection and the CasADi codegen blow-up). The deciding constraint: the user wants a Python backend they can modify/extend (for `mpc-rl`), which rules out reusing OCS2's own solver via bindings.

### Validation evidence (spikes, real T1 model)
- **Stand** (`spikes/croco_stand_spike.py`): ContactInvDynamics + SolverIntro holds the inverse-dynamics + contact equality to **‖h‖ = 1.1e-7**, foot force = exact body weight (338.6 N), **13 ms/iter** single-threaded.
- **Walk** (`spikes/croco_walk_spike.py`): single-support phases + contact switches (the case that killed acados) — CoM tracks forward, feet swing, **‖h‖ = 1.1e-3 through the switches**, per-node control dim varies natively (DS `nu=45`, SS `nu=39`), **9.4 ms/iter**.

### Formulation: Option A — Crocoddyl-native inverse dynamics (locked)
`DifferentialActionModelContactInvDynamics` + `SolverIntro`. Control `u = [a (nv joint accelerations); contact forces]`; the inverse-dynamics identity is an equality constraint resolved by nullspace, contact forces recovered as multipliers, joint torques τ exposed. Crocoddyl's `nh` for this model = **6 (underactuated floating-base rows) + contact-constraint rows** — i.e. *exactly* `t1_controller`'s kinodynamic equality structure (the actuated RNEA rows merely define τ and bind nothing). So Option A reproduces t1_controller's model natively, and additionally exposes τ (cleaner torque limits). Trajectory tracking is a cost-layer feature, formulation-independent, fully supported (confirmed against t1_controller's `ProceduralMpcMotionManager` and crocoddyl's `SimpleBipedGaitProblem`).

### Faithfulness boundary (honest)
- **Faithful:** the kinodynamic dynamics model (6 floating-base rows + contact constraints, nullspace-projected = OCS2 `projectStateInputEqualityConstraints`), the cost/constraint *set*, all weights/params (from `config_wb`), contact Baumgarte gains (= t1_controller foot-constraint feedback), single-RTI cadence (`maxiter=1` = `sqpIteration=1`), soft inequalities (penalty costs = t1_controller relaxed barriers).
- **Not faithful (accepted):** the solver is DDP-family (Riccati + nullspace), **not** OCS2's GN-SQP + HPIPM — abandoned with the pivot. The relaxed-barrier *shape* is approximated by `QuadraticBarrier` for M0 (exact relaxed barrier deferred to M1).

---

## 2. Repository layout decision

**Option C (locked):** everything stays in `t1_nmpc` (NMPC = nonlinear MPC, formulation-agnostic name). New Crocoddyl backend modules added in `t1_nmpc/wb/`; acados modules deleted in place. The `t1_cmpc` directory is left unused.

**Pre-teardown safety:** `git tag acados-port-final` on the current HEAD before any deletion — the full acados port stays recoverable as a comparison oracle via one `git checkout`. New work proceeds on a fresh branch (e.g. `crocoddyl-port`).

### File plan

**KEEP unchanged (infra reused by the builder):**
- `t1_nmpc/model.py`, `wb/model_wb.py` — pinocchio model (composite Translation+SphericalZYX base, 27 joints, foot contact frames)
- `wb/config_wb.py` — all domain params (`Q`, `R`, `friction_mu`, `foot_rect`, CoP, torque/joint limits, swing weights, nominal posture, foot-constraint gains, gait timing)
- `wb/gait_wb.py` — gait schedule + swing trajectories (`mode_to_stance`, `Gait`) — trivial for M0 (constant double-support), central for M1
- `sim/mujoco_runtime.py`, `sim/_sim_util.py`, `runtime/transport.py`, `runtime/mujoco_transport.py`, `runtime/sdk_transport.py`

**REFINE (keep domain logic, drop acados transcription):**
- `wb/reference_wb.py` — keep `build_reference`/`filter_command`/pose integration; adapt `x_ref` output from 68-dim acados state to 66-dim crocoddyl `[q;v]` (drop the `s,v_s` path slots — the deferred M2 contouring feature). **Scope note:** the M0 stand uses a *constant nominal* reference built trivially, so the full `build_reference` (velocity-command → per-node refs) adaptation is exercised at **M1**; M0 only requires the 68→66 state-shape helper.
- `wb/cost_wb.py` — **extract** `_relaxed_barrier`, `_contact_barrier_args` (friction/CoP residuals), `_swing_foot_residual`, `_foot_collision_residual` into the new crocoddyl cost module; **delete** the acados CONL builders (`build_cost_conl`, `build_residual*`)
- `wb/execution_wb.py` — adapt `to_joint_command_wb` to consume `CrocoResult` (recovered τ + predicted q/v) instead of the acados `u_phys_traj`
- `runtime/control_loop.py` — keep the MPC-rate/sim-rate decoupling loop structure; swap the acados solve for `CrocoMPC.step`

**DELETE (pure acados):**
- `wb/ocp_wb.py`, `wb/projection_wb.py`, `wb/mpc_wb.py`, `wb/grid_wb.py`, `wb/constraints_wb.py`, `mpc_result.py`
- `runtime/measure_deploy.py`
- `sim/wb_stand_gate.py`, `sim/wb_walk_gate.py`, `sim/wb_walk_view.py`
- the `.acados_wb` codegen cache (445 MB)
- acados-backend tests: `test_wb_ocp`, `test_wb_projection`, `test_wb_cost`, `test_wb_constraints*`, `test_wb_cost_walk`, `test_wb_mpc_walk`, `test_wb_default_discrete`, `test_wb_warmstart`, `test_wb_flow`, `test_wb_grid`, `test_wb_torque`
- (keep infra tests: `test_model`, `test_wb_config*`, `test_wb_gait`, `test_wb_swing`, `test_wb_reference`, `test_mujoco_runtime`, `test_runtime_*`, `test_execution`, `test_env`, `test_sim_util`, `test_sysid_friction`, `test_wb_model_rbd`)

**CREATE (new Crocoddyl backend):**
- `wb/croco_problem.py` — `T1ProblemBuilder`
- `wb/croco_costs.py` — cost/constraint residual builders
- `wb/croco_mpc.py` — `CrocoMPC` driver + `CrocoResult`
- `sim/wb_stand_croco.py` — closed-loop stand gate
- `tests/test_croco_problem.py`, `tests/test_croco_costs.py`, `tests/test_croco_mpc.py`

---

## 3. `T1ProblemBuilder` (`wb/croco_problem.py`)

Pure construction, no solver, no mutable state — independently testable.

**`__init__(cfg: WBConfig, wb: WBModel)`** caches, once: `state = StateMultibody(model)`, `actuation = ActuationModelFloatingBase(state)`, the two foot frame ids, `LWA = pin.LOCAL_WORLD_ALIGNED`, `dt = cfg.dt`, `N = cfg.N`.

**`make_node(stance_fids, refs, terminal=False) → IntegratedActionModelEuler`:**
- `nu = nv + 6*len(stance_fids)` (native per-node control dim: 45 double-support, 39 single — same method serves M0 and M1).
- `contacts = ContactModelMultiple(state, nu)`; per stance foot `ContactModel6D(state, fid, placement, LWA, nu, gains)`, where `placement` = the foot's planted SE(3) and **`gains = [kp, kd]` = `config_wb` foot-constraint feedback gains** (`foot_pos_err_gain_z`, `foot_linvel_err_gain`) — t1_controller's `foot_constraint` Baumgarte stabilization.
- `costs = build_costs(state, actuation, nu, refs, cfg, stance_fids)` (§4).
- `dam = DifferentialActionModelContactInvDynamics(state, actuation, contacts, costs)`.
- return `IntegratedActionModelEuler(dam, 0.0 if terminal else dt)`.

**`build_stand_problem(x0, refs) → ShootingProblem`** (M0): all `N` running nodes double-support (`stance=[L,R]`), references hold the nominal stand (`x_ref = nominal`, `com_ref = com0`, zero velocity, `swing_target = None`); terminal node `terminal=True`. Models built per-node (M1-ready; construction is ~µs).

---

## 4. Cost & constraint mapping (`wb/croco_costs.py`)

`build_costs(state, actuation, nu, refs, cfg, stance_fids) → CostModelSum`. All terms map to **native crocoddyl residuals** reading `config_wb`. M0 set:

| t1_controller term | crocoddyl residual | weight/param source |
|---|---|---|
| posture + base-pose + velocity tracking | `ResidualModelState(x_ref)` + `ActivationModelWeightedQuad(Q)` | `config_wb.Q` (68→66) |
| CoM/base target | `ResidualModelCoMPosition(com_ref)` | base-vel weight (`com0` for M0) |
| input reg (joint accels + contact forces) | `ResidualModelControl` + `WeightedQuad` (qdd weights on `a[6:33]`, wrench weights on force block, tiny on `a[0:6]`) | `config_wb.R` |
| joint-torque soft-cap | `ResidualModelJointEffort` (recovered τ) | `jointtorque_weight/scale` |
| friction cone + CoP-in-rect + unilateral (per stance foot) | `ResidualModelContactWrenchCone(WrenchCone(R_foot, μ, foot_rect))` + `ActivationModelQuadraticBarrier` | `friction_mu`, `foot_rect` |
| joint position limits | `ResidualModelState` (joint block) + `QuadraticBarrier(ActivationBounds(lo,hi))` | `joint_lower/upper` |
| torque limits | `ResidualModelJointEffort` + `QuadraticBarrier(±τ_lim)` | `torque_limit` |

All named residual/activation classes verified present in the installed crocoddyl 3.2.1.

**Deferred to M1** (`build_costs` skips the block when `refs.swing_target is None`): swing-foot tracking (`ResidualModelFrameTranslation/FramePlacement/FrameVelocity`), arm-swing reference, foot-collision (`ResidualModelPairCollision`), and the exact relaxed-barrier custom `ActivationModel` (math already extracted from `cost_wb._relaxed_barrier`).

---

## 5. `CrocoMPC` driver + MuJoCo closure (`wb/croco_mpc.py`)

**`CrocoMPC(cfg, wb)`** owns the builder, a persistent `ShootingProblem`, and one `SolverIntro`. `__init__`: build the stand problem at nominal `x0`, `solver = SolverIntro(problem)`, warm-start from `quasiStatic`, no callbacks.

**`step(x_meas, refs) → CrocoResult`:**
1. `problem.x0 = x_meas`.
2. Refresh references: mutate each running node's tracking-residual `.reference` in place (constant for M0; velocity→ref feed for M1).
3. Warm-start shift: `xs ← xs[1:]+[xs[-1]]`, `us ← us[1:]+[us[-1]]`.
4. **`solver.solve(xs, us, 1, False, reg)`** — `maxiter=1` single-RTI (= OCS2 `sqpIteration=1`; 13 ms/iter per spike).
5. Extract: `u0 = solver.us[0] = [a(33); forces]`; read the recovered joint torques τ from the solved ContactInvDynamics node data (τ is a byproduct of the inverse-dynamics calc; the exact data attribute is pinned in the plan); read predicted `q,v` at sample-ahead time.
6. Return `CrocoResult(xs, us, tau0, a0, q_des, v_des, diag={iters, stop, cost, isFeasible})`.

**State mapping (reuse):** crocoddyl `[q(33: xyz, euler-ZYX, 27 joints); v(33)]` is the port's existing pinocchio mapping; dropping the acados `s,v_s` slots, the MuJoCo↔pinocchio conversion in `model.py`/`mujoco_transport` carries over unchanged.

**Control extraction (refine `execution_wb`):** ContactInvDynamics hands τ directly, so `tau_ff = τ`, PD targets = predicted `q,v` — cleaner than the acados path (which reconstructed `tau_ff` via RNEA).

**Closed-loop gate (`sim/wb_stand_croco.py`):** read MuJoCo state → `CrocoMPC.step` → `to_joint_command_wb` → apply via `mujoco_transport` (joint PD + `tau_ff`) → step sim → repeat, with MPC-rate (28.5 Hz) / sim-rate (1 kHz) decoupling (ZOH command + PD between updates), via the refined `control_loop.py`.

**Error handling:** `step` checks `isFeasible` + non-finite on `solver.us[0]`; on failure returns the previous command (ZOH) and flags `diag.failed`; the gate counts these as `n_solver_failures`.

---

## 6. Testing & acceptance

**Unit tests (TDD, pure crocoddyl, no MuJoCo):**
- `test_croco_problem.py` — node count `N+1`; double-support node `nu=45`, `nh=18`; single-stance node `nu=39`, `nh=12`; stand problem solves with `‖h‖ < 1e-5`, foot force ≈ body weight, small base drift.
- `test_croco_costs.py` — residuals build at correct dims; weights come from `config_wb` (state-reg == mapped `Q`, control-reg == `R`, wrench-cone params == `foot_rect`/`μ`); cone-violating force produces a positive penalty.
- `test_croco_mpc.py` — `step()` returns finite τ at `maxiter=1`; warm-start shift + in-place reference mutation (no model rebuild); steps from a perturbed `x0` drive back toward the stand.
- update `test_execution` — `tau_ff` from `CrocoResult.tau`.

**M0 acceptance gate (`sim/wb_stand_croco.py --duration T --log out.npz`)** prints `STAND_GATE={...}`; thresholds mirror the acados M0 gate (direct baseline):

| metric | acados M0 (passed) | crocoddyl M0 target |
|---|---|---|
| `peak_tilt_rad` | 0.0279 | ≤ ~0.03 |
| `final_tilt` | 0.003 | < 0.01 |
| `base_z` steady | 0.6711 | ≈ 0.6734 |
| `n_solver_failures` | 0 | 0 |
| `max_abs_tau` | within limits | within `torque_limit` |

**Telemetry:** per-MPC-tick logger (adapting the deleted `wb_walk_gate` pattern) — solver diag (`iters, stop, cost, isFeasible, ‖h‖`) + physical (tilt, base_z, foot forces) → `out.npz`.

---

## 7. Roadmap beyond M0 (not in this spec)
- **M1 — walking:** gait scheduler drives per-node `stance_fids` (via `gait_wb.mode_to_stance`), swing-foot tracking costs, contact-switch node rebuilding, relaxed-barrier custom activation, foot-collision; closed-loop forward walk in MuJoCo.
- **M2 — contouring/tracking:** augment state with `s, v_s`, custom progress-coupled tracking residual; hand/joint motion tracking (the `t1_motion_tracking` feature); the `mpc-rl` policy-reference interface.
- **Hardware deploy:** the `sdk_transport` path.

---

## 8. Deviations & M1 backlog

Faithfulness deviations confirmed by the final M0 review (2026-06-25). Immaterial to the M0 stand; load-bearing for M1 walking.

- **Terminal cost (Q_final / terminal_scale):** The M0 terminal node reuses the running `cfg.Q` at weight 1.0. `cfg.Q_final` and `cfg.terminal_scale` (= 4.0) are defined in `config_wb` but are NOT wired into `croco_costs.build_costs` or `T1ProblemBuilder.make_node`. This is immaterial for the M0 double-support stand (terminal ≈ nominal throughout), but the receding-horizon terminal penalty shapes push-off timing during walking. **M1 action:** add a `terminal=True` branch in `build_costs` that scales state-tracking by `Q_final · terminal_scale`; thread it through `make_node(terminal=True)`.

- **WrenchCone foot rotation:** `croco_costs.build_costs` constructs each `WrenchConeResidual` with `R_foot = np.eye(3)` (world-aligned identity). This is valid at a flat, nominal-yaw stand where the foot frame is near-axis-aligned. During push-off / yawed walking the friction cone and CoP bounds should be expressed in the foot's actual (yawed) rotation frame. **M1 action:** pass the foot frame's rotation `data.oMf[fid].rotation` from `_planted` through `build_costs` (or compute it inside) and use it as `R_foot` in `WrenchConeResidual`.

- **`CrocoMPC.reset` contact placements:** `reset` re-initialises `_xs`/`_us` but does NOT rebuild the shooting problem with fresh contact placements. The contacts remain pinned to the analytical nominal foot placements computed at construction time (from `_nominal66()`), not the measured foot positions from the incoming `x0_68`. The Baumgarte stabilisation corrects the tiny offset, which is fine for M0. **M1 action:** on `reset`, call `self.builder.build_stand_problem(x66)` (or a targeted `update_planted`) to refresh contact placements from the measured state before the first warm-start solve.
