# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python **whole-body nonlinear MPC (NMPC)** for the **Booster T1 humanoid**, built on **acados** (SQP/RTI solver) + **pinocchio.casadi** (`cpin`, symbolic rigid-body dynamics) + **MuJoCo** (physics sim). The OCP is a full-order kinodynamic nonlinear program — the state carries the whole-body configuration and velocity, the control carries joint accelerations and contact wrenches, and acados solves the resulting NLP each tick. (An early centroidal-reduced formulation was abandoned; "centroidal" now survives only as the origin of a few shared-infrastructure files, never as the project's model.) It is a *faithful port* of the OCS2 `humanoid_mpc` controller (`t1_controller`, in `../wb_humanoid_mpc/`). The reference is hardware-proven, so **faithfulness to `t1_controller` is the governing design constraint**: every cost weight, constraint gain, gait timing, and execution rule traces to a cited source in the C++ reference. The north star is world-frame hand tracking while walking (milestone M2); standing (M0) and forward walking (M1) de-risk the foundation.

When you change a numerical value or formulation, find and cite its source in `t1_controller`, or document the deliberate divergence (see `docs/2026-06-25-t1controller-divergences.md` for the ledger format).

## Environment & commands

There is **no `git` repo, no build step, and no Makefile**. Everything runs through a load-bearing command preamble in the conda env `t1mpc` (Python 3.10: pinocchio 4.0 + cpin, casadi 3.7, numpy 2.2, mujoco 3.10, acados_template). The env has `t1_nmpc` installed editable (`pip install -e . --no-deps`); use conda exclusively — there is no other environment.

Always run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc` with this exact preamble:

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc python <args>
```

- `PYTHONPATH=` (empty) is **load-bearing**: it keeps `/opt/ros/humble`'s numpy<2 pinocchio off the path, which otherwise segfaults the conda pinocchio.
- `ACADOS_SOURCE_DIR` / `LD_LIBRARY_PATH` point acados at its install.
- `OMP_NUM_THREADS=1` is the single-thread default for reproducible tests; deployment harnesses set it higher (e.g. `=4`) to match their MPC core pool.

Common invocations (prefix each with the preamble above):

```bash
# Test suite (the authoritative regression gate — run before claiming anything works)
python -m pytest tests/ -q -p no:cacheprovider

# M0 standing gate — closed-loop 5 s stand in MuJoCo; prints WB_M0={...} JSON, expects PASS
python sim/wb_stand_gate.py

# M1 forward-walk gate — closed-loop 10 s walk; prints WALK_GATE={...} JSON (--view for live viewer)
python sim/wb_walk_gate.py --vx 0.3 --duration 10.0 [--view --speed 1.0]

# Honest deployment timing (async MPC pool + 500 Hz control thread)
OMP_NUM_THREADS=4 ... python t1_nmpc/runtime/measure_deploy.py
```

Run a single test: `python -m pytest tests/test_wb_mpc_walk.py -v`.

**Known test baseline (as of 2026-06-25):** 73 tests, ~60–61 pass, **12 deterministic failures are pre-existing**, not regressions — they are stale-dimension contract tests trailing the M1 constraint/cost refactor (`test_wb_constraints*`, `test_wb_cost*`, `test_wb_ocp::test_build_and_solve_stand` assert an older row layout like a "36-row `con_h`" that the current design emits as 14; `test_mujoco_runtime` waist-skip contracts). `test_runtime_loop::test_threaded_loop_runs_and_reports` is **flaky** (timing-sensitive thread test, passes ~half the time). A failure *outside* this set is a real regression.

**acados codegen is cached, not rebuilt every run.** The first solver build (or any change to `config_wb/model_wb/cost_wb/constraints_wb/ocp_wb`, the URDF, or compile flags) triggers a ~100 s cpin+C codegen into `.acados_wb/c_generated_code/` (override dir via `ACADOS_WB_CODEGEN_DIR`). `build_solver()` (`t1_nmpc/wb/ocp_wb.py`) hashes those inputs (SHA256) and reloads the cached `.so` instantly when the hash matches. Compile flags are baked into the hash *by value*, never read from ambient env — so a config change forces a deliberate rebuild and an unchanged config never silently regenerates. If you see unexpected long rebuilds, something in the hashed set changed.

## Architecture

### The live code path: `t1_nmpc/wb/` (whole-body, the active formulation)

The whole-body subpackage is the proven, in-development controller. Data flow:

```
WBConfig (config_wb.py)            numbers: Q/R weights, gains, limits, gait, arm-swing — all cited to t1_controller
   │
WBModel (model_wb.py)             cpin symbolic RBD: M(q), nle(q,v), foot Jacobians Jl/Jr, joint_torque(x,u) for tau_ff
   │
make_ocp(cfg) / build_solver()    acados OCP assembly + cached codegen   (ocp_wb.py)
   ├─ build_cost_conl / build_residual          (cost_wb.py)   CONVEX_OVER_NONLINEAR: LS tracking + relaxed barriers
   └─ build_con_h / stage_constraint_bounds     (constraints_wb.py)  contact equalities, per-node gated
   │
WholeBodyMPC.step(x_meas, t)      single-RTI solve + warm-start          (mpc_wb.py)
   ├─ build_node_params(...)      folds reference + gait schedule into per-node acados params
   ├─ build_reference(...)        velocity command → per-node base pose/posture/wrench  (reference_wb.py)
   └─ Gait.contact_flags / swing_z / impact_proximity                    (gait_wb.py)
   │
to_joint_command_wb(result)       extracts q_des/qd_des (sampled t+5ms) + tau_ff + foot wrenches  (execution_wb.py)
```

**State `x ∈ ℝ⁶⁸`** = `[q_base(6), q_joints(27), v_base(6), v_joints(27), s(1), v_s(1)]`. The base uses a **`Translation + SphericalZYX` Euler floating base, not a quaternion free-flyer** — so the MPC state maps *directly* onto pinocchio `q`/`v` with no conversion: `q = [pos(3), θ_zyx(3), q_j(27)]`, `v = [v_world(3), euler-rate(3), v_j(27)]` (sphericalZYX's tangent *is* the euler-rate). The head's 2 joints are excluded → 27 actuated joints. `(s, v_s)` are inactive path/contouring slots reserved for M2.

**Control `u ∈ ℝ⁴⁰`** = `[W_l(6), W_r(6), qdd_joints(27), vdot_s(1)]` — left/right foot contact wrenches, joint accelerations, path-rate. This is a kinodynamic formulation: contact wrenches are decision variables, with **no WBC layer** — `to_joint_command_wb` emits joint commands directly.

**acados regime** (`ocp_wb.py`): single-phase `N=31`, `dt=0.035` (~1.085 s horizon, 60 Hz), DISCRETE integrator over a hand-rolled CasADi-differentiated **RK4** step (~2× faster than ERK; do not "fix" this to continuous). Solver is **SQP with `max_iter=1` default (single-RTI)** — the 12-iter ceiling is baked in only for memory sizing; you can raise iters at runtime via `solver.options_set("max_iter", k)` *without* a rebuild. `PARTIAL_CONDENSING_HPIPM` (SPEED), `regularize_method=PROJECT`, `levenberg_marquardt=1e-3`, `MERIT_BACKTRACKING`. Hessian is Gauss-Newton via NONLINEAR_LS (acados numeric `JᵀWJ`), **not** exact — the exact Hessian is indefinite at contact switches and the symbolic version does not compile at -O2.

**Contact handling is the most delicate part.** Stance feet get `ZeroAccel` (6 equality rows/foot, soft Baumgarte PD regulation) + friction-cone + CoP relaxed barriers; swing feet get `ZeroWrench` input bounds + a `SwingZ` cost row tracking a Hermite swing spline. The equality rows are emitted *ungated* (full-rank `con_h`) and **activated per-node via bounds** (`stage_constraint_bounds`: active foot → `lh=uh=0`, inactive → `±∞`). Gating the *expression* to zero instead would make rows rank-deficient → singular KKT → HPIPM `MINSTEP`. In-model algebraic elimination of the contact constraint was tried and proven intractable (the generated Jacobian explodes to 94–255 MB); see `docs/acados_exact_elimination_pipeline.md` and `docs/2026-06-24-zeroaccel-elimination-design.md`.

### Runtime deployment: `t1_nmpc/runtime/`

A `Transport` protocol (`transport.py`: `read_state()`, `write_command()`, `now()`) decouples the controller from its target. Two impls: **`mujoco_transport.py`** (sim) and **`sdk_transport.py`** (Booster B1 SDK — **untested, hardware bring-up pending**; its B1JointIndex↔27-MPC-joint map is flagged as the highest on-robot risk). `control_loop.py::run_loop(...)` runs the async deployment pattern: a free-running MPC thread pinned to a core pool and a separate 500 Hz wall-clock-paced control thread, handing off via lock-free atomic tuple-cell assignment (state→MPC, plan→control). The control thread resamples the latest plan at `t + sample_ahead_s` (5 ms) and applies `τ = τ_ff + kp(q_des−q) + kd(qd_des−qd)`. This is what makes the stand hold even when a solve exceeds the 16.7 ms / 60 Hz budget.

### Simulation & acceptance gates: `sim/`

`sim/` is intentionally small — the sim backend plus the two acceptance gates. `mujoco_runtime.py` wraps MuJoCo at 2000 Hz with system-ID (per-joint armature + viscous damping) baked in to match the MPC's `tau_ff` model; without it, `tau_ff` over-commands the sim. `_sim_util.py` holds shared helpers (`tilt_from_quat_wxyz`, `upright_ok`). **`wb_stand_gate.py` (M0)** and **`wb_walk_gate.py` (M1)** are pass/fail harnesses — each `__main__` prints a `WB_M0=`/`WALK_GATE=` JSON verdict (tilt, base-z, solver-fail count, walk velocity/lateral-drift). Note `wb_stand_gate.py` is also a **shared helper module**: it exports `wb_state_estimate`, `_wb_reset`, `_sample_plan`, `_HEAD_KP/_KD`, which `wb_walk_gate.py` *and* the runtime `mujoco_transport.py` import — don't treat it as a throwaway entrypoint. `wb_walk_view.py` is the live MuJoCo viewer used by `wb_walk_gate.py --view`. (A family of one-off `wb_walk_*` diagnostic/oracle/probe/instrument scripts existed during M1 debugging and was removed in a cleanup; recover from history if you need that style of analysis again.)

### Shared robot config/model: `t1_nmpc/config.py`, `model.py`

`config.py` and `model.py` originated as the Phase-1 centroidal (41-dim, 40 Hz) formulation, but they are **not** vestigial — the whole WB path and both gates import shared pieces from them: `JointCommand`, `make_config`/`MPCConfig` (robot constants: mass, nominal height, joint names, PD gains, friction/contact params), `load_model`, `T1_URDF_PATH`, `pd_torque`, `MPCResult`. New MPC behavior still goes in `wb/` (e.g. `config_wb.py`), not here, but these two files are load-bearing infrastructure. The URDF is the single source of truth at `model.py`'s `T1_URDF_PATH` (`../wb_humanoid_mpc/robot_models/booster_t1/.../t1.urdf`); the canonical 29-joint order (`EXPECTED_JOINT_NAMES`) is enforced at model load. (`RobotModel`'s centroidal compute methods — `centroidal_map`, `inverse_dynamics_torque`, `frame_jacobian` — were pruned as dead code; WB dynamics live in `wb/model_wb.py`. The remaining `com`/`frame_placement` methods are currently unused too.)

## Invariants to respect

- **Faithfulness over cleverness.** Match `t1_controller`; cite sources; log deliberate divergences in the ledger. Resist speculative abstraction — the M2 contouring/cascade machinery is intentionally absent until its milestone needs it (YAGNI).
- **Euler base, direct pinocchio mapping.** Never introduce quaternion↔euler or world↔local velocity conversions into the WB state — the whole point is that `dq/dt = v` holds exactly.
- **Contact equalities stay full-rank and bound-gated**, never expression-gated, never in-model-eliminated.
- **Gait cycle must stay longer than the MPC horizon.** `SLOW_WALK` is 1.7 s by design — a 1.0 s cycle wraps a full gait period inside the horizon, producing a near-periodic reference the few-iter solver cannot satisfy (it failed catastrophically: tilt 1.97, NaN). See the note in `gait_wb.py`.
- **Single-RTI is rate-dependent.** Stable at 60 Hz (`dt=0.035`); diverges at ≤40 Hz. The vestigial `config.py` runs 40 Hz and is not deployable as-is.
- **Codegen flags are hashed by value.** Don't reintroduce env-var-driven compile flags — build/load disagreement silently nukes the `.so` cache.

## Status & docs

- **M0 (stand): PASS.** ~15 ms median solve on a 4-core pool (~64 Hz effective); holds upright, peak tilt ~0.025 rad.
- **M1 (forward walk): machinery complete, closed-loop gate FAILS.** The OCP solves clean (status 0) and the oracle rollout is stable, but the robot falls around step 2 in closed loop — a balance-authority / execution issue, not solver convergence. Open suspects (see `docs/2026-06-25-t1controller-divergences.md`): warm-start ratchet (partial fix shipped), clock-vs-measured contact mode timing, and `tau_ff` model mismatch (armature/viscous damping). **If you work on M1, start by reading that divergences doc and the oracle scripts.**
- **M2 (walk + hand tracking): deferred** until M1 closes.

This project follows a **spec-driven (SDD) workflow** documented in `docs/`: a dated `*-design.md` (the spec — decision table, faithfulness mapping, validation gates), a `*-plan.md` (task-by-task breakdown with source citations and failing-test-first), and `.sdd-progress-*.md` / `sdd-m*-task*-report.md` (per-task completion ledgers). Read the latest design + plan for the milestone you're touching before changing code.
