# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python **whole-body nonlinear MPC (NMPC)** for the **Booster T1 humanoid**, built on **aligator (ProxDDP / proximal augmented-Lagrangian DDP)** + **pinocchio** + **MuJoCo** (physics sim). The OCP is a full-order kinodynamic nonlinear program: state carries the whole-body free-flyer configuration and velocity, control carries joint accelerations and 6D contact wrenches per foot, and aligator's ProxDDP solver runs each MPC tick. It is a faithful port of the OCS2 `humanoid_mpc` controller (`t1_controller`, in `../t1_controller/`). The reference is hardware-proven, so **faithfulness to `t1_controller` is the governing design constraint**: every cost weight, constraint gain, gait timing, and execution rule traces to a cited source in the C++ reference. The north star is world-frame hand tracking while walking (milestone M2); standing (M0) and forward walking (M1) de-risk the foundation.

When you change a numerical value or formulation, find and cite its source in `t1_controller`, or document the deliberate divergence (see `docs/2026-06-25-t1controller-divergences.md` for the ledger format).

## Environment & commands

This **is a git repo**. Everything runs through a load-bearing command preamble in the conda env `t1mpc` (Python 3.10: pinocchio 4.0, aligator, numpy 2.2, mujoco 3.10). The env has `t1_nmpc` installed editable (`pip install -e . --no-deps`); use conda exclusively.

Always run from `/home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc` with this exact preamble:

```bash
PYTHONPATH= conda run -n t1mpc python <args>
```

- `PYTHONPATH=` (empty) is **load-bearing**: it keeps `/opt/ros/humble`'s numpy<2 pinocchio off the path, which otherwise segfaults the conda pinocchio.
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for reproducible tests; raise for deployment.

Common invocations (prefix each with the preamble above):

```bash
# Test suite (the authoritative regression gate)
python -m pytest tests/ -q -p no:cacheprovider
# Expected: 51 passed, 1 xfailed

# Live sim runner (walk)
python sim/walk.py [--vx 0.3] [--duration 4.0] [--view] [--gif out.gif] [--threads 4]
```

## Architecture

### Layout

```
t1_nmpc/
  robot/          plant layer (shared across formulations)
    config.py       MPCConfig, JointCommand, WBConfig (all robot numbers)
    model.py        T1_URDF_PATH, EXPECTED_JOINT_NAMES, RobotModel, load_model
    execution.py    pd_torque
  wb/             controller layer (aligator ProxDDP kinodynamic MPC)
    config.py       WBConfig, AligatorConfig, make_wb_config, make_aligator_config
    dynamics.py     WBModel (pinocchio.casadi cpin symbolic RBD for accel-level terms)
    ode.py          AligatorModel, build_aligator_model, make_ode, nominal_stand_x
    ocp.py          make_stage, build_problem, build_gait_cycle (StageModel factory)
    swing.py        SwingZBaumgarte (accel-level Baumgarte hard swing-z constraint)
    mpc.py          AligatorMPC.reset/step, AligatorResult
    state.py        mujoco_to_freeflyer, freeflyer_command (free-flyer state <-> MuJoCo)
    execution.py    extract_tau_ff (RNEA-based feedforward torque)
    gait.py         Gait, SLOW_WALK, WALK, STANCE_GAIT
  runtime/        transport layer
    transport.py    Transport protocol (read_state/write_command/now)
    mujoco_transport.py  MujocoTransport (sim-backed transport for threaded loop)
    sdk_transport.py     SdkTransport (Booster SDK, robot-only, UNTESTED)
sim/              simulation & runner
    mujoco_runtime.py   MujocoRuntime (2000 Hz physics, sysID armature + viscous damping)
    state.py            wb_state_estimate, wb_reset (free-flyer state from MuJoCo)
    walk.py             Aligator walk runner: headless metrics, GIF, live viewer
    _sim_util.py        tilt_from_quat_wxyz, upright_ok
```

### State and control

**Free-flyer state `x ∈ ℝ⁶⁷`** (aligator path) = `[q(34), v(33)]` where `q = [pos(3), quat_xyzw(4), joints(27)]`, `v = [lin_world(3), ang_local(3), jvel(27)]`. Uses a **JointModelFreeFlyer** base (quaternion), with MuJoCo ↔ pinocchio conversion handled in `wb/state.py`. The 27 MPC joints = the canonical 29 (§A.5) minus the 2 head joints.

**Control `u ∈ ℝ³⁹`** = `[W_l(6), W_r(6), qdd_joints(27)]` — left/right foot 6D contact wrenches + joint accelerations. Kinodynamic formulation: contact wrenches are decision variables, `extract_tau_ff` recovers joint torques via RNEA.

### Solver

`AligatorMPC` builds a `TrajOptProblem` (N=20 nodes, dt=0.035 s, ~0.7 s horizon) and runs ProxDDP with `max_iters=2` by default. The walk path adds `SwingZBaumgarte` — a custom Python `StageFunction` implementing an accel-level Baumgarte hard swing-z equality. Because this is a Python residual, the C++ parallel Riccati solver cannot call it across threads (GIL segfault): **walk is forced SERIAL** (`LQ_SOLVER_SERIAL`). Stand (gait=None, all-C++ residuals) keeps the parallel Riccati. Pass `--threads 4` only for stand or when no gait is set.

### Contact handling

Stance feet get a `FrameVelocityResidual` hard equality (zero velocity, AL-enforced) + `CentroidalFrictionConeResidual` + `CentroidalWrenchConeResidual` (hard `NegativeOrthant` by default). Swing feet get `SwingZBaumgarte` (hard accel-level z equality via AL) + an optional soft forward foot-placement x cost. Contact flags are determined per-node by `Gait.contact_flags(t)` and baked into the stage at build time — no bound-gating needed (aligator AL handles rank changes natively).

### Runtime

`AligatorMPC.step(x_meas, t)` builds (or recedes) the problem and runs ProxDDP. The recede path (`_recede`) calls `replaceStageCircular` + `cycleProblem` to advance the rolling horizon one knot with tip-stage gait time `t + N*dt`. The stand path just updates `x0_init`.

`sim/walk.py` is the sim runner: it wraps `MujocoTransport` (which uses `MujocoRuntime`) and drives the `AligatorMPC` in a closed loop, printing fall time, CoM advance, lateral drift, and p90 solve time.

## Invariants to respect

- **Faithfulness over cleverness.** Match `t1_controller`; cite sources; log deliberate divergences in the ledger.
- **Hard stagewise constraints native to aligator** — friction cone / CoP / contact velocity equality are hard `NegativeOrthant` or `EqualityConstraintSet` constraints, not penalty hacks.
- **Accel-level Baumgarte swing-z** (`swing.py`) is input-coupled so AL enforces it; a position-level constraint is not AL-enforceable at low iteration budgets.
- **Walk is SERIAL.** `SwingZBaumgarte` is a custom Python residual; the C++ parallel LQ solver cannot call Python across threads (GIL segfault). Only disable serial when `gait is None` (stand path).
- **Gait cycle must stay longer than the MPC horizon.** `SLOW_WALK` is 1.7 s; the horizon is ~0.7 s (N=20, dt=0.035). A 1.0 s cycle wraps a full gait period inside the horizon, producing a near-periodic reference the few-iter solver cannot satisfy.
- **Faithfulness, no YAGNI speculation.** The M2 hand-tracking / contouring machinery is absent until its milestone. Keep the codebase lean.

## Status & docs

- **M0 (stand): PASS.** fz/mg ∈ [0.9, 1.1], p90 solve ≈ 14 ms < 25 ms budget; holds upright.
- **M1 (forward walk): advances but topples laterally ~1.5 s.** The robot steps forward (CoM advances, foot lifts confirmed), but falls sideways around 1.5 s. The open problem is **lateral CoM-sway reference**: the few-iteration ProxDDP solver needs the base-y reference explicitly shifted over the stance foot in single support, and the current `_stage_x_ref` lateral shift is not yet sufficient to fully stabilize single-leg balance. This is a reference/planning gap (same gap identified in the divergences ledger), not solver convergence — the OCP solves cleanly. The forward locomotion mechanism in place: velocity-driven base velocity reference (`w_base_vel=3`) drives CoM forward, and an explicit forward foot-placement x target (`foot_place_lookahead * vx`) prevents the swing foot from scuffing backward.
- **M2 (walk + hand tracking): deferred** until M1 closes.

### Docs

- `docs/superpowers/specs/2026-06-26-aligator-native-port-design.md` — aligator port design spec
- `docs/superpowers/specs/2026-06-26-aligator-port-scoping.md` — port scoping notes
- `docs/superpowers/specs/2026-06-26-clean-base-design.md` — clean-base refactor spec
- `docs/superpowers/plans/2026-06-26-aligator-native-port.md` — aligator port task plan
- `docs/superpowers/plans/2026-06-26-clean-base.md` — clean-base task plan
- `docs/2026-06-25-t1controller-divergences.md` — ledger of deliberate divergences from t1_controller
