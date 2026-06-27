# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python **whole-body nonlinear MPC (NMPC)** for the **Booster T1 humanoid**, built on **CasADi `Opti` + Fatrop** (`whole_body_rnea` formulation) + **pinocchio.casadi** + **MuJoCo** (physics sim). The OCP is a full-order whole-body inverse-dynamics NLP: state carries the FreeFlyer configuration and velocity, control carries joint accelerations, 8-corner 3D contact forces (`nf=24`), and — on the first `tau_nodes` nodes — joint torques. Fatrop (interior-point) solves the sparse NLP each MPC tick.

The formulation is a port of **wb-mpc-locoman** (Molnar et al., RA-L 2025, *Whole-Body Inverse Dynamics MPC*, arXiv:2511.19709) adapted to T1 kinematics. `t1_controller` (OCS2 `humanoid_mpc`) remains a **data** source for T1 numbers (joint limits, standing pose, foot geometry) but is **not** a formulation authority for this backend. Deliberate divergences are logged in `docs/2026-06-25-t1controller-divergences.md`.

The north star is world-frame hand tracking while walking. **M0 (stand) is PASS.**

## Environment & commands

This **is a git repo**. Everything runs through a load-bearing command preamble in the conda env `t1mpc` (Python 3.10: pinocchio 4.0, casadi 3.7.2+Fatrop, numpy 2.2, mujoco 3.10). The env has `t1_nmpc` installed editable (`pip install -e . --no-deps`); use conda exclusively.

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
# Expected: 10 passed

# Live sim runner (stand)
python sim/stand.py [--duration 4.0] [--view] [--gif out.gif]
```

## Architecture

### Layout

```
t1_nmpc/
  robot/          plant layer
    assets/t1_description/   vendored t1.urdf + 30 meshes
                             (package_dirs root = robot/assets)
    model.py      FreeFlyer T1 load (package_dirs); add 8 corner contact frames;
                  mass via computeTotalMass; q0 from nominal_joint_pos; base = 'Trunk'
    config.py     T1WBConfig: horizon (nodes, dt_min, dt_max, tau_nodes), Q/R diagonals,
                  corner geometry, μ=0.4, nominal_joint_pos (shallow crouch),
                  base height 0.6734, joint limits
  wb/             whole-body-RNEA CasADi/Fatrop controller
    dynamics.py   cpin: state_integrate/difference (FreeFlyer manifold),
                  rnea_dynamics (8-corner force application via f_ext accumulation)
    ocp.py        Opti transcription: adaptive vars, params, RNEA path constraint,
                  contact/friction/velocity constraints, tracking objective,
                  gap-closing-equality-first Fatrop ordering invariant
    gait.py       biped contact schedule: stand = all 8 corners in contact;
                  2 physical-foot swing groups, each expanded ×4 to corner flags
    mpc.py        WholeBodyMPC: build Opti, init Fatrop solver_function,
                  update_params, solve, warm-start (prev solution incl. lam_g)
    state.py      MuJoCo <-> pinocchio FreeFlyer map (the §9 conversion —
                  single source of truth); first-node command extraction (q_des, v_des, τ_j)
  runtime/        transport layer (protocol unchanged; repointed to new mpc/state)
    transport.py           Transport protocol (read_state/write_command/now)
    mujoco_transport.py    MujocoTransport (sim-backed transport)
    sdk_transport.py       SdkTransport (Booster SDK, UNTESTED)
sim/
    mujoco_runtime.py   2000 Hz physics; settle to FK-consistent stand height 0.6734
    stand.py            closed-loop stand runner: metrics (Σfz/mg, tilt, solve p90),
                        viewer, GIF
    _sim_util.py        tilt_from_quat_wxyz, upright_ok
```

### State and control

**State `x ∈ ℝ⁷¹`** = `[q(36), v(35)]` where `q = [pos(3), quat_xyzw(4), joints(29)]`,
`v = [v_lin_local(3), v_ang_local(3), jvel(29)]`. Uses **JointModelFreeFlyer** base (quaternion). Both base-velocity parts are in the **body-local** frame (pinocchio FreeFlyer convention). MuJoCo ↔ pinocchio conversion is in `wb/state.py` — see invariants below.

**Decision delta-state `dx ∈ ℝ⁷⁰`** = `[dq(35), dv(35)]`, reconstructed per node via `cpin.integrate` on the manifold.

**Input (adaptive per node):**
- Nodes `i < tau_nodes`: `u_i = [a(35), forces(24), τ_j(29)]` — width 88.
- Nodes `i ≥ tau_nodes`: `u_i = [a(35), forces(24)]` — width 59.

`forces(24)` = 8 corner 3D forces in world frame (nf=24). `a(35)` = joint accelerations (base 6 + joints 29).

### Solver

`WholeBodyMPC` builds a CasADi `Opti` NLP and calls `opti.to_function(...)` to produce a `solver_function` backed by **Fatrop** (`opti.solver('fatrop', opts)`). Each MPC tick calls the solver function with the current state as `x_init` parameter and the previous solution as warm-start. The horizon uses a geometric time grid: `dt_i = dt_min · γ^i`, `γ = (dt_max/dt_min)^{1/(nodes-1)}`.

### Contact model

**8 unilateral 3D corner forces** (4 per foot) — each `f_z ≥ 0` and inside a linearised friction cone (`μ=0.4`). The 4 corners of a foot share one ankle-roll parent joint; their world forces are rotated to the body frame and accumulated into the same `f_ext` slot in `rnea_dynamics`. Stand: all 8 corners always in contact.

### Runtime

`WholeBodyMPC.solve(x_meas)` updates `x_init`, calls the Fatrop solver function, and returns the first-node torque command via `state.first_node_command`. The sim runner `sim/stand.py` drives the closed loop at ~50 Hz MPC / 2000 Hz physics, printing Σfz/mg, max tilt, and p90 solve time.

## Invariants to respect

- **MuJoCo ↔ pinocchio FreeFlyer state map — single source of truth in `wb/state.py`.**
  The critical non-obvious rule: MuJoCo `qvel[0:3]` is **world**-frame; pinocchio FreeFlyer `v[0:3]` is **body-local**. The conversion is `v[0:3] = R(q)ᵀ · qvel[0:3]`. Omitting this rotation is a latent bug (only harmless exactly at upright stand). Do not re-derive this anywhere else.

- **Fatrop gap-closing-equality-first.** At each Opti stage the state-transition (gap-closing) equality **must be added before any inequality** — otherwise Fatrop aborts with `Constraint Jacobian must start with gap-closing constraint`. The OCP code must maintain this ordering.

- **Fatrop staircase variable structure.** `include_acc=True` (accelerations in the input) is required for Fatrop's structure detection. The adaptive input width (`τ_j` present only on the first `tau_nodes` nodes) changes stage width mid-horizon; `structure_detection='auto'` must correctly segment this.

- **f_ext accumulation.** The 4 corners of each foot share one `parentJoint` (the ankle-roll joint). `rnea_dynamics` must accumulate all 4 corner contributions into the same `f_ext` slot — not overwrite. This is verified correct.

- **No aligator/ProxDDP.** The aligator backend is removed. Do not re-introduce it.

- **No YAGNI speculation.** Walking gait, arm/EE manipulation, Ipopt/OSQP, hardware/SDK, MJCF vendoring are deferred. Keep the codebase lean.

- **Cite or log divergences.** When a numerical value or formulation choice diverges from `t1_controller`, log it in `docs/2026-06-25-t1controller-divergences.md`.

## Status & docs

- **M0 (stand): PASS.** `fz_ratio_p50=0.9999`, `max_tilt=1.95°`, no fall, `solve_p90≈28.7 ms`.
- **M1 (forward walk): deferred** until stand is rock-solid and walking gait (swing/contact-schedule + Fatrop structure detection under changing contact rank) is validated.
- **M2 (walk + hand tracking): deferred** until M1 closes.

### Docs

- `docs/superpowers/specs/2026-06-27-wb-rnea-t1-port-design.md` — wb-rnea port design spec (authoritative architecture)
- `docs/2026-06-25-t1controller-divergences.md` — ledger of deliberate divergences from t1_controller
