# Clean base rewrite — design

**Date:** 2026-06-26
**Branch:** aligator-port
**Status:** design (approved); plan to follow.

## Problem

The `aligator-port` branch carries three tangled lineages: the **live aligator**
controller, a **dead crocoddyl** baseline (now preserved in branch `crocoddyl-walk-m1`),
and **vestigial acados** references. `CLAUDE.md` still documents the long-abandoned acados
formulation (wrong solver, files that no longer exist, even "no git repo" is false). The
result is bloat: dead modules, a `_aligator`/`_wb` suffix soup left over from when multiple
formulations coexisted, three different files all named some variant of "model", 652 K of
crocoddyl-era SDD process history, and a stack of abandoned-approach design docs.

**Goal:** a clean, structured base form of the project to start future sessions on — the
live aligator whole-body NMPC, nothing else, organised so the structure reveals intent.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Depth | **Prune + restructure + simplify** (deepest) |
| Docs / SDD history | **Aggressive prune** — delete abandoned-approach docs and the entire `.superpowers/sdd/` history |
| Restructure reach | **Full two-layer split** — `robot/` (plant) + `wb/` (controller); top-level shared files move into `robot/` |
| Controller package name | Keep **`wb/`** (whole-body — domain term, stable across solver swaps, matches the OCS2 reference) |
| Config objects | **Co-locate** `WBConfig` + `AligatorConfig` in one `wb/config.py` (two dataclasses, one module) — *not* fused into one object (deferred; YAGNI + risk) |

## Key facts established (verified, not assumed)

- The live aligator path **imports zero croco modules**. Every "croco" string inside
  aligator files is an explanatory comment. Croco is cleanly severable.
- `reference_wb.py` is **dead** — imported only by `croco_mpc.py` / `croco_walk.py` (both
  deleted) and its own test.
- The live aligator path has **no env toggles** (`T1_U*` toggles live only in `croco_walk.py`).
- **The async deployment loop is croco-interface and unused by aligator** (discovered
  during planning, user-confirmed for removal): `to_joint_command_wb` reads the croco
  `MPCResult` fields `.x_traj`/`.u_traj`; `AligatorResult` has `.xs`/`.us` lists instead, so
  they are incompatible. The live aligator path uses `freeflyer_command` (reads `.us`) +
  its own synchronous loop in `sim/wb_walk_aligator.py`. `run_loop` (`control_loop.py`) is
  used only by the deleted croco sims and `test_runtime_loop`; `to_joint_command_wb` only by
  `run_loop`, deleted croco code, and `test_wb_execution` (which fakes a croco `_Result`).
  → the whole cluster (`control_loop.py`, `to_joint_command_wb`, `test_runtime_loop`,
  `test_wb_execution`) is **deleted**, not ported. `sdk_transport.py` (hardware target) uses
  neither. The async loop is rebuilt against `AligatorResult` when deployment is a milestone.
- **`MPCResult` (`mpc_result.py`) is dead** (discovered during planning): its only consumers
  are `croco_mpc.py` (deleted) and `test_mpc_result.py`. Aligator uses inline
  `AligatorResult`. → `mpc_result.py` + `test_mpc_result.py` are **deleted**, not moved to
  `robot/result.py`. `robot/` therefore has only `model.py` / `config.py` / `execution.py`.
- **`execution_wb.py` is dead once `to_joint_command_wb` is removed**: its only other content
  is an unused `pd_torque` re-export (`test_execution` imports `pd_torque` directly from the
  real module). → `execution_wb.py` is **deleted**; `wb/execution.py` is simply
  `aligator_exec.py` (`extract_tau_ff`) renamed — no merge.
- `[tool.setuptools.packages.find]` already globs `t1_nmpc*`, so the new `robot/`
  subpackage needs only an `__init__.py` — no pyproject change (only the stale "acados"
  description string is fixed).
- **Regression gate: 51 passed, 1 xfailed** (~13 s) — established empirically by running the
  suite with every deletion-targeted test file excluded (baseline was 55/1 with croco
  present; −1 `test_runtime_loop`, −2 `test_wb_execution`, −1 `test_mpc_result`). The 1 xfail
  is the Phase-2 lateral-balance gate (known-open).

## Target structure

```
t1_nmpc/
  robot/                 # NEW: formulation-agnostic plant
    __init__.py
    model.py             # was t1_nmpc/model.py      (load_model, T1_URDF_PATH, joint/frame conventions)
    config.py            # was t1_nmpc/config.py     (MPCConfig, JointCommand, make_config, PD gains, mass)
    execution.py         # was t1_nmpc/execution.py  (pd_torque)
  wb/                    # the whole-body aligator NMPC controller
    __init__.py
    config.py            # was config_wb.py + config_aligator.py   (WBConfig + AligatorConfig co-located)
    dynamics.py          # was model_wb.py     (cpin symbolic RBD: M, nle, foot Jacobians)
    ode.py               # was aligator_model.py (manifold + RK4 ODE + nominal_stand_x + build)
    ocp.py               # was aligator_walk.py  (make_stage, build_problem, build_gait_cycle)
    swing.py             # was aligator_swingz.py (SwingZBaumgarte)
    mpc.py               # was aligator_mpc.py   (AligatorMPC)
    state.py             # was aligator_state.py (mujoco<->freeflyer map, freeflyer_command)
    execution.py         # was aligator_exec.py  (extract_tau_ff)   [execution_wb.py deleted: dead]
    gait.py              # was gait_wb.py        (Gait, SLOW_WALK, contact flags, swing_z)
  runtime/               # kept; imports repointed to robot/ + wb/  [control_loop.py deleted: croco-only]
    __init__.py
    transport.py
    mujoco_transport.py
    sdk_transport.py
sim/
  __init__.py
  mujoco_runtime.py      # kept; imports repointed to robot/
  state.py               # was wb_state.py
  walk.py                # was wb_walk_aligator.py  (the live runner: stand + walk)
  _sim_util.py
tests/                   # renamed to mirror modules; suite stays 51 passed / 1 xfailed
docs/
  superpowers/specs/2026-06-26-aligator-native-port-design.md   (kept)
  superpowers/specs/2026-06-26-aligator-port-scoping.md          (kept)
  superpowers/specs/2026-06-26-clean-base-design.md              (this doc)
  superpowers/plans/2026-06-26-aligator-native-port.md           (kept)
  2026-06-25-t1controller-divergences.md                          (kept)
CLAUDE.md                # fully rewritten for the aligator reality
```

## Deletions (all recoverable via git / branch `crocoddyl-walk-m1`)

- **Modules:** `wb/croco_{activations,collision,contact,costs,mpc,problem,swingz,walk}.py`
  (8), `wb/reference_wb.py`, `wb/execution_wb.py` (dead after `to_joint_command_wb` removal),
  `t1_nmpc/mpc_result.py` (dead `MPCResult`), `runtime/control_loop.py` (croco-interface
  async loop, unused by aligator).
- **Sim/spikes:** `sim/wb_stand_croco.py`, `sim/wb_walk_croco.py`, the whole `spikes/`
  directory (3 croco spikes).
- **Tests (12 deleted):** `test_croco_costs`, `test_croco_mpc`, `test_croco_problem`,
  `test_croco_walk_costs`, `test_croco_walk_mpc`, `test_croco_walk_problem`,
  `test_wb_reference`, `test_wb_stand_croco`, `test_wb_walk_croco`, `test_runtime_loop`
  (croco smoke), `test_wb_execution` (`to_joint_command_wb` croco-interface),
  `test_mpc_result` (dead `MPCResult`).
- **Docs:** `docs/acados_exact_elimination_pipeline.md`, and under `docs/superpowers/`:
  `*reduced-basis*`, `*d4-event*`, `*crocoddyl-port*`, `*crocoddyl-walk-m1*` (plans + specs).
- **History:** the entire `.superpowers/sdd/` directory (43 files, 652 K of review diffs +
  crocoddyl-era task ledgers). The aligator progress ledger is summarised forward into the
  rewritten CLAUDE.md before deletion.

## Renames / merges (the restructure)

Every move is a **pure rename or co-location — no logic edits**. One file absorbs a sibling:

- `wb/config.py` = `config_wb.py` body + `config_aligator.py` body (both dataclasses +
  both `make_*` factories). Consumers change import path only; object identities unchanged.
- `wb/execution.py` = `aligator_exec.py` (`extract_tau_ff`) renamed — no merge
  (`execution_wb.py` is deleted as dead). `state.py` imports `extract_tau_ff` from `.execution`.

The three `robot/` files move verbatim; import sites across `wb/`, `runtime/`, `sim/`,
and tests repoint `t1_nmpc.{model,config,execution}` →
`t1_nmpc.robot.{model,config,execution}` (exact site list is enumerated in the plan).

## Simplify pass — strict rule: behavior-preserving only

Only remove what is **provably dead** (zero-reference local vars, F401 imports, config
fields with no reader, dead files) and consolidate **duplicated** helpers. **No numeric or
formulation change whatsoever** — every cost weight, gait timing, solver knob, terminal
weight, Baumgarte gain stays byte-identical. Known-safe candidates: dead `ok`/`ndx` vars,
unused `field` import, the config co-location above. The hot-path `extract_tau_ff`
ode-reallocation is a *possible* perf cache (numerically identical) — flagged, optional,
not required for this rewrite. If a simplification cannot be proven behavior-preserving, it
is **out of scope** and left alone.

## CLAUDE.md + docs

`CLAUDE.md` is **fully rewritten** to describe reality: aligator (ProxDDP) whole-body NMPC,
the `robot/` + `wb/` layout, the conda `t1mpc` env + run preamble (acados env vars dropped),
the real test/sim commands, the live invariants (Euler→freeflyer state map, hard-constraint
contact handling, accel-level Baumgarte swing-z, gait-cycle-longer-than-horizon, serial-only
walk due to the GIL+C++ parallel-LQ segfault), and current status (stand PASS; walk steps +
goes forward but topples laterally ~1.5 s — lateral CoM-sway reference is the open problem).
The stale `pyproject.toml` description string is updated. The 4 kept docs stay as-is.

## Validation gate (run after **every** step)

1. `pytest tests/ -q` reproduces **exactly 51 passed, 1 xfailed** (after the deletions; test
   files renamed but the surviving test bodies are byte-identical apart from import paths).
2. `grep -rn` finds **zero** references to any deleted module or symbol (`croco_`,
   `reference_wb`, `execution_wb`, `mpc_result`, `to_joint_command_wb`, `control_loop`,
   `run_loop`, old `t1_nmpc.config`/`t1_nmpc.model`/`_wb`/`_aligator` paths) across
   `t1_nmpc/`, `sim/`, `tests/`.
3. Closed-loop stand (the Phase-1 gate test) holds unchanged: planned `fz/(m·g) ∈ [0.9,1.1]`,
   solve p90 < 25 ms.

Test command (project preamble):

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
```

## Sequencing (each step gated by the validation suite, each a separate commit)

1. **Delete dead code** (croco modules + sims + spikes + tests; `reference_wb`,
   `execution_wb`, `mpc_result`, `control_loop`; `test_runtime_loop`, `test_wb_execution`,
   `test_mpc_result`) → verify **51 passed, 1 xfailed**.
2. **Move/rename** into `robot/` + `wb/` (+ sim renames; config co-location) with all imports
   repointed and test files renamed → verify green.
3. **Simplify** (provably-dead removal only) → verify green.
4. **Prune docs + rewrite CLAUDE.md** (+ pyproject description); delete `.superpowers/sdd/`.

Delete, move, and simplify are kept as distinct commits so any regression bisects to the
exact change.

## Out of scope

- Any change to MPC numerics, formulation, gait, or cost weights.
- Fixing the open lateral-balance walk problem (tracked separately; the xfail stays xfail).
- Fully fusing `WBConfig` + `AligatorConfig` into one object.
- Hardware/SDK bring-up.
```
