# Clean Base Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the dead crocoddyl/acados lineage from the `aligator-port` branch and restructure the live aligator whole-body NMPC into a clean `robot/` (plant) + `wb/` (controller) two-layer package, with a rewritten CLAUDE.md — all behavior-preserving.

**Architecture:** Four sequential, individually-green commits: (1) delete dead code, (2) two-layer `robot/` split, (3) `wb/` + `sim/` restructure, (4) simplify + docs/CLAUDE.md. This is a **refactor**, not a feature: there is no new behavior to test. The regression gate is the existing test suite staying at a fixed pass count plus a grep proving zero references to deleted modules. Renames are mechanical `git mv` + word-boundary `sed` sweeps, each followed by a verification grep and the gate suite, so any misfire is caught immediately and bisects to one commit.

**Tech Stack:** Python 3.10, conda env `t1mpc`, pinocchio.casadi (`cpin`), aligator (ProxDDP), MuJoCo, pytest. No build step; package installed editable (`pip install -e . --no-deps`).

## Global Constraints

- **Run everything through the project preamble** (load-bearing — `PYTHONPATH=` empty keeps ROS's numpy<2 pinocchio off the path):
  ```bash
  PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc <args>
  ```
- **Regression gate (the test that governs every task):** `python -m pytest tests/ -q -p no:cacheprovider` must report **`51 passed, 1 xfailed`** after Task 1 and at the end of every task thereafter. The 1 xfail is the Phase-2 lateral-balance gate (known-open) — it must stay xfailed, never xpassed or failed.
- **Behavior-preserving only.** No change to any cost weight, gait timing, solver knob, terminal weight, Baumgarte gain, or numeric constant. Surviving test bodies stay byte-identical except for import paths.
- **Spec:** `docs/superpowers/specs/2026-06-26-clean-base-design.md`. Read it before starting.
- **Use `git mv` / `git rm`** (not raw `mv`/`rm`) so history follows renames.
- Work on the current `aligator-port` branch. There are pre-existing uncommitted working-tree changes to `sim/wb_walk_aligator.py`, `t1_nmpc/wb/aligator_mpc.py`, `t1_nmpc/wb/aligator_walk.py`, `t1_nmpc/wb/config_aligator.py`, `tests/test_aligator_config.py` — **leave them; do not stage or revert them.** Stage only the files each task touches.

---

### Task 1: Delete the dead crocoddyl/acados lineage

Removes all crocoddyl modules, the orphaned async-deployment cluster (`control_loop.run_loop` + `to_joint_command_wb`), the dead `MPCResult`, and their tests. Nothing surviving imports any of these (verified during planning). After this task the suite drops from the croco-present baseline (55) to the clean target **51 passed, 1 xfailed**.

**Files:**
- Delete (modules): `t1_nmpc/wb/croco_activations.py`, `croco_collision.py`, `croco_contact.py`, `croco_costs.py`, `croco_mpc.py`, `croco_problem.py`, `croco_swingz.py`, `croco_walk.py`, `t1_nmpc/wb/reference_wb.py`, `t1_nmpc/wb/execution_wb.py`, `t1_nmpc/mpc_result.py`, `t1_nmpc/runtime/control_loop.py`
- Delete (sim/spikes): `sim/wb_stand_croco.py`, `sim/wb_walk_croco.py`, `spikes/` (whole directory)
- Delete (tests): `tests/test_croco_costs.py`, `test_croco_mpc.py`, `test_croco_problem.py`, `test_croco_walk_costs.py`, `test_croco_walk_mpc.py`, `test_croco_walk_problem.py`, `test_wb_reference.py`, `test_wb_stand_croco.py`, `test_wb_walk_croco.py`, `test_runtime_loop.py`, `test_wb_execution.py`, `test_mpc_result.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a tree where the live aligator path (`wb/aligator_*`, `wb/*_wb` survivors, `runtime/{transport,mujoco_transport,sdk_transport}`, `sim/{mujoco_runtime,wb_state,_sim_util,wb_walk_aligator}`) is intact and self-contained.

- [ ] **Step 1: Delete the dead files**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git rm t1_nmpc/wb/croco_activations.py t1_nmpc/wb/croco_collision.py \
       t1_nmpc/wb/croco_contact.py t1_nmpc/wb/croco_costs.py \
       t1_nmpc/wb/croco_mpc.py t1_nmpc/wb/croco_problem.py \
       t1_nmpc/wb/croco_swingz.py t1_nmpc/wb/croco_walk.py \
       t1_nmpc/wb/reference_wb.py t1_nmpc/wb/execution_wb.py \
       t1_nmpc/mpc_result.py t1_nmpc/runtime/control_loop.py \
       sim/wb_stand_croco.py sim/wb_walk_croco.py \
       spikes/croco_m1_faithful_spike.py spikes/croco_stand_spike.py spikes/croco_walk_spike.py \
       tests/test_croco_costs.py tests/test_croco_mpc.py tests/test_croco_problem.py \
       tests/test_croco_walk_costs.py tests/test_croco_walk_mpc.py tests/test_croco_walk_problem.py \
       tests/test_wb_reference.py tests/test_wb_stand_croco.py tests/test_wb_walk_croco.py \
       tests/test_runtime_loop.py tests/test_wb_execution.py tests/test_mpc_result.py
rmdir spikes 2>/dev/null || true
```

- [ ] **Step 2: Verify no surviving file imports a deleted module/symbol**

```bash
grep -rnE '(\bcroco_|reference_wb|execution_wb|\bmpc_result\b|MPCResult|control_loop|run_loop|to_joint_command_wb)' \
     t1_nmpc sim tests --include='*.py'
```
Expected: **no output** (exit 1). Any hit is a dangling reference to a deleted module — fix before proceeding. (Note: prose comments containing the word "crocoddyl" do NOT match `croco_`; those are scrubbed in Task 4.)

- [ ] **Step 3: Run the regression gate**

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
```
Expected: **`51 passed, 1 xfailed`**.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(clean-base): delete dead crocoddyl/acados lineage

Remove croco_* modules, reference_wb, the croco-interface async deploy loop
(control_loop.run_loop + execution_wb.to_joint_command_wb), dead MPCResult,
croco sims/spikes, and their tests. Live aligator path untouched.
Suite: 51 passed, 1 xfailed. Croco preserved in branch crocoddyl-walk-m1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Two-layer split — create `robot/`

Move the three formulation-agnostic plant files into a `robot/` subpackage and repoint every importer. `robot/{model,config,execution}.py` keep their internal relative imports (`from .config import ...`) valid because they move together. `mpc_result.py` was already deleted (Task 1), so `robot/` has no `result.py`.

**Files:**
- Move: `t1_nmpc/model.py` → `t1_nmpc/robot/model.py`; `t1_nmpc/config.py` → `t1_nmpc/robot/config.py`; `t1_nmpc/execution.py` → `t1_nmpc/robot/execution.py`
- Create: `t1_nmpc/robot/__init__.py` (empty)
- Modify (import repoint): `t1_nmpc/wb/model_wb.py`, `t1_nmpc/wb/config_wb.py`, `t1_nmpc/wb/aligator_state.py`, `t1_nmpc/runtime/transport.py`, `t1_nmpc/runtime/mujoco_transport.py`, `t1_nmpc/runtime/sdk_transport.py`, `sim/mujoco_runtime.py`, `tests/test_sysid_friction.py`, `tests/test_mujoco_runtime.py`
- Rename + repoint (tests): `tests/test_model.py` → `tests/test_robot_model.py`; `tests/test_execution.py` → `tests/test_robot_execution.py`

**Interfaces:**
- Consumes: the post-Task-1 tree.
- Produces: importable `t1_nmpc.robot.model` (`load_model`, `T1_URDF_PATH`, `EXPECTED_JOINT_NAMES`, `CONTACT_FRAME_NAMES`, `CONTACT_PARENT_JOINTS`, `RobotModel`), `t1_nmpc.robot.config` (`MPCConfig`, `JointCommand`, `make_config`, `load_config`), `t1_nmpc.robot.execution` (`pd_torque`). The old `t1_nmpc.{model,config,execution}` paths no longer exist.

- [ ] **Step 1: Create the package and move the files**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
mkdir -p t1_nmpc/robot && touch t1_nmpc/robot/__init__.py
git add t1_nmpc/robot/__init__.py
git mv t1_nmpc/model.py     t1_nmpc/robot/model.py
git mv t1_nmpc/config.py    t1_nmpc/robot/config.py
git mv t1_nmpc/execution.py t1_nmpc/robot/execution.py
git mv tests/test_model.py     tests/test_robot_model.py
git mv tests/test_execution.py tests/test_robot_execution.py
```

- [ ] **Step 2: Repoint absolute imports (`from t1_nmpc.X` and `import t1_nmpc.X`)**

```bash
grep -rlE 'from t1_nmpc\.(model|config|execution) import|import t1_nmpc\.(model|config|execution)\b' \
     t1_nmpc sim tests --include='*.py' \
| xargs sed -i -E \
  -e 's/from t1_nmpc\.model import/from t1_nmpc.robot.model import/g' \
  -e 's/from t1_nmpc\.config import/from t1_nmpc.robot.config import/g' \
  -e 's/from t1_nmpc\.execution import/from t1_nmpc.robot.execution import/g' \
  -e 's/import t1_nmpc\.model as/import t1_nmpc.robot.model as/g'
```

- [ ] **Step 3: Repoint relative imports (`from ..X` inside `wb/` and `runtime/`)**

These are the two-dot relatives that pointed at the old top-level modules. `robot/` internals use one-dot (`.config`) and are NOT matched.

```bash
grep -rlE 'from \.\.(config|model|execution) import' t1_nmpc --include='*.py' \
| xargs sed -i -E \
  -e 's/from \.\.config import/from ..robot.config import/g' \
  -e 's/from \.\.model import/from ..robot.model import/g' \
  -e 's/from \.\.execution import/from ..robot.execution import/g'
```

- [ ] **Step 4: Verify no stale top-level references remain**

```bash
grep -rnE 'from t1_nmpc\.(model|config|execution) import|from \.\.(config|model|execution) import|import t1_nmpc\.(model|config|execution)\b' \
     t1_nmpc sim tests --include='*.py'
```
Expected: **no output**. (Sanity that the move is complete; `robot/`'s own `from .config import ...` one-dot imports are intentionally not matched here.)

- [ ] **Step 5: Run the regression gate**

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
```
Expected: **`51 passed, 1 xfailed`**.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(clean-base): two-layer split — t1_nmpc/robot/ plant package

Move model.py/config.py/execution.py into t1_nmpc/robot/; repoint all
importers across wb/, runtime/, sim/, tests/. Pure move, no logic change.
Suite: 51 passed, 1 xfailed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Restructure `wb/` and `sim/` — drop the suffix soup

Rename the live aligator controller files to intention-revealing names, co-locate the two config dataclasses into one `wb/config.py`, and rename the two sim files. All cross-references are fixed with one word-boundary `sed` sweep (the to-be-deleted files that also carried these tokens are already gone after Task 1, so the sweep only touches survivors).

**Files:**
- Rename (modules): `model_wb.py`→`dynamics.py`, `aligator_model.py`→`ode.py`, `aligator_walk.py`→`ocp.py`, `aligator_swingz.py`→`swing.py`, `aligator_mpc.py`→`mpc.py`, `aligator_state.py`→`state.py`, `aligator_exec.py`→`execution.py`, `gait_wb.py`→`gait.py`, `config_wb.py`→`config.py` (all under `t1_nmpc/wb/`)
- Merge + delete: append `config_aligator.py` body into `wb/config.py`, then `git rm t1_nmpc/wb/config_aligator.py`
- Rename (sim): `sim/wb_state.py`→`sim/state.py`, `sim/wb_walk_aligator.py`→`sim/walk.py`
- Rename (tests): `test_aligator_model.py`→`test_wb_ode.py`, `test_aligator_exec.py`→`test_wb_execution.py`, `test_aligator_walk.py`→`test_wb_ocp.py`, `test_aligator_gait_cycle.py`→`test_wb_ocp_cycle.py`, `test_aligator_mpc.py`→`test_wb_mpc.py`, `test_aligator_recede.py`→`test_wb_mpc_recede.py`, `test_aligator_config.py`→`test_wb_config_solver.py`, `test_aligator_phase1_gate.py`→`test_wb_phase1_gate.py`, `test_aligator_phase2_gate.py`→`test_wb_phase2_gate.py`, `test_wb_model_rbd.py`→`test_wb_dynamics.py`
- Modify (sweep, no rename): `test_wb_config.py`, `test_wb_config_walk.py`, `test_wb_gait.py`, `test_wb_swing.py`, `test_runtime_mujoco_transport.py`, `test_runtime_sdk_skeleton.py`, `t1_nmpc/runtime/mujoco_transport.py`, `t1_nmpc/runtime/sdk_transport.py`, `sim/mujoco_runtime.py`

**Interfaces:**
- Consumes: the post-Task-2 tree.
- Produces: `t1_nmpc.wb.{config,dynamics,ode,ocp,swing,mpc,state,execution,gait}`. Public symbols unchanged (e.g. `wb.config.make_wb_config`/`make_aligator_config`/`WBConfig`/`AligatorConfig`, `wb.ode.build_aligator_model`/`make_ode`/`nominal_stand_x`, `wb.mpc.AligatorMPC`, `wb.state.mujoco_to_freeflyer`/`freeflyer_command`, `wb.execution.extract_tau_ff`). `sim.state.wb_state_estimate`/`wb_reset` keep their function names; only the module file is renamed.

- [ ] **Step 1: Rename the module and sim files**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
git mv t1_nmpc/wb/model_wb.py       t1_nmpc/wb/dynamics.py
git mv t1_nmpc/wb/aligator_model.py t1_nmpc/wb/ode.py
git mv t1_nmpc/wb/aligator_walk.py  t1_nmpc/wb/ocp.py
git mv t1_nmpc/wb/aligator_swingz.py t1_nmpc/wb/swing.py
git mv t1_nmpc/wb/aligator_mpc.py   t1_nmpc/wb/mpc.py
git mv t1_nmpc/wb/aligator_state.py t1_nmpc/wb/state.py
git mv t1_nmpc/wb/aligator_exec.py  t1_nmpc/wb/execution.py
git mv t1_nmpc/wb/gait_wb.py        t1_nmpc/wb/gait.py
git mv t1_nmpc/wb/config_wb.py      t1_nmpc/wb/config.py
git mv sim/wb_state.py              sim/state.py
git mv sim/wb_walk_aligator.py      sim/walk.py
```

- [ ] **Step 2: Co-locate `AligatorConfig` into `wb/config.py`, then delete `config_aligator.py`**

`config_aligator.py` lines 1–4 are a docstring + `from __future__` + `from dataclasses import dataclass, field` + blank. `wb/config.py` (formerly `config_wb.py`) already imports `dataclass`, and `AligatorConfig` uses no `field(...)`, so append only the body from line 5 onward:

```bash
printf '\n\n' >> t1_nmpc/wb/config.py
tail -n +5 t1_nmpc/wb/config_aligator.py >> t1_nmpc/wb/config.py
git rm t1_nmpc/wb/config_aligator.py
```
Then confirm the appended block is the `@dataclass class AligatorConfig` … `def make_aligator_config()` (visually check the tail of `t1_nmpc/wb/config.py`).

- [ ] **Step 3: Rename the test files**

```bash
git mv tests/test_aligator_model.py      tests/test_wb_ode.py
git mv tests/test_aligator_exec.py       tests/test_wb_execution.py
git mv tests/test_aligator_walk.py       tests/test_wb_ocp.py
git mv tests/test_aligator_gait_cycle.py tests/test_wb_ocp_cycle.py
git mv tests/test_aligator_mpc.py        tests/test_wb_mpc.py
git mv tests/test_aligator_recede.py     tests/test_wb_mpc_recede.py
git mv tests/test_aligator_config.py     tests/test_wb_config_solver.py
git mv tests/test_aligator_phase1_gate.py tests/test_wb_phase1_gate.py
git mv tests/test_aligator_phase2_gate.py tests/test_wb_phase2_gate.py
git mv tests/test_wb_model_rbd.py        tests/test_wb_dynamics.py
```

- [ ] **Step 4: Sweep every module-path reference to the renamed modules**

Word-boundary substitutions over the surviving tree. Tokens are mutually non-overlapping; each maps old module name → new. `wb_state` → `state` matches the module token in `from sim.wb_state import ...` but NOT the function `wb_state_estimate` (the trailing `_` blocks the `\b`).

```bash
grep -rlE '\b(aligator_model|aligator_walk|aligator_swingz|aligator_mpc|aligator_state|aligator_exec|model_wb|gait_wb|config_wb|config_aligator|wb_state)\b' \
     t1_nmpc sim tests --include='*.py' \
| xargs sed -i -E \
  -e 's/\baligator_model\b/ode/g' \
  -e 's/\baligator_walk\b/ocp/g' \
  -e 's/\baligator_swingz\b/swing/g' \
  -e 's/\baligator_mpc\b/mpc/g' \
  -e 's/\baligator_state\b/state/g' \
  -e 's/\baligator_exec\b/execution/g' \
  -e 's/\bmodel_wb\b/dynamics/g' \
  -e 's/\bgait_wb\b/gait/g' \
  -e 's/\bconfig_wb\b/config/g' \
  -e 's/\bconfig_aligator\b/config/g' \
  -e 's/\bwb_state\b/state/g'
```

- [ ] **Step 5: Verify no old module token survives**

```bash
grep -rnE '\b(aligator_model|aligator_walk|aligator_swingz|aligator_mpc|aligator_state|aligator_exec|model_wb|gait_wb|config_wb|config_aligator|wb_state)\b' \
     t1_nmpc sim tests --include='*.py'
```
Expected: **no output**. (`WBModel`, `wb_state_estimate`, `wb_reset`, and the `wb` package name are different tokens and correctly untouched.)

- [ ] **Step 6: Run the regression gate**

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
```
Expected: **`51 passed, 1 xfailed`**.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(clean-base): rename wb/ + sim/ to intention-revealing names

Drop the _aligator/_wb suffix soup: model_wb->dynamics, aligator_model->ode,
aligator_walk->ocp, aligator_swingz->swing, aligator_mpc->mpc,
aligator_state->state, aligator_exec->execution, gait_wb->gait; co-locate
WBConfig+AligatorConfig in wb/config.py. sim: wb_state->state,
wb_walk_aligator->walk. Tests renamed to mirror. Pure rename.
Suite: 51 passed, 1 xfailed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Simplify, prune docs, rewrite CLAUDE.md

Behavior-preserving dead-code removal in the surviving live files, scrub stale crocoddyl/acados prose comments, prune abandoned docs + the SDD history, fix the stale pyproject description, and rewrite CLAUDE.md to describe the real aligator codebase.

**Files:**
- Modify: surviving `t1_nmpc/wb/*.py` (remove provably-dead `ok`/`ndx` locals, F401 imports, scrub stale "acados/crocoddyl" prose comments where they no longer describe anything live)
- Modify: `pyproject.toml` (description string)
- Rewrite: `CLAUDE.md`
- Delete (docs): `docs/acados_exact_elimination_pipeline.md`, `docs/superpowers/plans/2026-06-25-crocoddyl-port-m0.md`, `docs/superpowers/plans/2026-06-25-crocoddyl-walk-m1.md`, `docs/superpowers/plans/2026-06-25-d4-event-aligned-grid.md`, `docs/superpowers/plans/2026-06-25-reduced-basis-projection.md`, `docs/superpowers/specs/2026-06-25-crocoddyl-port-design.md`, `docs/superpowers/specs/2026-06-25-crocoddyl-walk-m1-design.md`, `docs/superpowers/specs/2026-06-25-d4-event-aligned-grid-design.md`, `docs/superpowers/specs/2026-06-25-reduced-basis-projection-design.md`
- Delete (history): `.superpowers/sdd/` (whole directory)

**Interfaces:**
- Consumes: the post-Task-3 tree.
- Produces: the final clean base. Kept docs: `docs/superpowers/specs/2026-06-26-aligator-native-port-design.md`, `docs/superpowers/specs/2026-06-26-aligator-port-scoping.md`, `docs/superpowers/specs/2026-06-26-clean-base-design.md`, `docs/superpowers/plans/2026-06-26-aligator-native-port.md`, `docs/superpowers/plans/2026-06-26-clean-base.md` (this file), `docs/2026-06-25-t1controller-divergences.md`.

- [ ] **Step 1: Find dead-code simplification candidates in surviving files**

```bash
cd /home/yoonwoo/humanoid_mpc_ws/src/t1_nmpc
PYTHONPATH= conda run -n t1mpc python -m pyflakes t1_nmpc/wb/*.py t1_nmpc/robot/*.py t1_nmpc/runtime/*.py sim/*.py 2>&1 | grep -vE 'unable to detect undefined' || true
grep -rnE '\b(ok|ndx)\b\s*=' t1_nmpc/wb/*.py
```
This lists unused imports/locals (pyflakes) and the known dead `ok`/`ndx` assignments. Review each hit: remove only assignments/imports with **zero** downstream readers. **Do not** touch any line that participates in a numeric computation, a solver call, or a returned value.

- [ ] **Step 2: Apply the behavior-preserving removals + comment scrub**

For each confirmed-dead item from Step 1, delete the line (e.g. the unused `field` import flagged historically, dead `ok = ...` / `ndx = ...` locals). Then scrub now-meaningless prose: comments that describe the deleted acados/crocoddyl machinery as if live. Keep comments that legitimately cite the OCS2/crocoddyl *reference* for a faithfulness rationale (those are still true). Verify nothing numeric changed:

```bash
git diff --stat
```
Expected: only deletions of blank/comment/dead-assignment lines and import lines; no edits to numeric literals or expressions.

- [ ] **Step 3: Run the regression gate (proves the simplify was behavior-preserving)**

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
```
Expected: **`51 passed, 1 xfailed`**.

- [ ] **Step 4: Prune abandoned docs and the SDD history**

```bash
git rm docs/acados_exact_elimination_pipeline.md \
       docs/superpowers/plans/2026-06-25-crocoddyl-port-m0.md \
       docs/superpowers/plans/2026-06-25-crocoddyl-walk-m1.md \
       docs/superpowers/plans/2026-06-25-d4-event-aligned-grid.md \
       docs/superpowers/plans/2026-06-25-reduced-basis-projection.md \
       docs/superpowers/specs/2026-06-25-crocoddyl-port-design.md \
       docs/superpowers/specs/2026-06-25-crocoddyl-walk-m1-design.md \
       docs/superpowers/specs/2026-06-25-d4-event-aligned-grid-design.md \
       docs/superpowers/specs/2026-06-25-reduced-basis-projection-design.md
git rm -r .superpowers/sdd
```

- [ ] **Step 5: Fix the stale pyproject description**

In `pyproject.toml`, change the `description` (currently `"wb_humanoid-faithful whole-body nonlinear MPC (NMPC) for Booster T1 (acados + pinocchio.casadi)"`) to read `"wb_humanoid-faithful whole-body nonlinear MPC (NMPC) for Booster T1 (aligator ProxDDP + pinocchio.casadi)"`.

- [ ] **Step 6: Rewrite CLAUDE.md**

Replace `CLAUDE.md` wholesale. It must describe the **real** current project (no acados, no `ocp_wb.py`/`cost_wb.py`/`mpc_wb.py`, and the repo IS a git repo). Required content:
- **What this is:** Python whole-body NMPC for Booster T1 on **aligator (ProxDDP / proximal augmented-Lagrangian DDP)** + pinocchio.casadi (`cpin`) + MuJoCo; a faithful port of OCS2 `humanoid_mpc` (`t1_controller`). Faithfulness to `t1_controller` is the governing constraint.
- **Env & commands:** the conda `t1mpc` preamble (drop the now-irrelevant `ACADOS_*` vars from the description, or note they are inert); `python -m pytest tests/ -q -p no:cacheprovider` (gate: 51 passed, 1 xfailed); the live sim runner `python sim/walk.py` (note `--view`/`--gif`/`--threads` flags).
- **Architecture:** the `robot/` (plant: `model`, `config`, `execution`) + `wb/` (controller: `config`, `dynamics`, `ode`, `ocp`, `swing`, `mpc`, `state`, `execution`, `gait`) + `runtime/` (`transport`, `mujoco_transport`, `sdk_transport`) + `sim/` (`mujoco_runtime`, `state`, `walk`, `_sim_util`) layout. State `x` = Translation+SphericalZYX freeflyer mapping directly onto pinocchio q/v; control = joint accels + 6D contact wrenches per foot.
- **Invariants:** hard stagewise constraints (friction cone / CoP / contact equality) native to aligator — not penalty hacks; accel-level Baumgarte swing-z (`swing.py`, input-coupled so AL enforces it); gait cycle longer than the horizon (`SLOW_WALK` 1.7 s); walk forced **serial** (custom python residual + C++ parallel-LQ = GIL segfault) — parallel only when `gait is None`; behavior-preserving faithfulness over cleverness.
- **Status:** M0 stand PASS (fz/mg ∈ [0.9,1.1], p90 ≈ 14 ms < 25 ms). Walk steps + advances forward but topples laterally ~1.5 s — **lateral CoM-sway reference is the open problem** (a reference/planning gap, not solver convergence; same gap noted in the crocoddyl ledger). Summarize the forward-locomotion mechanism (velocity-driven base pose target + explicit forward foot-placement catch) so the next session has the context that was in the deleted SDD ledger.
- **Docs:** point at the kept specs/plans under `docs/superpowers/`.

- [ ] **Step 7: Final gate + clean-tree verification**

```bash
PYTHONPATH= LD_LIBRARY_PATH=$HOME/acados/lib ACADOS_SOURCE_DIR=$HOME/acados \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n t1mpc \
python -m pytest tests/ -q -p no:cacheprovider
grep -rnE '\b(croco_|reference_wb|execution_wb|mpc_result|control_loop|to_joint_command_wb|aligator_model|aligator_walk|aligator_mpc|aligator_state|aligator_exec|model_wb|gait_wb|config_wb)\b' \
     t1_nmpc sim tests --include='*.py'
```
Expected: **`51 passed, 1 xfailed`**, and the grep produces **no output**.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(clean-base): simplify, prune docs, rewrite CLAUDE.md

Behavior-preserving dead-code removal in surviving wb/ files; scrub stale
acados/crocoddyl comments; delete abandoned-approach docs + .superpowers/sdd
history; fix pyproject description; rewrite CLAUDE.md for the aligator reality
(robot/ + wb/ layout, ProxDDP, current stand-PASS/walk-lateral-open status).
Suite: 51 passed, 1 xfailed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (against the spec)

**Spec coverage:** Deletions (croco/reference/execution_wb/mpc_result/control_loop + sims/spikes/tests) → Task 1. `robot/` two-layer split → Task 2. `wb/` rename + config co-location + sim renames + test renames → Task 3. Simplify + docs prune + CLAUDE.md + pyproject → Task 4. Validation gate (51/1 + grep-clean) → every task's verify steps. Sequencing (delete→move→restructure→simplify, separate commits) → task structure. All spec sections map to a task.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step is an exact command. The only prose-judgment steps (Task 4 Steps 2 & 6 — which dead lines to cut, CLAUDE.md content) enumerate explicit acceptance criteria and a numeric-invariance check rather than leaving it open.

**Type/name consistency:** Module paths are consistent across tasks — `t1_nmpc.robot.{model,config,execution}` (Task 2) and `t1_nmpc.wb.{config,dynamics,ode,ocp,swing,mpc,state,execution,gait}` (Task 3) match the Interfaces blocks and the spec's target tree. The sweep token list in Task 3 Step 4 matches the rename list in Task 3's Files block and the planning-verified inventory.

**Note (planning-time refinements folded into the spec):** `mpc_result.py`/`MPCResult` and `execution_wb.py` were found dead and are deleted (not moved/merged); the async deploy loop is removed (user-confirmed); the gate is **51/1** (empirically measured), not 55/1.
