"""Single-phase whole-body acados OCP for the T1 M0 stand.

Mirrors t1_controller: SQP, max_iter=1 (single RTI), PARTIAL_CONDENSING_HPIPM, ERK4,
N=31 / dt=0.035 (tf=1.085), GN custom Hessian, EXTERNAL cost. f_expl = WBModel.flow_expr;
cost from cost_wb (Q/R + JointTorque, GN Hessian); con_h from constraints_wb (ZeroAccel +
friction + CoP), soft-slacked so the QP never dies (acados has no OCS2 equality projection).
"""
from __future__ import annotations

import hashlib
import os
import subprocess

import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel as AcadosTemplateModel, ACADOS_INFTY

from .config_wb import WBConfig
from .model_wb import WBModel
from .cost_wb import build_cost_conl, build_residual_terminal, N_PARAM_WB, P_DT
from .constraints_wb import build_con_h, NH, NBU

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../t1_nmpc
# Persistent (not /tmp, which is wiped on reboot) so the compiled solver survives; env overrides for experiments.
_CODEGEN_DIR = os.environ.get("ACADOS_WB_CODEGEN_DIR", os.path.join(_PKG_ROOT, ".acados_wb"))

# Committed, deterministic compile flags — baked into the build config, NOT read from ambient env.
# (An env-var flag previously let a build and a later load disagree, so acados's reuse check fired
# and silently regenerated+rebuilt — nuking the cached .so. Flags are now part of the OCP and the
# build hash, so same config -> instant load, changed config -> a DELIBERATE rebuild.) Override
# explicitly for flag experiments via make_ocp(compile_flags=...) + build_solver(force_rebuild=True).
# DISCRETE needs -O1: its 18 MB fused-RK4 Jacobian is intractable to compile at -O2. ERK uses -O2.
COMPILE_FLAGS_ERK = "-O2 -march=native -mtune=native"
COMPILE_FLAGS_DISCRETE = ("-O2 -fno-schedule-insns -fno-schedule-insns2 -fno-gcse -fno-tree-pre "
                          "-fno-code-hoisting -march=native -mtune=native")


class WBBundle:
    """CasADi evaluators exposed for tests."""

    def __init__(self, model, cost_fun, con_h_fun):
        self.model = model
        self.cost_fun = cost_fun
        self.con_h_fun = con_h_fun


def _rk4(model, x, u, dt):
    """Explicit RK4 of the continuous flow over dt (CasADi expr) for a DISCRETE model."""
    k1 = model.flow_expr(x, u)
    k2 = model.flow_expr(x + 0.5 * dt * k1, u)
    k3 = model.flow_expr(x + 0.5 * dt * k2, u)
    k4 = model.flow_expr(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_ocp(cfg: WBConfig, discrete: bool = True, compile_flags: str | None = None):
    """Build (AcadosOcp, WBBundle). Reuses WBModel for the flow/cost/constraint exprs.

    discrete=True: hand-rolled RK4 disc_dyn_expr (acados differentiates the whole step via CasADi
    AD/CSE, no runtime forward-VDE) instead of ERK — ~2x faster solve (validated DISCRETE-O1 = 25ms).
    compile_flags: object/ext-fun compile flags baked into the OCP (defaults to the per-integrator
    COMPILE_FLAGS_* constant); pass explicitly to experiment with flags (e.g. a reduced-O2 set).
    (MX model — sharing the RBD sub-functions instead of SX-inlining — was tried 2026-06-24 and
    REVERTED: disc_dyn_jac halved 18.4->9.6MB but the call overhead made the solve +46% slower.)"""
    if compile_flags is None:
        compile_flags = COMPILE_FLAGS_DISCRETE if discrete else COMPILE_FLAGS_ERK
    model = WBModel(cfg)
    x = cs.SX.sym("x", cfg.nx)
    u = cs.SX.sym("u", cfg.nu)
    p = cs.SX.sym("p", N_PARAM_WB)

    # RAW input (no projection): the contact equalities are con_h rows toggled active/inactive per node.
    f_expl = model.flow_expr(x, u)
    y, yref, psi, r_psi = build_cost_conl(x, u, p, cfg, model)
    y_e, yref_e, Wdiag_e = build_residual_terminal(x, p, cfg)
    con_h, lh_def, uh_def = build_con_h(x, u, p, cfg, model)     # 14 raw ZeroAccel/SwingZ rows; lh/uh default all-inactive

    am = AcadosTemplateModel()
    am.name = "t1_wb"
    xdot = cs.SX.sym("xdot", cfg.nx)
    am.x, am.u, am.xdot, am.p = x, u, xdot, p
    am.f_expl_expr = f_expl
    am.f_impl_expr = xdot - f_expl
    if discrete:
        am.disc_dyn_expr = _rk4(model, x, u, p[P_DT])
    am.cost_y_expr = y                                           # CONL inner residual = [LS tracking; 10 contact barrier margins]
    am.cost_r_in_psi_expr = r_psi
    am.cost_psi_expr = psi
    am.cost_y_expr_e = y_e
    am.con_h_expr_0 = con_h                                      # stage 0 too: u_0 must produce contact-consistent accel
    am.con_h_expr = con_h                                        # stages 1..N-1: ZeroAccel (stance) / SwingZ (swing)

    ocp = AcadosOcp()
    ocp.model = am
    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf = cfg.N * cfg.dt                       # 1.085
    ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"                 # faithful OCS2: LS tracking + interior-repulsive contact barrier
    ocp.cost.cost_type_e = "NONLINEAR_LS"                        # terminal: pure LS (no contact barrier at tf)
    ocp.cost.W_e = np.diag(Wdiag_e)
    ocp.cost.yref = yref                                         # psi(y - yref); yref=0 (LS rows are deviations, margins direct)
    ocp.cost.yref_e = yref_e

    # Contact EQUALITIES as con_h, toggled per node via bounds (mpc_wb sets lh/uh each tick): a STANCE foot
    # activates ZeroAccel (6 rows -> [0,0]); a SWING foot activates SwingZ (1 row -> [0,0]); inactive rows
    # are [-INFTY,+INFTY]. Raw (ungated) expr keeps the Jacobian FULL RANK (the prior gate-to-zero left
    # zero rows -> singular KKT -> MINSTEP). Default here = all-inactive; the runtime bounds make it bite.
    ocp.constraints.lh_0 = lh_def.copy(); ocp.constraints.uh_0 = uh_def.copy()
    ocp.constraints.lh = lh_def.copy(); ocp.constraints.uh = uh_def.copy()
    # SWING-foot ZeroWrench as input box bounds on u[0:12]=[W_l,W_r]: swing -> 0, stance -> free (per node).
    ocp.constraints.idxbu = np.arange(NBU)
    ocp.constraints.lbu = -ACADOS_INFTY * np.ones(NBU)
    ocp.constraints.ubu = ACADOS_INFTY * np.ones(NBU)
    # joint-position box limits on q_joints (state idx 6..32)
    ocp.constraints.idxbx = np.arange(6, 33)
    ocp.constraints.lbx = cfg.joint_lower.copy()
    ocp.constraints.ubx = cfg.joint_upper.copy()

    x0 = model.nominal_state()
    ocp.constraints.x0 = x0
    pv0 = np.zeros(N_PARAM_WB); pv0[P_DT] = cfg.dt
    ocp.parameter_values = pv0

    # Solver regime = faithful port of arXiv:2605.04607 (Stark/DFKI) Appendix B — a WORKING acados biped.
    # The key change vs our prior single-RTI: SQP capped at 3 iters + projection regularization + BALANCE
    # HPIPM + warm-start, which is what lets a few-iter SQP carry hard-ish contact constraints across
    # touchdowns (where our single-RTI hit the infeasibility wall). Justified deviations: we keep
    # DISCRETE-RK4 (more accurate than their forward-Euler) and our N=31 single-phase (their N=10 WB->SRB
    # cascaded-fidelity split is a separate speed feature, not the contact handling).
    so = ocp.solver_options
    so.nlp_solver_type = "SQP"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"                    # sparsity-exploiting HPIPM (paper App.B)
    so.hpipm_mode = "SPEED"                                      # FAITHFUL to OCS2 (HpipmInterfaceSettings default = SPEED, never overridden)
    so.integrator_type = "DISCRETE" if discrete else "ERK"
    so.sim_method_num_stages = 4                                 # RK4 (ERK only); kept over paper's fwd-Euler
    so.sim_method_num_steps = 1
    so.levenberg_marquardt = 1e-3                               # base LM (paper: adaptive LM)
    so.regularize_method = "PROJECT"                            # projection-based Hessian reg (paper App.B)
    so.qp_solver_iter_max = 30                                  # FAITHFUL to OCS2 HPIPM default iter_max=30
    so.nlp_solver_max_iter = 12                                # CEILING, baked ONCE for solver memory sizing. The ACTUAL
    # max_iter is set at RUNTIME via options_set in WholeBodyMPC (acados can LOWER it), so changing SQP iters NEVER triggers
    # a codegen rebuild -- only model/cost/constraint/flag/integrator edits (which change the generated C) do. Default runtime
    # value = 1 (OCS2-faithful single-RTI). Convergence probe: more iters cut sink/tilt but NOT forward progress (mean_vx~0
    # at 1 and 8) -> the walk blocker is the emergent-stepping formulation, not convergence.
    so.globalization = "MERIT_BACKTRACKING"                    # line search needed for the multi-iter SQP (FIXED_STEP
    # under no-projection gave 537 MINSTEP — the QP steps need backtracking when the equality conditioning is poor).
    so.qp_solver_warm_start = 1                                 # warm-start QP (paper App.B)
    so.nlp_solver_warm_start_first_qp = True                    # init 1st QP from prev NLP soln (paper App.B)
    so.nlp_solver_tol_stat = 1e-3
    so.nlp_solver_tol_eq = 1e-3
    so.nlp_solver_tol_ineq = 1e-3
    so.nlp_solver_tol_comp = 1e-3
    # Baked compile flags = the single source of truth (build_solver reads them back for the Makefile
    # rewrite + the build hash). -march=native is build-CPU-specific (znver4) — revisit for HW
    # cross-compile. DISCRETE-O1 = 25ms (validated, ~2x ERK-O2's 52ms).
    # (p[P_DT]/dt in psi) is the SOLE time-weighting. acados' default cost_scaling for this OCP is 1
    # (verified Task 4 spike: the cost was NOT time-scaled), so ones() is a no-op here but pins it and
    # guards an acados version that might default to time_steps -> double-scale. Terminal unscaled. (D4)
    so.cost_scaling = np.ones(cfg.N + 1)
    so.ext_fun_compile_flags = compile_flags
    ocp.code_export_directory = os.path.join(_CODEGEN_DIR, "c_generated_code")

    bundle = WBBundle(
        model=model,
        cost_fun=cs.Function("resid_y", [x, u, p], [y]),
        con_h_fun=cs.Function("con_h", [x, u, p], [con_h]),
    )
    return ocp, bundle


def _build_hash(ocp: AcadosOcp) -> str:
    """Cache key over the ACTUAL build inputs, so a matching hash means the cached .so genuinely
    matches the requested build: the wb source files + the compile flags + the integrator type +
    the URDF (a model input the .py sources don't capture). An honest key lets us load purely
    (ocp=None) without acados re-deriving reuse — see build_solver."""
    here = os.path.dirname(__file__)
    srcs = ["config_wb.py", "model_wb.py", "cost_wb.py", "constraints_wb.py", "ocp_wb.py"]
    h = hashlib.sha256()
    for s in srcs:
        try:
            with open(os.path.join(here, s), "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    h.update(ocp.solver_options.ext_fun_compile_flags.encode())
    h.update(ocp.solver_options.integrator_type.encode())
    try:                                          # URDF: a model input not in the .py sources
        from ..model import T1_URDF_PATH
        with open(T1_URDF_PATH, "rb") as fh:
            h.update(fh.read())
    except (OSError, ImportError):
        pass
    return h.hexdigest()


def build_solver(ocp: AcadosOcp, force_rebuild: bool = False) -> AcadosOcpSolver:
    """Codegen + compile, reusing the cached .so when the build inputs (_build_hash) are unchanged,
    else a deliberate rebuild. acados hardcodes `-c -O2` in its Makefile (gcc never finishes on the
    big RBD/GN C); rewrite object flags to the OCP's baked compile_flags and force the UNUSED GN
    *_hess.o to -O0 (the build long-pole). Loads via acados's pure-load API (ocp=None, json_file=...)
    so a load is only ever a load — it never reconstructs the OCP, so it can never mismatch it and
    silently regenerate+rebuild (the failure the honest hash + this API together prevent)."""
    flags = ocp.solver_options.ext_fun_compile_flags
    base = os.path.dirname(ocp.code_export_directory)
    os.makedirs(base, exist_ok=True)
    json = os.path.join(base, "acados_ocp.json")
    cgd = ocp.code_export_directory
    so = os.path.join(cgd, f"libacados_ocp_solver_{ocp.model.name}.so")
    marker = os.path.join(base, ".build_hash")
    cur = _build_hash(ocp)

    rebuild = force_rebuild or os.environ.get("ACADOS_FORCE_REBUILD")
    if (not rebuild and os.path.exists(so) and os.path.exists(marker)
            and open(marker).read().strip() == cur):
        return AcadosOcpSolver(None, json_file=json, generate=False, build=False, verbose=False)

    AcadosOcpSolver.generate(ocp, json_file=json)
    mk = os.path.join(cgd, "Makefile")
    with open(mk) as fh:
        txt = fh.read()
    txt = txt.replace("-c -O2", "-c " + flags)
    # The NLS cost *_hess.c are generated but UNUSED for Gauss-Newton (acados wires the residual
    # Jacobian, not the Hessian — verified: 0 refs in acados_solver_t1_wb.c). They're the build
    # long-pole, so compile them at -O0 (instant) via a pattern-specific CFLAGS override — the .so is
    # functionally identical (the hess is never called at runtime under GAUSS_NEWTON).
    txt += "\n%_hess.o: CFLAGS := -fPIC -std=c99 -O0\n"
    with open(mk, "w") as fh:
        fh.write(txt)
    subprocess.run(["make", "-j", str(os.cpu_count() or 4), "ocp_shared_lib"], cwd=cgd, check=True)
    with open(marker, "w") as fh:
        fh.write(cur)
    return AcadosOcpSolver(None, json_file=json, generate=False, build=False, verbose=False)
