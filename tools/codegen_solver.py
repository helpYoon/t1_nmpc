"""Code-generate + persist a Fatrop solver_function so future runs LOAD a compiled
.so instead of re-building/JITting (ca.external).

Pattern (wb-mpc-locoman codegen/): opti.to_function -> .generate(C) -> gcc -shared
linking libfatrop+libblasfeo -> ca.external(name, .so).

IMPORTANT CAVEAT (measured 2026-06-27 on this OCP, conda env t1mpc):
  The whole-body RNEA OCP code-generates to a ~136 MB C file (it's the symbolic RNEA
  unrolled across N nodes inside opti.to_function; this is NOT fixed by expand=False
  -- expand=True was 150 MB, expand=False 136 MB). gcc -O3 on that does NOT finish in
  a couple minutes (it's a one-time, very long / memory-heavy compile). So codegen is
  IMPRACTICAL for the full OCP as-is. It becomes viable only for a much smaller problem
  (few nodes) or by code-generating the inner dynamics functions (RNEA) instead of the
  whole solver. Also note: codegen only speeds ms/iter -- it does NOT reduce the
  iteration count, which is the actual closed-loop wall.

Usage (you run the slow compile yourself, then load forever):
  PYTHONPATH= conda run -n t1mpc python tools/codegen_solver.py        # generates C + prints the gcc cmd
  # run the printed gcc command yourself (long); then in code:
  from tools.codegen_solver import load_compiled
  fn = load_compiled('solver_fn', 'tools/codegen')   # ca.external the persisted .so, or None
"""
from __future__ import annotations

import os
import casadi as ca


def fatrop_paths() -> tuple[str, str]:
    """(include_dir, lib_dir) for libfatrop/libblasfeo in the active conda env."""
    cp = os.environ["CONDA_PREFIX"]
    return os.path.join(cp, "include"), os.path.join(cp, "lib")


def generate_c(fn: ca.Function, out_dir: str) -> str:
    """Code-generate `fn` (a Fatrop solver_function) to <out_dir>/<fn.name()>.c.
    Build `fn` with expand=False so the C is as small as it gets (still large)."""
    os.makedirs(out_dir, exist_ok=True)
    cfile = os.path.join(out_dir, f"{fn.name()}.c")
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        fn.generate(f"{fn.name()}.c")
    finally:
        os.chdir(cwd)
    return cfile


def compile_command(name: str, out_dir: str, opt: str = "-O1") -> list[str]:
    """The gcc command to compile <name>.c -> <name>.so, linking Fatrop+Blasfeo.
    -O1 (not -O3) because the C is huge; -O3 may never finish. Run it yourself."""
    inc, lib = fatrop_paths()
    return ["gcc", opt, "-fPIC", "-shared",
            os.path.join(out_dir, f"{name}.c"), "-o", os.path.join(out_dir, f"{name}.so"),
            f"-I{inc}", f"-L{lib}", "-lfatrop", "-lblasfeo", f"-Wl,-rpath,{lib}"]


def load_compiled(name: str, out_dir: str) -> ca.Function | None:
    """ca.external the persisted .so if present (compiled function name must == `name`)."""
    so = os.path.join(out_dir, f"{name}.so")
    return ca.external(name, so) if os.path.isfile(so) else None


def _stand_solver_expand_false() -> ca.Function:
    """Build the StandOCP Fatrop solver_function with expand=False (codegen-friendlier)."""
    from t1_nmpc.robot.config import make_config
    from t1_nmpc.robot.model import load_model
    from t1_nmpc.wb.ocp import StandOCP

    cfg = make_config()
    ocp = StandOCP(cfg, load_model(cfg))
    ocp.set_weights()
    opts = {"expand": False, "structure_detection": "auto", "print_time": False,
            "fatrop": {"print_level": 0, "max_iter": cfg.fatrop_max_iter,
                       "tol": cfg.fatrop_tol, "mu_init": cfg.fatrop_mu_init,
                       "warm_start_init_point": True,
                       "warm_start_mult_bound_push": 1e-7, "bound_push": 1e-7}}
    ocp.opti.solver("fatrop", opts)
    return ocp.opti.to_function(
        "solver_fn", [ocp.x_init, ocp.Q_diag, ocp.R_diag, ocp.opti.x], [ocp.opti.x])


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codegen")
    fn = _stand_solver_expand_false()
    cfile = generate_c(fn, out)
    mb = os.path.getsize(cfile) / 1e6
    cmd = compile_command(fn.name(), out)
    print(f"generated {cfile}  ({mb:.0f} MB)")
    print("\nNOW RUN THIS (one-time, SLOW — minutes; -O1 to make it finish at all):\n")
    print("  " + " ".join(cmd))
    print(f"\nthen next session: load_compiled('{fn.name()}', '{out}')  -> ca.external the .so")
    if mb > 50:
        print(f"\nWARNING: {mb:.0f} MB C file. -O3 likely will NOT finish; even -O1 is heavy. "
              "Codegen is impractical for the full OCP at this size (see module docstring).")
