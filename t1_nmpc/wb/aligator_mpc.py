"""Persistent warm-started SolverProxDDP for the kinodynamic walk. Phase 1: fixed double-support
stand problem (gait=None). Phase 2: rolling contact schedule via replaceStageCircular + dynamic
tip-stage construction when gait is set. Mirrors CrocoMPC's reset/step interface so the runner +
diagnostics swap in directly.

Phase 2 ring-buffer alignment fix
-----------------------------------
Original approach: pre-built ring of n_cycle stages at fixed gait times [0, dt, 2dt, ...,
(n_cycle-1)*dt], advanced once per MPC call. Bug: ring wraps after n_cycle/MPC_Hz seconds but the
gait period is gait.duration — these don't match (e.g. 49/41.7=1.15s vs 1.7s), so after 1.15s the
ring cycles back to the WRONG contact phase.

Fix: build the tip stage on-the-fly in _recede using gait_t = t + N*dt (real simulation time + full
horizon depth). This ensures the horizon always looks at the correct future gait phase regardless of
the MPC rate / node-spacing mismatch. ODEs are cached by contact-mode tuple so only one StageModel
is allocated per recede call (cheap compared to the solve itself).
"""
from __future__ import annotations
from dataclasses import dataclass
import contextlib
import ctypes
import os
import time
import numpy as np
import pinocchio as pin
import aligator

_libc = ctypes.CDLL(None)
from .aligator_model import make_ode, nominal_stand_x
from .aligator_walk import build_problem, build_gait_cycle, build_problem_from_stages, make_stage

@contextlib.contextmanager
def _suppress_cxx_stderr():
    """Redirect C-level fd 1 and fd 2 to /dev/null for the duration of the block.
    Python sys.stdout/sys.stderr are unaffected. Both fds are restored in finally
    even on exception. fflush(NULL) is called while fds point at /dev/null so that
    any buffered C-library output (aligator uses fmt::vprint via stdout FILE*) is
    discarded before the fds are restored.

    Suppresses the benign aligator 'Resize happened' log printed during
    cycleProblem/run when the active constraint dimension changes at a contact-mode
    transition (DS↔SS: 50↔25 constraints)."""
    _libc.fflush(None)  # flush any pending C output before we redirect
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved1 = os.dup(1)
    saved2 = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        _libc.fflush(None)  # discard buffered C output to /dev/null before restore
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)
        os.close(devnull_fd)


@dataclass
class AligatorResult:
    xs: list
    us: list
    status: int
    num_iters: int

class AligatorMPC:
    def __init__(self, wb_cfg, al_cfg, am, gait=None):
        self.cfg = wb_cfg; self.al = al_cfg; self.am = am; self.gait = gait
        self.model = am  # exec uses .am-like fields; alias for runner compatibility
        self.last_solve_s = 0.0
        self._built = False
        self._ode_cache: dict = {}          # (bool,bool) -> KinodynamicsFwdDynamics
        self._x_ref_rdata = None            # cached pinocchio data for nominal forward kinematics

    def _get_ode(self, flags):
        """Return cached ODE for this contact mode; create on first access."""
        key = tuple(flags)
        if key not in self._ode_cache:
            self._ode_cache[key] = make_ode(self.am, flags, self.al.FS)
        return self._ode_cache[key]

    def _configure(self, problem):
        s = aligator.SolverProxDDP(self.al.tol, self.al.mu_init, max_iters=self.al.max_iters, verbose=aligator.QUIET)
        # The walk path (gait set) adds the custom python SwingZBaumgarte residual, which the C++ parallel
        # LQ solver cannot call across threads (GIL) -> force SERIAL when walking. The stand path (gait=None,
        # all-C++ residuals) keeps the parallel Riccati. (RT TODO: a C++ swing-z residual to regain parallel.)
        parallel = self.al.num_threads >= 2 and self.gait is None
        try:
            s.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL if parallel else aligator.LQ_SOLVER_SERIAL
        except Exception: pass
        if parallel:
            try: s.setNumThreads(self.al.num_threads)
            except Exception: pass
        for attr, val in [("max_al_iters", self.al.max_al_iters),
                          ("rollout_type", getattr(aligator, "ROLLOUT_LINEAR", None)),
                          ("sa_strategy", getattr(aligator, "SA_FILTER", None))]:
            if val is not None:
                try: setattr(s, attr, val)
                except Exception: pass
        s.setup(problem)
        return s

    def reset(self, x0):
        x0 = np.asarray(x0, float)
        N = self.al.N
        self._x_ref = nominal_stand_x(self.am, self.cfg)
        # Cache pinocchio data for foot-position look-up (x_ref is fixed)
        self._x_ref_rdata = self.am.model.createData()
        pin.framesForwardKinematics(self.am.model, self._x_ref_rdata, self._x_ref[:self.am.nq])
        self._last_gait_flags = None        # for transition-budget detection
        # Cache stance-foot lateral positions from the nominal x_ref kinematics.
        # Used to shift x_ref[y] over the stance foot during single-support stages so
        # the MPC cost stabilises lateral balance instead of fighting against it.
        self._foot_y = [
            float(self._x_ref_rdata.oMf[int(self.am.foot_ids[i])].translation[1])
            for i in range(len(self.am.foot_ids))
        ]

        if self.gait is not None:
            # ---- Walk path ----
            # Build the initial N stages with lateral-shifted x_refs appropriate for each
            # stage's contact mode. Subsequent stages are built dynamically in _recede.
            node_times = np.arange(N) * self.cfg.dt
            from .aligator_walk import make_stage as _ms
            import pinocchio as _pin
            _rdata = self._x_ref_rdata
            _ode_cache: dict = {}
            def _ode_for(flags):
                key = tuple(flags)
                if key not in _ode_cache: _ode_cache[key] = make_ode(self.am, flags, self.al.FS)
                return _ode_cache[key]
            init_models = []
            for t_node in node_times:
                flags = [bool(b) for b in self.gait.contact_flags(float(t_node))]
                x_ref_node = self._lateral_x_ref(flags)
                sw = []
                for i, on in enumerate(flags):
                    if not on:
                        z, _, _ = self.gait.swing_z(float(t_node), i)
                        p = _rdata.oMf[int(self.am.foot_ids[i])].translation.copy(); p[2] = z
                        sw.append((i, p))
                init_models.append(_ms(self.am, self.cfg, self.al, flags, x_ref_node, sw, _ode_for(flags), self.al.FS))
            self.problem = build_problem_from_stages(
                self.am, self.al, x0, self._lateral_x_ref([True, True]), init_models)
        else:
            # ---- Stand path: fixed double-support horizon ----
            sched = [[True, True]] * N
            self.problem = build_problem(self.am, self.cfg, self.al, x0, self._x_ref, sched, [[] for _ in range(N)])

        self.solver = self._configure(self.problem)
        ode = make_ode(self.am, [True, True]); mg = self.am.mass * 9.81
        u_grav = np.zeros(ode.nu); u_grav[2] = u_grav[8] = mg / 2
        self.xs = [x0.copy() for _ in range(N + 1)]; self.us = [u_grav.copy() for _ in range(N)]
        self.vs = []; self.lams = []
        self._built = True

    def _lateral_x_ref(self, flags) -> np.ndarray:
        """Return a copy of x_ref with the base lateral (y) position shifted to the centroid
        of the active stance feet. For single-support this places the CoM reference over the
        stance foot instead of the symmetric midpoint, removing the lateral cost pull that
        destabilises single-leg balance. For double-support the centroid ≈ 0 (symmetric).
        """
        x_ref = self._x_ref.copy()
        nst = max(1, sum(flags))
        y_support = sum(self._foot_y[k] for k, on in enumerate(flags) if on) / nst
        x_ref[1] = y_support      # base y = stance centroid
        return x_ref

    def _make_tip_stage(self, gait_t: float):
        """Build a StageModel for the tip of the horizon at future gait time gait_t.

        Uses cached ODE by contact-mode tuple and lateral-shifted x_ref so that the
        terminal cost references the CoM over the support foot (single-leg stability).
        """
        flags = [bool(b) for b in self.gait.contact_flags(gait_t)]
        x_ref = self._lateral_x_ref(flags)
        swing_refs = []
        for i, on in enumerate(flags):
            if not on:
                z, _, _ = self.gait.swing_z(gait_t, i)
                p = self._x_ref_rdata.oMf[int(self.am.foot_ids[i])].translation.copy()
                p[2] = z
                swing_refs.append((i, p))
        ode = self._get_ode(flags)
        return make_stage(self.am, self.cfg, self.al, flags, x_ref, swing_refs, ode, self.al.FS)

    def _recede(self, x_meas, t):
        """Advance the rolling horizon by one knot.

        Tip-stage gait time = t + N*dt (real simulation time + full horizon depth).
        This keeps the contact schedule correctly aligned with the future gait regardless
        of the MPC call rate vs. node-spacing mismatch (fixes the ring-wrap bug where the
        original ring of n_cycle=round(gait.duration/dt) stages wrapped in
        n_cycle/MPC_Hz < gait.duration seconds, forcing the wrong contact phase).

        Order is critical: replaceStageCircular FIRST (updates problem structure),
        then cycleProblem (shifts solver internal arrays to match new structure)."""
        gait_t = t + self.al.N * self.cfg.dt   # future time at horizon tip
        m = self._make_tip_stage(gait_t)
        d = m.createData()
        self.problem.replaceStageCircular(m)
        with _suppress_cxx_stderr():
            self.solver.cycleProblem(self.problem, d)
        self.problem.x0_init = x_meas
        # Shift primal warm-start: drop first (solved) state/control, extrapolate at horizon end
        self.xs = self.xs[1:] + [self.xs[-1].copy()]
        self.xs[0] = x_meas.copy()
        self.us = self.us[1:] + [self.us[-1].copy()]

    def _reseed_us0_vertical(self, flags_now) -> None:
        """Fix the vertical (fz) component of us[0] during SINGLE-SUPPORT only.

        Single-support: set stance-foot fz = mg, zero swing-foot wrench entirely.
        This corrects the wrong-foot vertical force without discarding lateral forces
        (which come from the warm-start shift of the previous us[1]).

        Double-support: do NOT modify us[0]. Leave the full warm-start intact so the
        solver can compute asymmetric corrective forces to recover accumulated roll.
        Forcing 50/50 symmetry would suppress the differential that arrests roll.
        """
        if all(flags_now):      # double support — let solver use warm-start freely
            return
        mg = self.am.mass * 9.81
        nst = max(1, sum(flags_now))
        FS = self.al.FS
        for k, on in enumerate(flags_now):
            if on:
                self.us[0][k * FS + 2] = mg / nst   # stance fz = mg/nst; keep lateral
            else:
                self.us[0][k * FS:(k+1) * FS] = 0.0  # swing: zero all (decoupled)

    def _grav_control(self, flags) -> np.ndarray:
        """Gravity-compensating control for the given contact mode: equal weight split over
        stance feet, zero for swing feet. Used to reseed us[0] at every walk step so the
        solver always starts from the correct force distribution regardless of warm-start
        history (avoids the LF→RF warm-start mismatch that causes slow convergence)."""
        mg = self.am.mass * 9.81
        nst = max(1, sum(flags))
        ode = self._get_ode(flags)
        u = np.zeros(ode.nu)
        for k, on in enumerate(flags):
            if on:
                u[k * self.al.FS + 2] = mg / nst
        return u

    def step(self, x_meas, t, command=None) -> AligatorResult:
        x_meas = np.asarray(x_meas, float)
        if not self._built:
            self.reset(x_meas)
        if self.gait is not None:
            self._recede(x_meas, t)
            flags_now = self.gait.contact_flags(t)
            # Variable iteration budget: give the solver more iterations at contact transitions
            at_transition = (self._last_gait_flags is not None and flags_now != self._last_gait_flags)
            self._last_gait_flags = flags_now
            iters = self.al.max_iters_transition if at_transition else self.al.max_iters
            try: self.solver.max_iters = iters
            except Exception: pass
            # Every step: fix only the vertical component of us[0] so the correct stance
            # foot carries the weight. Lateral forces from the warm-start shift (old us[1])
            # carry over for continuity — do NOT do a full us reset which would cause a
            # brief force-shed (fz→0) at the transition tick.
            self._reseed_us0_vertical(flags_now)
        else:
            # Stand path: only update x0; no horizon shift needed
            self.problem.x0_init = x_meas; self.xs[0] = x_meas.copy()
        t0 = time.perf_counter()
        with _suppress_cxx_stderr():
            ok = self.solver.run(self.problem, self.xs, self.us, self.vs, self.lams)
        self.last_solve_s = time.perf_counter() - t0
        R = self.solver.results
        self.xs = [np.asarray(a).copy() for a in R.xs]; self.us = [np.asarray(a).copy() for a in R.us]
        self.vs = [np.asarray(a).copy() for a in R.vs]; self.lams = [np.asarray(a).copy() for a in R.lams]
        finite = all(np.all(np.isfinite(a)) for a in self.xs)
        return AligatorResult(self.xs, self.us, 0 if finite else 1, R.num_iters)
