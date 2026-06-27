"""CasADi Opti transcription of whole_body_rnea for T1 walk (gated 8-corner contact).

WalkOCP generalizes the M0 StandOCP: per-foot contact/swing schedules are opti
parameters that gate the per-corner force constraints, the stance corner-velocity
constraints, and the swing foot-center z-velocity (cubic spline) constraint.
Schedules + base-velocity + footstep targets are passed to the compiled solver_fn
as ARGUMENTS so the one function is reused across MPC ticks (Task 6).

StandOCP is kept as a thin subclass that bakes all-stance schedules and exposes the
original 4-argument solve_function so the M0 stand path (mpc.py) is unchanged.
"""
from __future__ import annotations

import numpy as np
import casadi as ca

from ..robot.config import MPCConfig
from ..robot.model import RobotModel, nominal_x
from .dynamics import WBDynamics
from .spline import get_spline_vel_z


class WalkOCP:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, uniform_width: bool = False):
        self.cfg = cfg
        self.rm = rm
        self.dyn = WBDynamics(rm.model, rm.corner_frame_ids)
        self.mass = rm.mass
        self.nq, self.nv, self.nj, self.nf = self.dyn.nq, self.dyn.nv, self.dyn.nj, self.dyn.nf
        self.na = self.nv
        self.nodes, self.tau_nodes = cfg.nodes, cfg.tau_nodes
        self.mu, self.uniform_width = cfg.friction_mu, uniform_width
        self.nx = self.nq + self.nv
        self.ndx = 2 * self.nv
        self.f_idx, self.tau_idx = self.na, self.na + self.nf
        self.tau_max = rm.tau_max
        # foot/corner topology (corners 0-3 = foot 0 Left, 4-7 = foot 1 Right)
        self.corner_ids = rm.corner_frame_ids               # 8, first 4 Left, last 4 Right
        self.foot_center_ids = rm.foot_center_frame_ids      # 2 (Left, Right)
        self.n_feet = cfg.n_feet
        self.swing_period = cfg.switching_times[1]            # 0.6 s (do NOT add a config field)
        ratio = cfg.dt_max / cfg.dt_min
        gamma = ratio ** (1.0 / (self.nodes - 1))
        self.dts = [cfg.dt_min * gamma ** i for i in range(self.nodes)]
        self.opti = ca.Opti()
        self._build()

    def _nu(self, i):
        return self.na + self.nf + (self.nj if (self.uniform_width or i < self.tau_nodes) else 0)

    def _has_tau(self, i):
        return self.uniform_width or (i < self.tau_nodes)

    @staticmethod
    def _corner_foot(c):       # corner index -> foot index (0 Left:0-3, 1 Right:4-7)
        return 0 if c < 4 else 1

    def _build(self):
        opti = self.opti
        self.DX, self.U = [], []
        for i in range(self.nodes):                       # staircase order
            self.DX.append(opti.variable(self.ndx))
            self.U.append(opti.variable(self._nu(i)))
        self.DX.append(opti.variable(self.ndx))

        self.x_init = opti.parameter(self.nx)
        self.Q_diag = opti.parameter(self.ndx)
        self.R_diag = opti.parameter(self.na + self.nf + self.nj)

        # per-tick schedules + commands (passed as solver_fn arguments)
        self.contact_sched = opti.parameter(self.n_feet, self.nodes)
        self.swing_sched = opti.parameter(self.n_feet, self.nodes)
        self.base_vx = opti.parameter(1)
        self.footstep_tgt = opti.parameter(2 * self.n_feet, self.nodes)
        # defaults => WalkOCP defaults to an all-stance problem (keeps opti.value(p) valid)
        opti.set_value(self.contact_sched, np.ones((self.n_feet, self.nodes)))
        opti.set_value(self.swing_sched, np.zeros((self.n_feet, self.nodes)))
        opti.set_value(self.base_vx, 0.0)
        opti.set_value(self.footstep_tgt, np.zeros((2 * self.n_feet, self.nodes)))

        x_des = ca.DM(nominal_x(self.cfg, self.rm.model))  # nominal stand: base@0.6734 upright, nominal joints, zero vel
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_grav = self.mass * 9.81 / self.cfg.n_corners
        self.f_des = ca.vertcat(*[ca.DM([0, 0, f_grav]) for _ in range(self.cfg.n_corners)])
        self.u_des_full = ca.vertcat(ca.DM.zeros(self.na), self.f_des, ca.DM.zeros(self.nj))

        self._constraints()
        self._init_guess()
        opti.minimize(self._objective())

    # accessors
    def _x(self, i): return self.dyn.state_integrate()(self.x_init, self.DX[i])
    def _q(self, i): return self._x(i)[:self.nq]
    def _v(self, i): return self._x(i)[self.nq:]
    def _a(self, i): return self.U[i][:self.na]
    def _f(self, i): return self.U[i][self.f_idx:self.tau_idx]
    def _tau(self, i): return self.U[i][self.tau_idx:]

    def _init_guess(self):
        """Set gravity-comp initial guess for forces so the solver starts near feasibility."""
        u_des_np = np.array(self.u_des_full).flatten()  # DM -> numpy (na + nf + nj)
        opti = self.opti
        for i in range(self.nodes):
            opti.set_initial(self.DX[i], np.zeros(self.ndx))
            nu = self._nu(i)
            opti.set_initial(self.U[i], u_des_np[:nu])
        opti.set_initial(self.DX[self.nodes], np.zeros(self.ndx))

    def _constraints(self):
        opti = self.opti
        opti.subject_to(self.DX[0] == np.zeros(self.ndx))
        rnea = self.dyn.rnea_dynamics()
        corner_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.corner_ids}
        center_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.foot_center_ids}
        for i in range(self.nodes):
            dq, dv = self.DX[i][:self.nv], self.DX[i][self.nv:]
            dq_n, dv_n = self.DX[i + 1][:self.nv], self.DX[i + 1][self.nv:]
            q, v, a, forces, dt = self._q(i), self._v(i), self._a(i), self._f(i), self.dts[i]
            opti.subject_to(dq_n == dq + v * dt)                    # (1) gap-closing FIRST
            opti.subject_to(dv_n == dv + a * dt)
            tau_rnea = rnea(q, v, a, forces)
            opti.subject_to(tau_rnea[:6] == np.zeros(6))            # (2) base underactuation
            if self._has_tau(i):                                    # (3) torque eq + box
                tau_j = self._tau(i)
                opti.subject_to(tau_rnea[6:] == tau_j)
                opti.subject_to(opti.bounded(-self.tau_max, tau_j, self.tau_max))
            for c in range(self.cfg.n_corners):                    # (4) gated friction / swing zero-force
                fe = forces[c * 3:(c + 1) * 3]
                ic = self.contact_sched[self._corner_foot(c), i]
                opti.subject_to(ic * fe[2] >= 0)
                opti.subject_to(ic * self.mu**2 * fe[2]**2 >= ic * (fe[0]**2 + fe[1]**2))
                opti.subject_to((1 - ic) * fe == np.zeros(3))
            if i == 0:
                continue                                            # (5) node-0: NO velocity constraints
            for c, fid in enumerate(self.corner_ids):               # stance: gated corner velocity
                ic = self.contact_sched[self._corner_foot(c), i]
                opti.subject_to(ic * corner_vel[fid](q, v)[:3] == np.zeros(3))
            for f, fid in enumerate(self.foot_center_ids):          # swing: gated foot-center z-velocity
                ic = self.contact_sched[f, i]
                vz = center_vel[fid](q, v)[2]
                vz_ref = get_spline_vel_z(self.swing_sched[f, i], self.swing_period,
                                          self.cfg.swing_height, self.cfg.v_liftoff,
                                          self.cfg.v_touchdown)
                opti.subject_to((1 - ic) * (vz - vz_ref) == 0)

    def _objective(self):
        # base-vx fix (Task 6): the state-tracking Q drives every velocity -> 0 (its target).
        # Leaving the base local x-velocity weight on (Q[v_x]=2000) fought the dedicated
        # base-vx term (was w_bvx=200) below, so the optimum was v_x ~= base_vx/11 (~91%
        # attenuated -> the robot barely moved forward). Fix: ZERO the base-v_x weight in the
        # to-zero state tracking so the dedicated w_bvx term ALONE governs forward velocity,
        # and raise w_bvx to a tracking-grade weight (2000). The base-vx term stays STAGE-only
        # (nodes 0..N-1): pinning the TERMINAL v_x to base_vx conflicts with the pinned-foot
        # horizon end and destabilises the walk solve. Lateral (v_y) tracking is untouched
        # (no v_y command -> still prevents sideways drift). Stand (base_vx=0) keeps v_x -> 0
        # via the same term, so M0 is preserved. (With v_x un-tracked, the hard cold start
        # converges a little slower; warm MPC ticks and the closed loop are unaffected.)
        track_mask = np.ones(self.ndx)
        track_mask[self.nv + 0] = 0.0                  # dv index 0 = base local v_x
        Q = ca.diag(self.Q_diag * ca.DM(track_mask))
        R = ca.diag(self.R_diag)
        obj = 0
        for i in range(self.nodes):
            u = self.U[i]
            if not self._has_tau(i):
                u = ca.vertcat(u, ca.MX.zeros(self.nj))
            e_dx = self.DX[i] - self.dx_des
            obj += e_dx.T @ Q @ e_dx + (u - self.u_des_full).T @ R @ (u - self.u_des_full)
        e_dx = self.DX[self.nodes] - self.dx_des
        obj = obj + e_dx.T @ Q @ e_dx
        # base forward-velocity tracking: track base local x-velocity (v[0]) to base_vx
        w_bvx = 2000.0
        for i in range(self.nodes):
            vx = self._v(i)[0]
            obj = obj + w_bvx * (vx - self.base_vx)**2
        # Raibert footstep (soft): swing foot-center xy -> target, only where swinging
        getpos = {fid: self.dyn.frame_position(fid) for fid in self.foot_center_ids}
        for i in range(self.nodes):
            for f, fid in enumerate(self.foot_center_ids):
                sw = self.swing_sched[f, i]
                pxy = getpos[fid](self._q(i))[:2]
                tgt = self.footstep_tgt[2 * f:2 * f + 2, i]
                obj = obj + self.cfg.footstep_weight * sw * ca.sumsqr(pxy - tgt)
        return obj

    # --- API ---
    def set_weights(self):
        self.opti.set_value(self.Q_diag, self.cfg.Q_diag)
        self.opti.set_value(self.R_diag, self.cfg.R_diag)

    def set_x_init(self, x71):
        self.opti.set_value(self.x_init, np.asarray(x71, dtype=np.float64))

    def set_schedules(self, contact, swing):
        self.opti.set_value(self.contact_sched, np.asarray(contact, dtype=np.float64))
        self.opti.set_value(self.swing_sched, np.asarray(swing, dtype=np.float64))

    def set_base_vx(self, vx):
        self.opti.set_value(self.base_vx, float(vx))

    def set_footstep_targets(self, targets):       # (2*n_feet, N)
        self.opti.set_value(self.footstep_tgt, np.asarray(targets, dtype=np.float64))

    def x_initial(self):
        return self.opti.value(self.opti.x, self.opti.initial())

    def _fatrop_opts(self, max_iter):
        return {"expand": True, "structure_detection": "auto", "debug": False,
                "fatrop": {"print_level": 0, "max_iter": int(max_iter),
                           "tol": self.cfg.fatrop_tol, "mu_init": self.cfg.fatrop_mu_init,
                           "warm_start_init_point": True,
                           "warm_start_mult_bound_push": 1e-7, "bound_push": 1e-7}}

    def solve_function(self, max_iter):
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "solver_fn",
            [self.x_init, self.Q_diag, self.R_diag, self.contact_sched, self.swing_sched,
             self.base_vx, self.footstep_tgt, self.opti.x],
            [self.opti.x])

    def g_data(self):
        return ca.Function("g_data", [self.opti.x, self.opti.p],
                           [self.opti.g, self.opti.lbg, self.opti.ubg])

    @staticmethod
    def constr_viol_inf(g, lbg, ubg):
        viol = np.concatenate([np.maximum(0, lbg - g), np.maximum(0, g - ubg)])
        return float(np.max(np.abs(viol))) if viol.size else 0.0

    def retract(self, sol_x):
        sol_x = np.asarray(sol_x).flatten()
        out = {"q_sol": [], "v_sol": [], "a_sol": [], "forces_sol": [], "tau_sol": []}
        x_init = self.opti.value(self.x_init)
        integ = self.dyn.state_integrate()
        idx = 0
        for i in range(self.nodes):
            nu = self._nu(i)
            dx = sol_x[idx:idx + self.ndx]
            u = sol_x[idx + self.ndx: idx + self.ndx + nu]
            idx += self.ndx + nu
            x = np.array(integ(x_init, dx)).flatten()
            out["q_sol"].append(x[:self.nq]); out["v_sol"].append(x[self.nq:])
            out["a_sol"].append(u[:self.na])
            out["forces_sol"].append(u[self.f_idx:self.tau_idx])
            out["tau_sol"].append(u[self.tau_idx:] if self._has_tau(i) else np.zeros(self.nj))
        return out


class StandOCP(WalkOCP):
    """M0 stand: all-stance schedules baked in; original 4-arg solve_function preserved
    so the M0 stand path (wb/mpc.py, sim/stand.py) is unchanged."""

    def __init__(self, cfg: MPCConfig, rm: RobotModel, uniform_width: bool = False):
        super().__init__(cfg, rm, uniform_width=uniform_width)
        self.set_schedules(np.ones((self.n_feet, self.nodes)),
                           np.zeros((self.n_feet, self.nodes)))
        self.set_base_vx(0.0)
        self.set_footstep_targets(np.zeros((2 * self.n_feet, self.nodes)))

    def solve_function(self, max_iter):
        # all-stance schedules / zero commands are baked via set_value (not args).
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "solver_fn", [self.x_init, self.Q_diag, self.R_diag, self.opti.x], [self.opti.x])
