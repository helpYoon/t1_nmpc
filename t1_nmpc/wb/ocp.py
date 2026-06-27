"""CasADi Opti transcription of whole_body_rnea for T1 stand (8 corners all in contact)."""
from __future__ import annotations

import numpy as np
import casadi as ca

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .dynamics import WBDynamics


class StandOCP:
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
        ratio = cfg.dt_max / cfg.dt_min
        gamma = ratio ** (1.0 / (self.nodes - 1))
        self.dts = [cfg.dt_min * gamma ** i for i in range(self.nodes)]
        self.opti = ca.Opti()
        self._build()

    def _nu(self, i):
        return self.na + self.nf + (self.nj if (self.uniform_width or i < self.tau_nodes) else 0)

    def _has_tau(self, i):
        return self.uniform_width or (i < self.tau_nodes)

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

        q0 = self.x_init[:self.nq]
        x_des = ca.vertcat(q0, ca.MX.zeros(self.nv))      # track current q, zero velocity
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
        velfn = {fid: self.dyn.frame_velocity(fid) for fid in self.dyn.ee_frames}
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
            for c in range(self.cfg.n_corners):                    # (4) friction cone
                fe = forces[c*3:(c+1)*3]
                opti.subject_to(fe[2] >= 0)
                opti.subject_to(self.mu**2 * fe[2]**2 >= fe[0]**2 + fe[1]**2)
            if i == 0:
                continue
            for fid in self.dyn.ee_frames:                         # (5) zero contact velocity
                opti.subject_to(velfn[fid](q, v)[:3] == np.zeros(3))

    def _objective(self):
        Q, R = ca.diag(self.Q_diag), ca.diag(self.R_diag)
        obj = 0
        for i in range(self.nodes):
            u = self.U[i]
            if not self._has_tau(i):
                u = ca.vertcat(u, ca.MX.zeros(self.nj))
            e_dx = self.DX[i] - self.dx_des
            obj += e_dx.T @ Q @ e_dx + (u - self.u_des_full).T @ R @ (u - self.u_des_full)
        e_dx = self.DX[self.nodes] - self.dx_des
        return obj + e_dx.T @ Q @ e_dx

    # --- API ---
    def set_weights(self):
        self.opti.set_value(self.Q_diag, self.cfg.Q_diag)
        self.opti.set_value(self.R_diag, self.cfg.R_diag)

    def set_x_init(self, x71):
        self.opti.set_value(self.x_init, np.asarray(x71, dtype=np.float64))

    def x_initial(self):
        return self.opti.value(self.opti.x, self.opti.initial())

    def _fatrop_opts(self, max_iter):
        return {"expand": True, "structure_detection": "auto", "debug": True,
                "fatrop": {"print_level": 0, "max_iter": int(max_iter),
                           "tol": self.cfg.fatrop_tol, "mu_init": self.cfg.fatrop_mu_init,
                           "warm_start_init_point": True,
                           "warm_start_mult_bound_push": 1e-7, "bound_push": 1e-7}}

    def solve_function(self, max_iter):
        self.opti.solver("fatrop", self._fatrop_opts(max_iter))
        return self.opti.to_function(
            "solver_fn", [self.x_init, self.Q_diag, self.R_diag, self.opti.x], [self.opti.x])

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
