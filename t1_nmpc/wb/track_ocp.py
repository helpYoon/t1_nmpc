"""All-stance whole-body RNEA tracking OCP for the floor-pickup (CasADi Opti + Fatrop).

Generalizes the M0 stand OCP: the fixed nominal target becomes a per-node, time-varying
reference (x_ref). Both feet are always planted (V_lin=0, flat foot). Hands are tracked in
task-space via a slack trick: hand_pos - hand_ref == (1-grasp_gate)*s  -> soft (s free + cost)
off-grasp, hard (==ref) at grasp keyframes. Legs are SOLVED here (not tracked) against the hard
contact, with leg joint-position limits (anti-hyperextension); the leg-pitch seed lives in x_ref
at low Q weight. Built once and compiled with opti.to_function (NO jit). See the design spec."""
from __future__ import annotations

import numpy as np
import casadi as ca

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .dynamics import WBDynamics


class PickupOCP:
    def __init__(self, cfg: MPCConfig, rm: RobotModel):
        self.cfg, self.rm = cfg, rm
        self.dyn = WBDynamics(rm.model, rm.corner_frame_ids)
        self.mass = rm.mass
        self.nq, self.nv, self.nj, self.nf = self.dyn.nq, self.dyn.nv, self.dyn.nj, self.dyn.nf
        self.na = self.nv                                    # 35
        self.ns = 6                                          # hand slacks: L(3)+R(3)
        self.nodes, self.tau_nodes = cfg.nodes, cfg.tau_nodes
        self.mu = cfg.friction_mu
        self.nx, self.ndx = self.nq + self.nv, 2 * self.nv
        self.f_idx = self.na                                 # 35
        self.s_idx = self.na + self.nf                       # 59
        self.tau_idx = self.na + self.nf + self.ns           # 65
        self.tau_max = rm.tau_max
        self.n_corners = cfg.n_corners
        self.corner_ids = rm.corner_frame_ids
        self.foot_center_ids = rm.foot_center_frame_ids
        self.hand_ids = rm.hand_frame_ids
        self.dt = cfg.dt_min
        # leg joint full-q indices (for position box) and limits
        self.leg_q_idx = list(range(7 + 17, 7 + 23)) + list(range(7 + 23, 7 + 29))   # 12
        self.leg_lo = np.asarray(rm.model.lowerPositionLimit, dtype=np.float64)[self.leg_q_idx]
        self.leg_hi = np.asarray(rm.model.upperPositionLimit, dtype=np.float64)[self.leg_q_idx]
        self.opti = ca.Opti()
        self._build()

    def _nu(self, i):
        return self.na + self.nf + self.ns + (self.nj if i < self.tau_nodes else 0)

    def _has_tau(self, i):
        return i < self.tau_nodes

    def _build(self):
        opti = self.opti
        self.DX, self.U = [], []
        for i in range(self.nodes):                          # interleaved (Fatrop staircase)
            self.DX.append(opti.variable(self.ndx))
            self.U.append(opti.variable(self._nu(i)))
        self.DX.append(opti.variable(self.ndx))

        self.x_init = opti.parameter(self.nx)
        self.Q_diag = opti.parameter(self.ndx)
        self.R_diag = opti.parameter(self.na + self.nf + self.nj)     # [a, forces, tau]
        self.x_ref = opti.parameter(self.nx, self.nodes + 1)
        self.hand_ref = opti.parameter(6, self.nodes + 1)
        self.grasp_gate = opti.parameter(2, self.nodes + 1)
        # valid defaults so opti.value(p) works before set_refs
        opti.set_value(self.hand_ref, np.zeros((6, self.nodes + 1)))
        opti.set_value(self.grasp_gate, np.zeros((2, self.nodes + 1)))

        f_grav = self.mass * 9.81 / self.n_corners
        self.f_des = ca.vertcat(*[ca.DM([0, 0, f_grav]) for _ in range(self.n_corners)])
        self._integ = self.dyn.state_integrate()
        self._diff = self.dyn.state_difference()
        self._constraints()
        self._init_guess()
        opti.minimize(self._objective())

    # accessors
    def _x(self, i): return self._integ(self.x_init, self.DX[i])
    def _q(self, i): return self._x(i)[:self.nq]
    def _v(self, i): return self._x(i)[self.nq:]
    def _a(self, i): return self.U[i][:self.na]
    def _f(self, i): return self.U[i][self.f_idx:self.s_idx]
    def _s(self, i): return self.U[i][self.s_idx:self.tau_idx]
    def _tau(self, i): return self.U[i][self.tau_idx:]

    def _init_guess(self):
        f_np = np.array(self.f_des).flatten()
        u0 = np.concatenate([np.zeros(self.na), f_np, np.zeros(self.ns), np.zeros(self.nj)])
        for i in range(self.nodes):
            self.opti.set_initial(self.DX[i], np.zeros(self.ndx))
            self.opti.set_initial(self.U[i], u0[:self._nu(i)])
        self.opti.set_initial(self.DX[self.nodes], np.zeros(self.ndx))

    def _constraints(self):
        opti = self.opti
        opti.subject_to(self.DX[0] == np.zeros(self.ndx))
        rnea = self.dyn.rnea_dynamics()                      # ungated: all-stance -> every corner always applied
        center_vel = {fid: self.dyn.frame_velocity(fid) for fid in self.foot_center_ids}
        hand_pos = {fid: self.dyn.frame_position(fid) for fid in self.hand_ids}
        for i in range(self.nodes):
            dq, dv = self.DX[i][:self.nv], self.DX[i][self.nv:]
            dq_n, dv_n = self.DX[i + 1][:self.nv], self.DX[i + 1][self.nv:]
            q, v, a, forces, dt = self._q(i), self._v(i), self._a(i), self._f(i), self.dt
            # (1) gap-closing FIRST (forward Euler; slow quasi-static motion -> adequate)
            opti.subject_to(dq_n == dq + v * dt)
            opti.subject_to(dv_n == dv + a * dt)
            tau_rnea = rnea(q, v, a, forces)
            # (2) base underactuation
            opti.subject_to(tau_rnea[:6] == np.zeros(6))
            # (3) torque equality (first tau_nodes)
            if self._has_tau(i):
                opti.subject_to(tau_rnea[6:] == self._tau(i))
            # (4) contact (i>=1): planted + flat foot
            if i >= 1:
                for fid in self.foot_center_ids:
                    V = center_vel[fid](q, v)
                    opti.subject_to(V[0] == 0); opti.subject_to(V[1] == 0); opti.subject_to(V[2] == 0)
                    opti.subject_to(V[3] == 0); opti.subject_to(V[4] == 0)    # roll/pitch rate (flat)
                # (5) hand task (slack trick): pos - ref == (1-gate)*s
                for h, fid in enumerate(self.hand_ids):
                    p = hand_pos[fid](q)
                    s = self._s(i)[3 * h:3 * h + 3]
                    g = self.grasp_gate[h, i]
                    opti.subject_to(p - self.hand_ref[3 * h:3 * h + 3, i] == (1 - g) * s)
            # --- inequalities AFTER all equalities ---
            if self._has_tau(i):
                opti.subject_to(opti.bounded(-self.tau_max, self._tau(i), self.tau_max))
            for c in range(self.n_corners):
                fe = forces[c * 3:(c + 1) * 3]
                opti.subject_to(fe[2] >= 0)
                opti.subject_to(self.mu**2 * fe[2]**2 >= fe[0]**2 + fe[1]**2)
            if i >= 1:                                        # leg joint-position box (anti-hyperextension)
                q_leg = ca.vertcat(*[q[idx] for idx in self.leg_q_idx])
                opti.subject_to(opti.bounded(self.leg_lo, q_leg, self.leg_hi))

    def _objective(self):
        Q = ca.diag(self.Q_diag)
        R = ca.diag(self.R_diag)
        obj = 0
        for i in range(self.nodes + 1):
            dx_des = self._diff(self.x_init, self.x_ref[:, i])
            e = self.DX[i] - dx_des
            obj = obj + e.T @ Q @ e
            if i < self.nodes:
                u = self.U[i]
                a = u[:self.na]; forces = u[self.f_idx:self.s_idx]; s = u[self.s_idx:self.tau_idx]
                tau = u[self.tau_idx:] if self._has_tau(i) else ca.MX.zeros(self.nj)
                u_track = ca.vertcat(a, forces - self.f_des, tau)
                obj = obj + u_track.T @ R @ u_track
                obj = obj + self.cfg.w_hand * ca.sumsqr(s)        # hand task (soft off-grasp)
        return obj

    # --- API ---
    def set_weights(self):
        self.opti.set_value(self.Q_diag, self.cfg.Q_diag)
        self.opti.set_value(self.R_diag, self.cfg.R_diag)

    def set_refs(self, x_init, x_ref, hand_ref, grasp_gate):
        self.opti.set_value(self.x_init, np.asarray(x_init, dtype=np.float64))
        self.opti.set_value(self.x_ref, np.asarray(x_ref, dtype=np.float64))
        self.opti.set_value(self.hand_ref, np.asarray(hand_ref, dtype=np.float64))
        self.opti.set_value(self.grasp_gate, np.asarray(grasp_gate, dtype=np.float64))

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
            "pickup_fn",
            [self.x_init, self.Q_diag, self.R_diag, self.x_ref, self.hand_ref,
             self.grasp_gate, self.opti.x],
            [self.opti.x])

    def retract(self, sol_x, x_init):
        sol_x = np.asarray(sol_x).flatten()
        x_init = np.asarray(x_init, dtype=np.float64)
        out = {"q_sol": [], "v_sol": [], "a_sol": [], "forces_sol": [], "tau_sol": []}
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
            out["forces_sol"].append(u[self.f_idx:self.s_idx])
            out["tau_sol"].append(u[self.tau_idx:] if self._has_tau(i) else np.zeros(self.nj))
        # terminal node DX[nodes] (state only, no control)
        dx = sol_x[idx:idx + self.ndx]
        x = np.array(integ(x_init, dx)).flatten()
        out["q_sol"].append(x[:self.nq]); out["v_sol"].append(x[self.nq:])
        out["a_sol"].append(np.zeros(self.na))
        out["forces_sol"].append(np.zeros(self.nf))
        out["tau_sol"].append(np.zeros(self.nj))
        return out
