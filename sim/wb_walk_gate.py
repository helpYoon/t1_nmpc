"""M1 GATE — whole-body closed-loop FORWARD WALK in MuJoCo.

Reuses the proven wb_stand_gate harness (wb_state_estimate / _wb_reset / _sample_plan, MujocoRuntime,
60 Hz MPC, MRT t+5ms resample, kp/kd + tau_ff) but drives a forward velocity command so the SLOW_WALK
gait engages, and records the base x/y trajectory for the walking metrics. Single-thread/deterministic
(behavior gate; the deployed async loop is for timing, not behavior).

Run FOREGROUND:
  ... conda run -n t1mpc python sim/wb_walk_gate.py
  ... conda run -n t1mpc python sim/wb_walk_gate.py --log walk.npz --gap-probe-every 10
      -> also dumps per-MPC-tick solver-health + plan-vs-reality + contact telemetry and prints a
         convergence summary (WALK_CONV=...). --gap-probe-every K runs, every K-th MPC tick, a
         CONVERGED re-solve (max_iter=12) from the single-RTI step to measure how far 1 iter is from
         optimal (the definitive 'is the MPC converging' check).
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from t1_nmpc.config import make_config
from t1_nmpc.model import load_model, T1_URDF_PATH
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.mpc_wb import WholeBodyMPC
from sim.mujoco_runtime import MujocoRuntime
from sim.wb_stand_gate import wb_state_estimate, _wb_reset, _sample_plan, _HEAD_KP, _HEAD_KD
from sim._sim_util import tilt_from_quat_wxyz

SLOW_WALK_PERIOD = 1.7

# MuJoCo foot-link body names (for the premature-contact + GRF/contact instruments).
_FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")

# state indices (68): base theta_zyx = x[3:6]; waist = MPC joint 14 -> x[20]; hip-yaw L/R -> x[23]/x[29].
_WAIST, _LHIPY, _RHIPY = 20, 23, 29

# ---- per-MPC-tick log schema: every row carries every key (nan default) so the npz is rectangular ----
_LOG_KEYS = (
    # Tier 1 — solver convergence health
    "t", "status", "res_stat", "res_eq", "res_ineq", "res_comp", "qp_iter", "sqp_iter", "cost",
    "stepnorm_x", "stepnorm_u",
    # Tier 3 — plan-vs-reality (filled one tick LATE, into the prior row)
    "pred_err_full", "pred_err_base", "pred_err_yaw",
    # Tier 4 — physical / gait context
    "base_x", "base_y", "base_z", "base_yaw", "base_pitch", "base_roll", "tilt",
    "base_vx", "base_vy", "base_yawrate", "cmd_vx",
    "waist_q", "Lhipyaw", "Rhipyaw",
    "cplan_L", "cplan_R", "cact_L", "cact_R", "footL_z", "footR_z", "footL_yaw", "footR_yaw",
    "grfL", "grfR", "pd_err_inf", "tau_sat_max",
    # Tier 2 — gap probe (only on probe ticks)
    "gp_sqp_iter", "gp_status", "gp_res_eq_rti", "gp_res_eq_conv",
    "gp_cost_rti", "gp_cost_conv", "gp_x_gap",
)


def _solver_stats(solver) -> dict:
    """KKT residuals + QP/SQP iteration counts + cost from the LAST solve (all guarded -> nan)."""
    out = {k: np.nan for k in ("res_stat", "res_eq", "res_ineq", "res_comp", "qp_iter", "sqp_iter", "cost")}
    try:
        r = np.asarray(solver.get_stats("residuals"), float).ravel()
        if r.size >= 4:
            out["res_stat"], out["res_eq"], out["res_ineq"], out["res_comp"] = (float(v) for v in r[:4])
    except Exception:
        pass
    try:
        out["qp_iter"] = float(np.sum(np.asarray(solver.get_stats("qp_iter"))))
    except Exception:
        pass
    try:
        out["sqp_iter"] = float(np.sum(np.asarray(solver.get_stats("sqp_iter"))))
    except Exception:
        pass
    try:
        out["cost"] = float(solver.get_cost())
    except Exception:
        pass
    return out


def _foot_contacts(rt, foot_bids):
    """(grf[2], touch[2]) from MuJoCo: per-foot summed contact NORMAL force + boolean actual contact."""
    m, d = rt.mj_model, rt.mj_data
    grf = [0.0, 0.0]; touch = [False, False]
    w = np.zeros(6, dtype=np.float64)
    for ci in range(d.ncon):
        c = d.contact[ci]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        for i, fb in enumerate(foot_bids):
            if b1 == fb or b2 == fb:
                mujoco.mj_contactForce(m, d, ci, w)
                grf[i] += abs(float(w[0]))   # contact-frame x = normal
                touch[i] = True
    return grf, touch


def _foot_pose(rt, foot_bids):
    """(z[2], yaw[2]) world foot height + yaw (xmat row-major: R00=idx0, R10=idx3)."""
    d = rt.mj_data
    z = [float(d.xpos[fb, 2]) for fb in foot_bids]
    yaw = [float(np.arctan2(d.xmat[fb, 3], d.xmat[fb, 0])) for fb in foot_bids]
    return z, yaw


def _gap_probe(mpc, x_rti, probe_iter=12):
    """CONTINUE the just-finished single-RTI solve to convergence (max_iter=probe_iter) from the SAME
    subproblem (params/bounds unchanged) and measure the gap. Safe: mpc._x_prev already holds the RTI
    solution, and the next step() overwrites the solver primal with the warm-start, so this is purely
    observational and does NOT enter the closed loop."""
    s = mpc.solver
    rti = _solver_stats(s)                       # residual/cost at the RTI step
    s.options_set("max_iter", int(probe_iter))
    st = int(s.solve())                          # iterate to tol from the RTI primal
    conv = _solver_stats(s)
    x_conv = np.array([s.get(k, "x") for k in range(mpc.cfg.N + 1)])
    s.options_set("max_iter", 1)                 # restore single-RTI for the closed loop
    return {
        "gp_status": float(st),
        "gp_sqp_iter": conv["sqp_iter"],         # extra SQP iters to converge from the RTI step
        "gp_res_eq_rti": rti["res_eq"], "gp_res_eq_conv": conv["res_eq"],
        "gp_cost_rti": rti["cost"], "gp_cost_conv": conv["cost"],
        "gp_x_gap": float(np.max(np.abs(x_conv - x_rti))),
    }


def _predict_next(res, mpc_dt):
    """Plan's predicted state one MPC tick ahead (interp x_traj at node_times[0]+mpc_dt)."""
    nt = getattr(res, "node_times", None)
    nt = np.asarray(nt, float) if nt is not None else None
    if nt is None:
        return None
    tq = nt[0] + mpc_dt
    return np.array([np.interp(tq, nt, res.x_traj[:, j]) for j in range(res.x_traj.shape[1])])


def run_wb_walk(duration_s: float = 10.0, vx: float = 0.3, sample_ahead_s: float = 0.005,
                mpc=None, log_path: str | None = None, gap_probe_every: int = 0,
                probe_iter: int = 12) -> dict:
    cmd = np.array([vx, 0.0, 0.0, 0.0, 0.0])
    ccfg = make_config(mpc_hz=60.0)                  # 60 Hz MPC (single-RTI is rate-dependent)
    rt = MujocoRuntime(ccfg, load_model(T1_URDF_PATH, ccfg))

    if mpc is None:
        wb_cfg = make_wb_config(); wb_model = WBModel(wb_cfg)
        mpc = WholeBodyMPC(wb_cfg, wb_model)         # loads the cached walking solver (Task 8 build)
    else:
        wb_cfg, wb_model = mpc.cfg, mpc.model

    _foot_bids = tuple(
        mujoco.mj_name2id(rt.mj_model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in _FOOT_BODY_NAMES
    )

    _wb_reset(rt, wb_cfg)
    mpc.set_command(cmd)                             # speed>1e-3 -> SLOW_WALK gait
    x0 = wb_state_estimate(rt)
    mpc.reset(x0)
    res = mpc.step(x0, rt.t)
    x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t
    node_times_plan = res.node_times if getattr(res, "node_times", None) is not None else wb_cfg.dt

    n_phys = int(round(duration_s * ccfg.physics_hz))
    cdecim, mdecim = rt.control_decim, rt.mpc_decim
    mpc_dt = 1.0 / ccfg.mpc_hz
    n_fail = 0
    tilts, base_z, base_x, base_y, solve_tot = [], [], [], [], []
    kp, kd = wb_cfg.kp, wb_cfg.kd
    tlim = wb_cfg.torque_limit

    _contact_prev = [None, None]
    _min_foot_z_at_activation = float("inf")

    # ---- logging state ----
    do_log = log_path is not None
    rows = []                 # list of full-schema dicts
    _pending_pred = [None]    # (t_target, x_pred) carried to the NEXT mpc tick for prediction error
    _last_ctrl = {"pd_err_inf": np.nan, "tau_sat_max": np.nan}   # most-recent control-tick tracking
    _mpc_tick = [0]

    def _log_mpc(t, x_meas, res):
        if not do_log:
            return
        row = {k: np.nan for k in _LOG_KEYS}
        row["t"] = t; row["status"] = float(res.status)
        row.update(_solver_stats(mpc.solver))
        wx = getattr(mpc, "_last_warmstart_x", None); wu = getattr(mpc, "_last_warmstart_u", None)
        if wx is not None:
            row["stepnorm_x"] = float(np.max(np.abs(res.x_traj - wx)))
            row["stepnorm_u"] = float(np.max(np.abs(res.u_traj - wu)))
        # plan-vs-reality: write the PREVIOUS tick's prediction error now that x_meas is known
        if _pending_pred[0] is not None and rows:
            xp = _pending_pred[0]
            rows[-1]["pred_err_full"] = float(np.linalg.norm(x_meas - xp))
            rows[-1]["pred_err_base"] = float(np.linalg.norm(x_meas[0:6] - xp[0:6]))
            rows[-1]["pred_err_yaw"] = float(abs(x_meas[3] - xp[3]))
        xp_next = _predict_next(res, mpc_dt)
        _pending_pred[0] = xp_next
        # physical / gait context
        d = rt.mj_data
        row["base_x"], row["base_y"], row["base_z"] = float(x_meas[0]), float(x_meas[1]), float(x_meas[2])
        row["base_yaw"], row["base_pitch"], row["base_roll"] = float(x_meas[3]), float(x_meas[4]), float(x_meas[5])
        row["tilt"] = float(tilt_from_quat_wxyz(d.qpos[3:7]))
        row["base_vx"], row["base_vy"], row["base_yawrate"] = float(x_meas[33]), float(x_meas[34]), float(x_meas[36])
        row["cmd_vx"] = float(mpc._comm_filt[0])
        row["waist_q"], row["Lhipyaw"], row["Rhipyaw"] = float(x_meas[_WAIST]), float(x_meas[_LHIPY]), float(x_meas[_RHIPY])
        lf, rf = mpc._gait.contact_flags(t)
        row["cplan_L"], row["cplan_R"] = float(lf), float(rf)
        grf, touch = _foot_contacts(rt, _foot_bids)
        z, yaw = _foot_pose(rt, _foot_bids)
        row["cact_L"], row["cact_R"] = float(touch[0]), float(touch[1])
        row["grfL"], row["grfR"] = grf[0], grf[1]
        row["footL_z"], row["footR_z"] = z[0], z[1]
        row["footL_yaw"], row["footR_yaw"] = yaw[0], yaw[1]
        row["pd_err_inf"], row["tau_sat_max"] = _last_ctrl["pd_err_inf"], _last_ctrl["tau_sat_max"]
        # convergence GAP probe (periodic)
        if gap_probe_every and (_mpc_tick[0] % gap_probe_every == 0):
            row.update(_gap_probe(mpc, res.x_traj, probe_iter))
        rows.append(row)
        _mpc_tick[0] += 1

    _log_mpc(rt.t, x0, res)   # log the initial solve

    for k in range(n_phys):
        if k % mdecim == 0 and k > 0:
            x_meas = wb_state_estimate(rt)
            res = mpc.step(x_meas, rt.t)             # gait clock = rt.t -> SLOW_WALK advances
            if res.status not in (0, 2):
                n_fail += 1
            x_plan, u_plan, t_solve = res.x_traj, res.u_traj, rt.t
            node_times_plan = res.node_times if getattr(res, "node_times", None) is not None else wb_cfg.dt
            try:
                solve_tot.append(float(mpc.solver.get_stats("time_tot")) * 1e3)
            except Exception:
                pass

            contact_now = [bool(c) for c in mpc._gait.contact_flags(rt.t)]
            for i in range(2):
                if _contact_prev[i] is not None and not _contact_prev[i] and contact_now[i]:
                    foot_z = float(rt.mj_data.xpos[_foot_bids[i], 2])
                    _min_foot_z_at_activation = min(_min_foot_z_at_activation, foot_z)
                _contact_prev[i] = contact_now[i]

            _log_mpc(rt.t, x_meas, res)

        if k % cdecim == 0:
            q_pin, v_pin = rt._pin_q_v()
            q_meas = q_pin[8:35]; qd_meas = v_pin[8:35]
            x_star, u_star = _sample_plan(x_plan, u_plan, rt.t + sample_ahead_s - t_solve, node_times_plan, wb_cfg.N)
            q_des, qd_des = x_star[6:33], x_star[39:66]
            tau_ff = wb_model.joint_torque(x_star, u_star)
            tau_wb = kp * (q_des - q_meas) + kd * (qd_des - qd_meas) + tau_ff
            tau29 = np.zeros(29)
            tau29[2:29] = tau_wb
            tau29[0:2] = _HEAD_KP * (0.0 - q_pin[6:8]) - _HEAD_KD * v_pin[6:8]
            rt._apply_torque(tau29)
            if do_log:
                _last_ctrl["pd_err_inf"] = float(np.max(np.abs(q_des - q_meas)))
                _last_ctrl["tau_sat_max"] = float(np.max(np.abs(tau_wb) / tlim))
        rt.step_physics()
        d = rt.mj_data
        tilts.append(tilt_from_quat_wxyz(d.qpos[3:7]))
        base_z.append(float(d.qpos[2])); base_x.append(float(d.qpos[0])); base_y.append(float(d.qpos[1]))

    peak_tilt = float(np.max(tilts)); final_z = float(base_z[-1])
    nominal = float(wb_cfg.nominal_base_height)
    mean_vx = (base_x[-1] - base_x[0]) / duration_s
    lateral_pkpk = float(max(base_y) - min(base_y))
    n_steps = int(duration_s / SLOW_WALK_PERIOD * 2) if n_fail == 0 else 0
    min_foot_z = None if _min_foot_z_at_activation == float("inf") else round(_min_foot_z_at_activation, 4)
    passed = bool(peak_tilt < 0.2 and final_z > 0.85 * nominal and n_fail == 0
                  and mean_vx > 0.20 and lateral_pkpk < 0.10)
    out = {
        "mean_vx": round(mean_vx, 3), "peak_tilt_rad": round(peak_tilt, 4),
        "lateral_pkpk_m": round(lateral_pkpk, 4), "n_steps": n_steps, "n_fail": n_fail,
        "final_base_z": round(final_z, 4),
        "median_acados_tot_ms": round(float(np.median(solve_tot)), 2) if solve_tot else None,
        "min_foot_z_at_stance_activation": min_foot_z,
        "passed": passed,
    }

    if do_log:
        data = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in _LOG_KEYS}
        np.savez(log_path, **data)
        out["log_path"] = log_path
        out["convergence"] = _convergence_summary(data, gap_probe_every)
    return out


def _convergence_summary(d: dict, gap_probe_every: int) -> dict:
    """Reduce the per-tick log to the 'is the MPC converging' headline numbers."""
    t = d["t"]; n = len(t)
    if n == 0:
        return {}
    early = t <= (t[0] + 0.2 * (t[-1] - t[0]))
    late = t >= (t[0] + 0.8 * (t[-1] - t[0]))
    def med(key, mask=None):
        v = d[key]; v = v[mask] if mask is not None else v
        v = v[np.isfinite(v)]
        return round(float(np.median(v)), 5) if v.size else None
    def mx(key):
        v = d[key][np.isfinite(d[key])]
        return round(float(np.max(v)), 5) if v.size else None
    status = d["status"]
    failed = ~np.isin(status, [0.0, 2.0]) & np.isfinite(status)
    first_fail_t = round(float(t[failed][0]), 3) if failed.any() else None
    summ = {
        "n_mpc_ticks": int(n),
        "n_fail": int(failed.sum()), "first_fail_t": first_fail_t,
        "res_eq_median": med("res_eq"), "res_eq_max": mx("res_eq"),
        "res_eq_early_median": med("res_eq", early), "res_eq_late_median": med("res_eq", late),
        "res_stat_median": med("res_stat"), "res_stat_max": mx("res_stat"),
        "qp_iter_median": med("qp_iter"), "qp_iter_max": mx("qp_iter"),
        "stepnorm_x_early_median": med("stepnorm_x", early), "stepnorm_x_late_median": med("stepnorm_x", late),
        "pred_err_yaw_early_median": med("pred_err_yaw", early), "pred_err_yaw_late_median": med("pred_err_yaw", late),
        "pred_err_base_early_median": med("pred_err_base", early), "pred_err_base_late_median": med("pred_err_base", late),
    }
    if gap_probe_every:
        gp = np.isfinite(d["gp_sqp_iter"])
        summ["gap_probe"] = {
            "n_probes": int(gp.sum()),
            "conv_sqp_iter_median": med("gp_sqp_iter"), "conv_sqp_iter_max": mx("gp_sqp_iter"),
            "x_gap_median": med("gp_x_gap"), "x_gap_max": mx("gp_x_gap"),
            "res_eq_rti_median": med("gp_res_eq_rti"), "res_eq_conv_median": med("gp_res_eq_conv"),
            "cost_rti_median": med("gp_cost_rti"), "cost_conv_median": med("gp_cost_conv"),
        }
    return summ


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WB forward-walk gate: headless PASS/metrics, or live viewer with --view.")
    ap.add_argument("--vx", type=float, default=0.3, help="forward velocity command (0 -> stand)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--view", action="store_true", help="open the live MuJoCo viewer instead of printing metrics")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (<1 = slow-mo)")
    ap.add_argument("--log", type=str, default=None, metavar="OUT.npz",
                    help="dump per-MPC-tick solver-health + plan-vs-reality + contact telemetry")
    ap.add_argument("--gap-probe-every", type=int, default=0, metavar="K",
                    help="every K-th MPC tick, run a converged re-solve to measure the single-RTI gap (needs --log)")
    a = ap.parse_args()
    if a.view:
        from sim.wb_walk_view import run_view      # live viewer (same control law); needs a display
        run_view(a.vx, a.duration, a.speed)
    else:
        out = run_wb_walk(duration_s=a.duration, vx=a.vx, log_path=a.log, gap_probe_every=a.gap_probe_every)
        conv = out.pop("convergence", None)
        print("WALK_GATE=" + json.dumps(out))
        if conv is not None:
            print("WALK_CONV=" + json.dumps(conv, indent=2))
