# tests/test_croco_walk_mpc.py
import numpy as np
import pytest
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK, STANCE_GAIT
from t1_nmpc.wb.croco_mpc import CrocoMPC

def test_walk_step_rebuilds_and_emits_stance_aware_u():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0)
    res = mpc.step(x0, 0.0, command=np.array([0.3, 0.0, cfg.nominal_base_height, 0.0]))
    assert res.x_traj.shape == (cfg.N+1, 68) and res.u_traj.shape == (cfg.N, 40)
    assert res.status == 0 and np.all(np.isfinite(res.u_traj))

@pytest.mark.slow
def test_walk_path_holds_double_support_stand():
    """REGRESSION (closed-loop, fix #2): the walk path rebuilds SolverIntro every cycle. With a
    shifted previous-solution warm-start a fresh solver diverges at maxiter=1 (stoppingCriteria
    grows, robot falls); the quasiStatic re-anchored warm-start holds. Drive STANCE_GAIT (always
    double support) at vx=0 for 4 s and require the robot stays upright. Together with the
    well-conditioned-OCP test this pins both walk-path root causes."""
    from t1_nmpc.runtime.mujoco_transport import MujocoTransport
    from t1_nmpc.wb.execution_wb import to_joint_command_wb
    cfg = make_wb_config(); wb = WBModel(cfg)
    transport = MujocoTransport(cfg, mpc_hz=40.0); rt = transport.rt
    mpc = CrocoMPC(cfg, wb, max_iter=1, gait=STANCE_GAIT)
    x0 = transport.read_state(); mpc.reset(x0)
    ctrl_hz = float(rt.cfg.control_hz); solve_every = max(1, int(round(ctrl_hz / 40.0)))
    n = int(round(4.0 * ctrl_hz)); cmd_v = np.array([0., 0., float(cfg.nominal_base_height), 0.])
    res = mpc.step(x0, transport.now(), command=cmd_v)
    cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=0.005); nf = 0
    for k in range(n):
        x = transport.read_state()
        if k % solve_every == 0:
            res = mpc.step(x, transport.now(), command=cmd_v)
            if res.status != 0:
                nf += 1
            cmd = to_joint_command_wb(res, cfg, mpc.model, sample_ahead_s=0.005)
        transport.write_command(cmd)
    from sim._sim_util import tilt_from_quat_wxyz
    d = rt.mj_data; tilt = float(tilt_from_quat_wxyz(d.qpos[3:7])); z = float(d.qpos[2])
    assert nf == 0, f"solver failed {nf} times (single-RTI diverged)"
    assert tilt < 0.2, f"double-support stand fell (tilt={tilt:.3f} rad)"
    assert z > 0.6, f"base z collapsed (z={z:.3f} m)"


def test_walk_advances_gait_clock_a_few_steps():
    cfg = make_wb_config(); wb = WBModel(cfg)
    mpc = CrocoMPC(cfg, wb, gait=SLOW_WALK)
    x0 = np.zeros(68); x0[2] = cfg.nominal_base_height; x0[6:6+cfg.n_joints] = cfg.nominal_joint_pos
    mpc.reset(x0); x = x0.copy()
    n_steps = 5
    for k in range(n_steps):
        t = k * float(cfg.dt)
        res = mpc.step(x, t, command=np.array([0.3,0.,cfg.nominal_base_height,0.]))
        assert np.all(np.isfinite(res.x_traj)); x = res.x_traj[1].copy()
    # gait clock must track the real sim time passed in, not self-increment
    assert mpc._t_gait == pytest.approx((n_steps - 1) * float(cfg.dt))
