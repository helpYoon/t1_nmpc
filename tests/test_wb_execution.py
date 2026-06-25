# tests/test_wb_execution.py
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.execution_wb import to_joint_command_wb


class _Result:
    def __init__(self, x_traj, u_traj, node_times=None):
        self.x_traj = x_traj
        self.u_traj = u_traj
        self.node_times = node_times  # None exercises uniform fallback


def test_tau_ff_from_lookahead_sample_not_node0():
    # OCS2 (MpcMrtJointController.cpp:256-262) computes tau_ff AND q_des/qd_des from the SAME t+dt
    # look-ahead policy sample; the deployed port previously took tau_ff from plan node 0. (audit 2026-06-25)
    cfg = make_wb_config(); m = WBModel(cfg)
    N = cfg.N
    x0 = m.nominal_state(); x1 = x0.copy(); x1[33] = 0.3            # node-1 state differs (base vx)
    u0 = np.zeros(40); u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
    u1 = u0.copy(); u1[12:39] = 0.1                                 # node-1 input differs (joint accel)
    x_traj = np.tile(x0, (N + 1, 1)); x_traj[1] = x1
    u_traj = np.tile(u0, (N, 1)); u_traj[1] = u1

    jc = to_joint_command_wb(_Result(x_traj, u_traj), cfg, m, sample_ahead_s=0.005)

    a = 0.005 / cfg.dt                                             # lo=0, hi=1
    xq = (1 - a) * x0 + a * x1
    uq = (1 - a) * u0 + a * u1
    np.testing.assert_allclose(jc.tau_ff, m.joint_torque(xq, uq), atol=1e-9)
    # must DIFFER from the old node-0 behavior (so we are genuinely sampling the look-ahead)
    assert not np.allclose(jc.tau_ff, m.joint_torque(x0, u0), atol=1e-6)
    # q_des/qd_des and the reported wrenches come from the same look-ahead sample
    np.testing.assert_allclose(jc.q_des, xq[6:33], atol=1e-12)
    np.testing.assert_allclose(jc.wrench_l, uq[0:6], atol=1e-12)
