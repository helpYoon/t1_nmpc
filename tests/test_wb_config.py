import numpy as np

from t1_nmpc.wb.config import make_wb_config


def test_wb_config_dims_and_weights():
    c = make_wb_config()
    assert (c.nx, c.nu, c.n_joints, c.N) == (68, 40, 27, 31)
    assert abs(c.dt - 0.035) < 1e-9 and abs(c.horizon - 1.1) < 1e-2
    assert c.Q.shape == (68,) and c.R.shape == (40,) and c.Q_final.shape == (68,)
    assert len(c.mpc_joint_names) == 27 and "AAHead_yaw" not in c.mpc_joint_names
    assert c.nominal_joint_pos.shape == (27,) and c.torque_limit.shape == (27,)
    assert c.friction_mu == 0.4 and c.foot_pos_err_gain_z == 100.0
    assert np.all(c.torque_limit > 0)  # finite-or-inf, all positive


def test_torque_limit_matches_t1_controller():
    # Soft-cap limits must mirror t1_controller's effortLimit (its t1.urdf), NOT the wb_humanoid URDF
    # the dynamics load from (which under-limits 11 leg/waist/wrist joints, over-penalizing push-off
    # up to 25x). (audit 2026-06-25)
    c = make_wb_config()
    expected = np.array([18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
                         90, 130, 30, 30, 130, 60, 12, 130, 30, 30, 130, 60, 12], dtype=float)
    np.testing.assert_array_equal(c.torque_limit, expected)


def test_wb_config_qr_blocks_match_taskinfo():
    c = make_wb_config()
    # Q: base z weight 10, trunk pitch/roll 20; path slots (s,v_s) zero
    assert c.Q[2] == 10.0 and c.Q[4] == 20.0 and c.Q[5] == 20.0
    assert c.Q[66] == 0.0 and c.Q[67] == 0.0
    # R scaled by 1e-3: left fz weight 0.001*1e-3; path-input slot zero
    assert abs(c.R[2] - 0.001 * 1e-3) < 1e-12
    assert abs(c.R[39] - 1e-6) < 1e-12   # vdot_s regularized (1e-3 * 1e-3); was 0 before c02f6b2
    # joint limits / pd gains length 27, head-excluded order (arms,waist,legs)
    assert c.joint_lower.shape == (27,) and c.joint_upper.shape == (27,)
    assert np.all(c.joint_lower < c.joint_upper)
    assert c.kp.shape == (27,) and c.kd.shape == (27,)
    assert c.kp[0] == 20.0 and c.kp[14] == 200.0 and c.kp[-1] == 50.0  # arm, waist, ankle
