import numpy as np
from t1_nmpc.robot.config import make_config, JOINT_NAMES, LOCKED_JOINTS


def test_dims():
    c = make_config()
    assert (c.n_joints, c.nq, c.nv, c.nx, c.ndx) == (27, 34, 33, 67, 66)
    assert (c.n_feet, c.nf, c.na, c.nu) == (2, 12, 33, 45)
    assert c.nodes == 31 and abs(c.dt - 0.035) < 1e-12
    assert abs(c.nodes * c.dt - 1.085) < 1e-9


def test_gait_params():
    c = make_config()
    assert abs(c.gait_cycle - 1.4) < 1e-12
    assert c.switching_times == (0.0, 0.6, 0.7, 1.3, 1.4)
    assert abs(c.swing_height - 0.08) < 1e-12
    assert (c.v_liftoff, c.v_touchdown) == (0.05, -0.05)


def test_weights_shapes():
    c = make_config()
    assert c.Q_diag.shape == (66,)        # ndx
    assert c.R_diag.shape == (45,)        # nu
    assert c.kp.shape == (27,) and c.kd.shape == (27,)
    assert c.nominal_joint_pos.shape == (27,)


def test_joint_name_tables():
    assert len(JOINT_NAMES) == 27
    assert LOCKED_JOINTS == ("AAHead_yaw", "Head_pitch")
    assert "AAHead_yaw" not in JOINT_NAMES


def test_aligator_params():
    c = make_config()
    assert c.al_tol == 1e-3
    assert c.warm_max_iters >= 1 and c.cold_max_iters > c.warm_max_iters
    assert 0.0 < c.mu_init <= 1.0
