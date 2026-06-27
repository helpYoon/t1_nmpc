import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.state import (mujoco_to_freeflyer, extract_command, MUJOCO_TO_PIN_JOINTS)


def test_joint_index_map_drops_head():
    assert MUJOCO_TO_PIN_JOINTS.shape == (27,)
    # MuJoCo actuated joints are [head2, Larm7, Rarm7, waist, Lleg6, Rleg6] (29);
    # reduced pin joints are the same minus the 2 head -> indices 2..28
    np.testing.assert_array_equal(MUJOCO_TO_PIN_JOINTS, np.arange(2, 29))


def test_base_linear_velocity_rotation():
    cfg = make_config(); rm = load_model(cfg)
    # 90deg yaw, world x-velocity -> body y-velocity (negative)
    qw = np.cos(np.pi / 4); qz = np.sin(np.pi / 4)
    qpos = np.zeros(36); qpos[2] = 0.6734; qpos[3:7] = [qw, 0, 0, qz]
    qvel = np.zeros(35); qvel[0] = 1.0      # world +x
    x = mujoco_to_freeflyer(qpos, qvel, rm.model)
    # body-local linear vel: R^T @ [1,0,0]
    R = pin.Quaternion(qw, 0, 0, qz).toRotationMatrix()
    np.testing.assert_allclose(x[34:37], R.T @ np.array([1, 0, 0]), atol=1e-9)


def test_extract_command_shapes():
    cfg = make_config(); rm = load_model(cfg)
    x1 = nominal_x(cfg, rm.model); tau0 = np.zeros(27)
    cmd = extract_command(x1, tau0, cfg, rm)
    assert cmd.q_des.shape == (27,) and cmd.qd_des.shape == (27,)
    assert cmd.tau_ff.shape == (27,) and cmd.kp.shape == (27,)
