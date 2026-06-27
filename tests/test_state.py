import numpy as np
import pinocchio as pin
from t1_nmpc.robot.config import make_config, T1_URDF_PATH
from t1_nmpc.wb.state import mujoco_to_freeflyer, freeflyer_to_mujoco

def test_state_roundtrip_under_yaw():
    model = pin.buildModelFromUrdf(T1_URDF_PATH, pin.JointModelFreeFlyer())
    rng = np.random.default_rng(0); nj = model.nq - 7
    yaw = np.pi/2
    qpos = np.empty(36)
    qpos[0:3] = [0.31, -0.22, 0.6734]
    qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]   # (w,x,y,z) about z
    qpos[7:] = rng.uniform(-0.3, 0.3, nj)
    qvel = np.empty(35)
    qvel[0:3] = [0.7, -0.4, 0.15]    # WORLD linear
    qvel[3:6] = [0.2, -0.13, 0.5]    # LOCAL angular
    qvel[6:] = rng.uniform(-0.5, 0.5, nj)
    x = mujoco_to_freeflyer(qpos, qvel, model)
    assert x.shape == (71,)
    # the rotation must actually be applied: body-linear differs from world-linear under yaw
    assert np.max(np.abs(x[36:39] - qvel[0:3])) > 1e-3
    qpos2, qvel2 = freeflyer_to_mujoco(x, model)
    assert np.max(np.abs(qpos2 - qpos)) < 1e-12
    assert np.max(np.abs(qvel2 - qvel)) < 1e-12
