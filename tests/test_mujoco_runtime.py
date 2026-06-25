import numpy as np
import pytest

from t1_nmpc.config import load_config
from t1_nmpc.model import load_model
from sim.mujoco_runtime import (
    MujocoRuntime,
    T1_MJCF_PATH,
    MJ_JOINT_QPOS0,
    MJ_JOINT_QVEL0,
)


def _urdf_path():
    return ("/home/yoonwoo/humanoid_mpc_ws/src/wb_humanoid_mpc/"
            "robot_models/booster_t1/t1_description/urdf/t1.urdf")


def test_rate_decimation_constants():
    cfg = load_config()
    rt = MujocoRuntime(cfg, model=None, mjcf_path=T1_MJCF_PATH)
    # physics 2000 Hz, control 500 Hz, mpc 40 Hz
    assert rt.physics_dt == pytest.approx(1.0 / 2000.0)
    assert rt.control_decim == 4      # 2000/500
    assert rt.mpc_decim == 50         # 2000/40


def test_mujoco_model_layout_matches_contract():
    rt = MujocoRuntime(load_config(), model=None, mjcf_path=T1_MJCF_PATH)
    m = rt.mj_model
    assert m.nq == 36 and m.nv == 35 and m.nu == 28
    # base free joint, joints 1..29 are the §A.5 order at qpos[7:36]
    assert MJ_JOINT_QPOS0 == 7
    assert MJ_JOINT_QVEL0 == 6
    # total mass within 1e-2 of the contract robot_mass
    assert abs(sum(m.body_mass) - 34.5135) < 0.02


def test_actuator_map_skips_waist():
    rt = MujocoRuntime(load_config(), model=None, mjcf_path=T1_MJCF_PATH)
    # 28 actuators, Waist (state local idx 16) maps to -1 (unactuated)
    assert len(rt.act_to_state_idx) == 28
    assert 16 not in rt.act_to_state_idx          # Waist joint not actuated
    # every actuated index is a valid joint-local index 0..28
    assert all(0 <= i <= 28 for i in rt.act_to_state_idx)


def test_apply_pd_writes_ctrl_skipping_waist():
    rt = MujocoRuntime(load_config(), model=None, mjcf_path=T1_MJCF_PATH)
    tau29 = np.arange(29, dtype=np.float64)       # local-joint torque vector
    rt._apply_torque(tau29)
    ctrl = np.array(rt.mj_data.ctrl)
    assert ctrl.shape == (28,)
    # actuator i receives tau29[act_to_state_idx[i]]; none receives tau29[16]
    for a, j in enumerate(rt.act_to_state_idx):
        assert ctrl[a] == tau29[j]
    assert 16.0 not in ctrl.tolist()
