# tests/test_sysid_friction.py
import numpy as np
import mujoco
from t1_nmpc.config import make_config
from t1_nmpc.model import load_model, T1_URDF_PATH
from sim.mujoco_runtime import MujocoRuntime


def _rt(fric):
    return MujocoRuntime(make_config(mpc_hz=60.0), load_model(T1_URDF_PATH, make_config()),
                         apply_joint_friction=fric)


def test_friction_off_by_default():
    rt = MujocoRuntime(make_config(mpc_hz=60.0), load_model(T1_URDF_PATH, make_config()))
    assert rt.apply_joint_friction is False
    rt.mj_data.qvel[:] = 1.0
    rt.step_physics()
    assert np.allclose(rt.mj_data.qfrc_applied, 0.0)   # default: no custom friction force


def test_stribeck_static_to_dynamic_and_opposes_motion():
    rt = _rt(True)
    m = rt.mj_model
    adr = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Left_Knee_Pitch")])
    Fs, Fc = 0.998922, 0.722987                        # knee static, dynamic (sysID)
    rt.mj_data.qvel[:] = 0.0
    rt.mj_data.qvel[adr] = 2.0                          # well above stribeck/smoothing vel
    tau_hi = rt._joint_friction_torque()[adr]
    assert tau_hi < 0                                   # opposes +v
    assert abs(abs(tau_hi) - Fc) < 1e-2                 # high speed -> dynamic friction
    rt.mj_data.qvel[adr] = 0.015                        # breakaway band -> static-dominated
    tau_lo = abs(rt._joint_friction_torque()[adr])
    assert Fc < tau_lo <= Fs + 1e-9                     # static > dynamic near breakaway


def test_friction_stand_stays_upright():
    rt = _rt(True)
    rt.reset_to_nominal()                               # 0.6 s PD settle exercises step_physics+friction
    assert np.all(np.isfinite(rt.mj_data.qpos))         # no NaN/instability
    assert rt.mj_data.qpos[2] > 0.5                     # did not collapse
