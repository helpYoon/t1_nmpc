import numpy as np

from t1_nmpc.robot.config import load_config, JointCommand
from t1_nmpc.robot.execution import pd_torque


def test_pd_torque_law():
    cfg = load_config()
    q_des = 0.1 + 0.01 * np.arange(29, dtype=np.float64)
    qd_des = 0.5 + 0.02 * np.arange(29, dtype=np.float64)
    jc = JointCommand(
        q_des=q_des,
        qd_des=qd_des,
        kp=cfg.kp,
        kd=cfg.kd,
        tau_ff=np.zeros(cfg.n_joints, dtype=np.float64),
    )
    q_meas = np.zeros(29)
    qd_meas = np.zeros(29)
    tau = pd_torque(jc, q_meas, qd_meas)
    expected = cfg.kp * (jc.q_des - q_meas) + cfg.kd * (jc.qd_des - qd_meas)
    np.testing.assert_allclose(tau, expected)
    assert tau.shape == (29,)
