import numpy as np
from t1_nmpc.mpc_result import MPCResult

def test_mpc_result_fields():
    r = MPCResult(x_traj=np.zeros((2, 68)), u_traj=np.zeros((1, 40)),
                  feasible=True, solve_time=0.01, mode_schedule=None, status=0)
    assert r.feasible and r.status == 0 and r.x_traj.shape == (2, 68)
