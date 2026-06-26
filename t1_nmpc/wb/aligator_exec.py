"""Extract the robot command (contact wrenches + feed-forward joint torque) from a kinodynamic
solution. Kinodynamics returns NO torque -> recover via RNEA(q,v,a) - sum J_LWA^T f. Validated:
|tau[:6]| ~ 1e-13."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from .aligator_model import make_ode

def extract_tau_ff(am, x_meas, u0, FS: int = 6):
    q = np.asarray(x_meas[:am.nq]); v = np.asarray(x_meas[am.nq:])
    # base-6 accel is SOLVED by the dynamics; read it from the continuous ode's xdot
    ode = make_ode(am, [True, True], FS)            # contact mask irrelevant to the accel passthrough
    od = ode.createData(); ode.forward(x_meas, u0, od)
    a = np.asarray(od.xdot)[am.nv:].copy()
    a[6:] = np.asarray(u0[2 * FS:])                 # joint accels pass straight from the control
    rdata = am.model.createData()
    tau = pin.rnea(am.model, rdata, q, v, a)
    pin.computeJointJacobians(am.model, rdata, q); pin.framesForwardKinematics(am.model, rdata, q)
    for k, fid in enumerate(am.foot_ids):
        J = pin.getFrameJacobian(am.model, rdata, fid, pin.LOCAL_WORLD_ALIGNED)
        tau -= J.T @ np.asarray(u0[k * FS:(k + 1) * FS])
    return tau[6:].copy(), np.asarray(u0[0:6]).copy(), np.asarray(u0[6:12]).copy()
