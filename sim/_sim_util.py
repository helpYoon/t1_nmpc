"""Shared sim metrics: base tilt from a wxyz quat + an upright/standing check."""
import numpy as np
import pinocchio as pin

M0B_TILT_MAX = 0.2
M0B_Z_FRAC = 0.9


def tilt_from_quat_wxyz(q) -> float:
    R = pin.Quaternion(q[0], q[1], q[2], q[3]).normalized().toRotationMatrix()
    cos_t = float(np.clip(R[2, 2], -1.0, 1.0))
    return float(np.arccos(cos_t))


def upright_ok(z, tilt, nominal) -> bool:
    return bool(z > M0B_Z_FRAC * nominal and tilt < M0B_TILT_MAX)
