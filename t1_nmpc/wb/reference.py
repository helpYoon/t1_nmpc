"""motion_plan.pkl -> pinocchio reference. Joint mapping authority: t1_kd_mpc.

Maps the plan's reduced channels onto the full 29-joint FreeFlyer state: base z=trunk_height,
base lean=yaw-anchored trunk_quat, arms linear, Waist=-trunk_yaw, leg-pitch broadcast to both legs
(SEED only -- low Q weight; the OCP solves the real legs against planted-feet contact). Head and
hip-roll/yaw + ankle-roll stay nominal. Hands are exported as task-space targets. See
docs/superpowers/specs/2026-06-28-pickup-trajectory-tracking-design.md."""
from __future__ import annotations

import pickle

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

from ..robot.config import MPCConfig
from ..robot.model import RobotModel

# joint-local indices (pinocchio order); full-q index = 7 + j
_J_LARM = slice(2, 9)
_J_RARM = slice(9, 16)
_J_WAIST = 16
_J_LHIP_P, _J_LKNEE, _J_LANK_P = 17, 20, 21
_J_RHIP_P, _J_RKNEE, _J_RANK_P = 23, 26, 27


def _anchor_xyz(p, x0, y0, yaw0):
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.array([c * p[0] - s * p[1] + x0, s * p[0] + c * p[1] + y0, p[2]], dtype=np.float64)


def _anchor_quat(quat_xyzw, yaw0):
    if yaw0 == 0.0:
        return np.asarray(quat_xyzw, dtype=np.float64)
    return (Rsc.from_euler("z", yaw0) * Rsc.from_quat(quat_xyzw)).as_quat()


class MotionPlanReference:
    def __init__(self, plan_path: str, cfg: MPCConfig, rm: RobotModel,
                 x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
        self.cfg = cfg
        self.model = rm.model
        self.nomj = np.asarray(cfg.nominal_joint_pos, dtype=np.float64)
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.grasp_hw = float(cfg.grasp_halfwidth)
        with open(plan_path, "rb") as f:
            plan = pickle.load(f)
        self.segments = plan["segments"]
        self._build_timeline()

    def frame_to_xref(self, seg: dict, k: int) -> np.ndarray:
        """One plan frame -> pinocchio q (36,). Velocities are added later by finite difference."""
        P = seg["position"]
        q = np.empty(36, dtype=np.float64)
        base_xyz = _anchor_xyz(P["trunk_xyz"][k], self.x0, self.y0, self.yaw0)
        q[0] = base_xyz[0]; q[1] = base_xyz[1]
        q[2] = float(P["trunk_height"][k])                                   # base z = trunk_height
        q[3:7] = _anchor_quat(P["trunk_quat_xyzw"][k], self.yaw0)            # lean
        j = self.nomj.copy()
        j[_J_LARM] = P["left_arm"][k]; j[_J_RARM] = P["right_arm"][k]
        j[_J_WAIST] = -float(P["trunk_yaw"][k])                              # Waist = -trunk_yaw
        tp, kn, an = float(P["trunk_pitch"][k]), float(P["knee_pitch"][k]), float(P["ankle_pitch"][k])
        j[_J_LHIP_P] = tp; j[_J_RHIP_P] = tp                                 # broadcast both legs
        j[_J_LKNEE] = kn; j[_J_RKNEE] = kn
        j[_J_LANK_P] = an; j[_J_RANK_P] = an
        q[7:] = j
        return q

    def _hand_frame(self, seg: dict, k: int) -> np.ndarray:
        P = seg["position"]
        lh = _anchor_xyz(P["left_hand_xyz"][k], self.x0, self.y0, self.yaw0)
        rh = _anchor_xyz(P["right_hand_xyz"][k], self.x0, self.y0, self.yaw0)
        return np.concatenate([lh, rh])

    def _build_timeline(self):
        qs, hs, ts = [], [], []
        t = 0.0
        seg_start_t = []      # phase time at each segment's first frame
        for si, seg in enumerate(self.segments):
            seg_start_t.append(t)
            for k in range(seg["T"]):
                qs.append(self.frame_to_xref(seg, k))
                hs.append(self._hand_frame(seg, k))
                ts.append(t)
                t += float(seg["dt"])
        self.q_frame = np.asarray(qs)        # (F,36)
        self.hand_frame = np.asarray(hs)     # (F,6)
        self.t_frame = np.asarray(ts)        # (F,)
        self.duration_phase = float(self.t_frame[-1])
        # grasp/release events: where a hand's hold-state flips at a segment boundary
        self.events = {0: [], 1: []}
        prev = {0: False, 1: False}          # left, right currently held
        for si, seg in enumerate(self.segments):
            ho = seg["held_objs"]
            cur = {0: ("left" in ho), 1: ("right" in ho)}
            for h in (0, 1):
                if cur[h] != prev[h]:
                    self.events[h].append(seg_start_t[si])
            prev = cur
