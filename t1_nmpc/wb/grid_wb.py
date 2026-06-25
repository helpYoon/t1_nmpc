"""Event-aligned variable-dt shooting grid — faithful fixed-N adaptation of OCS2
timeDiscretizationWithEvents (TimeDiscretization.cpp:60-114). Marches at ~uniform dt but lands a
node exactly on every contact switch in the horizon. Single node per switch (no jump duplication;
identity jump map -> benign). Pure: no acados/casadi."""
from __future__ import annotations

import numpy as np


def event_aligned_grid(t0: float, gait, cfg) -> np.ndarray:
    N, dt = cfg.N, cfg.dt
    T = N * dt
    eps = 1e-10
    # Real contact-mode switches in the horizon: drop phase-0 cycle wraps that don't change the mode
    # (e.g. STANCE_GAIT, whose switch_times_in returns period wraps), and drop any switch within
    # 0.5*dt of a boundary (it would force a sub-0.5*dt segment; it is re-aligned on earlier ticks
    # while deeper in the horizon).
    switches = [s for s in gait.switch_times_in(t0, t0 + T)
                if gait.contact_flags(s - eps) != gait.contact_flags(s + eps)
                and (t0 + 0.5 * dt) < s < (t0 + T - 0.5 * dt)]
    bounds = np.array([t0, *switches, t0 + T], dtype=np.float64)
    seg_len = np.diff(bounds)                                  # M segments, each >= 0.5*dt
    n = np.maximum(1, np.round(seg_len / dt).astype(int))     # intervals per segment ~ uniform dt
    while n.sum() != N:                                        # reconcile to exactly N via longest segment
        if n.sum() > N:
            cand = np.where(n > 1)[0]
            j = cand[np.argmax(seg_len[cand])]
            n[j] -= 1
        else:
            j = int(np.argmax(seg_len))
            n[j] += 1
    nodes = [np.linspace(bounds[k], bounds[k + 1], n[k] + 1)[:-1] for k in range(len(n))]
    node_times = np.concatenate([*nodes, [bounds[-1]]])
    return np.ascontiguousarray(node_times, dtype=np.float64)
