"""Event-aligned variable-dt shooting grid — faithful fixed-N adaptation of OCS2
timeDiscretizationWithEvents (TimeDiscretization.cpp:60-114). Marches at ~uniform dt but lands a
node exactly on every contact switch in the horizon. Single node per switch (no jump duplication;
identity jump map -> benign). Pure: no acados/casadi."""
from __future__ import annotations

import numpy as np


def event_aligned_grid(t0: float, gait, cfg) -> np.ndarray:
    N, dt = cfg.N, cfg.dt
    T = N * dt
    all_switches = gait.switch_times_in(t0, t0 + T)

    # Filter to only actual contact-mode changes (not pure phase boundaries)
    switches = []
    eps = 1e-10
    for s in all_switches:
        before = gait.contact_flags(s - eps)
        after = gait.contact_flags(s + eps)
        if before != after:
            switches.append(s)

    bounds = np.array([t0, *switches, t0 + T], dtype=np.float64)
    seg_len = np.diff(bounds)                                  # M segments

    # Merge very small segments (< 0.5 * dt) with the adjacent segment on the right
    # This prevents degenerate intervals that would be < dt/2
    while True:
        tiny = np.where(seg_len < 0.5 * dt)[0]
        if len(tiny) == 0:
            break
        # Merge the first tiny segment (at index i) with the next (at index i+1)
        i = tiny[0]
        bounds = np.concatenate([bounds[:i+1], bounds[i+2:]])
        seg_len = np.diff(bounds)

    # intervals per segment ~ uniform dt (round), >=1
    n = np.maximum(1, np.round(seg_len / dt).astype(int))
    # reconcile to exactly N by adjusting the LONGEST segment (never below 1 interval)
    while n.sum() != N:
        if n.sum() > N:
            cand = np.where(n > 1)[0]
            j = cand[np.argmax(seg_len[cand])]
            n[j] -= 1
        else:
            j = int(np.argmax(seg_len))
            n[j] += 1
    # place each segment's nodes uniformly; drop the shared right endpoint between segments
    nodes = [np.linspace(bounds[k], bounds[k + 1], n[k] + 1)[:-1] for k in range(len(n))]
    node_times = np.concatenate([*nodes, [bounds[-1]]])
    return np.ascontiguousarray(node_times, dtype=np.float64)
