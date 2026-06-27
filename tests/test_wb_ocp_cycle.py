import numpy as np
from t1_nmpc.wb.config import make_wb_config
from t1_nmpc.wb.config import make_aligator_config
from t1_nmpc.wb.ode import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.gait import SLOW_WALK
from t1_nmpc.wb.ocp import build_gait_cycle

def test_gait_cycle_has_all_modes_with_correct_nu():
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    x = nominal_stand_x(am, cfg)
    node_times = np.arange(al.N) * cfg.dt
    models, sched = build_gait_cycle(am, cfg, al, SLOW_WALK, x, node_times)
    assert len(models) == len(sched) >= al.N
    assert all(m.nu == 39 for m in models)
    # at least one single-support node exists in a walking schedule
    assert any(sum(f) == 1 for f in sched)
