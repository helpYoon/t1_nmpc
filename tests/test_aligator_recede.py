import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.config_aligator import make_aligator_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.aligator_mpc import AligatorMPC

def test_full_contact_cycle_solves_finite(capfd):
    cfg = make_wb_config(); al = make_aligator_config(); am = build_aligator_model(cfg)
    mpc = AligatorMPC(cfg, al, am, gait=SLOW_WALK)
    x = nominal_stand_x(am, cfg); mpc.reset(x)
    # advance through more than one full gait period (DS -> Lswing -> DS -> Rswing -> DS)
    n = int(round(2.0 / cfg.dt))
    for k in range(n):
        res = mpc.step(x, k * cfg.dt)
        assert res.status == 0, f"non-finite solve at k={k}"
        x = np.asarray(res.xs[1]).copy()   # roll plan forward (open-loop here; closed-loop in Task 9)
    assert True  # reached the end with finite solves across all contact transitions
    # Pristine-output guard: the benign aligator 'Resize happened' log must be suppressed
    out, err = capfd.readouterr()
    assert "Resize happened" not in err and "Resize happened" not in out, (
        "aligator resize log leaked to output — check _suppress_cxx_stderr in aligator_mpc.py"
    )
