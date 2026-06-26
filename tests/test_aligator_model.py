import numpy as np
import pytest
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.aligator_model import build_aligator_model, nominal_stand_x, make_ode

@pytest.mark.filterwarnings("ignore:already registered:RuntimeWarning")  # pinocchio DeprecatedBool binding registration
def test_faithful_model_dims_and_dynamics():
    am = build_aligator_model(make_wb_config())
    assert am.nq == 34 and am.nv == 33 and am.ndx == 66
    assert abs(am.mass - 34.51) < 0.1            # m*g ~ 338.6 N
    assert len(am.foot_ids) == 2
    ode = make_ode(am, [True, True])
    assert ode.nu == 2 * 6 + (am.nv - 6) == 39
    x = nominal_stand_x(am, make_wb_config())
    assert x.shape[0] == am.nq + am.nv == 67
    import aligator
    disc = aligator.dynamics.IntegratorSemiImplEuler(ode, 0.02)
    d = disc.createData(); u = np.zeros(ode.nu); u[2] = u[8] = am.mass * 9.81 / 2
    disc.forward(x, u, d)
    assert np.all(np.isfinite(np.asarray(d.xnext)))
