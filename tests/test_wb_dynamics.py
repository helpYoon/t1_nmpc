import numpy as np

from t1_nmpc.wb.config import make_wb_config, MPC_JOINT_NAMES
from t1_nmpc.wb.dynamics import WBModel


def test_cpin_mass_matrix_matches_pinocchio_numeric():
    cfg = make_wb_config()
    m = WBModel(cfg)
    # composite base (translation + sphericalZYX): nq == nv == 33 = 6 base + 27 joints
    assert m.nq == 33 and m.nv == 33
    q = m.neutral_q()
    M = m.M(q)
    assert M.shape == (33, 33)
    assert np.allclose(M, M.T, atol=1e-8)              # symmetric
    assert np.all(np.linalg.eigvalsh(M) > 0)           # PD
    assert abs(m.total_mass() - M[0, 0]) < 1e-6        # top-left translation block = total mass
    assert np.allclose(M, m.M_numeric_pin(q), atol=1e-6)   # cpin == numeric pin (same model)


def test_joint_order_and_contact_frames():
    cfg = make_wb_config()
    m = WBModel(cfg)
    # reduced model joint order (head excluded) == the MPC joint set
    assert tuple(m.model.names[2:]) == MPC_JOINT_NAMES
    assert len(m.contact_fids) == 2


def test_nle_and_jacobian_shapes_and_numeric_match():
    cfg = make_wb_config()
    m = WBModel(cfg)
    rng = np.random.default_rng(0)
    q = m.neutral_q().copy()
    q[6:] += 0.05 * rng.standard_normal(27)            # perturb joints only (keep base sane)
    v = 0.1 * rng.standard_normal(33)
    assert m.nle(q, v).shape == (33,)
    assert m.Jl(q).shape == (6, 33) and m.Jr(q).shape == (6, 33)
    # cpin nle == numeric pin rnea(q,v,0)
    assert np.allclose(m.nle(q, v), m.nle_numeric_pin(q, v), atol=1e-6)
