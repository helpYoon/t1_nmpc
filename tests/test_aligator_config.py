from t1_nmpc.wb.config_aligator import AligatorConfig, make_aligator_config

def test_defaults_match_validated_operating_point():
    c = make_aligator_config()
    assert c.N == 20 and c.max_iters == 2 and c.num_threads == 4
    assert c.hard_cones is True and c.FS == 6
    assert c.mu_init == 1e-2 and c.tol == 1e-3 and c.max_al_iters == 2
    # cost-weight vectors are present and finite-sized
    assert c.w_base_pose > 0 and c.w_joint_pos > 0 and c.w_force_reg > 0
