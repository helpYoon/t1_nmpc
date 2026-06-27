import numpy as np
from t1_nmpc.robot.config import make_config, T1_URDF_PATH, T1_PACKAGE_DIRS, JOINT_NAMES
import os

def test_config_dims_and_pose():
    cfg = make_config()
    assert cfg.nx == 71 and cfg.ndx == 70 and cfg.nf == 24 and cfg.na == 35
    assert cfg.nodes == 31 and cfg.tau_nodes == 3
    assert abs(cfg.nominal_base_height - 0.6734) < 1e-12
    assert cfg.nominal_joint_pos.shape == (29,)
    # shallow-crouch legs (knee 0.10)
    assert abs(cfg.nominal_joint_pos[20] - 0.10) < 1e-12  # Left_Knee_Pitch idx in 29-order
    assert cfg.Q_diag.shape == (70,)
    assert cfg.R_diag.shape == (35 + 24 + 29,)

def test_walk_grid_and_gait():
    from t1_nmpc.robot.config import make_config
    c = make_config()
    assert c.nodes == 31 and c.dt_min == 0.035 and c.dt_max == 0.035
    assert abs(c.nodes * c.dt_min - 1.085) < 1e-9
    assert c.n_feet == 2
    assert c.gait_cycle == 1.4
    assert c.switching_times == (0.0, 0.6, 0.7, 1.3, 1.4)
    assert (c.swing_height, c.v_liftoff, c.v_touchdown) == (0.08, 0.05, -0.05)
    assert hasattr(c, "footstep_k") and hasattr(c, "footstep_weight")
    # M0 dims unchanged
    assert (c.nq, c.nv, c.n_corners, c.nf, c.na, c.tau_nodes) == (36, 35, 8, 24, 35, 3)

def test_vendored_urdf_present():
    assert os.path.isfile(T1_URDF_PATH)
    mesh_dir = os.path.join(T1_PACKAGE_DIRS[0], "t1_description", "meshes")
    assert len([f for f in os.listdir(mesh_dir) if f.endswith(".STL")]) == 30
    assert len(JOINT_NAMES) == 29
