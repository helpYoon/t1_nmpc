import numpy as np


def test_numpy_is_2x():
    assert np.__version__.startswith("2."), np.__version__


def test_pinocchio_imports_without_segfault():
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    data = model.createData()
    q = pin.neutral(model)
    pin.centerOfMass(model, data, q)
    assert data.com[0].shape == (3,)


def test_mujoco_imports():
    import mujoco

    assert hasattr(mujoco, "MjModel")
