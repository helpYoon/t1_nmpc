import importlib

def test_sdk_module_imports_without_sdk():
    m = importlib.import_module("t1_nmpc.runtime.sdk_transport")
    assert hasattr(m, "SdkTransport")
    assert hasattr(m, "SDK_JOINT_ORDER")

def test_sdk_construct_without_sdk_raises_clear_error():
    from t1_nmpc.runtime.sdk_transport import SdkTransport
    from t1_nmpc.wb.config_wb import make_wb_config
    import pytest
    with pytest.raises(RuntimeError, match="booster SDK"):
        SdkTransport(make_wb_config())
