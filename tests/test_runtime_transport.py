from t1_nmpc.runtime.transport import Transport

def test_transport_protocol_surface():
    assert hasattr(Transport, "read_state")
    assert hasattr(Transport, "write_command")
    assert hasattr(Transport, "now")
