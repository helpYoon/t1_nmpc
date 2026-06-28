import pytest
import warnings

# Suppress pinocchio DeprecatedBool binding registration warning (upstream pinocchio/aligator)
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Add filter before any test collection/execution
    warnings.filterwarnings("ignore", message=".*already registered.*", category=RuntimeWarning)
    config.addinivalue_line("markers", "slow: long-running closed-loop tests")
