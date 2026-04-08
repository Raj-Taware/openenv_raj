# conftest.py — pytest root configuration
import warnings

def pytest_configure(config):
    """Suppress known Qiskit PendingDeprecationWarnings from internal QFT class."""
    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        module=r"qiskit.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"qiskit.*",
    )
