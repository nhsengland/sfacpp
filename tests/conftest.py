import pytest
import numpy as np
from pysfacpp.monte_carlo import DGPConfig, ClassSpec, InefficiencyDist, generate_data


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def single_class_config():
    """DGP config with a single latent class (equivalent to TRE)."""
    return DGPConfig(
        n_firms=50,
        n_periods=5,
        classes=[
            ClassSpec(
                beta=np.array([1.0, 0.5, 0.3]),
                ln_sigma_u=np.log(0.3),
                ln_sigma_v=np.log(0.2),
                ln_sigma_0=np.log(0.15),
            ),
        ],
        prod_cost=1,
        dist=InefficiencyDist.HALF_NORMAL,
        seed=42,
    )


@pytest.fixture(scope="session")
def two_class_config():
    """DGP config with two well-separated latent classes."""
    return DGPConfig(
        n_firms=80,
        n_periods=5,
        classes=[
            ClassSpec(
                beta=np.array([1.0, 0.5, 0.3]),
                ln_sigma_u=np.log(0.2),
                ln_sigma_v=np.log(0.15),
                ln_sigma_0=np.log(0.1),
            ),
            ClassSpec(
                beta=np.array([2.0, 0.8, 0.5]),
                ln_sigma_u=np.log(0.5),
                ln_sigma_v=np.log(0.2),
                ln_sigma_0=np.log(0.15),
            ),
        ],
        delta=np.array([0.5]),
        prod_cost=1,
        dist=InefficiencyDist.HALF_NORMAL,
        seed=123,
    )


@pytest.fixture(scope="session")
def single_class_data(single_class_config):
    """Generated data from single-class DGP."""
    return generate_data(single_class_config)


@pytest.fixture(scope="session")
def two_class_data(two_class_config):
    """Generated data from two-class DGP."""
    return generate_data(two_class_config)
