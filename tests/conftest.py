"""Global pytest configuration and fixtures for TrafficMetry tests.

This module provides shared pytest fixtures used across all test modules,
including configuration reset for test isolation.
"""

from typing import Generator

import pytest


@pytest.fixture(autouse=True, scope="function")
def reset_config_fixture() -> Generator[None, None, None]:
    """Automatically reset configuration before and after each test.

    This fixture ensures test isolation by clearing the global configuration
    singleton before each test runs, and cleaning up after test completion.

    Yields:
        Generator: Control to the test function
    """
    from backend.config import reset_config

    # Reset config before test
    reset_config()

    # Yield control to the test
    yield

    # Reset config after test (cleanup)
    reset_config()
