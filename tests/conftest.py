# tests/conftest.py
"""
Shared pytest configuration and fixtures for HandBook.ai tests.
"""
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False,
        help="Run integration tests that hit NVIDIA API (slow, requires API key)"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="Use --integration to run API-heavy tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
