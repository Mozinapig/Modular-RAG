"""
Pytest configuration and fixtures for all tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def tests_dir():
    """Return the tests directory."""
    return Path(__file__).parent


@pytest.fixture
def fixtures_dir(tests_dir):
    """Return the fixtures directory."""
    fixtures = tests_dir / "fixtures"
    fixtures.mkdir(exist_ok=True, parents=True)
    return fixtures


@pytest.fixture
def sample_documents_dir(fixtures_dir):
    """Return the sample documents directory."""
    sample_docs = fixtures_dir / "sample_documents"
    sample_docs.mkdir(exist_ok=True, parents=True)
    return sample_docs


@pytest.fixture
def config_dir(project_root):
    """Return the config directory."""
    return project_root / "config"


@pytest.fixture
def settings_yaml_path(config_dir):
    """Return path to settings.yaml."""
    return config_dir / "settings.yaml"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir
