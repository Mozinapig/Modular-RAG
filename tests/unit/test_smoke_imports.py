"""
Smoke tests for basic package imports.
"""

import pytest


@pytest.mark.unit
@pytest.mark.smoke
def test_import_mcp_server():
    """Test importing mcp_server package."""
    import src.mcp_server


@pytest.mark.unit
@pytest.mark.smoke
def test_import_core():
    """Test importing core package."""
    import src.core


@pytest.mark.unit
@pytest.mark.smoke
def test_import_ingestion():
    """Test importing ingestion package."""
    import src.ingestion


@pytest.mark.unit
@pytest.mark.smoke
def test_import_libs():
    """Test importing libs package."""
    import src.libs


@pytest.mark.unit
@pytest.mark.smoke
def test_import_observability():
    """Test importing observability package."""
    import src.observability


@pytest.mark.unit
@pytest.mark.smoke
def test_import_settings():
    """Test importing settings module."""
    from src.core import settings
    assert hasattr(settings, 'Settings')
    assert hasattr(settings, 'load_settings')
    assert hasattr(settings, 'validate_settings')


@pytest.mark.unit
@pytest.mark.smoke
def test_settings_dataclass():
    """Test Settings dataclass can be instantiated."""
    from src.core.settings import (
        Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings, VisionLLMSettings
    )

    llm = LLMSettings(provider="openai", model="gpt-4", api_key="test-key", base_url="https://api.openai.com/v1")
    embedding = EmbeddingSettings(provider="openai", model="text-embedding-3", api_key="test-key", base_url="https://api.openai.com/v1")
    vs = VectorStoreSettings(backend="chroma", persist_path="./data/db/chroma")
    vision = VisionLLMSettings(provider="openai", model="gpt-4-vision", api_key="test-key", base_url="https://api.openai.com/v1")

    settings = Settings(llm=llm, embedding=embedding, vector_store=vs, vision_llm=vision)
    assert settings.llm.provider == "openai"
    assert settings.llm.base_url == "https://api.openai.com/v1"
    assert settings.embedding.model == "text-embedding-3"
    assert settings.embedding.base_url == "https://api.openai.com/v1"
    assert settings.vision_llm.provider == "openai"


@pytest.mark.unit
@pytest.mark.smoke
def test_load_settings():
    """Test loading settings from YAML file."""
    from src.core.settings import load_settings, validate_settings

    # Load from actual config file
    settings = load_settings("config/settings.yaml")

    # Validate
    validate_settings(settings)

    # Check OpenAI configuration
    assert settings.llm.provider == "openai"
    assert settings.llm.base_url == "https://api.openai.com/v1"
    assert settings.embedding.provider == "openai"
    assert settings.embedding.base_url == "https://api.openai.com/v1"
    assert settings.vision_llm is not None
    assert settings.vision_llm.provider == "openai"
    assert settings.vision_llm.base_url == "https://api.openai.com/v1"

