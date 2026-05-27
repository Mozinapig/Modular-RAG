"""Tests for LLMFactory vision LLM creation with OpenAI provider."""

import pytest
from unittest.mock import patch, MagicMock
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
from src.core.settings import VisionLLMSettings


@pytest.fixture
def factory():
    """Create LLMFactory instance."""
    return LLMFactory()


@pytest.fixture
def vision_settings():
    """Create OpenAI Vision settings."""
    return VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
    )


def test_factory_create_vision_llm_openai(factory, vision_settings):
    """Test factory creates OpenAI Vision LLM."""
    vision_llm = factory.create_vision_llm(vision_settings)

    assert isinstance(vision_llm, OpenAIVisionLLM)
    assert vision_llm.settings.provider == "openai"
    assert vision_llm.settings.model == "gpt-4-vision-preview"


def test_factory_create_vision_llm_fake(factory):
    """Test factory creates Fake Vision LLM for testing."""
    fake_settings = VisionLLMSettings(
        provider="fake",
        model="fake-vision",
        api_key="fake-key",
    )
    vision_llm = factory.create_vision_llm(fake_settings)

    assert vision_llm is not None
    assert vision_llm.settings.provider == "fake"


def test_factory_create_vision_llm_case_insensitive(factory, vision_settings):
    """Test factory handles provider names case-insensitively."""
    settings = VisionLLMSettings(
        provider="OpenAI",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
    )
    vision_llm = factory.create_vision_llm(settings)

    assert isinstance(vision_llm, OpenAIVisionLLM)


def test_factory_create_vision_llm_missing_provider():
    """Test factory raises error for missing provider."""
    factory = LLMFactory()
    settings = VisionLLMSettings(
        provider="",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
    )

    with pytest.raises(ValueError, match="provider is required"):
        factory.create_vision_llm(settings)


def test_factory_create_vision_llm_missing_model():
    """Test factory raises error for missing model."""
    factory = LLMFactory()
    settings = VisionLLMSettings(
        provider="openai",
        model="",
        api_key="test-api-key",
    )

    with pytest.raises(ValueError, match="Model name is required"):
        factory.create_vision_llm(settings)


def test_factory_create_vision_llm_missing_api_key():
    """Test factory raises error for missing api_key."""
    factory = LLMFactory()
    settings = VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="",
    )

    with pytest.raises(ValueError, match="api_key is required"):
        factory.create_vision_llm(settings)


def test_factory_create_vision_llm_unknown_provider(factory):
    """Test factory raises error for unknown provider."""
    settings = VisionLLMSettings(
        provider="unknown",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
    )

    with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
        factory.create_vision_llm(settings)


def test_factory_register_custom_vision_provider():
    """Test registering a custom vision LLM provider."""
    from src.libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse

    class CustomVisionLLM(BaseVisionLLM):
        def __init__(self, settings):
            self.settings = settings

        def chat_with_image(self, text, image_path=None, image_base64=None, trace=None):
            return VisionChatResponse(
                content="Custom response",
                model=self.settings.model,
                finish_reason="stop",
            )

        def validate_config(self):
            pass

    LLMFactory.register_vision_provider("custom", CustomVisionLLM)

    factory = LLMFactory()
    settings = VisionLLMSettings(
        provider="custom",
        model="custom-vision",
        api_key="test-key",
    )
    vision_llm = factory.create_vision_llm(settings)

    assert isinstance(vision_llm, CustomVisionLLM)


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_factory_openai_vision_llm_validates_config(mock_openai, factory, vision_settings):
    """Test factory validates config when creating OpenAI Vision LLM."""
    # This should succeed
    vision_llm = factory.create_vision_llm(vision_settings)
    assert vision_llm is not None

    # Invalid config should fail
    invalid_settings = VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="",
    )
    with pytest.raises(ValueError):
        factory.create_vision_llm(invalid_settings)
