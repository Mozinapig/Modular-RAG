"""Unit tests for Vision LLM factory and interface."""

import pytest
from pathlib import Path
from src.libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMSettings


class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM for testing."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.temperature = settings.temperature

    def chat_with_image(
        self,
        text: str,
        image_path: str = None,
        image_base64: str = None,
        trace=None,
    ) -> VisionChatResponse:
        """Return a fixed fake response."""
        if not text:
            raise ValueError("Text is required")
        if not image_path and not image_base64:
            raise ValueError("Either image_path or image_base64 is required")

        return VisionChatResponse(
            content=f"Fake vision response for: {text}",
            model=self.settings.model,
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    def validate_config(self) -> None:
        """Validate config."""
        if not self.settings.model:
            raise ValueError("Model name is required")


class TestBaseVisionLLMInterface:
    """Test BaseVisionLLM abstract interface."""

    def test_vision_llm_requires_text(self):
        """Test that chat_with_image requires text."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        with pytest.raises(ValueError, match="Text is required"):
            llm.chat_with_image(text="", image_path="/path/to/image.png")

    def test_vision_llm_requires_image(self):
        """Test that chat_with_image requires image."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        with pytest.raises(ValueError, match="Either image_path or image_base64 is required"):
            llm.chat_with_image(text="What is in this image?")

    def test_vision_llm_with_image_path(self):
        """Test chat_with_image with image path."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        response = llm.chat_with_image(
            text="What is in this image?",
            image_path="/path/to/image.png",
        )

        assert response.content == "Fake vision response for: What is in this image?"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30

    def test_vision_llm_with_image_base64(self):
        """Test chat_with_image with base64 image."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        response = llm.chat_with_image(
            text="Describe this image",
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        )

        assert response.content == "Fake vision response for: Describe this image"
        assert response.model == "test-model"

    def test_vision_response_serialization(self):
        """Test VisionChatResponse can be converted to dict."""
        response = VisionChatResponse(
            content="Test response",
            model="test-model",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        response_dict = {
            "content": response.content,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }

        assert response_dict["content"] == "Test response"
        assert response_dict["model"] == "test-model"


class TestLLMFactoryVisionSupport:
    """Test LLMFactory vision LLM creation."""

    def test_factory_create_vision_llm_with_fake_provider(self):
        """Test factory can create fake vision LLM."""
        # Register fake vision provider for testing
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        factory = LLMFactory()
        settings = LLMSettings(
            provider="fake",
            model="test-vision-model",
            api_key="test-key",
        )

        vision_llm = factory.create_vision_llm(settings)

        assert isinstance(vision_llm, BaseVisionLLM)
        assert vision_llm.settings.model == "test-vision-model"

    def test_factory_create_vision_llm_validates_config(self):
        """Test factory validates vision LLM config."""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        factory = LLMFactory()
        settings = LLMSettings(
            provider="fake",
            model="",  # Invalid: empty model
            api_key="test-key",
        )

        with pytest.raises(ValueError, match="Model name is required"):
            factory.create_vision_llm(settings)

    def test_factory_create_vision_llm_unknown_provider(self):
        """Test factory raises error for unknown vision provider."""
        factory = LLMFactory()
        settings = LLMSettings(
            provider="unknown_vision_provider",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
            factory.create_vision_llm(settings)

    def test_factory_vision_provider_registration(self):
        """Test registering custom vision provider."""
        class CustomVisionLLM(BaseVisionLLM):
            def __init__(self, settings):
                self.settings = settings

            def chat_with_image(self, text, image_path=None, image_base64=None, trace=None):
                return VisionChatResponse(
                    content="Custom response",
                    model=self.settings.model,
                )

            def validate_config(self):
                pass

        LLMFactory.register_vision_provider("custom", CustomVisionLLM)

        factory = LLMFactory()
        settings = LLMSettings(
            provider="custom",
            model="custom-model",
            api_key="test-key",
        )

        vision_llm = factory.create_vision_llm(settings)
        assert isinstance(vision_llm, CustomVisionLLM)

    def test_factory_vision_llm_requires_provider(self):
        """Test factory requires provider in settings."""
        factory = LLMFactory()
        settings = LLMSettings(
            provider="",  # Invalid: empty provider
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValueError, match="Vision LLM provider is required"):
            factory.create_vision_llm(settings)

    def test_factory_vision_llm_requires_model(self):
        """Test factory requires model in settings."""
        factory = LLMFactory()
        settings = LLMSettings(
            provider="fake",
            model="",  # Invalid: empty model
            api_key="test-key",
        )

        with pytest.raises(ValueError, match="Model name is required"):
            factory.create_vision_llm(settings)

    def test_factory_vision_llm_requires_api_key(self):
        """Test factory requires api_key in settings."""
        factory = LLMFactory()
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="",  # Invalid: empty api_key
        )

        with pytest.raises(ValueError, match="Vision LLM api_key is required"):
            factory.create_vision_llm(settings)

    def test_vision_llm_with_both_image_inputs(self):
        """Test that image_path takes precedence when both are provided."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        # Both image_path and image_base64 provided - should succeed
        response = llm.chat_with_image(
            text="What is in this image?",
            image_path="/path/to/image.png",
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        )

        assert response.content == "Fake vision response for: What is in this image?"

    def test_vision_llm_trace_parameter(self):
        """Test that trace parameter is accepted."""
        settings = LLMSettings(
            provider="fake",
            model="test-model",
            api_key="test-key",
        )
        llm = FakeVisionLLM(settings)

        # Mock trace object
        class MockTrace:
            def record_stage(self, stage_name, details):
                pass

        trace = MockTrace()

        response = llm.chat_with_image(
            text="What is in this image?",
            image_path="/path/to/image.png",
            trace=trace,
        )

        assert response.content == "Fake vision response for: What is in this image?"
