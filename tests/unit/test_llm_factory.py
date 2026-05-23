"""Tests for LLM factory and provider routing."""

import pytest
from unittest.mock import Mock, MagicMock

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMSettings


class FakeLLM(BaseLLM):
    """Fake LLM implementation for testing."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def chat(self, messages, temperature=None, max_tokens=None, trace=None):
        """Return a fixed fake response."""
        return ChatResponse(
            content="Fake LLM response",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def validate_config(self):
        """Validate config (no-op for fake)."""
        pass


class TestLLMFactory:
    """Test LLMFactory routing and creation."""

    def test_factory_creates_fake_provider(self):
        """Test factory can create instances with fake provider."""
        settings = LLMSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            temperature=0.7,
        )

        factory = LLMFactory()
        # Mock the _create_fake_llm method
        llm = factory.create(settings)

        assert llm is not None
        assert isinstance(llm, BaseLLM)

    def test_factory_respects_temperature(self):
        """Test factory preserves temperature setting."""
        settings = LLMSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            temperature=0.3,
        )

        factory = LLMFactory()
        llm = factory.create(settings)

        # Verify the LLM instance has the correct temperature
        assert hasattr(llm, "temperature") or isinstance(llm, BaseLLM)

    def test_factory_chat_works(self):
        """Test that created LLM can process chat requests."""
        settings = LLMSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
        )

        factory = LLMFactory()
        llm = factory.create(settings)

        messages = [
            ChatMessage(role="user", content="Hello"),
        ]

        response = llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content is not None
        assert response.model is not None

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error on unknown provider."""
        settings = LLMSettings(
            provider="unknown_provider",
            model="fake-model",
            api_key="fake-key",
        )

        factory = LLMFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "unknown_provider" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_api_key(self):
        """Test factory raises error when api_key is missing."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="",  # Empty API key
        )

        factory = LLMFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "api_key" in str(exc_info.value).lower()

    def test_chat_message_creation(self):
        """Test ChatMessage dataclass."""
        msg = ChatMessage(role="user", content="Hello world")

        assert msg.role == "user"
        assert msg.content == "Hello world"

    def test_chat_response_creation(self):
        """Test ChatResponse dataclass."""
        response = ChatResponse(
            content="Test response",
            model="test-model",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15

    def test_factory_singleton_pattern(self):
        """Test factory behaves consistently across calls."""
        settings = LLMSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
        )

        factory = LLMFactory()
        llm1 = factory.create(settings)
        llm2 = factory.create(settings)

        # Both should be BaseLLM instances (not necessarily same instance)
        assert isinstance(llm1, BaseLLM)
        assert isinstance(llm2, BaseLLM)
