"""Ollama LLM provider tests."""

import pytest
from unittest.mock import patch, MagicMock
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.base_llm import ChatMessage
from src.core.settings import LLMSettings


class TestOllamaLLMInitialization:
    """Test OllamaLLM initialization."""

    def test_init_with_valid_settings(self):
        """Test initialization with valid settings."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)
        assert llm.settings == settings

    def test_init_without_base_url(self):
        """Test initialization without base_url uses default."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key"
        )
        llm = OllamaLLM(settings)
        assert llm.settings == settings


class TestOllamaLLMValidation:
    """Test OllamaLLM configuration validation."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)
        llm.validate_config()  # Should not raise

    def test_validate_config_missing_api_key(self):
        """Test validation fails with missing api_key."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)
        with pytest.raises(ValueError, match="api_key is required"):
            llm.validate_config()

    def test_validate_config_missing_model(self):
        """Test validation fails with missing model."""
        settings = LLMSettings(
            provider="ollama",
            model="",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)
        with pytest.raises(ValueError, match="model is required"):
            llm.validate_config()

    def test_validate_config_missing_base_url(self):
        """Test validation succeeds with missing base_url (uses default)."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url=""
        )
        llm = OllamaLLM(settings)
        llm.validate_config()  # Should not raise, uses default


class TestOllamaLLMChat:
    """Test OllamaLLM chat functionality."""

    def test_chat_empty_messages(self):
        """Test chat with empty messages list."""
        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)
        with pytest.raises(ValueError, match="messages list cannot be empty"):
            llm.chat([])

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_success(self, mock_post):
        """Test successful chat call."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        llm = OllamaLLM(settings)

        messages = [ChatMessage(role="user", content="Hello")]
        response = llm.chat(messages)

        assert response.content == "Hello! How can I help you?"
        assert response.model == "llama2"
        mock_post.assert_called_once()

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_with_temperature_override(self, mock_post):
        """Test chat with temperature override."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        llm = OllamaLLM(settings)

        messages = [ChatMessage(role="user", content="Hello")]
        response = llm.chat(messages, temperature=0.3)

        # Verify the temperature was passed in the request
        call_args = mock_post.call_args
        assert call_args[1]["json"]["temperature"] == 0.3

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_connection_error(self, mock_post):
        """Test chat with connection error."""
        mock_post.side_effect = Exception("Connection refused")

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)

        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RuntimeError, match="Ollama API error.*ollama"):
            llm.chat(messages)

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_timeout_error(self, mock_post):
        """Test chat with timeout error."""
        import requests
        mock_post.side_effect = requests.Timeout("Request timeout")

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)

        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RuntimeError, match="Ollama API error.*ollama"):
            llm.chat(messages)

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_with_max_tokens(self, mock_post):
        """Test chat with max_tokens parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)

        messages = [ChatMessage(role="user", content="Hello")]
        response = llm.chat(messages, max_tokens=100)

        # Verify max_tokens was passed
        call_args = mock_post.call_args
        assert call_args[1]["json"]["num_predict"] == 100

    @patch('src.libs.llm.ollama_llm.requests.post')
    def test_chat_multiple_messages(self, mock_post):
        """Test chat with multiple messages."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )
        llm = OllamaLLM(settings)

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?")
        ]
        response = llm.chat(messages)

        # Verify all messages were sent
        call_args = mock_post.call_args
        sent_messages = call_args[1]["json"]["messages"]
        assert len(sent_messages) == 3


class TestOllamaLLMFactory:
    """Test OllamaLLM integration with factory."""

    def test_factory_creates_ollama_llm(self):
        """Test that factory can create OllamaLLM."""
        from src.libs.llm.llm_factory import LLMFactory

        settings = LLMSettings(
            provider="ollama",
            model="llama2",
            api_key="dummy_key",
            base_url="http://localhost:11434"
        )

        factory = LLMFactory()
        llm = factory.create(settings)

        assert isinstance(llm, OllamaLLM)
        assert llm.settings == settings

    def test_factory_unknown_provider(self):
        """Test that factory raises error for unknown provider."""
        from src.libs.llm.llm_factory import LLMFactory

        settings = LLMSettings(
            provider="unknown_provider",
            model="model",
            api_key="key"
        )

        factory = LLMFactory()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            factory.create(settings)
