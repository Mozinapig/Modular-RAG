"""Smoke tests for OpenAI-compatible LLM providers (OpenAI/Azure/DeepSeek)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.libs.llm.base_llm import ChatMessage, ChatResponse
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.llm_factory import LLMFactory
from src.core.settings import LLMSettings


class TestOpenAILLM:
    """Test OpenAI LLM provider."""

    def test_init_with_valid_settings(self):
        """Test OpenAI LLM initialization with valid settings."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
            temperature=0.7,
            max_tokens=1000,
        )
        llm = OpenAILLM(settings)
        assert llm.settings == settings
        assert llm.settings.model == "gpt-4"

    def test_validate_config_missing_api_key(self):
        """Test validation fails when api_key is missing."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="",
            temperature=0.7,
        )
        llm = OpenAILLM(settings)
        with pytest.raises(ValueError, match="api_key"):
            llm.validate_config()

    def test_validate_config_missing_model(self):
        """Test validation fails when model is missing."""
        settings = LLMSettings(
            provider="openai",
            model="",
            api_key="sk-test-key",
            temperature=0.7,
        )
        llm = OpenAILLM(settings)
        with pytest.raises(ValueError, match="model"):
            llm.validate_config()

    @patch("src.libs.llm.openai_llm.OpenAI")
    def test_chat_success(self, mock_openai_class):
        """Test successful chat call with mocked OpenAI client."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_client.chat.completions.create.return_value = mock_response

        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
            temperature=0.7,
        )
        llm = OpenAILLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        response = llm.chat(messages, temperature=0.5)

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15

    @patch("src.libs.llm.openai_llm.OpenAI")
    def test_chat_with_invalid_messages(self, mock_openai_class):
        """Test chat raises ValueError for invalid messages."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
        )
        llm = OpenAILLM(settings)

        with pytest.raises(ValueError, match="messages"):
            llm.chat([])  # Empty messages list

    @patch("src.libs.llm.openai_llm.OpenAI")
    def test_chat_api_error(self, mock_openai_class):
        """Test chat handles API errors gracefully."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
        )
        llm = OpenAILLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="openai"):
            llm.chat(messages)


class TestAzureLLM:
    """Test Azure OpenAI LLM provider."""

    def test_init_with_valid_settings(self):
        """Test Azure LLM initialization with valid settings."""
        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
            api_version="2024-02-15-preview",
            temperature=0.7,
        )
        llm = AzureLLM(settings)
        assert llm.settings == settings
        assert llm.settings.azure_endpoint == "https://test.openai.azure.com/"

    def test_validate_config_missing_azure_endpoint(self):
        """Test validation fails when azure_endpoint is missing."""
        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="",
            azure_deployment="test-deployment",
        )
        llm = AzureLLM(settings)
        with pytest.raises(ValueError, match="azure_endpoint"):
            llm.validate_config()

    def test_validate_config_missing_azure_deployment(self):
        """Test validation fails when azure_deployment is missing."""
        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="",
        )
        llm = AzureLLM(settings)
        with pytest.raises(ValueError, match="azure_deployment"):
            llm.validate_config()

    @patch("src.libs.llm.azure_llm.AzureOpenAI")
    def test_chat_success(self, mock_azure_class):
        """Test successful chat call with mocked Azure client."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Azure response"))]
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=20, completion_tokens=10, total_tokens=30
        )
        mock_client.chat.completions.create.return_value = mock_response

        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
        )
        llm = AzureLLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        response = llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content == "Azure response"
        assert response.usage["total_tokens"] == 30

    @patch("src.libs.llm.azure_llm.AzureOpenAI")
    def test_chat_api_error(self, mock_azure_class):
        """Test chat handles API errors gracefully."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Azure API error")

        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
        )
        llm = AzureLLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="azure"):
            llm.chat(messages)


class TestDeepSeekLLM:
    """Test DeepSeek LLM provider."""

    def test_init_with_valid_settings(self):
        """Test DeepSeek LLM initialization with valid settings."""
        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-deepseek-key",
            temperature=0.7,
        )
        llm = DeepSeekLLM(settings)
        assert llm.settings == settings
        assert llm.settings.model == "deepseek-chat"

    def test_validate_config_missing_api_key(self):
        """Test validation fails when api_key is missing."""
        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="",
        )
        llm = DeepSeekLLM(settings)
        with pytest.raises(ValueError, match="api_key"):
            llm.validate_config()

    @patch("src.libs.llm.deepseek_llm.OpenAI")
    def test_chat_success(self, mock_openai_class):
        """Test successful chat call with mocked DeepSeek client."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="DeepSeek response"))]
        mock_response.model = "deepseek-chat"
        mock_response.usage = MagicMock(
            prompt_tokens=15, completion_tokens=8, total_tokens=23
        )
        mock_client.chat.completions.create.return_value = mock_response

        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-deepseek-key",
        )
        llm = DeepSeekLLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        response = llm.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content == "DeepSeek response"
        assert response.model == "deepseek-chat"

    @patch("src.libs.llm.deepseek_llm.OpenAI")
    def test_chat_api_error(self, mock_openai_class):
        """Test chat handles API errors gracefully."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("DeepSeek API error")

        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-deepseek-key",
        )
        llm = DeepSeekLLM(settings)
        messages = [ChatMessage(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="deepseek"):
            llm.chat(messages)


class TestLLMFactoryRouting:
    """Test LLMFactory provider routing."""

    def test_factory_creates_openai_llm(self):
        """Test factory creates OpenAI LLM instance."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key",
        )
        factory = LLMFactory()
        llm = factory.create(settings)
        assert isinstance(llm, OpenAILLM)

    def test_factory_creates_azure_llm(self):
        """Test factory creates Azure LLM instance."""
        settings = LLMSettings(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            azure_deployment="test-deployment",
        )
        factory = LLMFactory()
        llm = factory.create(settings)
        assert isinstance(llm, AzureLLM)

    def test_factory_creates_deepseek_llm(self):
        """Test factory creates DeepSeek LLM instance."""
        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-deepseek-key",
        )
        factory = LLMFactory()
        llm = factory.create(settings)
        assert isinstance(llm, DeepSeekLLM)

    def test_factory_provider_case_insensitive(self):
        """Test factory provider name is case insensitive."""
        settings = LLMSettings(
            provider="OpenAI",
            model="gpt-4",
            api_key="sk-test-key",
        )
        factory = LLMFactory()
        llm = factory.create(settings)
        assert isinstance(llm, OpenAILLM)

    def test_factory_unknown_provider_error(self):
        """Test factory raises error for unknown provider."""
        settings = LLMSettings(
            provider="unknown_provider",
            model="gpt-4",
            api_key="sk-test-key",
        )
        factory = LLMFactory()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            factory.create(settings)

    def test_factory_missing_provider_error(self):
        """Test factory raises error when provider is missing."""
        settings = LLMSettings(
            provider="",
            model="gpt-4",
            api_key="sk-test-key",
        )
        factory = LLMFactory()
        with pytest.raises(ValueError, match="provider is required"):
            factory.create(settings)
