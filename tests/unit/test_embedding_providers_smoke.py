"""OpenAI and Azure Embedding provider tests."""

import pytest
from unittest.mock import patch, MagicMock
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.azure_embedding import AzureEmbedding
from src.core.settings import EmbeddingSettings


class TestOpenAIEmbeddingInitialization:
    """Test OpenAIEmbedding initialization."""

    def test_init_with_valid_settings(self):
        """Test initialization with valid settings."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536
        )
        embedding = OpenAIEmbedding(settings)
        assert embedding.settings == settings

    def test_init_with_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-large",
            api_key="sk-test",
            dimensions=3072
        )
        embedding = OpenAIEmbedding(settings)
        assert embedding.settings.dimensions == 3072


class TestOpenAIEmbeddingValidation:
    """Test OpenAIEmbedding configuration validation."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test"
        )
        embedding = OpenAIEmbedding(settings)
        embedding.validate_config()  # Should not raise

    def test_validate_config_missing_api_key(self):
        """Test validation fails with missing api_key."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key=""
        )
        embedding = OpenAIEmbedding(settings)
        with pytest.raises(ValueError, match="api_key is required"):
            embedding.validate_config()

    def test_validate_config_missing_model(self):
        """Test validation fails with missing model."""
        settings = EmbeddingSettings(
            provider="openai",
            model="",
            api_key="sk-test"
        )
        embedding = OpenAIEmbedding(settings)
        with pytest.raises(ValueError, match="model is required"):
            embedding.validate_config()


class TestOpenAIEmbeddingEmbed:
    """Test OpenAIEmbedding embed functionality."""

    def test_embed_empty_texts(self):
        """Test embed with empty texts list."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test"
        )
        embedding = OpenAIEmbedding(settings)
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding.embed([])

    @patch('src.libs.embedding.openai_embedding.OpenAI')
    def test_embed_single_text(self, mock_openai_class):
        """Test embedding a single text."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response

        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536
        )
        embedding = OpenAIEmbedding(settings)

        result = embedding.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_client.embeddings.create.assert_called_once()

    @patch('src.libs.embedding.openai_embedding.OpenAI')
    def test_embed_multiple_texts(self, mock_openai_class):
        """Test embedding multiple texts."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create mock embeddings for 3 texts
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
            dimensions=1536
        )
        embedding = OpenAIEmbedding(settings)

        texts = ["Hello", "World", "Test"]
        result = embedding.embed(texts)

        assert len(result) == 3
        assert all(len(vec) == 1536 for vec in result)

    @patch('src.libs.embedding.openai_embedding.OpenAI')
    def test_embed_api_error(self, mock_openai_class):
        """Test embed with API error."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API error")

        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test"
        )
        embedding = OpenAIEmbedding(settings)

        with pytest.raises(RuntimeError, match="OpenAI Embedding error.*openai"):
            embedding.embed(["Hello"])


class TestAzureEmbeddingInitialization:
    """Test AzureEmbedding initialization."""

    def test_init_with_valid_settings(self):
        """Test initialization with valid settings."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            dimensions=1536
        )
        embedding = AzureEmbedding(settings)
        assert embedding.settings == settings

    def test_init_with_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            dimensions=1536
        )
        embedding = AzureEmbedding(settings)
        assert embedding.settings.dimensions == 1536


class TestAzureEmbeddingValidation:
    """Test AzureEmbedding configuration validation."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        embedding = AzureEmbedding(settings)
        embedding.validate_config()  # Should not raise

    def test_validate_config_missing_api_key(self):
        """Test validation fails with missing api_key."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="",
            azure_endpoint="https://test.openai.azure.com/"
        )
        embedding = AzureEmbedding(settings)
        with pytest.raises(ValueError, match="api_key is required"):
            embedding.validate_config()

    def test_validate_config_missing_model(self):
        """Test validation fails with missing model."""
        settings = EmbeddingSettings(
            provider="azure",
            model="",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        embedding = AzureEmbedding(settings)
        with pytest.raises(ValueError, match="model is required"):
            embedding.validate_config()

    def test_validate_config_missing_endpoint(self):
        """Test validation fails with missing azure_endpoint."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint=""
        )
        embedding = AzureEmbedding(settings)
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            embedding.validate_config()


class TestAzureEmbeddingEmbed:
    """Test AzureEmbedding embed functionality."""

    def test_embed_empty_texts(self):
        """Test embed with empty texts list."""
        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        embedding = AzureEmbedding(settings)
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding.embed([])

    @patch('src.libs.embedding.azure_embedding.AzureOpenAI')
    def test_embed_single_text(self, mock_azure_class):
        """Test embedding a single text."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response

        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            dimensions=1536
        )
        embedding = AzureEmbedding(settings)

        result = embedding.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 1536
        mock_client.embeddings.create.assert_called_once()

    @patch('src.libs.embedding.azure_embedding.AzureOpenAI')
    def test_embed_multiple_texts(self, mock_azure_class):
        """Test embedding multiple texts."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/",
            dimensions=1536
        )
        embedding = AzureEmbedding(settings)

        texts = ["Hello", "World"]
        result = embedding.embed(texts)

        assert len(result) == 2
        assert all(len(vec) == 1536 for vec in result)

    @patch('src.libs.embedding.azure_embedding.AzureOpenAI')
    def test_embed_api_error(self, mock_azure_class):
        """Test embed with API error."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API error")

        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )
        embedding = AzureEmbedding(settings)

        with pytest.raises(RuntimeError, match="Azure Embedding error.*azure"):
            embedding.embed(["Hello"])


class TestEmbeddingFactory:
    """Test Embedding factory integration."""

    def test_factory_creates_openai_embedding(self):
        """Test that factory can create OpenAIEmbedding."""
        from src.libs.embedding.embedding_factory import EmbeddingFactory

        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test"
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.settings == settings

    def test_factory_creates_azure_embedding(self):
        """Test that factory can create AzureEmbedding."""
        from src.libs.embedding.embedding_factory import EmbeddingFactory

        settings = EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com/"
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        assert isinstance(embedding, AzureEmbedding)
        assert embedding.settings == settings

    def test_factory_unknown_provider(self):
        """Test that factory raises error for unknown provider."""
        from src.libs.embedding.embedding_factory import EmbeddingFactory

        settings = EmbeddingSettings(
            provider="unknown_provider",
            model="model",
            api_key="key"
        )

        factory = EmbeddingFactory()
        with pytest.raises(ValueError, match="Unknown Embedding provider"):
            factory.create(settings)
