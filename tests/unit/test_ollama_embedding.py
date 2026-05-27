"""Ollama Embedding provider tests."""

import pytest
from unittest.mock import patch, MagicMock
import requests
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.core.settings import EmbeddingSettings


class TestOllamaEmbeddingInitialization:
    """Test OllamaEmbedding initialization."""

    def test_init_with_valid_settings(self):
        """Test initialization with valid settings."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        assert embedding.settings == settings

    def test_init_with_default_base_url(self):
        """Test initialization with default base_url."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        assert embedding._get_base_url() == "http://localhost:11434"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base_url."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            base_url="http://ollama.example.com:11434",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        assert embedding._get_base_url() == "http://ollama.example.com:11434"


class TestOllamaEmbeddingValidation:
    """Test OllamaEmbedding configuration validation."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        embedding.validate_config()  # Should not raise

    def test_validate_config_missing_api_key(self):
        """Test validation fails with missing api_key."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key=""
        )
        embedding = OllamaEmbedding(settings)
        with pytest.raises(ValueError, match="api_key is required"):
            embedding.validate_config()

    def test_validate_config_missing_model(self):
        """Test validation fails with missing model."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        with pytest.raises(ValueError, match="model is required"):
            embedding.validate_config()


class TestOllamaEmbeddingEmbed:
    """Test OllamaEmbedding embed functionality."""

    def test_embed_empty_texts(self):
        """Test embed with empty texts list."""
        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            embedding.embed([])

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_single_text(self, mock_post):
        """Test embedding a single text."""
        # Mock the Ollama API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 768]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama",
            base_url="http://localhost:11434"
        )
        embedding = OllamaEmbedding(settings)

        result = embedding.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768
        mock_post.assert_called_once()

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_multiple_texts(self, mock_post):
        """Test embedding multiple texts."""
        # Mock the Ollama API response for 3 texts
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1] * 768,
                [0.2] * 768,
                [0.3] * 768,
            ]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        texts = ["Hello", "World", "Test"]
        result = embedding.embed(texts)

        assert len(result) == 3
        assert all(len(vec) == 768 for vec in result)

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_request_structure(self, mock_post):
        """Test the structure of the POST request to Ollama."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 768, [0.2] * 768]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="mxbai-embed-large",
            api_key="ollama",
            base_url="http://custom-ollama:11434"
        )
        embedding = OllamaEmbedding(settings)

        texts = ["text1", "text2"]
        embedding.embed(texts)

        # Verify the request was made to the correct endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "http://custom-ollama:11434/api/embed" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "mxbai-embed-large"
        assert call_args[1]["json"]["input"] == texts

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_connection_error(self, mock_post):
        """Test embed with connection error."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        with pytest.raises(RuntimeError, match="Ollama API error"):
            embedding.embed(["test"])

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_timeout_error(self, mock_post):
        """Test embed with timeout error."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        with pytest.raises(RuntimeError, match="Ollama API error"):
            embedding.embed(["test"])

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_http_error(self, mock_post):
        """Test embed with HTTP error response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        with pytest.raises(RuntimeError, match="Ollama API error"):
            embedding.embed(["test"])

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_malformed_response(self, mock_post):
        """Test embed with malformed API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}  # Wrong format
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        with pytest.raises(RuntimeError, match="Ollama API error"):
            embedding.embed(["test"])

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_with_trace_context(self, mock_post):
        """Test embed with trace context parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 768]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        # Mock trace context
        mock_trace = MagicMock()

        result = embedding.embed(["test"], trace=mock_trace)

        assert len(result) == 1
        assert len(result[0]) == 768

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_different_dimensions(self, mock_post):
        """Test embed models with different output dimensions."""
        # Different models return different dimensions
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 1024]  # 1024 dimensions
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="mxbai-embed-large",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        result = embedding.embed(["test"])

        assert len(result) == 1
        assert len(result[0]) == 1024

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_large_batch(self, mock_post):
        """Test embedding a large batch of texts."""
        num_texts = 100
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 768 for _ in range(num_texts)]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        texts = [f"text_{i}" for i in range(num_texts)]
        result = embedding.embed(texts)

        assert len(result) == num_texts
        assert all(len(vec) == 768 for vec in result)

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_uses_correct_timeout(self, mock_post):
        """Test that embed uses correct timeout setting."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 768]
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        embedding.embed(["test"])

        # Verify timeout is set in the request
        call_kwargs = mock_post.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 30  # Default timeout

    @patch('src.libs.embedding.ollama_embedding.requests.post')
    def test_embed_response_order_preservation(self, mock_post):
        """Test that embed preserves the order of responses."""
        mock_response = MagicMock()
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_response.json.return_value = {
            "embeddings": embeddings
        }
        mock_post.return_value = mock_response

        settings = EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            api_key="ollama"
        )
        embedding = OllamaEmbedding(settings)

        texts = ["first", "second", "third"]
        result = embedding.embed(texts)

        # Verify order is preserved
        assert result == embeddings
