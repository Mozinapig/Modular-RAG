"""Integration tests for Cross-Encoder Reranker with API calls."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import Timeout, ConnectionError

from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.core.settings import Settings, RerankerSettings


@pytest.fixture
def api_reranker_settings():
    """Create settings with API-based Cross-Encoder Reranker configuration."""
    rerank_config = RerankerSettings(
        backend="cross_encoder",
        model="bge-reranker-v2-m3",
        api_key="test-api-key",
        base_url="http://localhost:8000",
    )
    return Settings(
        llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
        embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
        vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
        rerank=rerank_config,
    )


class TestCrossEncoderAPIIntegration:
    """Test Cross-Encoder with actual API calls (mocked)."""

    def test_api_call_with_valid_response(self, api_reranker_settings):
        """Test API call returns properly formatted scores."""
        candidates = [
            {"id": "doc1", "text": "First document"},
            {"id": "doc2", "text": "Second document"},
            {"id": "doc3", "text": "Third document"},
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "score": 0.95},
                {"index": 1, "score": 0.72},
                {"index": 2, "score": 0.88},
            ]
        }

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(api_reranker_settings)
            result = reranker.rerank("test query", candidates)

            # Verify API was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "http://localhost:8000/v1/rerank" in call_args[0][0]
            assert call_args[1]["json"]["query"] == "test query"
            assert call_args[1]["json"]["passages"] == [c["text"] for c in candidates]

            # Verify reordering by scores: doc1 (0.95), doc3 (0.88), doc2 (0.72)
            assert len(result) == 3
            assert result[0]["id"] == "doc1"
            assert result[0]["cross_encoder_score"] == 0.95
            assert result[1]["id"] == "doc3"
            assert result[1]["cross_encoder_score"] == 0.88
            assert result[2]["id"] == "doc2"
            assert result[2]["cross_encoder_score"] == 0.72

    def test_api_call_with_authorization_header(self, api_reranker_settings):
        """Test that API key is sent in Authorization header."""
        candidates = [{"id": "doc1", "text": "text"}]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"index": 0, "score": 0.9}]}

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(api_reranker_settings)
            reranker.rerank("query", candidates)

            # Verify Authorization header
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

    def test_api_call_http_error(self, api_reranker_settings):
        """Test handling of HTTP errors."""
        candidates = [{"id": "doc1", "text": "text"}]

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(api_reranker_settings)

            with pytest.raises(RuntimeError, match="Reranker API error"):
                reranker.rerank("query", candidates)

    def test_api_call_timeout(self, api_reranker_settings):
        """Test handling of timeout."""
        candidates = [{"id": "doc1", "text": "text"}]

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.side_effect = Timeout("Request timeout")

            reranker = CrossEncoderReranker(api_reranker_settings)

            with pytest.raises(RuntimeError, match="timeout"):
                reranker.rerank("query", candidates)

    def test_api_call_connection_error(self, api_reranker_settings):
        """Test handling of connection errors."""
        candidates = [{"id": "doc1", "text": "text"}]

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("Connection failed")

            reranker = CrossEncoderReranker(api_reranker_settings)

            with pytest.raises(RuntimeError, match="request failed"):
                reranker.rerank("query", candidates)

    def test_api_missing_api_key(self):
        """Test validation when api_key is missing."""
        settings = Settings(
            llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
            vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
            rerank=RerankerSettings(
                backend="cross_encoder",
                model="bge-reranker-v2-m3",
                base_url="http://localhost:8000",
            ),
        )
        candidates = [{"id": "doc1", "text": "text"}]

        reranker = CrossEncoderReranker(settings)

        with pytest.raises(ValueError, match="api_key"):
            reranker.rerank("query", candidates)

    def test_api_missing_base_url(self):
        """Test validation when base_url is missing."""
        settings = Settings(
            llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
            vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
            rerank=RerankerSettings(
                backend="cross_encoder",
                model="bge-reranker-v2-m3",
                api_key="test-api-key",
            ),
        )
        candidates = [{"id": "doc1", "text": "text"}]

        reranker = CrossEncoderReranker(settings)

        with pytest.raises(ValueError, match="base_url"):
            reranker.rerank("query", candidates)

    def test_api_invalid_response_format(self, api_reranker_settings):
        """Test handling of invalid response format."""
        candidates = [{"id": "doc1", "text": "text"}]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}  # Missing 'results'

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(api_reranker_settings)

            with pytest.raises(ValueError, match="Unexpected.*response.*format"):
                reranker.rerank("query", candidates)

    def test_api_invalid_result_item(self, api_reranker_settings):
        """Test handling of invalid result items."""
        candidates = [{"id": "doc1", "text": "text"}]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"index": 0}]}  # Missing 'score'

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(api_reranker_settings)

            with pytest.raises(ValueError, match="Invalid result item format"):
                reranker.rerank("query", candidates)

    def test_api_base_url_normalization(self):
        """Test that base_url is normalized (trailing slash removed)."""
        # Create settings with trailing slash
        settings = Settings(
            llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
            embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
            vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
            rerank=RerankerSettings(
                backend="cross_encoder",
                model="bge-reranker-v2-m3",
                api_key="test-api-key",
                base_url="http://localhost:8000/",
            ),
        )

        candidates = [{"id": "doc1", "text": "text"}]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"index": 0, "score": 0.9}]}

        with patch("src.libs.reranker.cross_encoder_reranker.requests.post") as mock_post:
            mock_post.return_value = mock_response

            reranker = CrossEncoderReranker(settings)
            reranker.rerank("query", candidates)

            # Verify URL is normalized (no double slash)
            call_args = mock_post.call_args
            url = call_args[0][0]
            assert url == "http://localhost:8000/v1/rerank"

    def test_reranker_settings_with_api_fields(self):
        """Test RerankerSettings accepts api_key and base_url."""
        settings = RerankerSettings(
            backend="cross_encoder",
            model="bge-reranker-v2-m3",
            api_key="test-key",
            base_url="http://localhost:8000",
        )

        assert settings.api_key == "test-key"
        assert settings.base_url == "http://localhost:8000"
