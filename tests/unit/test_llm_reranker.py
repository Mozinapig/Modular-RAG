"""Unit tests for LLM Reranker implementation."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.libs.reranker.llm_reranker import LLMReranker
from src.core.settings import Settings


@pytest.fixture
def reranker_settings():
    """Create settings with LLM Reranker configuration."""
    return Settings(
        llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
        embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
        vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
        rerank={"backend": "llm"},
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = Mock()
    return mock


class TestLLMRerankerInit:
    """Test LLMReranker initialization and configuration."""

    def test_init_with_valid_settings(self, reranker_settings):
        """Test initialization with valid settings."""
        reranker = LLMReranker(reranker_settings)
        assert reranker is not None
        assert reranker.settings == reranker_settings

    def test_validate_config_requires_llm(self):
        """Test that LLM configuration is required."""
        from src.core.settings import RerankerSettings
        settings = RerankerSettings(backend="llm")
        reranker = LLMReranker(settings)

        with pytest.raises(ValueError, match="LLM provider is required"):
            reranker.validate_config()

    def test_prompt_file_loading_with_default_path(self, reranker_settings, tmp_path):
        """Test loading prompt from default config path."""
        # Create a temporary prompt file
        prompt_dir = tmp_path / "config" / "prompts"
        prompt_dir.mkdir(parents=True)
        prompt_file = prompt_dir / "rerank.txt"
        prompt_file.write_text("Test rerank prompt: {query}\n{candidates}")

        with patch(
            "src.libs.reranker.llm_reranker.LLMReranker._get_prompt_path",
            return_value=str(prompt_file),
        ):
            reranker = LLMReranker(reranker_settings)
            # Prompt should be loaded during initialization
            assert reranker.prompt_template is not None

    def test_prompt_file_missing_fallback(self, reranker_settings):
        """Test fallback when prompt file is missing."""
        # Should not raise error, use default template
        reranker = LLMReranker(reranker_settings)
        assert reranker.prompt_template is not None
        assert "{query}" in reranker.prompt_template.lower() or "{candidates}" in reranker.prompt_template.lower()


class TestLLMRerankerBasicFunctionality:
    """Test basic reranking functionality."""

    def test_rerank_preserves_candidate_structure(self, reranker_settings):
        """Test that rerank preserves candidate dictionary structure."""
        candidates = [
            {"id": "doc1", "score": 0.9, "text": "First candidate"},
            {"id": "doc2", "score": 0.7, "text": "Second candidate"},
            {"id": "doc3", "score": 0.6, "text": "Third candidate"},
        ]

        with patch.object(
            LLMReranker,
            "_call_llm",
            return_value=["doc3", "doc1", "doc2"],
        ):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # All candidates should still be present
            assert len(result) == 3
            # All original fields should be preserved
            for item in result:
                assert "id" in item
                assert "score" in item
                assert "text" in item

    def test_rerank_reorders_by_llm_output(self, reranker_settings):
        """Test that rerank properly reorders candidates based on LLM output."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
            {"id": "doc3", "score": 0.6},
        ]

        # Mock LLM to return different order
        with patch.object(
            LLMReranker,
            "_call_llm",
            return_value=["doc3", "doc1", "doc2"],
        ):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # Check order is changed
            assert result[0]["id"] == "doc3"
            assert result[1]["id"] == "doc1"
            assert result[2]["id"] == "doc2"

    def test_rerank_empty_candidates(self, reranker_settings):
        """Test rerank with empty candidate list."""
        with patch.object(LLMReranker, "_call_llm"):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", [])
            assert result == []

    def test_rerank_single_candidate(self, reranker_settings):
        """Test rerank with single candidate."""
        candidates = [{"id": "doc1", "score": 0.9}]

        with patch.object(LLMReranker, "_call_llm", return_value=["doc1"]):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            assert len(result) == 1
            assert result[0]["id"] == "doc1"


class TestLLMRerankerOutputValidation:
    """Test LLM output validation and error handling."""

    def test_invalid_llm_output_wrong_count(self, reranker_settings):
        """Test handling when LLM returns wrong number of IDs."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]

        # LLM returns only 1 ID for 2 candidates
        with patch.object(LLMReranker, "_call_llm", return_value=["doc1"]):
            reranker = LLMReranker(reranker_settings)

            with pytest.raises(ValueError, match="count mismatch|invalid.*output"):
                reranker.rerank("test query", candidates)

    def test_invalid_llm_output_unknown_id(self, reranker_settings):
        """Test handling when LLM returns unknown document IDs."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]

        # LLM returns unknown ID
        with patch.object(
            LLMReranker,
            "_call_llm",
            return_value=["doc1", "doc_unknown"],
        ):
            reranker = LLMReranker(reranker_settings)

            with pytest.raises(ValueError, match="unknown.*id|invalid.*id"):
                reranker.rerank("test query", candidates)


class TestLLMRerankerPromptConstruction:
    """Test prompt construction logic."""

    def test_prompt_includes_candidates(self, reranker_settings):
        """Test that prompt includes candidate information."""
        candidates = [
            {"id": "doc1", "score": 0.9, "text": "Azure OpenAI setup"},
            {"id": "doc2", "score": 0.7, "text": "Cloud deployment"},
        ]

        with patch.object(LLMReranker, "_call_llm") as mock_call:
            mock_call.return_value = ["doc1", "doc2"]
            reranker = LLMReranker(reranker_settings)
            reranker.rerank("test query", candidates)

            # Verify prompt includes candidate info
            call_args = mock_call.call_args
            prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
            # Should include at least some candidate info
            assert "doc1" in prompt_arg or "Azure" in prompt_arg


class TestLLMRerankerWithTrace:
    """Test trace context recording."""

    def test_rerank_with_candidate_mapping(self, reranker_settings):
        """Test that reranking correctly maps candidates to output."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
            {"id": "doc3", "score": 0.5},
        ]

        # Reorder candidates to doc3, doc1, doc2
        with patch.object(LLMReranker, "_call_llm", return_value=["doc3", "doc1", "doc2"]):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # Verify order was changed
            assert result[0]["id"] == "doc3"
            assert result[1]["id"] == "doc1"
            assert result[2]["id"] == "doc2"


class TestLLMRerankerCustomPrompt:
    """Test custom prompt injection (for testing)."""

    def test_custom_prompt_injection(self, reranker_settings):
        """Test that custom prompt can be injected for testing."""
        custom_prompt = "Rank these documents: {candidates}. Return JSON: {{'ranked_ids': [...]}}"
        candidates = [{"id": "doc1", "score": 0.9}]

        reranker = LLMReranker(reranker_settings)
        reranker.prompt_template = custom_prompt

        with patch.object(LLMReranker, "_call_llm", return_value=["doc1"]):
            result = reranker.rerank("test query", candidates)
            assert result[0]["id"] == "doc1"


class TestLLMRerankerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_candidates_with_missing_id_field(self, reranker_settings):
        """Test handling of candidates missing 'id' field."""
        candidates = [
            {"score": 0.9},  # missing 'id'
            {"id": "doc2", "score": 0.7},
        ]

        reranker = LLMReranker(reranker_settings)

        with pytest.raises(ValueError, match="missing.*id|id.*required"):
            reranker.rerank("test query", candidates)

    def test_duplicate_candidate_ids(self, reranker_settings):
        """Test handling of duplicate candidate IDs."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc1", "score": 0.7},  # duplicate
        ]

        reranker = LLMReranker(reranker_settings)

        with pytest.raises(ValueError, match="duplicate|unique"):
            reranker.rerank("test query", candidates)

    def test_very_large_candidates_list(self, reranker_settings):
        """Test reranking a large list of candidates."""
        candidates = [
            {"id": f"doc{i}", "score": 1.0 - (i * 0.01)}
            for i in range(100)
        ]

        expected_order = [f"doc{i}" for i in range(100)]

        with patch.object(
            LLMReranker,
            "_call_llm",
            return_value=expected_order,
        ):
            reranker = LLMReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            assert len(result) == 100
            assert result[0]["id"] == "doc0"
            assert result[99]["id"] == "doc99"


class TestLLMRerankerIntegration:
    """Test integration with factory."""

    def test_factory_creates_llm_reranker(self, reranker_settings):
        """Test that factory can create LLMReranker."""
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()

        with patch.object(LLMReranker, "validate_config"):
            reranker = factory.create(reranker_settings.rerank)
            assert reranker is not None
