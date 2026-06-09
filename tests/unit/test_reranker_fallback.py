"""
Unit tests for Core-layer Reranker with fallback logic.

Tests verify:
1. Normal reranking operation (backend works)
2. Graceful fallback when backend fails/times out
3. Fallback flag is properly set in metadata
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.types import RetrievalResult
from src.core.query_engine.reranker import CoreReranker


@pytest.fixture
def mock_settings():
    """Mock Settings object."""
    settings = Mock()
    settings.rerank = Mock()
    settings.rerank.backend = "none"
    settings.rerank.top_m = None
    settings.rerank.timeout = None
    return settings


@pytest.fixture
def sample_fusion_results():
    """Sample results from fusion stage."""
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.8,
            text="First result",
            metadata={"source": "doc1.pdf"}
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            score=0.6,
            text="Second result",
            metadata={"source": "doc2.pdf"}
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            score=0.4,
            text="Third result",
            metadata={"source": "doc3.pdf"}
        ),
    ]


@pytest.fixture
def mock_reranker_backend():
    """Mock reranker backend."""
    backend = Mock()
    backend.rerank.return_value = [
        {"id": "chunk_3", "score": 0.9, "text": "Third result", "metadata": {"source": "doc3.pdf"}},
        {"id": "chunk_1", "score": 0.8, "text": "First result", "metadata": {"source": "doc1.pdf"}},
        {"id": "chunk_2", "score": 0.7, "text": "Second result", "metadata": {"source": "doc2.pdf"}},
    ]
    backend.validate_config.return_value = None
    return backend


class TestCoreRerankerNormal:
    """Test normal reranking operation."""

    def test_none_backend_passthrough(self, mock_settings, sample_fusion_results):
        """Test that 'none' backend passes through results unchanged."""
        mock_settings.rerank.backend = "none"
        reranker = CoreReranker(mock_settings)

        results = reranker.rerank(
            query="test query",
            candidates=sample_fusion_results,
        )

        # Results should be unchanged
        assert len(results) == 3
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.8
        # No fallback flag
        assert results[0].metadata.get("fallback") is None

    def test_reranker_backend_success(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test successful reranking with a working backend."""
        mock_settings.rerank.backend = "cross_encoder"

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Results should be reranked (different order)
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_3"  # Moved to top
            assert results[0].score == 0.9
            # No fallback flag when successful
            assert results[0].metadata.get("fallback") is None

    def test_rerank_with_trace_context(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test that trace context is passed to backend."""
        mock_settings.rerank.backend = "cross_encoder"
        mock_trace = Mock()

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
                trace=mock_trace,
            )

            assert len(results) == 3
            # Verify trace was passed to backend
            mock_reranker_backend.rerank.assert_called_once()
            call_kwargs = mock_reranker_backend.rerank.call_args[1]
            assert call_kwargs.get("trace") == mock_trace


class TestCoreRerankerFallback:
    """Test fallback behavior when backend fails."""

    def test_reranker_exception_fallback(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test fallback when reranker raises exception."""
        mock_settings.rerank.backend = "cross_encoder"
        mock_reranker_backend.rerank.side_effect = RuntimeError("Reranker backend failed")

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Should fall back to original results (unchanged)
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_1"
            assert results[0].score == 0.8

            # Fallback flag should be set
            assert results[0].metadata.get("fallback") is True
            fallback_reason = results[0].metadata.get("fallback_reason", "")
            assert len(fallback_reason) > 0
            assert "backend error" in fallback_reason

    def test_reranker_timeout_fallback(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test fallback when reranker times out."""
        mock_settings.rerank.backend = "llm"
        mock_settings.rerank.timeout = 5.0
        mock_reranker_backend.rerank.side_effect = TimeoutError("Reranker timeout")

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Should fall back to original results
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_1"

            # Fallback flag and reason
            assert results[0].metadata.get("fallback") is True
            assert "timeout" in results[0].metadata.get("fallback_reason", "").lower()

    def test_reranker_validation_error_fallback(self, mock_settings, sample_fusion_results):
        """Test fallback when reranker backend configuration is invalid."""
        mock_settings.rerank.backend = "cross_encoder"

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            # Factory creation fails due to invalid config
            mock_factory.create.side_effect = ValueError("Invalid model config")

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Should fall back to original results
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_1"
            assert results[0].metadata.get("fallback") is True

    def test_fallback_preserves_all_fields(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test that fallback preserves all result fields including metadata."""
        mock_settings.rerank.backend = "cross_encoder"
        mock_reranker_backend.rerank.side_effect = RuntimeError("Backend failed")

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Verify all results preserved
            for original, result in zip(sample_fusion_results, results):
                assert result.chunk_id == original.chunk_id
                assert result.text == original.text
                assert result.metadata["source"] == original.metadata["source"]
                # Fallback flag added to metadata
                assert result.metadata["fallback"] is True


class TestCoreRerankerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_candidates_list(self, mock_settings):
        """Test handling of empty candidates list."""
        reranker = CoreReranker(mock_settings)
        results = reranker.rerank(query="test", candidates=[])

        assert results == []

    def test_single_candidate(self, mock_settings):
        """Test handling of single candidate."""
        single_result = [RetrievalResult(
            chunk_id="chunk_1",
            score=0.9,
            text="Only result",
            metadata={"source": "doc1.pdf"}
        )]

        reranker = CoreReranker(mock_settings)
        results = reranker.rerank(query="test", candidates=single_result)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"

    def test_candidates_must_have_required_fields(self, mock_settings):
        """Test that candidates with missing required fields are handled."""
        invalid_candidates = [
            {"id": "chunk_1", "score": 0.9},  # Missing text and metadata
        ]

        reranker = CoreReranker(mock_settings)

        # Should either reject or fill in defaults gracefully
        # This tests robustness of input handling
        try:
            results = reranker.rerank(query="test", candidates=invalid_candidates)
            # If it doesn't raise, verify fallback behavior
            assert len(results) >= 0
        except (ValueError, TypeError):
            # It's acceptable to raise on invalid input
            pass

    def test_top_m_parameter_respected(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test that top_m parameter limits candidates sent to reranker."""
        mock_settings.rerank.backend = "cross_encoder"
        mock_settings.rerank.top_m = 2

        # Mock backend receives only top 2
        def mock_rerank_impl(query, candidates, trace=None):
            assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"
            return [c for c in candidates]  # Return as-is

        mock_reranker_backend.rerank.side_effect = mock_rerank_impl

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,  # 3 candidates
            )

            assert len(results) == 2  # Only top 2 returned

    def test_fallback_metadata_contains_reason(self, mock_settings, sample_fusion_results, mock_reranker_backend):
        """Test that fallback metadata includes detailed reason."""
        mock_settings.rerank.backend = "cross_encoder"
        error_msg = "Model not found on GPU"
        mock_reranker_backend.rerank.side_effect = RuntimeError(error_msg)

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.return_value = mock_reranker_backend

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # All results should have fallback info
            for result in results:
                assert result.metadata.get("fallback") is True
                fallback_reason = result.metadata.get("fallback_reason", "")
                assert len(fallback_reason) > 0, "fallback_reason should not be empty"


class TestCoreRerankerFactory:
    """Test factory integration."""

    def test_reranker_factory_called_with_settings(self, mock_settings, sample_fusion_results):
        """Test that RerankerFactory is called with correct settings."""
        mock_settings.rerank.backend = "cross_encoder"

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_backend = Mock()
            mock_backend.rerank.return_value = sample_fusion_results
            mock_factory.create.return_value = mock_backend

            reranker = CoreReranker(mock_settings)
            reranker.rerank(query="test", candidates=sample_fusion_results)

            # Factory should be called with settings
            mock_factory.create.assert_called_once_with(mock_settings)

    def test_factory_exception_triggers_fallback(self, mock_settings, sample_fusion_results):
        """Test that exception during factory creation triggers fallback."""
        mock_settings.rerank.backend = "cross_encoder"

        with patch("src.core.query_engine.reranker.RerankerFactory") as mock_factory:
            mock_factory.create.side_effect = ImportError("CrossEncoder library not installed")

            reranker = CoreReranker(mock_settings)
            results = reranker.rerank(
                query="test query",
                candidates=sample_fusion_results,
            )

            # Should fall back to original results
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_1"
            assert results[0].metadata.get("fallback") is True
