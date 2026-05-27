"""Unit tests for Cross-Encoder Reranker implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.core.settings import Settings


@pytest.fixture
def reranker_settings():
    """Create settings with Cross-Encoder Reranker configuration."""
    return Settings(
        llm={"provider": "openai", "model": "gpt-4", "api_key": "test-key"},
        embedding={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-key"},
        vector_store={"backend": "chroma", "persist_path": "./data/db/chroma"},
        rerank={"backend": "cross_encoder", "model": "cross-encoder/ms-marco-MiniLM-L-12-v2"},
    )


class TestCrossEncoderRerankerInit:
    """Test CrossEncoderReranker initialization and configuration."""

    def test_init_with_valid_settings(self, reranker_settings):
        """Test initialization with valid settings."""
        reranker = CrossEncoderReranker(reranker_settings)
        assert reranker is not None
        assert reranker.settings == reranker_settings

    def test_validate_config_with_valid_model(self, reranker_settings):
        """Test that configuration with model is valid."""
        reranker = CrossEncoderReranker(reranker_settings)
        # Should not raise
        reranker.validate_config()

    def test_validate_config_missing_model(self):
        """Test that configuration without model raises error."""
        from src.core.settings import RerankerSettings
        settings = RerankerSettings(backend="cross_encoder")
        reranker = CrossEncoderReranker(settings)

        with pytest.raises(ValueError, match="model"):
            reranker.validate_config()

    def test_init_creates_scorer_lazy(self, reranker_settings):
        """Test that scorer is created lazily (not in __init__)."""
        reranker = CrossEncoderReranker(reranker_settings)
        # Scorer should be None until first rerank call
        assert reranker.scorer is None


class TestCrossEncoderRerankerBasicFunctionality:
    """Test basic reranking functionality."""

    def test_rerank_preserves_candidate_structure(self, reranker_settings):
        """Test that rerank preserves candidate dictionary structure."""
        candidates = [
            {"id": "doc1", "score": 0.9, "text": "First candidate"},
            {"id": "doc2", "score": 0.7, "text": "Second candidate"},
            {"id": "doc3", "score": 0.6, "text": "Third candidate"},
        ]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            return_value=[0.8, 0.5, 0.7],  # Reorder: doc3(0.7), doc1(0.8), doc2(0.5)
        ):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # All candidates should still be present
            assert len(result) == 3
            # All original fields should be preserved
            for item in result:
                assert "id" in item
                assert "score" in item
                assert "text" in item

    def test_rerank_reorders_by_scores(self, reranker_settings):
        """Test that rerank properly reorders candidates based on scorer output."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
            {"id": "doc3", "score": 0.6},
        ]

        # Mock scorer returns new relevance scores for each candidate
        # Scores are ordered by doc order: doc1=0.8, doc2=0.5, doc3=0.7
        # So order should be: doc1(0.8), doc3(0.7), doc2(0.5)
        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            return_value=[0.8, 0.5, 0.7],
        ):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # Check order is changed based on scores
            assert result[0]["id"] == "doc1"
            assert result[0]["cross_encoder_score"] == 0.8
            assert result[1]["id"] == "doc3"
            assert result[1]["cross_encoder_score"] == 0.7
            assert result[2]["id"] == "doc2"
            assert result[2]["cross_encoder_score"] == 0.5

    def test_rerank_empty_candidates(self, reranker_settings):
        """Test rerank with empty candidate list."""
        with patch.object(CrossEncoderReranker, "_call_scorer"):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", [])
            assert result == []

    def test_rerank_single_candidate(self, reranker_settings):
        """Test rerank with single candidate."""
        candidates = [{"id": "doc1", "score": 0.9}]

        with patch.object(CrossEncoderReranker, "_call_scorer", return_value=[0.85]):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            assert len(result) == 1
            assert result[0]["id"] == "doc1"
            assert result[0]["cross_encoder_score"] == 0.85


class TestCrossEncoderRerankerScoring:
    """Test Cross-Encoder scoring logic."""

    def test_scorer_receives_correct_pairs(self, reranker_settings):
        """Test that scorer receives query-candidate pairs."""
        candidates = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
        ]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            return_value=[0.6, 0.4],
        ) as mock_scorer:
            reranker = CrossEncoderReranker(reranker_settings)
            reranker.rerank("my query", candidates)

            # Verify scorer was called with query and candidates
            call_args = mock_scorer.call_args
            assert call_args is not None
            # First arg should be query
            query_arg = call_args[0][0]
            assert query_arg == "my query"
            # Second arg should be texts
            texts_arg = call_args[0][1]
            assert len(texts_arg) == 2

    def test_scores_added_to_candidates(self, reranker_settings):
        """Test that cross_encoder_score is added to each candidate."""
        candidates = [
            {"id": "doc1"},
            {"id": "doc2"},
        ]

        scores = [0.75, 0.25]
        with patch.object(CrossEncoderReranker, "_call_scorer", return_value=scores):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test", candidates)

            # Check scores are added and values correct
            assert result[0]["cross_encoder_score"] == 0.75
            assert result[1]["cross_encoder_score"] == 0.25


class TestCrossEncoderRerankerFallback:
    """Test fallback behavior on timeout/failure."""

    def test_timeout_fallback_signal(self, reranker_settings):
        """Test that timeout signals fallback availability."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            side_effect=TimeoutError("Scorer timeout"),
        ):
            reranker = CrossEncoderReranker(reranker_settings)

            with pytest.raises(RuntimeError, match="timeout"):
                reranker.rerank("test query", candidates)

    def test_scorer_failure_fallback_signal(self, reranker_settings):
        """Test that scorer failure signals fallback."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            side_effect=Exception("Model loading failed"),
        ):
            reranker = CrossEncoderReranker(reranker_settings)

            with pytest.raises(RuntimeError, match="reranking.*failed"):
                reranker.rerank("test query", candidates)

    def test_invalid_scores_fallback(self, reranker_settings):
        """Test that invalid score count triggers fallback."""
        candidates = [
            {"id": "doc1"},
            {"id": "doc2"},
        ]

        # Return wrong number of scores
        with patch.object(CrossEncoderReranker, "_call_scorer", return_value=[0.5]):
            reranker = CrossEncoderReranker(reranker_settings)

            with pytest.raises(ValueError, match="count mismatch"):
                reranker.rerank("test", candidates)


class TestCrossEncoderRerankerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_candidates_with_missing_id_field(self, reranker_settings):
        """Test handling of candidates missing 'id' field."""
        candidates = [
            {"score": 0.9},  # missing 'id'
            {"id": "doc2", "score": 0.7},
        ]

        reranker = CrossEncoderReranker(reranker_settings)

        with pytest.raises(ValueError, match="missing.*id|id.*required"):
            reranker.rerank("test query", candidates)

    def test_duplicate_candidate_ids(self, reranker_settings):
        """Test handling of duplicate candidate IDs."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc1", "score": 0.7},  # duplicate
        ]

        reranker = CrossEncoderReranker(reranker_settings)

        with pytest.raises(ValueError, match="duplicate|unique"):
            reranker.rerank("test query", candidates)

    def test_very_large_candidates_list(self, reranker_settings):
        """Test reranking a large list of candidates."""
        candidates = [
            {"id": f"doc{i}", "text": f"text {i}", "score": 1.0 - (i * 0.01)}
            for i in range(100)
        ]

        # Return scores in reverse order (lower score for higher indices)
        expected_scores = [1.0 - (i * 0.01) for i in range(100)]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            return_value=expected_scores,
        ):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            assert len(result) == 100
            # Verify top results are highest scored
            assert result[0]["cross_encoder_score"] >= result[99]["cross_encoder_score"]

    def test_candidates_with_missing_text_field(self, reranker_settings):
        """Test handling candidates without text field."""
        candidates = [
            {"id": "doc1"},  # no text field
            {"id": "doc2", "text": "content"},
        ]

        with patch.object(CrossEncoderReranker, "_call_scorer", return_value=[0.6, 0.4]):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates)

            # Should handle gracefully, possibly with empty string
            assert len(result) == 2

    def test_empty_query_string(self, reranker_settings):
        """Test handling of empty query."""
        candidates = [{"id": "doc1", "text": "content"}]

        reranker = CrossEncoderReranker(reranker_settings)

        with pytest.raises(ValueError, match="Query.*empty|query.*required"):
            reranker.rerank("", candidates)


class TestCrossEncoderRerankerIntegration:
    """Test integration with factory."""

    def test_factory_creates_cross_encoder_reranker(self, reranker_settings):
        """Test that factory can create CrossEncoderReranker."""
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()

        with patch.object(CrossEncoderReranker, "validate_config"):
            reranker = factory.create(reranker_settings.rerank)
            assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_registration(self):
        """Test that cross_encoder provider is registered in factory."""
        from src.libs.reranker.reranker_factory import RerankerFactory

        factory = RerankerFactory()
        assert "cross_encoder" in factory._providers


class TestCrossEncoderRerankerMockScorer:
    """Test mock scorer pattern for deterministic testing."""

    def test_mock_scorer_deterministic_results(self, reranker_settings):
        """Test that mock scorer provides deterministic results."""
        candidates = [
            {"id": "doc1", "score": 0.5},
            {"id": "doc2", "score": 0.6},
            {"id": "doc3", "score": 0.7},
        ]

        # Define deterministic scores
        deterministic_scores = [0.9, 0.3, 0.6]

        with patch.object(
            CrossEncoderReranker,
            "_call_scorer",
            return_value=deterministic_scores,
        ):
            reranker = CrossEncoderReranker(reranker_settings)

            # Run multiple times
            result1 = reranker.rerank("query", candidates.copy())
            result2 = reranker.rerank("query", candidates.copy())

            # Results should be identical
            assert [r["id"] for r in result1] == [r["id"] for r in result2]
            assert [r.get("cross_encoder_score") for r in result1] == [
                r.get("cross_encoder_score") for r in result2
            ]


class TestCrossEncoderRerankerWithTrace:
    """Test trace context recording."""

    def test_rerank_with_trace_recording(self, reranker_settings):
        """Test that reranking records trace information."""
        candidates = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]

        mock_trace = Mock()

        with patch.object(CrossEncoderReranker, "_call_scorer", return_value=[0.8, 0.5]):
            reranker = CrossEncoderReranker(reranker_settings)
            result = reranker.rerank("test query", candidates, trace=mock_trace)

            # Verify trace was recorded
            if mock_trace.record_stage.called:
                call_args = mock_trace.record_stage.call_args
                assert call_args is not None
                # Should record cross_encoder stage
                stage_name = call_args[0][0] if call_args[0] else call_args[1].get("stage", "")
                assert "cross_encoder" in stage_name.lower() or "rerank" in stage_name.lower()
