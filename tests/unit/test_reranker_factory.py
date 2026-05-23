"""Tests for Reranker factory and provider routing."""

import pytest
from unittest.mock import Mock

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import RerankerFactory, NoneReranker


class TestRerankerFactory:
    """Test RerankerFactory routing and creation."""

    def test_factory_creates_none_provider(self):
        """Test factory can create instances with none provider."""
        settings = Mock()
        settings.provider = "none"

        factory = RerankerFactory()
        reranker = factory.create(settings)

        assert reranker is not None
        assert isinstance(reranker, BaseReranker)
        assert isinstance(reranker, NoneReranker)

    def test_none_reranker_preserves_order(self):
        """Test NoneReranker preserves original order."""
        settings = Mock()
        reranker = NoneReranker(settings)

        candidates = [
            {"id": "1", "text": "First", "score": 0.9},
            {"id": "2", "text": "Second", "score": 0.7},
            {"id": "3", "text": "Third", "score": 0.5},
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result) == 3
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"
        assert result[2]["id"] == "3"

    def test_none_reranker_with_empty_candidates(self):
        """Test NoneReranker handles empty candidate list."""
        settings = Mock()
        reranker = NoneReranker(settings)

        result = reranker.rerank("test query", [])

        assert result == []

    def test_none_reranker_with_single_candidate(self):
        """Test NoneReranker handles single candidate."""
        settings = Mock()
        reranker = NoneReranker(settings)

        candidates = [{"id": "1", "text": "Only one", "score": 0.8}]

        result = reranker.rerank("test query", candidates)

        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error on unknown provider."""
        settings = Mock()
        settings.provider = "unknown_provider"

        factory = RerankerFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "unknown_provider" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_provider(self):
        """Test factory raises error when provider is missing."""
        settings = Mock()
        settings.provider = None

        factory = RerankerFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "provider" in str(exc_info.value).lower()

    def test_none_reranker_case_insensitive(self):
        """Test factory is case insensitive for provider names."""
        settings = Mock()
        settings.provider = "NONE"

        factory = RerankerFactory()
        reranker = factory.create(settings)

        assert isinstance(reranker, NoneReranker)

    def test_none_reranker_accepts_trace(self):
        """Test NoneReranker accepts trace parameter."""
        settings = Mock()
        reranker = NoneReranker(settings)

        candidates = [
            {"id": "1", "text": "First", "score": 0.9},
        ]
        mock_trace = Mock()

        result = reranker.rerank("test query", candidates, trace=mock_trace)

        assert len(result) == 1

    def test_factory_provider_registration(self):
        """Test factory can register custom providers."""
        class CustomReranker(BaseReranker):
            def __init__(self, settings):
                self.settings = settings

            def rerank(self, query, candidates, trace=None):
                # Simple custom logic: reverse order
                return list(reversed(candidates))

            def validate_config(self):
                pass

        factory = RerankerFactory()
        factory.register_provider("custom", CustomReranker)

        settings = Mock()
        settings.provider = "custom"

        reranker = factory.create(settings)
        assert isinstance(reranker, BaseReranker)

        candidates = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        result = reranker.rerank("query", candidates)
        assert result[0]["id"] == "3"

    def test_none_reranker_validate_config(self):
        """Test NoneReranker validates config (should always pass)."""
        settings = Mock()
        reranker = NoneReranker(settings)

        # Should not raise
        reranker.validate_config()

    def test_none_reranker_returns_same_object_references(self):
        """Test NoneReranker returns same candidate objects."""
        settings = Mock()
        reranker = NoneReranker(settings)

        candidates = [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
        ]

        result = reranker.rerank("query", candidates)

        # Should be the same objects, not copies
        assert result[0] is candidates[0]
        assert result[1] is candidates[1]

    def test_factory_provider_list(self):
        """Test factory provides list of available providers."""
        factory = RerankerFactory()

        # Should have at least 'none' provider
        assert "none" in factory._providers
        assert NoneReranker == factory._providers["none"]
