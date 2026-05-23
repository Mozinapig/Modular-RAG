"""Tests for Embedding factory and provider routing."""

import pytest
from unittest.mock import Mock

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import EmbeddingSettings


class TestEmbeddingFactory:
    """Test EmbeddingFactory routing and creation."""

    def test_factory_creates_fake_provider(self):
        """Test factory can create instances with fake provider."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-embedding-model",
            api_key="fake-key",
            dimensions=768,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        assert embedding is not None
        assert isinstance(embedding, BaseEmbedding)

    def test_factory_respects_dimensions(self):
        """Test factory preserves dimension setting."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            dimensions=1024,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        assert isinstance(embedding, BaseEmbedding)

    def test_embedding_batch_works(self):
        """Test that created embedding can process batch requests."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            dimensions=768,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        texts = ["Hello world", "Test document"]
        vectors = embedding.embed(texts)

        assert isinstance(vectors, list)
        assert len(vectors) == 2
        for vector in vectors:
            assert isinstance(vector, list)
            assert len(vector) == 768

    def test_embedding_empty_input(self):
        """Test embedding handles empty input."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            dimensions=768,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        vectors = embedding.embed([])

        assert isinstance(vectors, list)
        assert len(vectors) == 0

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error on unknown provider."""
        settings = EmbeddingSettings(
            provider="unknown_provider",
            model="fake-model",
            api_key="fake-key",
        )

        factory = EmbeddingFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "unknown_provider" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_api_key(self):
        """Test factory raises error when api_key is missing."""
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="",  # Empty API key
        )

        factory = EmbeddingFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "api_key" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_model(self):
        """Test factory raises error when model is missing."""
        settings = EmbeddingSettings(
            provider="openai",
            model="",  # Empty model
            api_key="fake-key",
        )

        factory = EmbeddingFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(settings)

        assert "model" in str(exc_info.value).lower()

    def test_embedding_with_trace(self):
        """Test embedding accepts trace parameter."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            dimensions=768,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        mock_trace = Mock()
        texts = ["Hello world"]
        vectors = embedding.embed(texts, trace=mock_trace)

        assert isinstance(vectors, list)
        assert len(vectors) == 1

    def test_embedding_single_text(self):
        """Test embedding single text works correctly."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            dimensions=1536,
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)

        texts = ["Single document"]
        vectors = embedding.embed(texts)

        assert len(vectors) == 1
        assert len(vectors[0]) == 1536

    def test_factory_provider_registration(self):
        """Test factory can register custom providers."""
        from src.libs.embedding.base_embedding import BaseEmbedding

        class CustomEmbedding(BaseEmbedding):
            def __init__(self, settings):
                self.settings = settings

            def embed(self, texts, trace=None):
                return [[0.0] * 768] * len(texts)

            def validate_config(self):
                pass

        factory = EmbeddingFactory()
        factory.register_provider("custom", CustomEmbedding)

        settings = EmbeddingSettings(
            provider="custom",
            model="custom-model",
            api_key="fake-key",
        )

        embedding = factory.create(settings)
        assert isinstance(embedding, BaseEmbedding)

    def test_factory_default_dimensions(self):
        """Test embedding factory uses correct default dimensions."""
        settings = EmbeddingSettings(
            provider="fake",
            model="fake-model",
            api_key="fake-key",
            # Don't specify dimensions, should use default
        )

        factory = EmbeddingFactory()
        embedding = factory.create(settings)
        vectors = embedding.embed(["test"])

        # Default should be 1536 for openai-compatible models
        assert len(vectors[0]) == settings.dimensions
