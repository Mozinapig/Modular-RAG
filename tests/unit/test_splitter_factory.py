"""Tests for Splitter factory and provider routing."""

import pytest
from unittest.mock import Mock

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings


class TestSplitterFactory:
    """Test SplitterFactory routing and creation."""

    def test_factory_creates_fake_provider(self):
        """Test factory can create instances with fake provider."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 512
        splitter_settings.overlap = 50

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        assert splitter is not None
        assert isinstance(splitter, BaseSplitter)

    def test_factory_splits_text(self):
        """Test that created splitter can split text."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        text = "This is a test document. " * 20  # Repeat to get longer text
        chunks = splitter.split_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_splitter_empty_text(self):
        """Test splitter handles empty text."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        chunks = splitter.split_text("")

        assert isinstance(chunks, list)

    def test_splitter_respects_chunk_size(self):
        """Test splitter respects chunk_size setting."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 50
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        text = "a" * 1000  # Create long text
        chunks = splitter.split_text(text)

        # Most chunks should respect the chunk size limit
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 50 or len(chunk) == len(text)

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error on unknown provider."""
        splitter_settings = Mock()
        splitter_settings.provider = "unknown_provider"

        factory = SplitterFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(splitter_settings)

        assert "unknown_provider" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_provider(self):
        """Test factory raises error when provider is missing."""
        splitter_settings = Mock()
        splitter_settings.provider = None

        factory = SplitterFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(splitter_settings)

        assert "provider" in str(exc_info.value).lower()

    def test_splitter_with_trace(self):
        """Test splitter accepts trace parameter."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        mock_trace = Mock()
        text = "This is a test document."
        chunks = splitter.split_text(text, trace=mock_trace)

        assert isinstance(chunks, list)

    def test_splitter_single_small_text(self):
        """Test splitter handles single small text correctly."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        text = "Short text"
        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        assert chunks[0] == text or text in chunks[0]

    def test_factory_provider_registration(self):
        """Test factory can register custom providers."""
        from src.libs.splitter.base_splitter import BaseSplitter

        class CustomSplitter(BaseSplitter):
            def __init__(self, settings):
                self.settings = settings

            def split_text(self, text, trace=None):
                return [text]  # Simple strategy: return whole text

            def validate_config(self):
                pass

        factory = SplitterFactory()
        factory.register_provider("custom", CustomSplitter)

        splitter_settings = Mock()
        splitter_settings.provider = "custom"

        splitter = factory.create(splitter_settings)
        assert isinstance(splitter, BaseSplitter)

    def test_splitter_deterministic(self):
        """Test splitter produces consistent results."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        text = "This is a test document. " * 10
        chunks1 = splitter.split_text(text)
        chunks2 = splitter.split_text(text)

        assert chunks1 == chunks2

    def test_splitter_whitespace_handling(self):
        """Test splitter handles whitespace correctly."""
        splitter_settings = Mock()
        splitter_settings.provider = "fake"
        splitter_settings.chunk_size = 100
        splitter_settings.overlap = 0

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        text = "Text with   multiple    spaces\n\nand newlines\n\nand tabs\t\there"
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # At least some content should be preserved
        combined = " ".join(chunks)
        assert "Text" in combined
