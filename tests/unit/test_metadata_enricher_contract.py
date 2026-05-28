"""
Unit tests for MetadataEnricher - rule-based and LLM-enhanced metadata enrichment.
Tests focus on title/summary/tags generation, LLM enhancement, and graceful degradation.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.metadata_enricher import MetadataEnricher


class FakeSettings:
    """Fake settings for testing."""
    def __init__(self, use_llm=False, provider="openai", model="gpt-4o-mini"):
        self.use_llm = use_llm
        self.provider = provider
        self.model = model
        self.api_key = "test-key"


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk_1",
            text="Python is a high-level programming language known for its simplicity and readability. "
                 "It supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
            metadata={
                "source_path": "/test/doc.pdf",
                "chunk_index": 0,
                "source_ref": "doc_123"
            }
        ),
        Chunk(
            id="chunk_2",
            text="Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data. "
                 "Common algorithms include decision trees, random forests, and neural networks.",
            metadata={
                "source_path": "/test/doc.pdf",
                "chunk_index": 1,
                "source_ref": "doc_123"
            }
        ),
        Chunk(
            id="chunk_3",
            text="Data science combines statistics, programming, and domain knowledge to extract insights from data.",
            metadata={
                "source_path": "/test/doc.pdf",
                "chunk_index": 2,
                "source_ref": "doc_123"
            }
        ),
    ]


class TestMetadataEnricherRuleBased:
    """Test rule-based metadata enrichment."""

    @pytest.fixture
    def enricher(self):
        """Create enricher with rule-based mode."""
        settings = FakeSettings(use_llm=False)
        return MetadataEnricher(settings)

    def test_enricher_initialization(self, enricher):
        """Test that enricher initializes correctly."""
        assert enricher is not None
        assert hasattr(enricher, 'transform')

    def test_rule_based_generates_title(self, enricher, sample_chunks):
        """Test that rule-based enrichment generates title."""
        result = enricher.transform([sample_chunks[0]])

        assert len(result) == 1
        assert "title" in result[0].metadata
        assert len(result[0].metadata["title"]) > 0

    def test_rule_based_generates_summary(self, enricher, sample_chunks):
        """Test that rule-based enrichment generates summary."""
        result = enricher.transform([sample_chunks[0]])

        assert len(result) == 1
        assert "summary" in result[0].metadata
        assert len(result[0].metadata["summary"]) > 0

    def test_rule_based_generates_tags(self, enricher, sample_chunks):
        """Test that rule-based enrichment generates tags."""
        result = enricher.transform([sample_chunks[0]])

        assert len(result) == 1
        assert "tags" in result[0].metadata
        assert isinstance(result[0].metadata["tags"], list)
        assert len(result[0].metadata["tags"]) > 0

    def test_rule_based_preserves_existing_metadata(self, enricher, sample_chunks):
        """Test that existing metadata is preserved."""
        result = enricher.transform([sample_chunks[0]])

        assert result[0].metadata["source_path"] == "/test/doc.pdf"
        assert result[0].metadata["chunk_index"] == 0
        assert result[0].metadata["source_ref"] == "doc_123"

    def test_rule_based_marks_enriched_by(self, enricher, sample_chunks):
        """Test that enriched_by metadata is added."""
        result = enricher.transform([sample_chunks[0]])

        assert "enriched_by" in result[0].metadata
        assert result[0].metadata["enriched_by"] == "rule"

    def test_rule_based_multiple_chunks(self, enricher, sample_chunks):
        """Test processing multiple chunks."""
        result = enricher.transform(sample_chunks)

        assert len(result) == 3
        for chunk in result:
            assert "title" in chunk.metadata
            assert "summary" in chunk.metadata
            assert "tags" in chunk.metadata
            assert chunk.metadata["enriched_by"] == "rule"

    def test_rule_based_handles_empty_chunk(self, enricher):
        """Test handling of empty chunk."""
        chunk = Chunk(
            id="test_empty",
            text="",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert len(result) == 1
        # Empty text should still have metadata fields
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_rule_based_handles_short_text(self, enricher):
        """Test handling of very short text."""
        chunk = Chunk(
            id="test_short",
            text="Hello",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert len(result) == 1
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata


class TestMetadataEnricherLLM:
    """Test LLM-enhanced metadata enrichment with mocking."""

    @pytest.fixture
    def mock_llm_settings(self):
        """Create settings with LLM enabled."""
        return FakeSettings(use_llm=True, provider="openai")

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = Mock()
        # Mock LLM response with JSON structure
        mock.generate = Mock(return_value=json.dumps({
            "title": "Python Programming Language",
            "summary": "Overview of Python as a high-level programming language",
            "tags": ["python", "programming", "language"]
        }))
        return mock

    def test_llm_enricher_initialization(self, mock_llm_settings):
        """Test LLM enricher initialization."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_factory.return_value.create.return_value = Mock()
            enricher = MetadataEnricher(mock_llm_settings)
            assert enricher is not None

    def test_llm_enricher_calls_llm(self, mock_llm_settings, sample_chunks):
        """Test that LLM is called for enrichment."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value=json.dumps({
                "title": "Python Programming",
                "summary": "A language overview",
                "tags": ["python", "programming"]
            }))
            mock_factory.return_value.create.return_value = mock_llm

            enricher = MetadataEnricher(mock_llm_settings)
            result = enricher.transform([sample_chunks[0]])

            # LLM should have been called
            assert mock_llm.generate.called

    def test_llm_enricher_generates_semantic_metadata(self, mock_llm_settings, sample_chunks):
        """Test that LLM generates semantic metadata."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value=json.dumps({
                "title": "Python Programming Language",
                "summary": "A comprehensive overview of Python",
                "tags": ["python", "programming", "language", "oop"]
            }))
            mock_factory.return_value.create.return_value = mock_llm

            enricher = MetadataEnricher(mock_llm_settings)
            result = enricher.transform([sample_chunks[0]])

            assert len(result) == 1
            assert result[0].metadata["title"] == "Python Programming Language"
            assert "comprehensive" in result[0].metadata["summary"].lower()
            assert len(result[0].metadata["tags"]) >= 3

    def test_llm_enricher_metadata_marked(self, mock_llm_settings, sample_chunks):
        """Test that LLM-enriched chunks are marked in metadata."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value=json.dumps({
                "title": "Test Title",
                "summary": "Test Summary",
                "tags": ["test"]
            }))
            mock_factory.return_value.create.return_value = mock_llm

            enricher = MetadataEnricher(mock_llm_settings)
            result = enricher.transform([sample_chunks[0]])

            # Should be marked as enriched by LLM
            assert result[0].metadata.get("enriched_by") == "llm"

    def test_llm_fallback_on_error(self, mock_llm_settings, sample_chunks):
        """Test fallback to rule-based when LLM fails."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(side_effect=Exception("LLM error"))
            mock_factory.return_value.create.return_value = mock_llm

            enricher = MetadataEnricher(mock_llm_settings)
            result = enricher.transform([sample_chunks[0]])

            # Should fall back to rule-based
            assert result[0].metadata.get("enriched_by") == "rule"
            assert "fallback_reason" in result[0].metadata
            # Should still have metadata fields
            assert "title" in result[0].metadata
            assert "summary" in result[0].metadata
            assert "tags" in result[0].metadata

    def test_llm_fallback_on_invalid_json(self, mock_llm_settings, sample_chunks):
        """Test fallback when LLM returns invalid JSON."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value="Invalid JSON response")
            mock_factory.return_value.create.return_value = mock_llm

            enricher = MetadataEnricher(mock_llm_settings)
            result = enricher.transform([sample_chunks[0]])

            # Should fall back to rule-based
            assert result[0].metadata.get("enriched_by") == "rule"
            assert "fallback_reason" in result[0].metadata

    def test_llm_disabled_uses_rule_based(self, sample_chunks):
        """Test that disabled LLM uses rule-based."""
        settings = FakeSettings(use_llm=False)
        enricher = MetadataEnricher(settings)

        result = enricher.transform([sample_chunks[0]])

        assert result[0].metadata.get("enriched_by") == "rule"


class TestMetadataEnricherEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def enricher(self):
        """Create enricher."""
        settings = FakeSettings(use_llm=False)
        return MetadataEnricher(settings)

    def test_empty_chunks_list(self, enricher):
        """Test handling of empty chunks list."""
        result = enricher.transform([])
        assert result == []

    def test_chunk_with_special_characters(self, enricher):
        """Test handling of special characters."""
        chunk = Chunk(
            id="test_special",
            text="Text with émojis 🎉 and spëcial çhars",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert len(result) == 1
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_chunk_with_very_long_text(self, enricher):
        """Test handling of very long text."""
        long_text = "Word " * 10000  # 50k+ characters
        chunk = Chunk(
            id="test_long",
            text=long_text,
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert len(result) == 1
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata

    def test_chunk_with_only_whitespace(self, enricher):
        """Test handling of whitespace-only chunk."""
        chunk = Chunk(
            id="test_whitespace",
            text="   \n\n\n   \t\t\t   ",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert len(result) == 1
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_chunk_exception_doesnt_block_others(self, enricher):
        """Test that exception in one chunk doesn't block others."""
        chunks = [
            Chunk(id="c1", text="Normal text", metadata={"source_path": "/test"}),
            Chunk(id="c2", text="More text", metadata={"source_path": "/test"}),
            Chunk(id="c3", text="Another chunk", metadata={"source_path": "/test"}),
        ]

        # Should process all chunks without raising
        result = enricher.transform(chunks)
        assert len(result) == 3

    def test_trace_context_optional(self, enricher):
        """Test that trace context is optional."""
        chunk = Chunk(
            id="test_trace",
            text="Text to enrich",
            metadata={"source_path": "/test"}
        )

        # Should work without trace
        result = enricher.transform([chunk], trace=None)
        assert len(result) == 1

        # Should also work with trace
        trace = TraceContext(trace_type="ingestion")
        result = enricher.transform([chunk], trace=trace)
        assert len(result) == 1

    def test_metadata_tags_is_list(self, enricher):
        """Test that tags is always a list."""
        chunk = Chunk(
            id="test_tags",
            text="Sample text for tagging",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert isinstance(result[0].metadata["tags"], list)
        for tag in result[0].metadata["tags"]:
            assert isinstance(tag, str)

    def test_metadata_title_is_string(self, enricher):
        """Test that title is always a string."""
        chunk = Chunk(
            id="test_title",
            text="Sample text for title",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert isinstance(result[0].metadata["title"], str)
        assert len(result[0].metadata["title"]) > 0

    def test_metadata_summary_is_string(self, enricher):
        """Test that summary is always a string."""
        chunk = Chunk(
            id="test_summary",
            text="Sample text for summary generation",
            metadata={"source_path": "/test"}
        )
        result = enricher.transform([chunk])

        assert isinstance(result[0].metadata["summary"], str)
        assert len(result[0].metadata["summary"]) > 0


class TestMetadataEnricherIntegration:
    """Integration tests with real-like scenarios."""

    @pytest.fixture
    def enricher(self):
        """Create enricher."""
        settings = FakeSettings(use_llm=False)
        return MetadataEnricher(settings)

    def test_enrichment_preserves_chunk_structure(self, enricher, sample_chunks):
        """Test that enrichment preserves chunk structure."""
        original_ids = [c.id for c in sample_chunks]
        result = enricher.transform(sample_chunks)

        result_ids = [c.id for c in result]
        assert original_ids == result_ids

    def test_enrichment_preserves_chunk_text(self, enricher, sample_chunks):
        """Test that enrichment preserves chunk text."""
        original_texts = [c.text for c in sample_chunks]
        result = enricher.transform(sample_chunks)

        result_texts = [c.text for c in result]
        assert original_texts == result_texts

    def test_enrichment_with_trace_context(self, enricher, sample_chunks):
        """Test enrichment with trace context."""
        trace = TraceContext(trace_type="ingestion")
        result = enricher.transform(sample_chunks, trace=trace)

        assert len(result) == 3
        # Trace should have recorded the stage
        assert trace.stages is not None

    def test_multiple_enrichments_idempotent(self, enricher):
        """Test that multiple enrichments produce consistent results."""
        chunk = Chunk(
            id="test_idempotent",
            text="Python is a programming language",
            metadata={"source_path": "/test"}
        )

        result1 = enricher.transform([chunk])
        result2 = enricher.transform([chunk])

        # Results should be consistent
        assert result1[0].metadata["title"] == result2[0].metadata["title"]
        assert result1[0].metadata["summary"] == result2[0].metadata["summary"]
        assert result1[0].metadata["tags"] == result2[0].metadata["tags"]
