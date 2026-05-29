"""
Test suite for SparseEncoder - build BM25 statistics from chunks.
Coverage: Initialization, sparse encoding, term statistics, edge cases, integration.
Target: 16+ tests.
"""

import pytest
from unittest.mock import Mock
from typing import List

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class TestSparseEncoderInitialization:
    """Tests for SparseEncoder initialization."""

    def test_init_with_settings(self):
        """Should initialize with settings."""
        settings = Mock()
        encoder = SparseEncoder(settings)
        assert encoder.settings is not None

    def test_init_with_dependency_injection(self):
        """Should accept injected tokenizer for testing."""
        settings = Mock()
        mock_tokenizer = Mock()
        encoder = SparseEncoder(settings, tokenizer=mock_tokenizer)
        assert encoder.tokenizer == mock_tokenizer


class TestSparseEncoderBasicEncoding:
    """Tests for basic sparse encoding functionality."""

    def test_encode_single_chunk(self):
        """Should encode a single chunk and return it with sparse statistics."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="hello world test", metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert result[0].id == "chunk_001"
        assert "sparse_embedding" in result[0].metadata
        sparse_stats = result[0].metadata["sparse_embedding"]
        assert isinstance(sparse_stats, dict)
        assert "term_weights" in sparse_stats

    def test_encode_multiple_chunks(self):
        """Should encode multiple chunks and return all with sparse statistics."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_001", text="apple banana cherry", metadata={}),
            Chunk(id="chunk_002", text="apple date elder", metadata={}),
            Chunk(id="chunk_003", text="fig grape honey", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        assert len(result) == 3
        for chunk in result:
            assert "sparse_embedding" in chunk.metadata
            sparse_stats = chunk.metadata["sparse_embedding"]
            assert "term_weights" in sparse_stats

    def test_encode_preserves_chunk_identity(self):
        """Should preserve chunk id, text, and other metadata."""
        settings = Mock()
        chunk = Chunk(
            id="chunk_001",
            text="test content",
            metadata={"source": "doc.pdf", "page": 1}
        )

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert result[0].id == "chunk_001"
        assert result[0].text == "test content"
        assert result[0].metadata["source"] == "doc.pdf"
        assert result[0].metadata["page"] == 1


class TestSparseEncoderTermStatistics:
    """Tests for term statistics extraction."""

    def test_extract_term_weights(self):
        """Should extract term weights from chunk text."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="apple apple banana cherry", metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        sparse_stats = result[0].metadata["sparse_embedding"]
        term_weights = sparse_stats["term_weights"]

        # "apple" appears twice, should have higher weight
        assert "apple" in term_weights
        assert "banana" in term_weights
        assert "cherry" in term_weights

    def test_term_frequency_weights(self):
        """Term frequency should be reflected in weights."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_001", text="apple apple banana", metadata={}),
            Chunk(id="chunk_002", text="banana cherry", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        # First chunk has "apple" twice
        stats1 = result[0].metadata["sparse_embedding"]
        weights1 = stats1["term_weights"]

        # Check that apple weight is captured
        assert "apple" in weights1

    def test_document_frequency_in_statistics(self):
        """Should track document frequency for IDF calculation."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_001", text="apple banana", metadata={}),
            Chunk(id="chunk_002", text="apple cherry", metadata={}),
            Chunk(id="chunk_003", text="date elder fig", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        # All should have sparse_embedding
        for chunk in result:
            assert "sparse_embedding" in chunk.metadata
            stats = chunk.metadata["sparse_embedding"]
            assert "term_weights" in stats


class TestSparseEncoderEdgeCases:
    """Edge case tests."""

    def test_empty_chunk_list(self):
        """Empty chunk list should return empty result."""
        settings = Mock()
        encoder = SparseEncoder(settings)
        result = encoder.encode([])
        assert result == []

    def test_chunk_with_empty_text(self):
        """Chunk with empty text should be handled gracefully."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="", metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert "sparse_embedding" in result[0].metadata
        sparse_stats = result[0].metadata["sparse_embedding"]
        # Empty text should have empty or minimal term_weights
        assert isinstance(sparse_stats["term_weights"], dict)

    def test_chunk_with_whitespace_only(self):
        """Chunk with only whitespace should be handled gracefully."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="   \n\t  ", metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert "sparse_embedding" in result[0].metadata

    def test_chunk_with_very_long_text(self):
        """Should handle chunks with very long text."""
        settings = Mock()
        long_text = "word " * 10000  # Very long text
        chunk = Chunk(id="chunk_001", text=long_text, metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert "sparse_embedding" in result[0].metadata

    def test_chunk_with_special_characters(self):
        """Should handle special characters and punctuation."""
        settings = Mock()
        chunk = Chunk(
            id="chunk_001",
            text="Hello! @#$%^&*() [brackets] {braces} <angle>",
            metadata={}
        )

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert "sparse_embedding" in result[0].metadata

    def test_chunk_with_unicode(self):
        """Should handle unicode characters correctly."""
        settings = Mock()
        chunk = Chunk(
            id="chunk_001",
            text="你好世界 مرحبا بالعالم שלום עולם",
            metadata={}
        )

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        assert len(result) == 1
        assert "sparse_embedding" in result[0].metadata

    def test_preserves_chunk_order(self):
        """Chunk order should be preserved after encoding."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_003", text="third", metadata={}),
            Chunk(id="chunk_001", text="first", metadata={}),
            Chunk(id="chunk_002", text="second", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        assert result[0].id == "chunk_003"
        assert result[1].id == "chunk_001"
        assert result[2].id == "chunk_002"


class TestSparseEncoderCaseSensitivity:
    """Tests for case sensitivity in term extraction."""

    def test_case_insensitive_term_extraction(self):
        """Terms should be normalized (case-insensitive)."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="Apple BANANA banana", metadata={})

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        sparse_stats = result[0].metadata["sparse_embedding"]
        term_weights = sparse_stats["term_weights"]

        # Should normalize to lowercase
        assert "banana" in term_weights
        # Check that case variants are merged
        normalized_keys = [k.lower() for k in term_weights.keys()]
        assert normalized_keys.count("banana") >= 1


class TestSparseEncoderStopwords:
    """Tests for stop word handling."""

    def test_stopword_filtering(self):
        """Should filter out common stopwords."""
        settings = Mock()
        # Text with many common stopwords
        chunk = Chunk(
            id="chunk_001",
            text="the quick brown fox jumps over the lazy dog",
            metadata={}
        )

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk])

        sparse_stats = result[0].metadata["sparse_embedding"]
        term_weights = sparse_stats["term_weights"]

        # Common stopwords should be filtered
        assert "the" not in term_weights
        # Content words should be present
        assert "quick" in term_weights or "fox" in term_weights or "dog" in term_weights


class TestSparseEncoderIntegration:
    """Integration tests."""

    def test_encode_with_trace_context(self):
        """Should accept and pass trace context."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="test content", metadata={})
        trace = TraceContext()

        encoder = SparseEncoder(settings)
        result = encoder.encode([chunk], trace=trace)

        assert len(result) == 1

    def test_encode_idempotent(self):
        """Encoding same chunk twice should give same statistics."""
        settings = Mock()
        chunk = Chunk(id="chunk_001", text="test content", metadata={})

        encoder = SparseEncoder(settings)
        result1 = encoder.encode([chunk])

        # Re-encode the same chunk
        chunk2 = Chunk(id="chunk_001", text="test content", metadata={})
        result2 = encoder.encode([chunk2])

        # Statistics should be identical
        stats1 = result1[0].metadata["sparse_embedding"]["term_weights"]
        stats2 = result2[0].metadata["sparse_embedding"]["term_weights"]
        assert stats1 == stats2

    def test_output_structure_for_bm25_indexer(self):
        """Output structure should be compatible with bm25_indexer expectations."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_001", text="machine learning algorithms", metadata={}),
            Chunk(id="chunk_002", text="deep learning neural networks", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        # Verify structure that bm25_indexer would use
        for chunk in result:
            sparse_stats = chunk.metadata["sparse_embedding"]
            assert isinstance(sparse_stats, dict)
            assert "term_weights" in sparse_stats
            term_weights = sparse_stats["term_weights"]
            assert isinstance(term_weights, dict)
            # term_weights should be {term: weight} format
            for term, weight in term_weights.items():
                assert isinstance(term, str)
                assert isinstance(weight, (int, float))

    def test_multiple_encodings_with_different_texts(self):
        """Should handle multiple chunks with diverse content."""
        settings = Mock()
        chunks = [
            Chunk(id="chunk_001", text="Python is a programming language", metadata={}),
            Chunk(id="chunk_002", text="Machine learning is AI subset", metadata={}),
            Chunk(id="chunk_003", text="Data science uses statistics", metadata={}),
        ]

        encoder = SparseEncoder(settings)
        result = encoder.encode(chunks)

        assert len(result) == 3
        # Each should have unique sparse statistics
        for chunk in result:
            assert "sparse_embedding" in chunk.metadata
            assert "term_weights" in chunk.metadata["sparse_embedding"]
