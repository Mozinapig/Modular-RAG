"""
Test suite for DenseEncoder - batch embedding chunks with image caption fusion.
Coverage: Initialization, encoding, dimension validation, caption fusion, edge cases, integration.
Target: 18+ tests.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder


class TestDenseEncoderInitialization:
    """Tests for DenseEncoder initialization."""

    def test_init_with_settings(self):
        """Should initialize with settings and create embedding client."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 1536

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.1, 0.2]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)

        assert encoder.embedding_client is not None
        assert encoder.embedding_dim == 1536

    def test_init_with_dependency_injection(self):
        """Should accept injected embedding client for testing."""
        settings = Mock()
        mock_embedding = Mock()
        mock_embedding.embed = Mock(return_value=[[0.1]])

        encoder = DenseEncoder(settings, embedding_client=mock_embedding)

        assert encoder.embedding_client == mock_embedding

    def test_init_extracts_dimension_from_settings(self):
        """Should extract embedding dimension from settings."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 768

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)

        assert encoder.embedding_dim == 768


class TestDenseEncoderBasicEncoding:
    """Tests for basic encoding functionality."""

    def test_encode_single_chunk(self):
        """Should encode a single chunk and return it with embedding."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 3

        chunks = [
            Chunk(id="chunk_001", text="Hello world", metadata={"source": "test.pdf"})
        ]

        embedding_vector = [[0.1, 0.2, 0.3]]

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=embedding_vector)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode(chunks)

        assert len(result) == 1
        assert result[0].id == "chunk_001"
        assert "dense_embedding" in result[0].metadata
        assert result[0].metadata["dense_embedding"] == [0.1, 0.2, 0.3]

    def test_encode_multiple_chunks(self):
        """Should encode multiple chunks and return all with embeddings."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunks = [
            Chunk(id="chunk_001", text="First text", metadata={}),
            Chunk(id="chunk_002", text="Second text", metadata={}),
            Chunk(id="chunk_003", text="Third text", metadata={}),
        ]

        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=embeddings)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode(chunks)

        assert len(result) == 3
        assert result[0].metadata["dense_embedding"] == [0.1, 0.2]
        assert result[1].metadata["dense_embedding"] == [0.3, 0.4]
        assert result[2].metadata["dense_embedding"] == [0.5, 0.6]

    def test_encode_preserves_chunk_identity(self):
        """Should preserve chunk id, text, and other metadata."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(
            id="chunk_001",
            text="Original text",
            metadata={"source": "doc.pdf", "page": 1}
        )

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.1, 0.2]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert result[0].id == "chunk_001"
        assert result[0].text == "Original text"
        assert result[0].metadata["source"] == "doc.pdf"
        assert result[0].metadata["page"] == 1


class TestDenseEncoderDimensionValidation:
    """Tests for embedding dimension validation."""

    def test_embedding_dimension_matches_config(self):
        """Embedding vector dimension should match configured dimension."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 1536

        chunk = Chunk(id="chunk_001", text="test", metadata={})

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            # Create a proper 1536-dimensional vector
            proper_vector = [[i * 0.001 for i in range(1536)]]
            mock_embedding.embed = Mock(return_value=proper_vector)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert len(result[0].metadata["dense_embedding"]) == 1536

    def test_output_count_matches_input_count(self):
        """Number of embeddings should equal number of input chunks."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 5

        chunks = [
            Chunk(id=f"chunk_{i:03d}", text=f"Text {i}", metadata={})
            for i in range(10)
        ]

        embeddings = [
            [0.1 * (i + 1)] * 5 for i in range(10)
        ]

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=embeddings)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode(chunks)

        assert len(result) == 10


class TestDenseEncoderCaptionFusion:
    """Tests for fusing image captions into text during encoding."""

    def test_fuse_image_captions_into_embedding_text(self):
        """Should fuse image_captions into text when encoding."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 3

        chunk = Chunk(
            id="chunk_001",
            text="Original text with [IMAGE CAPTION: diagram of architecture]",
            metadata={
                "image_captions": {"img_001": "diagram of architecture"}
            }
        )

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            captured_texts = []

            def capture_texts(texts, trace=None):
                captured_texts.extend(texts)
                return [[0.1, 0.2, 0.3]] * len(texts)

            mock_embedding.embed = Mock(side_effect=capture_texts)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        # Verify the captured text includes both original and caption
        assert len(captured_texts) == 1
        assert "Original text" in captured_texts[0]
        assert "diagram of architecture" in captured_texts[0]

    def test_chunk_without_captions_embed_original_text(self):
        """Chunks without image_captions should embed original text."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(
            id="chunk_001",
            text="Just original text",
            metadata={}
        )

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            captured_texts = []

            def capture_texts(texts, trace=None):
                captured_texts.extend(texts)
                return [[0.1, 0.2]] * len(texts)

            mock_embedding.embed = Mock(side_effect=capture_texts)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert captured_texts[0] == "Just original text"

    def test_fuse_multiple_captions(self):
        """Should fuse multiple image captions if present."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(
            id="chunk_001",
            text="Text with [IMAGE CAPTION: first] and [IMAGE CAPTION: second]",
            metadata={
                "image_captions": {
                    "img_001": "first",
                    "img_002": "second"
                }
            }
        )

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            captured_texts = []

            def capture_texts(texts, trace=None):
                captured_texts.extend(texts)
                return [[0.1, 0.2]] * len(texts)

            mock_embedding.embed = Mock(side_effect=capture_texts)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert "first" in captured_texts[0]
        assert "second" in captured_texts[0]


class TestDenseEncoderEdgeCases:
    """Edge case tests."""

    def test_empty_chunk_list(self):
        """Empty chunk list should return empty result."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([])

        assert result == []
        # Should not call embedding on empty list
        mock_embedding.embed.assert_not_called()

    def test_chunk_with_empty_text(self):
        """Chunk with empty text should still be encoded."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(id="chunk_001", text="", metadata={})

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.0, 0.0]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert len(result) == 1
        assert "dense_embedding" in result[0].metadata

    def test_chunk_with_very_long_text(self):
        """Should handle chunks with very long text."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 3

        long_text = "word " * 10000  # Very long text
        chunk = Chunk(id="chunk_001", text=long_text, metadata={})

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk])

        assert len(result) == 1
        assert len(result[0].metadata["dense_embedding"]) == 3

    def test_preserves_chunk_order(self):
        """Chunk order should be preserved after encoding."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunks = [
            Chunk(id="chunk_003", text="Third", metadata={}),
            Chunk(id="chunk_001", text="First", metadata={}),
            Chunk(id="chunk_002", text="Second", metadata={}),
        ]

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=embeddings)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode(chunks)

        assert result[0].id == "chunk_003"
        assert result[1].id == "chunk_001"
        assert result[2].id == "chunk_002"


class TestDenseEncoderIntegration:
    """Integration tests."""

    def test_encode_with_trace_context(self):
        """Should accept and pass trace context."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(id="chunk_001", text="test", metadata={})
        trace = TraceContext()

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.1, 0.2]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result = encoder.encode([chunk], trace=trace)

        assert len(result) == 1

    def test_encode_idempotent(self):
        """Encoding same chunk twice should give same result."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunk = Chunk(id="chunk_001", text="test", metadata={})

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()
            mock_embedding.embed = Mock(return_value=[[0.1, 0.2]])
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            result1 = encoder.encode([chunk])
            result2 = encoder.encode(result1)

        assert result1[0].metadata["dense_embedding"] == result2[0].metadata["dense_embedding"]

    def test_embedding_call_receives_correct_texts(self):
        """Embedding client should receive correct text list."""
        settings = Mock()
        settings.embedding = Mock()
        settings.embedding.dimensions = 2

        chunks = [
            Chunk(id="chunk_001", text="First chunk", metadata={}),
            Chunk(id="chunk_002", text="Second chunk", metadata={}),
        ]

        received_texts = []

        with patch('src.ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_embedding = Mock()

            def capture_embed(texts, trace=None):
                received_texts.append(texts)
                return [[0.1, 0.2]] * len(texts)

            mock_embedding.embed = Mock(side_effect=capture_embed)
            mock_factory.return_value.create = Mock(return_value=mock_embedding)

            encoder = DenseEncoder(settings)
            encoder.encode(chunks)

        assert len(received_texts) == 1
        assert received_texts[0] == ["First chunk", "Second chunk"]
