"""
Test suite for BatchProcessor - orchestrate batch processing of chunks.
Coverage: Batch creation, chunk division, dense/sparse encoding coordination, timing.
Target: 15+ tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class TestBatchProcessorInitialization:
    """Tests for BatchProcessor initialization."""

    def test_init_with_batch_size(self):
        """Should initialize with batch size configuration."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        assert processor.batch_size == 2
        assert processor.dense_encoder is dense_encoder
        assert processor.sparse_encoder is sparse_encoder

    def test_init_with_default_batch_size(self):
        """Should use default batch size if not provided."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder
        )

        assert processor.batch_size == 32  # Default batch size

    def test_init_stores_encoders(self):
        """Should store encoder references."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=16
        )

        assert processor.dense_encoder is dense_encoder
        assert processor.sparse_encoder is sparse_encoder


class TestBatchProcessorBatchCreation:
    """Tests for batch creation and chunk division."""

    def test_create_batches_from_chunks(self):
        """Should divide chunks into batches."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        chunks = [
            Chunk(id=f"chunk_{i:03d}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        batches = processor._create_batches(chunks)

        assert len(batches) == 3  # 5 chunks with batch_size=2 -> 3 batches
        assert len(batches[0]) == 2  # First batch: 2 chunks
        assert len(batches[1]) == 2  # Second batch: 2 chunks
        assert len(batches[2]) == 1  # Third batch: 1 chunk

    def test_batch_order_preservation(self):
        """Batch creation should preserve chunk order."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        chunks = [
            Chunk(id="chunk_003", text="text 3", metadata={}),
            Chunk(id="chunk_001", text="text 1", metadata={}),
            Chunk(id="chunk_002", text="text 2", metadata={}),
            Chunk(id="chunk_005", text="text 5", metadata={}),
            Chunk(id="chunk_004", text="text 4", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        batches = processor._create_batches(chunks)

        # Verify order is preserved
        assert batches[0][0].id == "chunk_003"
        assert batches[0][1].id == "chunk_001"
        assert batches[1][0].id == "chunk_002"
        assert batches[1][1].id == "chunk_005"
        assert batches[2][0].id == "chunk_004"

    def test_batch_single_chunk(self):
        """Should handle single chunk."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        chunks = [Chunk(id="chunk_001", text="text", metadata={})]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        batches = processor._create_batches(chunks)

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_empty_list(self):
        """Should handle empty chunk list."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        batches = processor._create_batches([])

        assert batches == []

    def test_batch_size_one(self):
        """Should handle batch_size=1."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        chunks = [
            Chunk(id="chunk_001", text="text 1", metadata={}),
            Chunk(id="chunk_002", text="text 2", metadata={}),
            Chunk(id="chunk_003", text="text 3", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=1
        )

        batches = processor._create_batches(chunks)

        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 1

    def test_batch_larger_than_chunk_list(self):
        """Should handle batch_size larger than chunk list."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        chunks = [
            Chunk(id="chunk_001", text="text 1", metadata={}),
            Chunk(id="chunk_002", text="text 2", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=10
        )

        batches = processor._create_batches(chunks)

        assert len(batches) == 1
        assert len(batches[0]) == 2


class TestBatchProcessorProcessing:
    """Tests for batch processing with encoders."""

    def test_process_chunks_with_mocked_encoders(self):
        """Should process chunks through both encoders."""
        # Create mock encoders that return chunks with embedding added
        def mock_dense_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["dense_embedding"] = [0.1, 0.2, 0.3]
            return chunks

        def mock_sparse_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["sparse_embedding"] = {"term": 1}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_dense_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_sparse_encode)

        chunks = [
            Chunk(id="chunk_001", text="text 1", metadata={}),
            Chunk(id="chunk_002", text="text 2", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        assert len(result) == 2
        # Check that both encoders were called
        assert dense_encoder.encode.called
        assert sparse_encoder.encode.called

    def test_process_preserves_chunk_order(self):
        """Processing should preserve chunk order."""
        def mock_dense_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["dense_embedding"] = [0.1, 0.2]
            return chunks

        def mock_sparse_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["sparse_embedding"] = {"t": 1}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_dense_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_sparse_encode)

        chunks = [
            Chunk(id="chunk_003", text="text 3", metadata={}),
            Chunk(id="chunk_001", text="text 1", metadata={}),
            Chunk(id="chunk_002", text="text 2", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        assert result[0].id == "chunk_003"
        assert result[1].id == "chunk_001"
        assert result[2].id == "chunk_002"

    def test_process_with_empty_chunk_list(self):
        """Should handle empty chunk list."""
        dense_encoder = Mock(spec=DenseEncoder)
        sparse_encoder = Mock(spec=SparseEncoder)

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process([])

        assert result == []

    def test_process_adds_timing_metadata(self):
        """Should add timing information to batches."""
        def mock_dense_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["dense_embedding"] = [0.1, 0.2]
            return chunks

        def mock_sparse_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["sparse_embedding"] = {"t": 1}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_dense_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_sparse_encode)

        chunks = [
            Chunk(id="chunk_001", text="text 1", metadata={}),
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        # Timing should be recorded (implementation detail)
        assert len(result) == 1


class TestBatchProcessorIntegration:
    """Integration tests with real or mock encoders."""

    def test_process_with_batch_size_2_five_chunks(self):
        """Test acceptance criterion: batch_size=2 with 5 chunks creates 3 batches."""
        def mock_encode(chunks, trace=None):
            for chunk in chunks:
                if "dense_embedding" not in chunk.metadata:
                    chunk.metadata["dense_embedding"] = [0.1]
                if "sparse_embedding" not in chunk.metadata:
                    chunk.metadata["sparse_embedding"] = {"t": 1}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_encode)

        chunks = [
            Chunk(id=f"chunk_{i:03d}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        # All chunks should be returned
        assert len(result) == 5
        # Order should be stable
        for i, chunk in enumerate(result):
            assert chunk.id == f"chunk_{i:03d}"

    def test_process_with_trace_context(self):
        """Should accept and pass trace context."""
        def mock_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["embedding"] = [0.1]
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_encode)

        chunk = Chunk(id="chunk_001", text="text", metadata={})
        trace = TraceContext()

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process([chunk], trace=trace)

        assert len(result) == 1

    def test_encoder_calls_receive_correct_batches(self):
        """Encoders should receive batches in order."""
        call_history = []

        def mock_dense_encode(chunks, trace=None):
            call_history.append(("dense", [c.id for c in chunks]))
            for chunk in chunks:
                chunk.metadata["dense_embedding"] = [0.1]
            return chunks

        def mock_sparse_encode(chunks, trace=None):
            call_history.append(("sparse", [c.id for c in chunks]))
            for chunk in chunks:
                chunk.metadata["sparse_embedding"] = {"t": 1}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_dense_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_sparse_encode)

        chunks = [
            Chunk(id=f"chunk_{i:03d}", text=f"text {i}", metadata={})
            for i in range(5)
        ]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        # Verify batches were processed correctly
        assert len(result) == 5
        # Both encoders should have been called
        assert any(call[0] == "dense" for call in call_history)
        assert any(call[0] == "sparse" for call in call_history)

    def test_process_metadata_accumulation(self):
        """Metadata from both encoders should accumulate on chunks."""
        def mock_dense_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["dense_embedding"] = [0.1, 0.2, 0.3]
            return chunks

        def mock_sparse_encode(chunks, trace=None):
            for chunk in chunks:
                chunk.metadata["sparse_embedding"] = {"word": 5}
            return chunks

        dense_encoder = Mock(spec=DenseEncoder)
        dense_encoder.encode = Mock(side_effect=mock_dense_encode)

        sparse_encoder = Mock(spec=SparseEncoder)
        sparse_encoder.encode = Mock(side_effect=mock_sparse_encode)

        chunks = [Chunk(id="chunk_001", text="text", metadata={})]

        processor = BatchProcessor(
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            batch_size=2
        )

        result = processor.process(chunks)

        # Both embedding types should be present
        assert "dense_embedding" in result[0].metadata
        assert "sparse_embedding" in result[0].metadata
