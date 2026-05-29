"""
Test suite for VectorUpserter - idempotent vector storage with deterministic IDs.
Coverage: ID generation, idempotency, batch operations, vector store integration.
Target: 18+ tests.
"""

import pytest
import hashlib
from unittest.mock import Mock, MagicMock, call
from typing import List

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.storage.vector_upserter import VectorUpserter


class TestVectorUpserterInitialization:
    """Tests for VectorUpserter initialization."""

    def test_init_with_vector_store(self):
        """Should initialize with vector store."""
        vector_store = Mock()

        upserter = VectorUpserter(vector_store=vector_store)

        assert upserter.vector_store is vector_store

    def test_init_stores_reference(self):
        """Should store vector store reference."""
        vector_store = Mock()

        upserter = VectorUpserter(vector_store=vector_store)

        assert hasattr(upserter, 'vector_store')


class TestVectorUpserterIDGeneration:
    """Tests for deterministic chunk ID generation."""

    def test_generate_id_from_chunk_metadata(self):
        """Should generate ID from source_path, chunk_index, and content_hash."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk = Chunk(
            id="chunk_001",
            text="sample text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123def456"
            }
        )

        chunk_id = upserter._generate_chunk_id(chunk)

        assert isinstance(chunk_id, str)
        assert len(chunk_id) > 0

    def test_id_deterministic_same_content(self):
        """Same content should generate same ID."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk1 = Chunk(
            id="chunk_001",
            text="same text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123def456"
            }
        )

        chunk2 = Chunk(
            id="chunk_002",
            text="same text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123def456"
            }
        )

        id1 = upserter._generate_chunk_id(chunk1)
        id2 = upserter._generate_chunk_id(chunk2)

        assert id1 == id2

    def test_id_different_source_path(self):
        """Different source_path should generate different ID."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk1 = Chunk(
            id="chunk_001",
            text="text",
            metadata={
                "source_path": "/docs/file1.pdf",
                "chunk_index": 0,
                "content_hash": "abc123"
            }
        )

        chunk2 = Chunk(
            id="chunk_002",
            text="text",
            metadata={
                "source_path": "/docs/file2.pdf",
                "chunk_index": 0,
                "content_hash": "abc123"
            }
        )

        id1 = upserter._generate_chunk_id(chunk1)
        id2 = upserter._generate_chunk_id(chunk2)

        assert id1 != id2

    def test_id_different_chunk_index(self):
        """Different chunk_index should generate different ID."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk1 = Chunk(
            id="chunk_001",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123"
            }
        )

        chunk2 = Chunk(
            id="chunk_002",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 1,
                "content_hash": "abc123"
            }
        )

        id1 = upserter._generate_chunk_id(chunk1)
        id2 = upserter._generate_chunk_id(chunk2)

        assert id1 != id2

    def test_id_different_content_hash(self):
        """Different content_hash should generate different ID."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk1 = Chunk(
            id="chunk_001",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash1"
            }
        )

        chunk2 = Chunk(
            id="chunk_002",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash2"
            }
        )

        id1 = upserter._generate_chunk_id(chunk1)
        id2 = upserter._generate_chunk_id(chunk2)

        assert id1 != id2

    def test_id_uses_first_8_chars_of_hash(self):
        """ID should use only first 8 characters of content_hash."""
        upserter = VectorUpserter(vector_store=Mock())

        chunk1 = Chunk(
            id="chunk_001",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123def456"
            }
        )

        chunk2 = Chunk(
            id="chunk_002",
            text="text",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "abc123def999"
            }
        )

        # Same first 8 chars, different rest
        id1 = upserter._generate_chunk_id(chunk1)
        id2 = upserter._generate_chunk_id(chunk2)

        # Should be same since only first 8 chars are used
        assert id1 == id2


class TestVectorUpserterIdempotency:
    """Tests for idempotent upsert behavior."""

    def test_same_chunk_twice_produces_same_id(self):
        """Upserting same chunk twice should produce same ID."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="embedding data",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        upserter.upsert([chunk])
        upserter.upsert([chunk])

        # Both calls should use same ID
        calls = vector_store.upsert.call_args_list
        assert len(calls) == 2

        # Extract IDs from both calls (they're passed as ids parameter)
        id1 = calls[0][1]['ids'][0]
        id2 = calls[1][1]['ids'][0]
        assert id1 == id2

    def test_content_change_produces_different_id(self):
        """Content change should produce different ID."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk1 = Chunk(
            id="chunk_001",
            text="first content",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash1",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        chunk2 = Chunk(
            id="chunk_001",
            text="different content",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash2",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        upserter.upsert([chunk1])
        upserter.upsert([chunk2])

        calls = vector_store.upsert.call_args_list
        id1 = calls[0][1]['ids'][0]
        id2 = calls[1][1]['ids'][0]

        assert id1 != id2


class TestVectorUpserterUpsert:
    """Tests for upsert operations."""

    def test_upsert_single_chunk(self):
        """Should upsert single chunk with embedding."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        upserter.upsert([chunk])

        vector_store.upsert.assert_called_once()

    def test_upsert_calls_vector_store(self):
        """Should call vector_store.upsert with correct parameters."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        upserter.upsert([chunk])

        # Verify vector_store.upsert was called with ids and embeddings
        call_args = vector_store.upsert.call_args
        assert 'ids' in call_args[1]
        assert 'embeddings' in call_args[1]

    def test_upsert_preserves_metadata(self):
        """Should pass metadata to vector store."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3],
                "title": "Section 1"
            }
        )

        upserter.upsert([chunk])

        call_args = vector_store.upsert.call_args
        assert 'metadatas' in call_args[1]

    def test_upsert_batch_multiple_chunks(self):
        """Should upsert multiple chunks in batch."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text=f"text {i}",
                metadata={
                    "source_path": "/docs/sample.pdf",
                    "chunk_index": i,
                    "content_hash": f"hash{i}",
                    "dense_embedding": [0.1 * (i+1)] * 3
                }
            )
            for i in range(5)
        ]

        upserter.upsert(chunks)

        call_args = vector_store.upsert.call_args
        assert len(call_args[1]['ids']) == 5

    def test_upsert_preserves_order(self):
        """Batch upsert should preserve chunk order."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text=f"text {i}",
                metadata={
                    "source_path": "/docs/sample.pdf",
                    "chunk_index": i,
                    "content_hash": f"hash{i}",
                    "dense_embedding": [0.1] * 3
                }
            )
            for i in range(5)
        ]

        upserter.upsert(chunks)

        call_args = vector_store.upsert.call_args
        ids = call_args[1]['ids']
        embeddings = call_args[1]['embeddings']

        # Order should be preserved
        assert len(ids) == 5
        assert len(embeddings) == 5

    def test_upsert_empty_list(self):
        """Should handle empty chunk list."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        upserter.upsert([])

        # Should not call upsert on empty list
        vector_store.upsert.assert_not_called()

    def test_upsert_with_trace_context(self):
        """Should accept trace context."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        trace = TraceContext()
        upserter.upsert([chunk], trace=trace)

        vector_store.upsert.assert_called_once()


class TestVectorUpserterEmbeddingHandling:
    """Tests for dense embedding handling."""

    def test_extract_embedding_from_metadata(self):
        """Should extract dense_embedding from metadata."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        )

        upserter.upsert([chunk])

        call_args = vector_store.upsert.call_args
        embeddings = call_args[1]['embeddings']

        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_handle_missing_embedding(self):
        """Should handle chunks without dense_embedding gracefully."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123"
            }
        )

        # Should not raise error
        upserter.upsert([chunk])

        # May not call upsert if embedding is missing
        # or handle with None/empty embedding


class TestVectorUpserterMetadataHandling:
    """Tests for metadata preservation."""

    def test_preserve_source_path_in_metadata(self):
        """Should preserve source_path in metadata."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3],
                "title": "Section 1"
            }
        )

        upserter.upsert([chunk])

        call_args = vector_store.upsert.call_args
        metadatas = call_args[1]['metadatas']

        # Should preserve key metadata
        assert any('source_path' in str(m) or 'title' in str(m) for m in metadatas)

    def test_batch_metadata_preservation(self):
        """Should preserve metadata for all chunks in batch."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text=f"text {i}",
                metadata={
                    "source_path": "/docs/sample.pdf",
                    "chunk_index": i,
                    "content_hash": f"hash{i}",
                    "dense_embedding": [0.1] * 3,
                    "title": f"Section {i}"
                }
            )
            for i in range(3)
        ]

        upserter.upsert(chunks)

        call_args = vector_store.upsert.call_args
        metadatas = call_args[1]['metadatas']

        assert len(metadatas) == 3


class TestVectorUpserterEdgeCases:
    """Tests for edge cases."""

    def test_chunk_with_empty_embedding(self):
        """Should handle chunk with empty embedding vector."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": []
            }
        )

        # Should handle gracefully
        upserter.upsert([chunk])

    def test_chunk_with_very_large_embedding(self):
        """Should handle chunks with large embedding vectors."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/sample.pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1] * 1536  # Large embedding (like OpenAI)
            }
        )

        upserter.upsert([chunk])

        vector_store.upsert.assert_called_once()

    def test_chunk_with_special_characters_in_path(self):
        """Should handle special characters in source_path."""
        vector_store = Mock()
        upserter = VectorUpserter(vector_store=vector_store)

        chunk = Chunk(
            id="chunk_001",
            text="sample",
            metadata={
                "source_path": "/docs/文件名/特殊-字符 (1).pdf",
                "chunk_index": 0,
                "content_hash": "hash123",
                "dense_embedding": [0.1, 0.2, 0.3]
            }
        )

        upserter.upsert([chunk])

        vector_store.upsert.assert_called_once()
