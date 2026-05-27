"""Integration tests for ChromaStore roundtrip (upsert + query)."""

import tempfile
import shutil
from pathlib import Path
import pytest

from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.base_vector_store import VectorRecord
from src.core.settings import Settings


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma database."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def settings_with_chroma(temp_chroma_dir):
    """Create settings with temporary Chroma path."""
    settings = Settings(
        llm={"provider": "openai", "model": "gpt-4"},
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        vector_store={"backend": "chroma", "persist_path": temp_chroma_dir},
    )
    return settings


class TestChromaStoreRoundtrip:
    """Test complete upsert -> query roundtrip."""

    def test_upsert_and_query_basic(self, settings_with_chroma):
        """Test basic upsert and query functionality."""
        store = ChromaStore(settings_with_chroma)

        # Create test records
        records = [
            VectorRecord(
                id="doc1_chunk1",
                text="Machine learning is a branch of AI",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": "doc1.pdf", "page": 1}
            ),
            VectorRecord(
                id="doc1_chunk2",
                text="Deep learning uses neural networks",
                embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
                metadata={"source": "doc1.pdf", "page": 2}
            ),
            VectorRecord(
                id="doc2_chunk1",
                text="Python is a programming language",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                metadata={"source": "doc2.pdf", "page": 1}
            ),
        ]

        # Upsert records
        store.upsert(records)

        # Query with a vector similar to first record
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = store.query(query_vector, top_k=2)

        # Verify results
        assert len(results) == 2
        assert results[0].id == "doc1_chunk1"
        assert results[0].text == "Machine learning is a branch of AI"
        assert results[0].metadata["source"] == "doc1.pdf"

    def test_query_with_top_k(self, settings_with_chroma):
        """Test top_k parameter works correctly."""
        store = ChromaStore(settings_with_chroma)

        records = [
            VectorRecord(id=f"chunk_{i}", text=f"text_{i}", embedding=[float(i)] * 5, metadata={"index": i})
            for i in range(10)
        ]
        store.upsert(records)

        # Query with different top_k values
        results_5 = store.query([1.0, 1.0, 1.0, 1.0, 1.0], top_k=5)
        results_3 = store.query([1.0, 1.0, 1.0, 1.0, 1.0], top_k=3)

        assert len(results_5) == 5
        assert len(results_3) == 3

    def test_upsert_idempotency(self, settings_with_chroma):
        """Test that upserting same record twice produces consistent results."""
        store = ChromaStore(settings_with_chroma)

        record = VectorRecord(
            id="test_id",
            text="Test text",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"key": "value"}
        )

        # Upsert twice
        store.upsert([record])
        store.upsert([record])

        # Query should return only one record
        results = store.query([0.1, 0.2, 0.3, 0.4, 0.5], top_k=10)
        matching = [r for r in results if r.id == "test_id"]
        assert len(matching) == 1

    def test_metadata_filtering(self, settings_with_chroma):
        """Test metadata filtering in queries."""
        store = ChromaStore(settings_with_chroma)

        records = [
            VectorRecord(
                id="doc1_chunk1",
                text="Text from doc1",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": "doc1.pdf", "doc_type": "technical"}
            ),
            VectorRecord(
                id="doc2_chunk1",
                text="Text from doc2",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": "doc2.pdf", "doc_type": "blog"}
            ),
            VectorRecord(
                id="doc3_chunk1",
                text="Text from doc3",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": "doc3.pdf", "doc_type": "technical"}
            ),
        ]
        store.upsert(records)

        # Query with filter
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = store.query(query_vector, top_k=10, filters={"doc_type": "technical"})

        # All results should have doc_type="technical"
        for result in results:
            assert result.metadata.get("doc_type") == "technical"

    def test_empty_query_results(self, settings_with_chroma):
        """Test behavior with filters that match no records."""
        store = ChromaStore(settings_with_chroma)

        records = [
            VectorRecord(
                id="chunk1",
                text="Text 1",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"type": "A"}
            ),
        ]
        store.upsert(records)

        # Query with filter that doesn't match
        results = store.query([0.1, 0.2, 0.3, 0.4, 0.5], top_k=10, filters={"type": "B"})
        assert len(results) == 0

    def test_persistence(self, temp_chroma_dir):
        """Test that data persists across store instances."""
        settings = Settings(
            llm={"provider": "openai", "model": "gpt-4"},
            embedding={"provider": "openai", "model": "text-embedding-3-small"},
            vector_store={"backend": "chroma", "persist_path": temp_chroma_dir},
        )

        # Store 1: Upsert data
        store1 = ChromaStore(settings)
        record = VectorRecord(
            id="persistent_test",
            text="This should persist",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"persistent": True}
        )
        store1.upsert([record])

        # Store 2: Query data from same directory
        store2 = ChromaStore(settings)
        results = store2.query([0.1, 0.2, 0.3, 0.4, 0.5], top_k=10)

        # Should find the record
        matching = [r for r in results if r.id == "persistent_test"]
        assert len(matching) == 1
        assert matching[0].metadata["persistent"] is True

    def test_multiple_records_same_batch(self, settings_with_chroma):
        """Test upserting and querying multiple records in one batch."""
        store = ChromaStore(settings_with_chroma)

        # Create batch of records
        batch_size = 20
        records = [
            VectorRecord(
                id=f"batch_doc_{i}",
                text=f"Document {i}",
                embedding=[float(i % 10) * 0.1 + 0.05, 0.2, 0.3, 0.4, 0.5],
                metadata={"batch": 1, "index": i}
            )
            for i in range(batch_size)
        ]

        store.upsert(records)

        # Query and verify all records are stored
        query_vector = [0.55, 0.2, 0.3, 0.4, 0.5]
        results = store.query(query_vector, top_k=batch_size)

        assert len(results) >= 10  # Should get at least some results

    def test_validate_config(self, settings_with_chroma):
        """Test configuration validation."""
        store = ChromaStore(settings_with_chroma)
        # Should not raise
        store.validate_config()

    def test_query_vector_dimension_consistency(self, settings_with_chroma):
        """Test that vectors with consistent dimensions work correctly."""
        store = ChromaStore(settings_with_chroma)

        # All vectors are 5-dimensional
        records = [
            VectorRecord(
                id="dim_test_1",
                text="Text 1",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={}
            ),
            VectorRecord(
                id="dim_test_2",
                text="Text 2",
                embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
                metadata={}
            ),
        ]

        store.upsert(records)

        # Query with matching dimensions
        results = store.query([0.1, 0.2, 0.3, 0.4, 0.5], top_k=2)
        assert len(results) == 2
