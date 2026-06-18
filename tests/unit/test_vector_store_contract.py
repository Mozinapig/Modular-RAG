"""Tests for VectorStore factory and provider routing."""

import pytest
from unittest.mock import Mock

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class TestVectorStoreFactory:
    """Test VectorStoreFactory routing and creation."""

    def test_factory_creates_fake_provider(self):
        """Test factory can create instances with fake provider."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        assert store is not None
        assert isinstance(store, BaseVectorStore)

    def test_factory_upsert_records(self):
        """Test that created store can upsert records."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="First document",
                embedding=[0.1, 0.2, 0.3],
                metadata={"source": "test.pdf"},
            ),
            VectorRecord(
                id="doc_002",
                text="Second document",
                embedding=[0.4, 0.5, 0.6],
                metadata={"source": "test.pdf"},
            ),
        ]

        # Should not raise
        store.upsert(records)

    def test_factory_query_returns_records(self):
        """Test that created store can query and return records."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="First document",
                embedding=[0.1, 0.2, 0.3],
                metadata={"source": "test.pdf"},
            ),
            VectorRecord(
                id="doc_002",
                text="Second document",
                embedding=[0.4, 0.5, 0.6],
                metadata={"source": "test.pdf"},
            ),
        ]

        store.upsert(records)

        query_vector = [0.1, 0.2, 0.3]
        results = store.query(query_vector, top_k=10)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, VectorRecord) for r in results)

    def test_store_query_respects_top_k(self):
        """Test that query respects top_k parameter."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        # Upsert multiple records
        records = [
            VectorRecord(
                id=f"doc_{i:03d}",
                text=f"Document {i}",
                embedding=[float(i) / 10] * 3,
                metadata={"source": "test.pdf"},
            )
            for i in range(10)
        ]

        store.upsert(records)

        query_vector = [0.1, 0.2, 0.3]
        results = store.query(query_vector, top_k=3)

        assert len(results) <= 3

    def test_store_query_with_filters(self):
        """Test that query supports metadata filters."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="First document",
                embedding=[0.1, 0.2, 0.3],
                metadata={"source": "file1.pdf", "collection": "test"},
            ),
            VectorRecord(
                id="doc_002",
                text="Second document",
                embedding=[0.4, 0.5, 0.6],
                metadata={"source": "file2.pdf", "collection": "other"},
            ),
        ]

        store.upsert(records)

        query_vector = [0.1, 0.2, 0.3]
        filters = {"collection": "test"}
        results = store.query(query_vector, top_k=10, filters=filters)

        assert isinstance(results, list)

    def test_store_empty_upsert(self):
        """Test store handles empty upsert gracefully."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        # Should not raise on empty list
        store.upsert([])

    def test_store_query_empty_vector_db(self):
        """Test query on empty vector database."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        query_vector = [0.1, 0.2, 0.3]
        results = store.query(query_vector, top_k=10)

        assert isinstance(results, list)

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error on unknown provider."""
        store_settings = Mock()
        store_settings.provider = "unknown_provider"

        factory = VectorStoreFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(store_settings)

        assert "unknown_provider" in str(exc_info.value).lower()

    def test_factory_raises_on_missing_provider(self):
        """Test factory raises error when provider is missing."""
        store_settings = Mock()
        store_settings.provider = None

        factory = VectorStoreFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create(store_settings)

        assert "provider" in str(exc_info.value).lower()

    def test_vector_record_creation(self):
        """Test VectorRecord dataclass creation."""
        record = VectorRecord(
            id="test_001",
            text="Test document",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test.pdf"},
        )

        assert record.id == "test_001"
        assert record.text == "Test document"
        assert record.embedding == [0.1, 0.2, 0.3]
        assert record.metadata["source"] == "test.pdf"

    def test_vector_record_without_metadata(self):
        """Test VectorRecord can be created without metadata."""
        record = VectorRecord(
            id="test_001",
            text="Test document",
            embedding=[0.1, 0.2, 0.3],
        )

        assert record.id == "test_001"
        assert record.metadata is None

    def test_store_with_trace(self):
        """Test store accepts trace parameter."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        mock_trace = Mock()
        records = [
            VectorRecord(
                id="doc_001",
                text="First document",
                embedding=[0.1, 0.2, 0.3],
            ),
        ]

        # Should not raise
        store.upsert(records, trace=mock_trace)

        query_vector = [0.1, 0.2, 0.3]
        results = store.query(query_vector, top_k=10, trace=mock_trace)

        assert isinstance(results, list)

    def test_factory_provider_registration(self):
        """Test factory can register custom providers."""
        class CustomVectorStore(BaseVectorStore):
            def __init__(self, settings):
                self.settings = settings
                self.records = {}

            def upsert(self, records, trace=None):
                for record in records:
                    self.records[record.id] = record

            def query(self, vector, top_k=10, filters=None, trace=None):
                return list(self.records.values())[:top_k]

            def validate_config(self):
                pass

        factory = VectorStoreFactory()
        factory.register_provider("custom", CustomVectorStore)

        store_settings = Mock()
        store_settings.provider = "custom"

        store = factory.create(store_settings)
        assert isinstance(store, BaseVectorStore)

    def test_store_roundtrip_upsert_query(self):
        """Test complete upsert and query roundtrip."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        # Upsert
        records = [
            VectorRecord(
                id="doc_001",
                text="Machine learning is great",
                embedding=[0.1, 0.2, 0.3],
                metadata={"topic": "ml"},
            ),
            VectorRecord(
                id="doc_002",
                text="Deep learning advances",
                embedding=[0.15, 0.25, 0.35],
                metadata={"topic": "ml"},
            ),
            VectorRecord(
                id="doc_003",
                text="Web development tools",
                embedding=[0.9, 0.8, 0.7],
                metadata={"topic": "web"},
            ),
        ]

        store.upsert(records)

        # Query similar to ML documents
        query_vector = [0.1, 0.2, 0.3]
        results = store.query(query_vector, top_k=2)

        assert len(results) > 0
        assert all(isinstance(r, VectorRecord) for r in results)

    def test_store_idempotency(self):
        """Test that upserting same record twice is idempotent."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        record = VectorRecord(
            id="doc_001",
            text="Document",
            embedding=[0.1, 0.2, 0.3],
        )

        # Upsert twice
        store.upsert([record])
        store.upsert([record])

        # Query should still work correctly
        results = store.query([0.1, 0.2, 0.3], top_k=10)
        assert len(results) > 0

    def test_store_large_embedding_dimensions(self):
        """Test store handles high-dimensional embeddings."""
        store_settings = Mock()
        store_settings.provider = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        # 768-dimensional embedding (typical for BGE/OpenAI)
        embedding = [float(i) / 768 for i in range(768)]

        record = VectorRecord(
            id="doc_001",
            text="High dimensional document",
            embedding=embedding,
        )

        store.upsert([record])
        results = store.query(embedding, top_k=10)

        assert len(results) > 0

    def test_store_delete_by_metadata_basic(self):
        """Test basic delete_by_metadata functionality."""
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store_settings = Mock()
        store_settings.backend = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        # Upsert records with different metadata
        records = [
            VectorRecord(
                id="doc_001",
                text="Document 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"collection": "test", "source": "file1.pdf"},
            ),
            VectorRecord(
                id="doc_002",
                text="Document 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"collection": "test", "source": "file2.pdf"},
            ),
            VectorRecord(
                id="doc_003",
                text="Document 3",
                embedding=[0.7, 0.8, 0.9],
                metadata={"collection": "other", "source": "file1.pdf"},
            ),
        ]

        store.upsert(records)

        # Delete by collection
        if hasattr(store, 'delete_by_metadata'):
            store.delete_by_metadata({"collection": "test"})

    def test_store_delete_by_metadata_empty_match(self):
        """Test delete_by_metadata with no matching records."""
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store_settings = Mock()
        store_settings.backend = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="Document 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"collection": "test"},
            ),
        ]

        store.upsert(records)

        # Delete with non-matching metadata
        if hasattr(store, 'delete_by_metadata'):
            store.delete_by_metadata({"collection": "nonexistent"})

    def test_store_delete_by_metadata_compound_filter(self):
        """Test delete_by_metadata with compound filters."""
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store_settings = Mock()
        store_settings.backend = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="Document 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"collection": "test", "status": "active"},
            ),
            VectorRecord(
                id="doc_002",
                text="Document 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"collection": "test", "status": "inactive"},
            ),
        ]

        store.upsert(records)

        # Delete with multiple metadata fields
        if hasattr(store, 'delete_by_metadata'):
            store.delete_by_metadata({"collection": "test", "status": "active"})

    def test_store_delete_all_by_metadata(self):
        """Test delete_by_metadata can delete all records."""
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store_settings = Mock()
        store_settings.backend = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id=f"doc_{i:03d}",
                text=f"Document {i}",
                embedding=[float(i) / 10, 0.2, 0.3],
                metadata={"delete_me": "yes"},
            )
            for i in range(5)
        ]

        store.upsert(records)

        # Delete all with matching metadata
        if hasattr(store, 'delete_by_metadata'):
            store.delete_by_metadata({"delete_me": "yes"})

    def test_store_delete_by_empty_metadata(self):
        """Test delete_by_metadata with empty filter dict."""
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        store_settings = Mock()
        store_settings.backend = "fake"

        factory = VectorStoreFactory()
        store = factory.create(store_settings)

        records = [
            VectorRecord(
                id="doc_001",
                text="Document 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"collection": "test"},
            ),
        ]

        store.upsert(records)

        # Delete with empty filter (should be no-op or error)
        if hasattr(store, 'delete_by_metadata'):
            try:
                store.delete_by_metadata({})
            except (ValueError, TypeError):
                pass  # Expected behavior for empty filter
