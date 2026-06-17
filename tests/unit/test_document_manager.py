"""
Unit tests for DocumentManager (G2)
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json

from src.ingestion.document_manager import DocumentManager, DocumentInfo, DocumentDetail, DeleteResult, CollectionStats


class TestDocumentInfo:
    """Test DocumentInfo data structure."""

    def test_document_info_creation(self):
        """Test DocumentInfo creation."""
        doc_info = DocumentInfo(
            source_path="test.pdf",
            collection="test_col",
            chunk_count=10,
            image_count=2,
            created_at="2026-06-17T10:00:00",
            doc_id="doc_123"
        )

        assert doc_info.source_path == "test.pdf"
        assert doc_info.collection == "test_col"
        assert doc_info.chunk_count == 10
        assert doc_info.image_count == 2
        assert doc_info.doc_id == "doc_123"

    def test_document_detail_creation(self):
        """Test DocumentDetail creation."""
        detail = DocumentDetail(
            doc_id="doc_123",
            source_path="test.pdf",
            collection="test_col",
            title="Test Document",
            summary="Test summary",
            tags=["tag1", "tag2"],
            chunk_count=10,
            image_count=2,
            created_at="2026-06-17T10:00:00",
            updated_at="2026-06-17T11:00:00"
        )

        assert detail.doc_id == "doc_123"
        assert detail.title == "Test Document"
        assert detail.chunk_count == 10


class TestDocumentManager:
    """Test DocumentManager functionality."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock stores."""
        return {
            "chroma_store": Mock(),
            "bm25_indexer": Mock(),
            "image_storage": Mock(),
            "file_integrity": Mock(),
        }

    def test_document_manager_init(self, mock_stores):
        """Test DocumentManager initialization."""
        manager = DocumentManager(
            chroma_store=mock_stores["chroma_store"],
            bm25_indexer=mock_stores["bm25_indexer"],
            image_storage=mock_stores["image_storage"],
            file_integrity=mock_stores["file_integrity"],
        )

        assert manager.chroma_store == mock_stores["chroma_store"]
        assert manager.bm25_indexer == mock_stores["bm25_indexer"]

    def test_list_documents_empty(self, mock_stores):
        """Test listing documents when none exist."""
        mock_stores["chroma_store"].get_collection_stats = Mock(return_value={
            "collections": [],
            "total_chunks": 0,
            "total_images": 0,
        })
        mock_stores["chroma_store"].get_by_metadata = Mock(return_value=[])

        manager = DocumentManager(**mock_stores)
        docs = manager.list_documents()

        assert docs == []

    def test_list_documents_with_data(self, mock_stores):
        """Test listing documents with data."""
        # Mock ChromaStore's get_by_metadata to return chunks
        mock_stores["chroma_store"].get_by_metadata = Mock(return_value=[
            {
                "id": "chunk_001",
                "text": "Sample text",
                "metadata": {
                    "source_path": "test.pdf",
                    "collection": "test_col",
                    "chunk_index": 0,
                    "title": "Test Doc",
                }
            }
        ])

        manager = DocumentManager(**mock_stores)

        # Call list_documents
        docs = manager.list_documents(collection="test_col")

        # Verify the call was made
        mock_stores["chroma_store"].get_by_metadata.assert_called()

    def test_get_document_detail(self, mock_stores):
        """Test getting document detail."""
        mock_stores["chroma_store"].get_by_metadata = Mock(return_value=[
            {
                "id": "chunk_001",
                "text": "Sample text 1",
                "metadata": {
                    "doc_id": "doc_123",
                    "source_path": "test.pdf",
                    "title": "Test Doc",
                    "summary": "Test summary",
                    "tags": ["tag1"],
                    "image_refs": ["img_1", "img_2"],
                }
            },
            {
                "id": "chunk_002",
                "text": "Sample text 2",
                "metadata": {
                    "doc_id": "doc_123",
                    "source_path": "test.pdf",
                }
            }
        ])

        manager = DocumentManager(**mock_stores)
        detail = manager.get_document_detail("doc_123")

        assert detail.doc_id == "doc_123"
        assert detail.chunk_count == 2
        assert detail.image_count == 2

    def test_delete_document(self, mock_stores):
        """Test deleting a document."""
        mock_stores["chroma_store"].delete_by_metadata = Mock(return_value=5)
        mock_stores["chroma_store"].get_by_metadata = Mock(return_value=[
            {
                "id": "chunk_001",
                "metadata": {"doc_id": "doc_123", "source_path": "test.pdf"}
            }
        ])
        mock_stores["bm25_indexer"].remove_document = Mock(return_value=5)
        mock_stores["image_storage"].delete_by_doc_id = Mock(return_value=2)
        mock_stores["file_integrity"].remove_record = Mock()

        manager = DocumentManager(**mock_stores)
        result = manager.delete_document("test.pdf", "test_col")

        assert result.source_path == "test.pdf"
        assert result.chunks_deleted == 5
        assert result.images_deleted == 2

        # Verify all stores were called
        mock_stores["chroma_store"].delete_by_metadata.assert_called()
        mock_stores["bm25_indexer"].remove_document.assert_called()
        mock_stores["image_storage"].delete_by_doc_id.assert_called()

    def test_get_collection_stats(self, mock_stores):
        """Test getting collection statistics."""
        mock_stores["chroma_store"].get_collection_stats = Mock(return_value={
            "collections": [{"name": "test_col", "chunk_count": 100, "image_count": 5}],
            "total_chunks": 100,
            "total_images": 5,
        })
        mock_stores["chroma_store"].get_by_metadata = Mock(return_value=[])

        manager = DocumentManager(**mock_stores)
        stats = manager.get_collection_stats("test_col")

        assert stats.collection_name == "test_col"
        assert stats.chunk_count == 100
        assert stats.image_count == 5


class TestDocumentManagerIntegration:
    """Integration tests for DocumentManager."""

    def test_delete_and_relist(self):
        """Test that deleting a document removes it from listing."""
        mock_chroma = Mock()
        mock_bm25 = Mock()
        mock_images = Mock()
        mock_integrity = Mock()

        # First listing returns 1 document
        mock_chroma.get_by_metadata = Mock(side_effect=[
            # First call: list before delete
            [{
                "id": "chunk_001",
                "metadata": {"source_path": "test.pdf", "collection": "test_col"}
            }],
            # Second call: list after delete
            []
        ])

        manager = DocumentManager(
            chroma_store=mock_chroma,
            bm25_indexer=mock_bm25,
            image_storage=mock_images,
            file_integrity=mock_integrity,
        )

        # Before delete
        docs_before = manager.list_documents()
        assert len(docs_before) == 1

        # Delete
        mock_chroma.delete_by_metadata = Mock(return_value=1)
        mock_bm25.remove_document = Mock(return_value=1)
        mock_images.delete_by_doc_id = Mock(return_value=0)

        result = manager.delete_document("test.pdf", "test_col")
        assert result.chunks_deleted == 1

        # After delete
        docs_after = manager.list_documents()
        assert len(docs_after) == 0
