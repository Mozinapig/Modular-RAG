"""
Unit tests for DataService - encapsulates ChromaStore and ImageStorage reads
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.observability.dashboard.services.data_service import DataService


class TestDataServiceGetDocuments:
    """Test DataService.get_documents() - fetches documents grouped by source_path"""

    def test_get_documents_empty_store(self):
        """Should return empty list when ChromaStore has no documents"""
        mock_chroma = Mock()
        mock_chroma.get_by_metadata.return_value = []

        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_documents(collection="test_collection")

        assert result == []
        mock_chroma.get_by_metadata.assert_called_once()

    def test_get_documents_single_collection(self):
        """Should retrieve and group documents by source_path from ChromaStore"""
        mock_chroma = Mock()
        mock_records = [
            {
                "id": "chunk_1",
                "content": "text1",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            {
                "id": "chunk_2",
                "content": "text2",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 1
                }
            },
            {
                "id": "chunk_3",
                "content": "text3",
                "metadata": {
                    "source_path": "doc2.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            }
        ]
        mock_chroma.get_by_metadata.return_value = mock_records
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_documents(collection="test_collection")

        # Should group by source_path
        assert len(result) == 2
        assert result[0]["source_path"] == "doc1.pdf"
        assert result[0]["chunk_count"] == 2
        assert result[1]["source_path"] == "doc2.pdf"
        assert result[1]["chunk_count"] == 1

    def test_get_documents_with_collection_filter(self):
        """Should pass collection filter to ChromaStore"""
        mock_chroma = Mock()
        mock_chroma.get_by_metadata.return_value = []
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        service.get_documents(collection="my_collection")

        # Verify collection was passed in filter
        call_args = mock_chroma.get_by_metadata.call_args
        assert call_args is not None


class TestDataServiceGetChunks:
    """Test DataService.get_chunks() - retrieves all chunks for a document"""

    def test_get_chunks_for_document(self):
        """Should retrieve all chunks for a given source_path"""
        mock_chroma = Mock()
        mock_chunks = [
            {
                "id": "chunk_1",
                "content": "This is chunk 1",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "title": "Section 1",
                    "tags": ["intro"]
                }
            },
            {
                "id": "chunk_2",
                "content": "This is chunk 2",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 1,
                    "title": "Section 1",
                    "tags": ["content"]
                }
            }
        ]
        mock_chroma.get_by_metadata.return_value = mock_chunks
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_chunks(source_path="doc1.pdf", collection="test_collection")

        assert len(result) == 2
        assert result[0]["id"] == "chunk_1"
        assert result[1]["id"] == "chunk_2"

    def test_get_chunks_empty_result(self):
        """Should return empty list when no chunks found"""
        mock_chroma = Mock()
        mock_chroma.get_by_metadata.return_value = []
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_chunks(source_path="nonexistent.pdf", collection="test_collection")

        assert result == []


class TestDataServiceGetChunkImages:
    """Test DataService.get_chunk_images() - retrieves images for a chunk"""

    def test_get_chunk_images(self):
        """Should retrieve images associated with a chunk"""
        mock_chroma = Mock()
        mock_image_storage = Mock()

        # Mock image storage to return image metadata
        mock_image_storage.list_images.return_value = [
            {
                "image_id": "img_001",
                "chunk_id": "chunk_1",
                "path": "data/images/collection/img_001.png",
                "format": "png"
            }
        ]

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_chunk_images(chunk_id="chunk_1")

        assert len(result) == 1
        assert result[0]["image_id"] == "img_001"

    def test_get_chunk_images_no_images(self):
        """Should return empty list when chunk has no images"""
        mock_chroma = Mock()
        mock_image_storage = Mock()
        mock_image_storage.list_images.return_value = []

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_chunk_images(chunk_id="chunk_without_images")

        assert result == []


class TestDataServiceGetDocumentStats:
    """Test DataService.get_document_stats() - retrieves statistics for a document"""

    def test_get_document_stats(self):
        """Should calculate stats for a document (chunk count, ingestion time, etc.)"""
        mock_chroma = Mock()
        mock_chunks = [
            {
                "id": "chunk_1",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "ingestion_time": "2026-06-17T10:00:00Z"
                }
            },
            {
                "id": "chunk_2",
                "metadata": {
                    "source_path": "doc1.pdf",
                    "ingestion_time": "2026-06-17T10:00:00Z"
                }
            }
        ]
        mock_chroma.get_by_metadata.return_value = mock_chunks
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.get_document_stats(source_path="doc1.pdf", collection="test_collection")

        assert result["chunk_count"] == 2
        assert result["source_path"] == "doc1.pdf"


class TestDataServiceListCollections:
    """Test DataService.list_collections() - lists all available collections"""

    def test_list_collections(self):
        """Should list all available collections in ChromaStore"""
        mock_chroma = Mock()
        mock_chroma.list_collections.return_value = ["collection1", "collection2"]
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.list_collections()

        assert len(result) == 2
        assert "collection1" in result
        assert "collection2" in result

    def test_list_collections_empty(self):
        """Should return empty list when no collections exist"""
        mock_chroma = Mock()
        mock_chroma.list_collections.return_value = []
        mock_image_storage = Mock()

        service = DataService(chroma_store=mock_chroma, image_storage=mock_image_storage)
        result = service.list_collections()

        assert result == []
