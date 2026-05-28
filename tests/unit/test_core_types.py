"""
Unit tests for core data types (Document, Chunk, ChunkRecord).
Tests focus on serialization, field stability, and metadata contract.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.core.types import Chunk, ChunkRecord, Document, ImageRef


class TestDocument:
    """Test Document type."""

    def test_document_creation_minimal(self):
        """Test creating a Document with minimal required fields."""
        doc = Document(
            id="doc_001",
            text="Sample document text",
            metadata={"source_path": "test.pdf"}
        )
        assert doc.id == "doc_001"
        assert doc.text == "Sample document text"
        assert doc.metadata["source_path"] == "test.pdf"

    def test_document_creation_with_images(self):
        """Test creating a Document with image metadata."""
        images = [
            ImageRef(
                id="img_001",
                path="data/images/doc_001/img_001.png",
                page=1,
                text_offset=50,
                text_length=14,
                position={"x": 100, "y": 200, "width": 300, "height": 400}
            )
        ]
        doc = Document(
            id="doc_001",
            text="Sample text [IMAGE: img_001] more text",
            metadata={
                "source_path": "test.pdf",
                "images": images
            }
        )
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0].id == "img_001"

    def test_document_serialization_to_dict(self):
        """Test Document can be serialized to dict."""
        doc = Document(
            id="doc_001",
            text="Sample text",
            metadata={"source_path": "test.pdf", "doc_type": "pdf"}
        )
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "doc_001"
        assert doc_dict["text"] == "Sample text"
        assert doc_dict["metadata"]["source_path"] == "test.pdf"

    def test_document_serialization_to_json(self):
        """Test Document can be serialized to JSON."""
        doc = Document(
            id="doc_001",
            text="Sample text",
            metadata={"source_path": "test.pdf"}
        )
        json_str = json.dumps(doc.to_dict())
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["id"] == "doc_001"

    def test_document_metadata_extensibility(self):
        """Test that Document metadata can be extended with custom fields."""
        doc = Document(
            id="doc_001",
            text="Sample text",
            metadata={
                "source_path": "test.pdf",
                "custom_field": "custom_value",
                "nested": {"key": "value"}
            }
        )
        assert doc.metadata["custom_field"] == "custom_value"
        assert doc.metadata["nested"]["key"] == "value"

    def test_document_with_empty_images_list(self):
        """Test Document with empty images list."""
        doc = Document(
            id="doc_001",
            text="Sample text",
            metadata={"source_path": "test.pdf", "images": []}
        )
        assert doc.metadata["images"] == []

    def test_document_without_images_field(self):
        """Test Document without images field in metadata."""
        doc = Document(
            id="doc_001",
            text="Sample text",
            metadata={"source_path": "test.pdf"}
        )
        assert "images" not in doc.metadata or doc.metadata.get("images") is None


class TestChunk:
    """Test Chunk type."""

    def test_chunk_creation_minimal(self):
        """Test creating a Chunk with minimal required fields."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        assert chunk.id == "chunk_001"
        assert chunk.text == "Chunk text"
        assert chunk.metadata["source_path"] == "test.pdf"

    def test_chunk_with_chunk_index(self):
        """Test Chunk with chunk_index metadata."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={
                "source_path": "test.pdf",
                "chunk_index": 0
            }
        )
        assert chunk.metadata["chunk_index"] == 0

    def test_chunk_with_source_ref(self):
        """Test Chunk with source_ref pointing to parent Document."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={
                "source_path": "test.pdf",
                "source_ref": "doc_001"
            }
        )
        assert chunk.metadata["source_ref"] == "doc_001"

    def test_chunk_with_image_refs(self):
        """Test Chunk with image_refs list."""
        chunk = Chunk(
            id="chunk_001",
            text="Text [IMAGE: img_001] more text",
            metadata={
                "source_path": "test.pdf",
                "image_refs": ["img_001"],
                "images": [
                    ImageRef(
                        id="img_001",
                        path="data/images/doc_001/img_001.png",
                        page=1,
                        text_offset=5,
                        text_length=14
                    )
                ]
            }
        )
        assert chunk.metadata["image_refs"] == ["img_001"]
        assert len(chunk.metadata["images"]) == 1

    def test_chunk_without_images(self):
        """Test Chunk without images field (no image references)."""
        chunk = Chunk(
            id="chunk_001",
            text="Plain text without images",
            metadata={
                "source_path": "test.pdf",
                "chunk_index": 0
            }
        )
        assert "images" not in chunk.metadata or chunk.metadata.get("images") is None

    def test_chunk_serialization_to_dict(self):
        """Test Chunk can be serialized to dict."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf", "chunk_index": 0}
        )
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["id"] == "chunk_001"
        assert chunk_dict["text"] == "Chunk text"

    def test_chunk_serialization_to_json(self):
        """Test Chunk can be serialized to JSON."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        json_str = json.dumps(chunk.to_dict())
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["id"] == "chunk_001"

    def test_chunk_with_optional_offsets(self):
        """Test Chunk with optional start_offset and end_offset."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            start_offset=100,
            end_offset=110
        )
        assert chunk.start_offset == 100
        assert chunk.end_offset == 110

    def test_chunk_without_optional_offsets(self):
        """Test Chunk without optional offsets."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        assert chunk.start_offset is None
        assert chunk.end_offset is None


class TestChunkRecord:
    """Test ChunkRecord type."""

    def test_chunk_record_creation_minimal(self):
        """Test creating a ChunkRecord with minimal required fields."""
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        assert record.id == "chunk_001"
        assert record.text == "Chunk text"

    def test_chunk_record_with_dense_vector(self):
        """Test ChunkRecord with dense vector."""
        dense_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            dense_vector=dense_vector
        )
        assert record.dense_vector == dense_vector
        assert len(record.dense_vector) == 5

    def test_chunk_record_with_sparse_vector(self):
        """Test ChunkRecord with sparse vector (term weights)."""
        sparse_vector = {"term1": 0.5, "term2": 0.3, "term3": 0.2}
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            sparse_vector=sparse_vector
        )
        assert record.sparse_vector == sparse_vector

    def test_chunk_record_with_both_vectors(self):
        """Test ChunkRecord with both dense and sparse vectors."""
        dense_vector = [0.1, 0.2, 0.3]
        sparse_vector = {"term1": 0.5, "term2": 0.3}
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )
        assert record.dense_vector == dense_vector
        assert record.sparse_vector == sparse_vector

    def test_chunk_record_serialization_to_dict(self):
        """Test ChunkRecord can be serialized to dict."""
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )
        record_dict = record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict["id"] == "chunk_001"
        assert record_dict["dense_vector"] == [0.1, 0.2, 0.3]

    def test_chunk_record_serialization_to_json(self):
        """Test ChunkRecord can be serialized to JSON."""
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"},
            dense_vector=[0.1, 0.2, 0.3]
        )
        json_str = json.dumps(record.to_dict())
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["id"] == "chunk_001"

    def test_chunk_record_without_vectors(self):
        """Test ChunkRecord without vectors (before encoding)."""
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_chunk_record_with_image_refs(self):
        """Test ChunkRecord with image references."""
        record = ChunkRecord(
            id="chunk_001",
            text="Text [IMAGE: img_001]",
            metadata={
                "source_path": "test.pdf",
                "image_refs": ["img_001"],
                "images": [
                    ImageRef(
                        id="img_001",
                        path="data/images/doc_001/img_001.png",
                        page=1,
                        text_offset=5,
                        text_length=14
                    )
                ]
            },
            dense_vector=[0.1, 0.2, 0.3]
        )
        assert record.metadata["image_refs"] == ["img_001"]
        assert len(record.metadata["images"]) == 1


class TestImageRef:
    """Test ImageRef type."""

    def test_image_ref_creation_minimal(self):
        """Test creating an ImageRef with minimal required fields."""
        img_ref = ImageRef(
            id="img_001",
            path="data/images/doc_001/img_001.png",
            text_offset=50,
            text_length=14
        )
        assert img_ref.id == "img_001"
        assert img_ref.path == "data/images/doc_001/img_001.png"
        assert img_ref.text_offset == 50
        assert img_ref.text_length == 14

    def test_image_ref_creation_with_page(self):
        """Test creating an ImageRef with page number."""
        img_ref = ImageRef(
            id="img_001",
            path="data/images/doc_001/img_001.png",
            page=1,
            text_offset=50,
            text_length=14
        )
        assert img_ref.page == 1

    def test_image_ref_creation_with_position(self):
        """Test creating an ImageRef with position information."""
        position = {"x": 100, "y": 200, "width": 300, "height": 400}
        img_ref = ImageRef(
            id="img_001",
            path="data/images/doc_001/img_001.png",
            text_offset=50,
            text_length=14,
            position=position
        )
        assert img_ref.position == position

    def test_image_ref_serialization_to_dict(self):
        """Test ImageRef can be serialized to dict."""
        img_ref = ImageRef(
            id="img_001",
            path="data/images/doc_001/img_001.png",
            page=1,
            text_offset=50,
            text_length=14,
            position={"x": 100, "y": 200}
        )
        img_dict = img_ref.to_dict()
        assert isinstance(img_dict, dict)
        assert img_dict["id"] == "img_001"
        assert img_dict["page"] == 1

    def test_image_ref_serialization_to_json(self):
        """Test ImageRef can be serialized to JSON."""
        img_ref = ImageRef(
            id="img_001",
            path="data/images/doc_001/img_001.png",
            text_offset=50,
            text_length=14
        )
        json_str = json.dumps(img_ref.to_dict())
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["id"] == "img_001"


class TestTypeCompatibility:
    """Test compatibility and relationships between types."""

    def test_document_to_chunk_metadata_inheritance(self):
        """Test that Chunk can inherit metadata from Document."""
        doc = Document(
            id="doc_001",
            text="Document text",
            metadata={
                "source_path": "test.pdf",
                "doc_type": "pdf",
                "title": "Test Document"
            }
        )
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={
                **doc.metadata,
                "chunk_index": 0,
                "source_ref": doc.id
            }
        )
        assert chunk.metadata["source_path"] == doc.metadata["source_path"]
        assert chunk.metadata["doc_type"] == doc.metadata["doc_type"]
        assert chunk.metadata["source_ref"] == "doc_001"

    def test_chunk_to_chunk_record_conversion(self):
        """Test converting Chunk to ChunkRecord with vectors."""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf"}
        )
        record = ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"term1": 0.5}
        )
        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.metadata == chunk.metadata

    def test_document_with_multiple_images(self):
        """Test Document with multiple images in metadata."""
        images = [
            ImageRef(
                id="img_001",
                path="data/images/doc_001/img_001.png",
                page=1,
                text_offset=50,
                text_length=14
            ),
            ImageRef(
                id="img_002",
                path="data/images/doc_001/img_002.png",
                page=2,
                text_offset=200,
                text_length=14
            )
        ]
        doc = Document(
            id="doc_001",
            text="Text [IMAGE: img_001] more [IMAGE: img_002]",
            metadata={
                "source_path": "test.pdf",
                "images": images
            }
        )
        assert len(doc.metadata["images"]) == 2
        assert doc.metadata["images"][0].id == "img_001"
        assert doc.metadata["images"][1].id == "img_002"

    def test_chunk_with_subset_of_document_images(self):
        """Test Chunk containing only subset of Document images."""
        doc_images = [
            ImageRef(
                id="img_001",
                path="data/images/doc_001/img_001.png",
                page=1,
                text_offset=50,
                text_length=14
            ),
            ImageRef(
                id="img_002",
                path="data/images/doc_001/img_002.png",
                page=2,
                text_offset=200,
                text_length=14
            )
        ]
        chunk = Chunk(
            id="chunk_001",
            text="Text [IMAGE: img_001]",
            metadata={
                "source_path": "test.pdf",
                "image_refs": ["img_001"],
                "images": [doc_images[0]]  # Only first image
            }
        )
        assert len(chunk.metadata["images"]) == 1
        assert chunk.metadata["images"][0].id == "img_001"
        assert chunk.metadata["image_refs"] == ["img_001"]
