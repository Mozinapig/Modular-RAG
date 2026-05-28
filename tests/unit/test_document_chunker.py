"""
Unit tests for DocumentChunker - Document to Chunks conversion.
Tests focus on chunk ID generation, metadata inheritance, and image reference distribution.
"""

import hashlib
import pytest

from src.core.types import Document, Chunk, ImageRef
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.splitter_factory import SplitterFactory


class FakeSplitterSettings:
    """Fake Settings for testing."""
    def __init__(self, chunk_size=100, overlap=0, provider="fake"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.provider = provider


class TestDocumentChunker:
    """Test DocumentChunker implementation."""

    @pytest.fixture
    def fake_settings(self):
        """Create fake settings for splitter."""
        return FakeSplitterSettings(chunk_size=100, overlap=0)

    @pytest.fixture
    def chunker(self, fake_settings):
        """Create DocumentChunker instance."""
        return DocumentChunker(fake_settings)

    @pytest.fixture
    def simple_document(self):
        """Create a simple document without images."""
        text = "Hello world. " * 20  # ~260 chars
        return Document(
            id="doc_12345678",
            text=text,
            metadata={
                "source_path": "/path/to/doc.pdf",
            }
        )

    @pytest.fixture
    def document_with_images(self):
        """Create a document with image references."""
        text = ("Start of document. [IMAGE: img_001] Text after first image. "
                "More content here. [IMAGE: img_002] Text after second image. "
                "Final content.")
        images = [
            ImageRef(
                id="img_001",
                path="/data/images/doc_hash/img_001.png",
                text_offset=18,
                text_length=14,
                page=0,
                position={"x": 100, "y": 200}
            ),
            ImageRef(
                id="img_002",
                path="/data/images/doc_hash/img_002.png",
                text_offset=110,
                text_length=14,
                page=1,
                position={"x": 150, "y": 300}
            )
        ]
        return Document(
            id="doc_with_images",
            text=text,
            metadata={
                "source_path": "/path/to/doc_with_images.pdf",
                "images": images,
                "doc_type": "technical_doc",
                "title": "Test Document with Images"
            }
        )

    def test_chunker_initialization(self, chunker):
        """Test that DocumentChunker initializes correctly."""
        assert chunker is not None
        assert hasattr(chunker, 'split_document')

    def test_split_simple_document_returns_chunks(self, chunker, simple_document):
        """Test that splitting a simple document returns Chunk objects."""
        chunks = chunker.split_document(simple_document)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_id_format(self, chunker, simple_document):
        """Test that chunk IDs follow the format: {doc_id}_{index:04d}_{hash_8chars}."""
        chunks = chunker.split_document(simple_document)

        for idx, chunk in enumerate(chunks):
            # ID should contain doc_id
            assert simple_document.id in chunk.id
            # ID should contain index in 04d format
            assert f"_{idx:04d}_" in chunk.id
            # ID should be in expected format
            parts = chunk.id.split('_')
            assert len(parts) >= 3  # At least: doc_id_index_hash

    def test_chunk_id_deterministic(self, chunker, simple_document):
        """Test that chunk IDs are deterministic - same document produces same IDs."""
        chunks1 = chunker.split_document(simple_document)
        chunks2 = chunker.split_document(simple_document)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id

    def test_chunk_id_unique(self, chunker, simple_document):
        """Test that chunk IDs are unique within a document."""
        chunks = chunker.split_document(simple_document)
        chunk_ids = [chunk.id for chunk in chunks]

        # All IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_metadata_includes_source_path(self, chunker, simple_document):
        """Test that chunk metadata includes source_path from document."""
        chunks = chunker.split_document(simple_document)

        for chunk in chunks:
            assert "source_path" in chunk.metadata
            assert chunk.metadata["source_path"] == simple_document.metadata["source_path"]

    def test_chunk_metadata_includes_chunk_index(self, chunker, simple_document):
        """Test that chunk metadata includes chunk_index."""
        chunks = chunker.split_document(simple_document)

        for idx, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_index"] == idx

    def test_chunk_source_ref_points_to_document(self, chunker, simple_document):
        """Test that chunk source_ref points to parent document."""
        chunks = chunker.split_document(simple_document)

        for chunk in chunks:
            assert "source_ref" in chunk.metadata
            assert chunk.metadata["source_ref"] == simple_document.id

    def test_chunk_metadata_inherits_from_document(self, chunker, document_with_images):
        """Test that chunk metadata inherits document metadata fields."""
        chunks = chunker.split_document(document_with_images)

        for chunk in chunks:
            # Should have inherited metadata
            assert chunk.metadata["source_path"] == document_with_images.metadata["source_path"]
            assert chunk.metadata["doc_type"] == document_with_images.metadata["doc_type"]
            assert chunk.metadata["title"] == document_with_images.metadata["title"]

    def test_chunk_without_images_has_no_images_field(self, chunker, simple_document):
        """Test that chunks without image placeholders don't have images field."""
        chunks = chunker.split_document(simple_document)

        for chunk in chunks:
            # Should not have images field if no placeholders
            if "[IMAGE:" not in chunk.text:
                assert "images" not in chunk.metadata
                assert "image_refs" not in chunk.metadata

    def test_chunk_with_image_placeholder_includes_image_refs(self, chunker, document_with_images):
        """Test that chunks containing image placeholders have image_refs list."""
        chunks = chunker.split_document(document_with_images)

        has_images = False
        for chunk in chunks:
            if "[IMAGE:" in chunk.text:
                has_images = True
                assert "image_refs" in chunk.metadata
                assert isinstance(chunk.metadata["image_refs"], list)
                assert len(chunk.metadata["image_refs"]) > 0

        assert has_images, "Document should have chunks with image placeholders"

    def test_image_distribution_correctness(self, chunker, document_with_images):
        """Test that only referenced images are in chunk.metadata["images"]."""
        chunks = chunker.split_document(document_with_images)

        # Verify each chunk's images match its placeholders
        for chunk in chunks:
            if "image_refs" in chunk.metadata:
                image_refs = chunk.metadata["image_refs"]
                images = chunk.metadata.get("images", [])

                # Should have corresponding ImageRef objects
                assert len(images) == len(image_refs)

                # Each image in metadata should be referenced in text
                for img in images:
                    placeholder = f"[IMAGE: {img.id}]"
                    assert placeholder in chunk.text

    def test_chunk_text_contains_image_placeholders(self, chunker, document_with_images):
        """Test that chunk text contains image placeholders."""
        chunks = chunker.split_document(document_with_images)

        has_placeholder = False
        for chunk in chunks:
            if "[IMAGE:" in chunk.text:
                has_placeholder = True
                # Verify placeholder format matches ImageRef IDs
                import re
                placeholders = re.findall(r'\[IMAGE: (\w+)\]', chunk.text)
                image_refs = chunk.metadata.get("image_refs", [])
                assert set(placeholders) == set(image_refs)

        assert has_placeholder, "Should find at least one image placeholder"

    def test_chunk_text_content(self, chunker, simple_document):
        """Test that chunk text content is non-empty."""
        chunks = chunker.split_document(simple_document)

        for chunk in chunks:
            assert isinstance(chunk.text, str)
            assert len(chunk.text.strip()) > 0

    def test_chunk_serializable(self, chunker, simple_document):
        """Test that chunks can be serialized to dict."""
        chunks = chunker.split_document(simple_document)

        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            assert isinstance(chunk_dict, dict)
            assert "id" in chunk_dict
            assert "text" in chunk_dict
            assert "metadata" in chunk_dict

    def test_document_content_preserved(self, chunker, simple_document):
        """Test that when concatenating chunks, original content is largely preserved."""
        original_text = simple_document.text
        chunks = chunker.split_document(simple_document)

        # Concatenate chunk text
        concatenated = "".join(chunk.text for chunk in chunks)

        # Should preserve content
        assert concatenated == original_text

    def test_multiple_documents_different_ids(self, chunker, simple_document, document_with_images):
        """Test that different documents produce different chunk IDs."""
        chunks1 = chunker.split_document(simple_document)
        chunks2 = chunker.split_document(document_with_images)

        ids1 = set(c.id for c in chunks1)
        ids2 = set(c.id for c in chunks2)

        # No overlap in IDs
        assert len(ids1.intersection(ids2)) == 0

    def test_chunk_offset_tracking(self, chunker, simple_document):
        """Test that chunks can track offsets in original document."""
        chunks = chunker.split_document(simple_document)

        # Chunks should maintain order
        offset = 0
        for chunk in chunks:
            chunk_text = chunk.text
            assert simple_document.text[offset:offset + len(chunk_text)] == chunk_text
            offset += len(chunk_text)

    def test_empty_document(self, chunker):
        """Test handling of empty document."""
        doc = Document(
            id="empty_doc",
            text="",
            metadata={"source_path": "/path/to/empty.pdf"}
        )
        chunks = chunker.split_document(doc)
        assert chunks == []

    def test_image_metadata_structure_in_chunk(self, chunker, document_with_images):
        """Test that ImageRef objects in chunk metadata have correct structure."""
        chunks = chunker.split_document(document_with_images)

        for chunk in chunks:
            if "images" in chunk.metadata:
                for img in chunk.metadata["images"]:
                    if isinstance(img, ImageRef):
                        # Verify required ImageRef fields
                        assert img.id is not None
                        assert img.path is not None
                        assert img.text_offset >= 0
                        assert img.text_length > 0

    def test_single_image_in_chunk(self, chunker):
        """Test chunk containing exactly one image."""
        text = "Before image [IMAGE: single_img] after image"
        images = [
            ImageRef(
                id="single_img",
                path="/data/images/doc_hash/single_img.png",
                text_offset=14,
                text_length=14,
                page=0
            )
        ]
        doc = Document(
            id="doc_single",
            text=text,
            metadata={"source_path": "/path/doc", "images": images}
        )

        chunks = chunker.split_document(doc)

        # Should have image in metadata
        assert len(chunks) > 0
        chunk_with_img = next((c for c in chunks if "[IMAGE:" in c.text), None)
        assert chunk_with_img is not None
        assert "image_refs" in chunk_with_img.metadata
        assert "single_img" in chunk_with_img.metadata["image_refs"]

    def test_configuration_driven_chunk_size(self):
        """Test that chunk size from settings affects splitting."""
        # Create chunker with smaller chunk size
        settings_small = FakeSplitterSettings(chunk_size=50, overlap=0)
        chunker_small = DocumentChunker(settings_small)

        # Create chunker with larger chunk size
        settings_large = FakeSplitterSettings(chunk_size=200, overlap=0)
        chunker_large = DocumentChunker(settings_large)

        # Same document
        text = "Hello world. " * 20
        doc = Document(id="doc", text=text, metadata={"source_path": "/path"})

        chunks_small = chunker_small.split_document(doc)
        chunks_large = chunker_large.split_document(doc)

        # Small chunk size should produce more chunks
        assert len(chunks_small) > len(chunks_large)

    def test_chunk_index_sequential(self, chunker, simple_document):
        """Test that chunk_index is sequential from 0."""
        chunks = chunker.split_document(simple_document)

        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == idx

    def test_complex_document_with_multiple_images(self, chunker):
        """Test document with multiple images scattered throughout."""
        text = (
            "Intro [IMAGE: img1] text1 "
            "[IMAGE: img2] middle [IMAGE: img3] "
            "more [IMAGE: img4] end"
        )
        images = [
            ImageRef(id=f"img{i}", path=f"/data/images/img{i}.png",
                    text_offset=text.find(f"[IMAGE: img{i}]"),
                    text_length=14, page=0)
            for i in range(1, 5)
        ]

        doc = Document(
            id="complex_doc",
            text=text,
            metadata={"source_path": "/path", "images": images}
        )

        chunks = chunker.split_document(doc)

        # All chunks with images should have correct image_refs
        for chunk in chunks:
            if "image_refs" in chunk.metadata:
                for img_id in chunk.metadata["image_refs"]:
                    assert f"[IMAGE: {img_id}]" in chunk.text
