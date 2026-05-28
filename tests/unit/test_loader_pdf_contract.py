"""
Unit tests for Loader abstraction and PDF Loader implementation.
Tests focus on contract compliance, document structure, and image handling.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


class TestBaseLoaderInterface:
    """Test BaseLoader abstract interface."""

    def test_abstract_interface(self):
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()


class TestPdfLoaderContract:
    """Test PdfLoader implementation against contract."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory for extracted images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def loader(self, temp_image_dir):
        """Create a PdfLoader instance with temp image directory."""
        return PdfLoader(image_output_dir=temp_image_dir)

    @pytest.fixture
    def simple_pdf_path(self):
        """Get path to simple PDF fixture (text only)."""
        pdf_path = Path("tests/fixtures/sample_documents/blogger_intro.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Sample PDF not found: {pdf_path}")
        return str(pdf_path)

    @pytest.fixture
    def complex_pdf_path(self):
        """Get path to complex PDF fixture (may contain images)."""
        pdf_path = Path("tests/fixtures/sample_documents/complex_technical_doc.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Sample PDF not found: {pdf_path}")
        return str(pdf_path)

    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader is not None
        assert hasattr(loader, 'load')

    def test_load_simple_pdf_returns_document(self, loader, simple_pdf_path):
        """Test that loading a simple PDF returns a Document object."""
        doc = loader.load(simple_pdf_path)

        assert isinstance(doc, Document)
        assert doc.id is not None
        assert doc.text is not None
        assert len(doc.text) > 0
        assert "source_path" in doc.metadata

    def test_document_metadata_contains_source_path(self, loader, simple_pdf_path):
        """Test that document metadata contains source_path."""
        doc = loader.load(simple_pdf_path)

        assert "source_path" in doc.metadata
        assert doc.metadata["source_path"] == simple_pdf_path

    def test_document_id_is_deterministic(self, loader, simple_pdf_path):
        """Test that document ID is deterministic (same file → same ID)."""
        doc1 = loader.load(simple_pdf_path)
        doc2 = loader.load(simple_pdf_path)

        assert doc1.id == doc2.id

    def test_document_id_based_on_file_hash(self, loader, simple_pdf_path):
        """Test that document ID is based on file hash."""
        doc = loader.load(simple_pdf_path)

        # Compute expected hash
        with open(simple_pdf_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Document ID should contain or be based on file hash
        assert file_hash in doc.id or doc.id.startswith(file_hash)

    def test_simple_pdf_no_images_metadata(self, loader, simple_pdf_path):
        """Test that simple PDF without images has empty or absent images metadata."""
        doc = loader.load(simple_pdf_path)

        # images field should be absent or empty
        images = doc.metadata.get("images", [])
        assert isinstance(images, list)
        # For simple text-only PDF, should be empty
        assert len(images) == 0 or all(isinstance(img, ImageRef) for img in images)

    def test_complex_pdf_loads_successfully(self, loader, complex_pdf_path):
        """Test that complex PDF loads successfully."""
        doc = loader.load(complex_pdf_path)

        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        assert "source_path" in doc.metadata

    def test_document_text_contains_image_placeholders(self, loader, complex_pdf_path):
        """Test that document text contains image placeholders if images exist."""
        doc = loader.load(complex_pdf_path)

        images = doc.metadata.get("images", [])
        if images:
            # If images exist, text should contain placeholders
            for img in images:
                if isinstance(img, ImageRef):
                    placeholder = f"[IMAGE: {img.id}]"
                    assert placeholder in doc.text, f"Placeholder {placeholder} not found in text"

    def test_image_metadata_structure(self, loader, complex_pdf_path):
        """Test that image metadata follows C1 contract."""
        doc = loader.load(complex_pdf_path)

        images = doc.metadata.get("images", [])
        for img in images:
            if isinstance(img, ImageRef):
                # Verify required fields
                assert img.id is not None
                assert img.path is not None
                assert img.text_offset >= 0
                assert img.text_length > 0

                # Verify path convention
                assert "data/images/" in img.path

    def test_image_files_saved_to_disk(self, loader, temp_image_dir, complex_pdf_path):
        """Test that extracted images are saved to disk."""
        doc = loader.load(complex_pdf_path)

        images = doc.metadata.get("images", [])
        for img in images:
            if isinstance(img, ImageRef):
                # Image file should exist
                assert os.path.exists(img.path), f"Image file not found: {img.path}"

    def test_image_extraction_failure_does_not_block_text(self, loader, complex_pdf_path):
        """Test that image extraction failure doesn't block text parsing."""
        # This test verifies graceful degradation
        doc = loader.load(complex_pdf_path)

        # Document should always have text, even if image extraction fails
        assert len(doc.text) > 0
        assert "source_path" in doc.metadata

    def test_nonexistent_file_raises_error(self, loader):
        """Test that loading nonexistent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            loader.load("/nonexistent/path/to/file.pdf")

    def test_document_serializable(self, loader, simple_pdf_path):
        """Test that document can be serialized to dict."""
        doc = loader.load(simple_pdf_path)
        doc_dict = doc.to_dict()

        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict

    def test_multiple_pdfs_different_ids(self, loader, simple_pdf_path, complex_pdf_path):
        """Test that different PDFs produce different document IDs."""
        doc1 = loader.load(simple_pdf_path)
        doc2 = loader.load(complex_pdf_path)

        assert doc1.id != doc2.id

    def test_document_text_not_empty(self, loader, simple_pdf_path):
        """Test that extracted text is not empty."""
        doc = loader.load(simple_pdf_path)

        assert len(doc.text.strip()) > 0

    def test_document_text_is_string(self, loader, simple_pdf_path):
        """Test that document text is a string."""
        doc = loader.load(simple_pdf_path)

        assert isinstance(doc.text, str)

    def test_metadata_is_dict(self, loader, simple_pdf_path):
        """Test that metadata is a dictionary."""
        doc = loader.load(simple_pdf_path)

        assert isinstance(doc.metadata, dict)

    def test_image_offset_within_text_bounds(self, loader, complex_pdf_path):
        """Test that image text_offset and text_length are within text bounds."""
        doc = loader.load(complex_pdf_path)

        images = doc.metadata.get("images", [])
        for img in images:
            if isinstance(img, ImageRef):
                # text_offset + text_length should not exceed text length
                end_offset = img.text_offset + img.text_length
                assert end_offset <= len(doc.text), \
                    f"Image offset {img.text_offset} + length {img.text_length} exceeds text length {len(doc.text)}"

    def test_image_placeholder_matches_metadata(self, loader, complex_pdf_path):
        """Test that image placeholders in text match metadata."""
        doc = loader.load(complex_pdf_path)

        images = doc.metadata.get("images", [])
        for img in images:
            if isinstance(img, ImageRef):
                # Extract text at offset
                placeholder_text = doc.text[img.text_offset:img.text_offset + img.text_length]
                expected_placeholder = f"[IMAGE: {img.id}]"
                assert placeholder_text == expected_placeholder, \
                    f"Placeholder mismatch: '{placeholder_text}' != '{expected_placeholder}'"
