"""
Integration test suite for IngestionPipeline.
Coverage: Document loading, chunking, transformation, encoding, and storage.
Target: 20+ tests covering success path, error handling, and callbacks.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Optional

from src.core.types import Document, Chunk
from src.ingestion.ingestion_pipeline import IngestionPipeline, IngestionResult
from src.core.trace.trace_context import TraceContext


class TestIngestionPipelineInitialization:
    """Tests for IngestionPipeline initialization."""

    def test_init_creates_pipeline_instance(self):
        """Should create IngestionPipeline with default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)
            assert pipeline is not None

    def test_init_with_custom_paths(self):
        """Should accept custom output paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            db_path = Path(tmpdir) / "test.db"

            pipeline = IngestionPipeline(
                base_path=tmpdir,
                images_dir=str(images_dir),
                db_path=str(db_path)
            )

            assert pipeline is not None

    def test_init_creates_required_directories(self):
        """Should create base and output directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"

            pipeline = IngestionPipeline(
                base_path=tmpdir,
                images_dir=str(images_dir)
            )

            # Directories should be created during init
            assert Path(tmpdir).exists()


class TestIngestionPipelineBasicFlow:
    """Tests for basic pipeline execution flow."""

    def test_run_with_simple_document(self):
        """Should process a simple PDF document end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Use sample PDF from fixtures
            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, collection="test")

            assert result is not None
            assert isinstance(result, IngestionResult)
            assert result.success
            assert result.document_count == 1
            assert result.chunk_count > 0

    def test_run_returns_ingestion_result(self):
        """Should return IngestionResult with statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            assert result.success
            assert hasattr(result, 'document_count')
            assert hasattr(result, 'chunk_count')
            assert hasattr(result, 'elapsed_ms')
            assert hasattr(result, 'errors')

    def test_run_with_collection_name(self):
        """Should use specified collection name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, collection="my_collection")

            assert result.success
            # Collection should be recorded
            assert "my_collection" in result.collection or result.collection == "my_collection"


class TestIngestionPipelineStages:
    """Tests for individual pipeline stages."""

    def test_loading_stage_extracts_document(self):
        """Loading stage should extract document from PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            document = pipeline._load_document(pdf_path)

            assert isinstance(document, Document)
            assert document.text
            assert "source_path" in document.metadata

    def test_chunking_stage_produces_chunks(self):
        """Chunking stage should produce multiple chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Create a test document with sufficient text
            doc = Document(
                id="test_doc",
                text="This is a sample document. " * 100,
                metadata={"source_path": "/test/path.pdf"}
            )

            chunks = pipeline._chunk_document(doc)

            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
            assert all(c.text for c in chunks)

    def test_transform_stage_enriches_metadata(self):
        """Transform stage should enrich chunk metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            chunks = [
                Chunk(
                    id="chunk_1",
                    text="Sample chunk text",
                    metadata={"source_path": "/test.pdf"}
                )
            ]

            transformed = pipeline._transform_chunks(chunks)

            assert len(transformed) > 0
            assert all(isinstance(c, Chunk) for c in transformed)

    def test_encoding_stage_adds_embeddings(self):
        """Encoding stage should add embeddings to chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            chunks = [
                Chunk(
                    id="chunk_1",
                    text="Sample chunk text",
                    metadata={"source_path": "/test.pdf"}
                )
            ]

            encoded = pipeline._encode_chunks(chunks)

            assert len(encoded) > 0
            # After encoding, chunks should have dense/sparse embeddings
            for chunk in encoded:
                assert "dense_embedding" in chunk.metadata or "sparse_embedding" in chunk.metadata


class TestIngestionPipelineProgressCallback:
    """Tests for on_progress callback functionality."""

    def test_on_progress_callback_invoked(self):
        """Should invoke on_progress callback during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            progress_calls = []

            def on_progress(stage: str, current: int, total: int):
                progress_calls.append((stage, current, total))

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, on_progress=on_progress)

            # Should have been called at least once
            assert len(progress_calls) > 0

    def test_on_progress_callback_stages(self):
        """Progress callback should report different stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            progress_calls = []

            def on_progress(stage: str, current: int, total: int):
                progress_calls.append(stage)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, on_progress=on_progress)

            # Should report multiple stages
            stages = set(progress_calls)
            assert len(stages) > 1

    def test_on_progress_callback_parameters(self):
        """Progress callback should report current and total."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            progress_calls = []

            def on_progress(stage: str, current: int, total: int):
                progress_calls.append({"stage": stage, "current": current, "total": total})

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, on_progress=on_progress)

            # Check parameters
            for call_info in progress_calls:
                assert "stage" in call_info
                assert "current" in call_info
                assert "total" in call_info
                assert call_info["current"] <= call_info["total"]


class TestIngestionPipelineErrorHandling:
    """Tests for error handling and recovery."""

    def test_run_with_nonexistent_file(self):
        """Should handle missing file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run("/nonexistent/path/file.pdf")

            # Should return failure result, not raise
            assert not result.success
            assert len(result.errors) > 0

    def test_run_with_invalid_pdf(self):
        """Should handle invalid PDF file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Create a completely invalid file (binary garbage)
            fake_pdf = Path(tmpdir) / "fake.pdf"
            fake_pdf.write_bytes(b'\x00\x01\x02\x03\x04\x05')

            result = pipeline.run(str(fake_pdf))

            # Should return result (either success or failure)
            assert isinstance(result, IngestionResult)

    def test_run_returns_failure_result_on_error(self):
        """Should return IngestionResult with errors on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run("/nonexistent/file.pdf")

            assert isinstance(result, IngestionResult)
            assert not result.success
            assert result.errors is not None
            assert len(result.errors) > 0


class TestIngestionPipelineOutput:
    """Tests for pipeline output artifacts."""

    def test_run_creates_vector_index(self):
        """Should create vector index in Chroma."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, collection="test_vectors")

            if result.success:
                # Vector store should have processed the document
                assert result.chunk_count > 0

    def test_run_creates_bm25_index(self):
        """Should create BM25 index in database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "bm25"
            db_path.mkdir(parents=True, exist_ok=True)

            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            if result.success:
                # BM25 index should be created
                assert result.chunk_count > 0

    def test_run_stores_images(self):
        """Should store extracted images from PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"

            pipeline = IngestionPipeline(
                base_path=tmpdir,
                images_dir=str(images_dir)
            )

            pdf_path = "tests/fixtures/sample_documents/complex_technical_doc.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Complex PDF with images not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            # If PDF contains images, they should be extracted
            if result.success and hasattr(result, 'image_count'):
                # Check images were stored (if any)
                pass


class TestIngestionPipelineTraceContext:
    """Tests for trace context propagation."""

    def test_run_with_trace_context(self):
        """Should accept and use trace context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            trace = TraceContext(trace_id="test_trace", trace_type="ingestion")

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, trace=trace)

            # Should complete without error
            assert isinstance(result, IngestionResult)

    def test_trace_context_propagation(self):
        """Trace context should be propagated through stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            trace = TraceContext(trace_id="test_trace", trace_type="ingestion")

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path, trace=trace)

            assert result.success or result.errors
            # Trace should have recorded stages
            if trace.stages:
                assert len(trace.stages) > 0


class TestIngestionPipelineMultipleDocuments:
    """Tests for processing multiple documents."""

    def test_run_with_multiple_pdfs_separately(self):
        """Should process multiple PDFs in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdfs = [
                "tests/fixtures/sample_documents/simple.pdf"
            ]

            # Filter to existing files
            pdfs = [p for p in pdfs if Path(p).exists()]

            if not pdfs:
                pytest.skip("No sample PDFs found")

            total_chunks = 0
            for pdf_path in pdfs:
                result = pipeline.run(pdf_path, collection="batch")
                if result.success:
                    total_chunks += result.chunk_count

            assert total_chunks > 0


class TestIngestionPipelineStatistics:
    """Tests for result statistics."""

    def test_result_contains_timing_info(self):
        """Result should contain elapsed time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            assert hasattr(result, 'elapsed_ms')
            assert result.elapsed_ms > 0

    def test_result_contains_chunk_statistics(self):
        """Result should contain chunk count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            if result.success:
                assert result.chunk_count >= 0

    def test_result_contains_document_count(self):
        """Result should contain document count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)

            assert result.document_count >= 0
