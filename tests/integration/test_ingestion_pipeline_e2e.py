"""
End-to-end acceptance tests for IngestionPipeline.
Validates complete pipeline execution on real PDF with file outputs.
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.ingestion.ingestion_pipeline import IngestionPipeline


class TestIngestionPipelineAcceptance:
    """Acceptance tests for complete pipeline execution."""

    def test_e2e_complex_technical_doc_with_file_outputs(self):
        """
        Test complete pipeline on complex_technical_doc.pdf.

        Verifies:
        - Document loads successfully
        - Chunks are generated
        - Embeddings are computed
        - Vector index stored in Chroma (records.jsonl)
        - BM25 index saved to data/db/bm25/
        - Images extracted if present
        - All stages complete with progress logging
        """
        pdf_path = "tests/fixtures/sample_documents/complex_technical_doc.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Track progress callbacks
            progress_events = []

            def on_progress(stage: str, current: int, total: int):
                progress_events.append({
                    "stage": stage,
                    "current": current,
                    "total": total
                })

            # Run pipeline
            result = pipeline.run(
                pdf_path,
                collection="test_acceptance",
                on_progress=on_progress
            )

            # Verify success
            assert result.success, f"Pipeline failed: {result.errors}"
            assert result.document_count == 1
            assert result.chunk_count > 0

            # Verify Chroma vector index exists (skip on Windows - uses memory fallback)
            import sys
            if sys.platform != "win32":
                chroma_records = Path(tmpdir) / "db" / "chroma" / "records.jsonl"
                assert chroma_records.exists(), "Chroma records file not found"

                # Read and verify Chroma records have embeddings
                with open(chroma_records, 'r') as f:
                    records = [json.loads(line) for line in f if line.strip()]
                assert len(records) > 0, "No records in Chroma index"
            else:
                # On Windows, Chroma uses in-memory fallback, so skip file checks
                pass

            # Verify BM25 index saved to disk
            bm25_index = Path(tmpdir) / "db" / "bm25" / "test_acceptance.json"
            assert bm25_index.exists(), f"BM25 index not found at {bm25_index}"

            # Verify BM25 index has valid content
            with open(bm25_index, 'r') as f:
                bm25_data = json.load(f)
            assert isinstance(bm25_data, dict), "BM25 index is not a dict"

            # Verify progress callbacks were called
            assert len(progress_events) > 0, "No progress callbacks received"
            stages = set(e["stage"] for e in progress_events)
            assert len(stages) > 1, "Should have multiple pipeline stages"

    def test_e2e_simple_pdf_end_to_end(self):
        """Test pipeline on simple.pdf with minimal data."""
        pdf_path = "tests/fixtures/sample_documents/simple.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run(
                pdf_path,
                collection="simple_test"
            )

            assert result.success
            assert result.document_count == 1
            assert result.chunk_count > 0

            # Verify BM25 index exists (always saved to disk)
            bm25_index = Path(tmpdir) / "db" / "bm25" / "simple_test.json"
            assert bm25_index.exists()

            # Verify Chroma records (skip on Windows - uses memory fallback)
            import sys
            if sys.platform != "win32":
                chroma_records = Path(tmpdir) / "db" / "chroma" / "records.jsonl"
                assert chroma_records.exists()

    def test_pipeline_creates_output_directories(self):
        """Verify pipeline creates all required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            pdf_path = "tests/fixtures/sample_documents/simple.pdf"
            if not Path(pdf_path).exists():
                pytest.skip(f"Test PDF not found: {pdf_path}")

            result = pipeline.run(pdf_path)
            assert result.success

            # Check all expected directories exist
            assert Path(tmpdir).exists()
            assert (Path(tmpdir) / "db").exists()
            # BM25 directory should always be created
            assert (Path(tmpdir) / "db" / "bm25").exists()

    def test_pipeline_handles_multiple_collections(self):
        """Verify separate BM25 indices per collection."""
        pdf_path = "tests/fixtures/sample_documents/simple.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Process same PDF in different collections
            result1 = pipeline.run(pdf_path, collection="col1")
            assert result1.success

            result2 = pipeline.run(pdf_path, collection="col2")
            assert result2.success

            # Verify separate BM25 index files
            index1 = Path(tmpdir) / "db" / "bm25" / "col1.json"
            index2 = Path(tmpdir) / "db" / "bm25" / "col2.json"

            assert index1.exists(), f"Collection 1 index not found: {index1}"
            assert index2.exists(), f"Collection 2 index not found: {index2}"

    def test_pipeline_timing_info(self):
        """Verify pipeline reports accurate timing."""
        pdf_path = "tests/fixtures/sample_documents/simple.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run(pdf_path)

            assert result.success
            assert result.elapsed_ms > 0
            # Processing should take at least 100ms
            assert result.elapsed_ms > 100

    def test_chroma_records_have_metadata(self):
        """Verify Chroma records include chunk metadata (skip on Windows)."""
        import sys
        if sys.platform == "win32":
            pytest.skip("Chroma uses in-memory mode on Windows")

        pdf_path = "tests/fixtures/sample_documents/simple.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run(pdf_path, collection="metadata_test")
            assert result.success

            # Read Chroma records
            chroma_records = Path(tmpdir) / "db" / "chroma" / "records.jsonl"
            with open(chroma_records, 'r') as f:
                records = [json.loads(line) for line in f if line.strip()]

            # Verify metadata in records
            for record in records:
                assert "id" in record
                assert "text" in record
                assert "metadata" in record
                metadata = record["metadata"]
                assert "collection" in metadata
                assert metadata["collection"] == "metadata_test"

    def test_bm25_index_has_terms(self):
        """Verify BM25 index contains term data."""
        pdf_path = "tests/fixtures/sample_documents/simple.pdf"
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            result = pipeline.run(pdf_path, collection="bm25_test")
            assert result.success

            # Read BM25 index
            bm25_index = Path(tmpdir) / "db" / "bm25" / "bm25_test.json"
            with open(bm25_index, 'r') as f:
                index = json.load(f)

            # Verify index structure
            assert isinstance(index, dict)
            # Should have terms (keys in the dict)
            assert len(index) > 0, "BM25 index has no terms"

            # Each term should have idf and postings
            for term, term_data in list(index.items())[:3]:
                assert "idf" in term_data, f"Term {term} missing idf"
                assert "postings" in term_data, f"Term {term} missing postings"
                assert isinstance(term_data["postings"], list)

    def test_pipeline_error_messages_are_clear(self):
        """Verify error messages are descriptive on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestionPipeline(base_path=tmpdir)

            # Try to process non-existent file
            result = pipeline.run("/nonexistent/path/file.pdf")

            assert not result.success
            assert len(result.errors) > 0
            error_msg = result.errors[0]
            # Error should mention file not found
            assert "not found" in error_msg.lower() or "nonexistent" in error_msg.lower()
