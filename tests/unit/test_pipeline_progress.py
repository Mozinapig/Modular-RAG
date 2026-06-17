"""
Unit tests for Pipeline progress callback (F5).
Verifies that on_progress callback is correctly invoked during ingestion.
"""
import pytest
from unittest.mock import Mock, patch, call

from src.core.trace.trace_context import TraceContext
from src.ingestion.ingestion_pipeline import IngestionPipeline


class TestPipelineProgressCallback:
    """Test progress callback functionality in IngestionPipeline."""

    def test_on_progress_called_for_load_stage(self):
        """Test that on_progress is called for load stage."""
        pipeline = IngestionPipeline()
        mock_progress = Mock()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", on_progress=mock_progress)

        # Verify on_progress was called for load stage
        calls = [c for c in mock_progress.call_args_list if c[0][0] == "load"]
        assert len(calls) >= 1

    def test_on_progress_called_for_all_stages(self):
        """Test that on_progress is called for all pipeline stages."""
        pipeline = IngestionPipeline()
        mock_progress = Mock()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", on_progress=mock_progress)

        # Extract stage names from all calls
        stages_called = set()
        for call_obj in mock_progress.call_args_list:
            if call_obj[0]:  # Has positional args
                stages_called.add(call_obj[0][0])

        # All major stages should be represented
        expected_stages = {"load", "split", "transform", "encode", "store"}
        assert expected_stages.issubset(stages_called), f"Missing stages: {expected_stages - stages_called}"

    def test_on_progress_receives_correct_parameters(self):
        """Test that on_progress receives correct parameters (stage, current, total)."""
        pipeline = IngestionPipeline()
        mock_progress = Mock()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", on_progress=mock_progress)

        # Verify all calls have 3 arguments
        for call_obj in mock_progress.call_args_list:
            assert len(call_obj[0]) == 3, f"Expected 3 args, got {len(call_obj[0])}"
            stage, current, total = call_obj[0]
            assert isinstance(stage, str)
            assert isinstance(current, int)
            assert isinstance(total, int)
            assert 0 <= current <= total

    def test_on_progress_not_called_when_none(self):
        """Test that pipeline works when on_progress is None."""
        pipeline = IngestionPipeline()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            # Should not raise exception
                            result = pipeline.run("test.pdf", on_progress=None)
                            assert result is not None

    def test_on_progress_with_trace_context(self):
        """Test that on_progress and trace can coexist."""
        pipeline = IngestionPipeline()
        mock_progress = Mock()
        trace = TraceContext(trace_type="ingestion")

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", trace=trace, on_progress=mock_progress)

        # Both should have been used
        assert mock_progress.call_count > 0
        assert len(trace.stages) > 0

    def test_on_progress_called_before_and_after_stages(self):
        """Test that on_progress is called before (0,1) and after (1,1) each stage."""
        pipeline = IngestionPipeline()
        mock_progress = Mock()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", on_progress=mock_progress)

        # Check that for "load" stage we have (0,1) and (1,1) calls
        load_calls = [c for c in mock_progress.call_args_list if c[0][0] == "load"]
        assert len(load_calls) >= 2

        # Verify start and end calls
        start_call = load_calls[0]
        assert start_call[0][1] == 0 and start_call[0][2] == 1  # (load, 0, 1)

    def test_progress_callback_signature(self):
        """Test that progress callback has correct signature."""
        # Define a proper progress callback with correct signature
        progress_data = []

        def my_progress(stage: str, current: int, total: int):
            progress_data.append((stage, current, total))

        pipeline = IngestionPipeline()

        with patch.object(pipeline, '_load_document') as mock_load:
            with patch.object(pipeline, '_chunk_document') as mock_chunk:
                with patch.object(pipeline, '_transform_chunks') as mock_transform:
                    with patch.object(pipeline, '_encode_chunks') as mock_encode:
                        with patch.object(pipeline, '_store_chunks') as mock_store:
                            mock_load.return_value = Mock(text="Test", metadata={})
                            mock_chunk.return_value = [Mock(text="chunk")]
                            mock_transform.return_value = [Mock(text="chunk")]
                            mock_encode.return_value = [Mock(text="chunk")]
                            mock_store.return_value = Mock()

                            pipeline.run("test.pdf", on_progress=my_progress)

        # Verify callback was invoked with proper data
        assert len(progress_data) > 0
        for stage, current, total in progress_data:
            assert isinstance(stage, str)
            assert isinstance(current, int)
            assert isinstance(total, int)
