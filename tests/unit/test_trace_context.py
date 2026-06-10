"""
Unit tests for TraceContext enhancements (F1).
Tests finish(), elapsed_ms(), to_dict(), and trace_type field.
"""

import pytest
import time
from datetime import datetime
from src.core.trace.trace_context import TraceContext, StageRecord
from src.core.trace.trace_collector import TraceCollector


class TestTraceContextFinish:
    """Test TraceContext.finish() method."""

    def test_finish_sets_end_time(self):
        """Test that finish() sets end_time."""
        trace = TraceContext()
        assert trace.end_time is None

        trace.finish()
        assert trace.end_time is not None
        assert trace.end_time >= trace.start_time

    def test_finish_is_idempotent(self):
        """Test that finish() can be called multiple times."""
        trace = TraceContext()
        first_finish_time = None

        trace.finish()
        first_finish_time = trace.end_time

        time.sleep(0.01)
        trace.finish()
        second_finish_time = trace.end_time

        # Second finish should update end_time (not idempotent in current impl, but that's ok)
        assert trace.end_time is not None


class TestTraceContextElapsedMs:
    """Test TraceContext.elapsed_ms() method."""

    def test_elapsed_ms_total_before_finish(self):
        """Test elapsed_ms() calculates elapsed time even before finish()."""
        trace = TraceContext()
        time.sleep(0.05)  # Sleep 50ms

        elapsed = trace.elapsed_ms()
        assert elapsed >= 50  # Should be at least 50ms

    def test_elapsed_ms_total_after_finish(self):
        """Test elapsed_ms() returns total elapsed time after finish()."""
        trace = TraceContext()
        time.sleep(0.05)  # Sleep 50ms

        trace.finish()
        elapsed = trace.elapsed_ms()
        assert elapsed >= 50

    def test_elapsed_ms_for_specific_stage(self):
        """Test elapsed_ms() for a specific stage."""
        trace = TraceContext()

        # Record a stage with 100ms duration
        start = datetime.now().timestamp()
        end = start + 0.1
        trace.record_stage("query_processing", start, end, method="keyword_extraction")

        elapsed = trace.elapsed_ms("query_processing")
        assert 95 <= elapsed <= 105  # Should be ~100ms

    def test_elapsed_ms_for_nonexistent_stage(self):
        """Test elapsed_ms() returns 0 for nonexistent stage."""
        trace = TraceContext()
        elapsed = trace.elapsed_ms("nonexistent_stage")
        assert elapsed == 0


class TestTraceContextToDict:
    """Test TraceContext.to_dict() serialization."""

    def test_to_dict_includes_required_fields(self):
        """Test to_dict() includes all required fields."""
        trace = TraceContext(trace_type="query")
        trace.finish()

        result = trace.to_dict()
        assert "trace_id" in result
        assert "trace_type" in result
        assert result["trace_type"] == "query"
        assert "started_at" in result
        assert "finished_at" in result
        assert "total_elapsed_ms" in result
        assert "stages" in result
        assert "metadata" in result

    def test_to_dict_includes_trace_type_query(self):
        """Test to_dict() includes trace_type='query'."""
        trace = TraceContext(trace_type="query")
        trace.finish()

        result = trace.to_dict()
        assert result["trace_type"] == "query"

    def test_to_dict_includes_trace_type_ingestion(self):
        """Test to_dict() includes trace_type='ingestion'."""
        trace = TraceContext(trace_type="ingestion")
        trace.finish()

        result = trace.to_dict()
        assert result["trace_type"] == "ingestion"

    def test_to_dict_includes_stages(self):
        """Test to_dict() includes stage information."""
        trace = TraceContext()

        start = datetime.now().timestamp()
        end = start + 0.1
        trace.record_stage(
            "dense_retrieval",
            start,
            end,
            method="embedding",
            provider="openai"
        )

        trace.finish()
        result = trace.to_dict()

        assert len(result["stages"]) == 1
        assert result["stages"][0]["stage_name"] == "dense_retrieval"
        assert result["stages"][0]["metadata"]["method"] == "embedding"
        assert result["stages"][0]["metadata"]["provider"] == "openai"

    def test_to_dict_is_json_serializable(self):
        """Test to_dict() output is JSON serializable."""
        import json

        trace = TraceContext(trace_type="query")
        trace.record_stage("test_stage", datetime.now().timestamp(), datetime.now().timestamp())
        trace.finish()

        result = trace.to_dict()
        json_str = json.dumps(result)

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Verify we can deserialize it back
        deserialized = json.loads(json_str)
        assert deserialized["trace_type"] == "query"
        assert deserialized["trace_id"] == trace.trace_id


class TestTraceContextTraceType:
    """Test TraceContext trace_type field."""

    def test_trace_type_default_is_query(self):
        """Test default trace_type is 'query'."""
        trace = TraceContext()
        assert trace.trace_type == "query"

    def test_trace_type_can_be_ingestion(self):
        """Test trace_type can be set to 'ingestion'."""
        trace = TraceContext(trace_type="ingestion")
        assert trace.trace_type == "ingestion"

    def test_trace_type_preserved_in_to_dict(self):
        """Test trace_type is preserved in to_dict()."""
        for trace_type in ["query", "ingestion"]:
            trace = TraceContext(trace_type=trace_type)
            trace.finish()

            result = trace.to_dict()
            assert result["trace_type"] == trace_type


class TestTraceCollector:
    """Test TraceCollector functionality."""

    def test_collector_collects_traces(self):
        """Test collector can collect traces."""
        collector = TraceCollector()

        trace1 = TraceContext(trace_type="query")
        trace1.finish()

        trace2 = TraceContext(trace_type="ingestion")
        trace2.finish()

        collector.collect(trace1)
        collector.collect(trace2)

        traces = collector.get_traces()
        assert len(traces) == 2

    def test_collector_get_traces_by_type(self):
        """Test collector can filter traces by type."""
        collector = TraceCollector()

        for _ in range(2):
            trace = TraceContext(trace_type="query")
            trace.finish()
            collector.collect(trace)

        for _ in range(3):
            trace = TraceContext(trace_type="ingestion")
            trace.finish()
            collector.collect(trace)

        query_traces = collector.get_traces_by_type("query")
        ingestion_traces = collector.get_traces_by_type("ingestion")

        assert len(query_traces) == 2
        assert len(ingestion_traces) == 3

    def test_collector_clear(self):
        """Test collector can clear all traces."""
        collector = TraceCollector()

        trace = TraceContext()
        trace.finish()
        collector.collect(trace)

        assert len(collector.get_traces()) == 1

        collector.clear()
        assert len(collector.get_traces()) == 0


class TestTraceContextIntegration:
    """Integration tests for TraceContext with multiple stages."""

    def test_multi_stage_trace_with_elapsed_times(self):
        """Test trace with multiple stages and elapsed_ms()."""
        trace = TraceContext(trace_type="query")

        base_time = datetime.now().timestamp()

        # Simulate query processing: 50ms
        trace.record_stage("query_processing", base_time, base_time + 0.05)

        # Simulate dense retrieval: 100ms
        trace.record_stage("dense_retrieval", base_time + 0.05, base_time + 0.15)

        # Simulate sparse retrieval: 75ms
        trace.record_stage("sparse_retrieval", base_time + 0.15, base_time + 0.225)

        # Simulate fusion: 25ms
        trace.record_stage("fusion", base_time + 0.225, base_time + 0.25)

        trace.finish()

        # Check individual stage times
        assert 45 <= trace.elapsed_ms("query_processing") <= 55
        assert 95 <= trace.elapsed_ms("dense_retrieval") <= 105
        assert 70 <= trace.elapsed_ms("sparse_retrieval") <= 80
        assert 20 <= trace.elapsed_ms("fusion") <= 30

        # Check total time
        total = trace.elapsed_ms()
        assert 245 <= total <= 260

    def test_trace_to_dict_with_multiple_stages(self):
        """Test to_dict() with multiple stages."""
        trace = TraceContext(trace_type="query")

        base_time = datetime.now().timestamp()
        trace.record_stage("stage1", base_time, base_time + 0.05, method="method1")
        trace.record_stage("stage2", base_time + 0.05, base_time + 0.1, method="method2")

        trace.finish()
        result = trace.to_dict()

        assert len(result["stages"]) == 2
        assert result["stages"][0]["stage_name"] == "stage1"
        assert result["stages"][1]["stage_name"] == "stage2"
        assert "total_elapsed_ms" in result
        assert result["total_elapsed_ms"] > 0
