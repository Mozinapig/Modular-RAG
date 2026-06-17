"""
Unit tests for TraceService - reads and parses trace logs
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.observability.dashboard.services.trace_service import TraceService


class TestTraceServiceLoadTraces:
    """Test TraceService.load_traces() - reads traces.jsonl"""

    def test_load_traces_from_file(self):
        """Should load traces from JSONL file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"

            # Write test traces
            traces_data = [
                {
                    "trace_id": "trace_1",
                    "trace_type": "ingestion",
                    "timestamp": "2026-06-17T10:00:00Z",
                    "stages": [
                        {"name": "load", "start_ms": 0, "end_ms": 100},
                        {"name": "split", "start_ms": 100, "end_ms": 250},
                    ]
                },
                {
                    "trace_id": "trace_2",
                    "trace_type": "query",
                    "timestamp": "2026-06-17T10:05:00Z",
                    "stages": [
                        {"name": "process", "start_ms": 0, "end_ms": 50},
                    ]
                }
            ]

            with open(trace_file, "w") as f:
                for trace in traces_data:
                    f.write(json.dumps(trace) + "\n")

            # Load traces
            service = TraceService(log_path=str(tmpdir))
            traces = service.load_traces()

            assert len(traces) == 2
            assert traces[0]["trace_type"] == "ingestion"
            assert traces[1]["trace_type"] == "query"

    def test_load_traces_empty_file(self):
        """Should return empty list when trace file is empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = TraceService(log_path=str(tmpdir))
            traces = service.load_traces()

            assert traces == []

    def test_load_traces_file_not_found(self):
        """Should return empty list when trace file does not exist"""
        service = TraceService(log_path="/nonexistent/path")
        traces = service.load_traces()

        assert traces == []


class TestTraceServiceFilterByType:
    """Test TraceService.filter_by_type() - filters ingestion/query traces"""

    def test_filter_ingestion_traces(self):
        """Should filter traces by trace_type"""
        traces = [
            {"trace_id": "1", "trace_type": "ingestion"},
            {"trace_id": "2", "trace_type": "query"},
            {"trace_id": "3", "trace_type": "ingestion"},
        ]

        service = TraceService(log_path="/tmp")
        filtered = service.filter_by_type(traces, trace_type="ingestion")

        assert len(filtered) == 2
        assert all(t["trace_type"] == "ingestion" for t in filtered)

    def test_filter_query_traces(self):
        """Should filter query traces"""
        traces = [
            {"trace_id": "1", "trace_type": "ingestion"},
            {"trace_id": "2", "trace_type": "query"},
            {"trace_id": "3", "trace_type": "query"},
        ]

        service = TraceService(log_path="/tmp")
        filtered = service.filter_by_type(traces, trace_type="query")

        assert len(filtered) == 2
        assert all(t["trace_type"] == "query" for t in filtered)


class TestTraceServiceCalculateStageDuration:
    """Test TraceService.calculate_stage_duration()"""

    def test_calculate_stage_duration(self):
        """Should calculate duration for stages"""
        trace = {
            "stages": [
                {"name": "load", "start_ms": 0, "end_ms": 100},
                {"name": "split", "start_ms": 100, "end_ms": 250},
                {"name": "embed", "start_ms": 250, "end_ms": 500},
            ]
        }

        service = TraceService(log_path="/tmp")
        result = service.calculate_stage_duration(trace)

        assert result["load"] == 100
        assert result["split"] == 150
        assert result["embed"] == 250

    def test_calculate_stage_duration_empty_stages(self):
        """Should handle empty stages"""
        trace = {"stages": []}

        service = TraceService(log_path="/tmp")
        result = service.calculate_stage_duration(trace)

        assert result == {}


class TestTraceServiceSortByTimestamp:
    """Test TraceService.sort_by_timestamp()"""

    def test_sort_traces_by_timestamp(self):
        """Should sort traces in descending order by timestamp"""
        traces = [
            {"trace_id": "1", "timestamp": "2026-06-17T10:00:00Z"},
            {"trace_id": "2", "timestamp": "2026-06-17T10:05:00Z"},
            {"trace_id": "3", "timestamp": "2026-06-17T10:01:00Z"},
        ]

        service = TraceService(log_path="/tmp")
        sorted_traces = service.sort_by_timestamp(traces, descending=True)

        # Should be in descending order
        assert sorted_traces[0]["trace_id"] == "2"
        assert sorted_traces[1]["trace_id"] == "3"
        assert sorted_traces[2]["trace_id"] == "1"

    def test_sort_traces_ascending(self):
        """Should sort traces in ascending order"""
        traces = [
            {"trace_id": "1", "timestamp": "2026-06-17T10:00:00Z"},
            {"trace_id": "2", "timestamp": "2026-06-17T10:05:00Z"},
            {"trace_id": "3", "timestamp": "2026-06-17T10:01:00Z"},
        ]

        service = TraceService(log_path="/tmp")
        sorted_traces = service.sort_by_timestamp(traces, descending=False)

        # Should be in ascending order
        assert sorted_traces[0]["trace_id"] == "1"
        assert sorted_traces[1]["trace_id"] == "3"
        assert sorted_traces[2]["trace_id"] == "2"


class TestTraceServiceGetTraceDetail:
    """Test TraceService.get_trace_detail()"""

    def test_get_trace_detail(self):
        """Should retrieve detailed info for a trace"""
        traces = [
            {
                "trace_id": "trace_1",
                "trace_type": "ingestion",
                "timestamp": "2026-06-17T10:00:00Z",
                "stages": [
                    {"name": "load", "start_ms": 0, "end_ms": 100},
                    {"name": "split", "start_ms": 100, "end_ms": 250},
                ]
            }
        ]

        service = TraceService(log_path="/tmp")
        with patch.object(service, 'load_traces', return_value=traces):
            detail = service.get_trace_detail("trace_1")

            assert detail["trace_id"] == "trace_1"
            assert len(detail["stages"]) == 2

    def test_get_trace_detail_not_found(self):
        """Should return None when trace not found"""
        service = TraceService(log_path="/tmp")
        with patch.object(service, 'load_traces', return_value=[]):
            detail = service.get_trace_detail("nonexistent")

            assert detail is None
