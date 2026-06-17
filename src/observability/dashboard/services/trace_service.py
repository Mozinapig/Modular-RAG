"""
TraceService - Reads and parses trace logs from traces.jsonl
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class TraceService:
    """
    Service layer for trace data access.
    Reads and parses traces.jsonl for Dashboard visualization.
    """

    def __init__(self, log_path: str = "logs"):
        """
        Initialize TraceService.

        Args:
            log_path: Path to logs directory containing traces.jsonl
        """
        self.log_path = Path(log_path)
        self.trace_file = self.log_path / "traces.jsonl"

    def load_traces(self) -> List[Dict[str, Any]]:
        """
        Load all traces from traces.jsonl.

        Returns:
            List of trace dictionaries
        """
        if not self.trace_file.exists():
            return []

        traces = []
        try:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            trace = json.loads(line)
                            traces.append(trace)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading traces: {e}")
            return []

        return traces

    def filter_by_type(
        self,
        traces: List[Dict[str, Any]],
        trace_type: str
    ) -> List[Dict[str, Any]]:
        """
        Filter traces by type (ingestion/query).

        Args:
            traces: List of traces
            trace_type: Type to filter by ("ingestion" or "query")

        Returns:
            Filtered traces
        """
        return [t for t in traces if t.get("trace_type") == trace_type]

    def sort_by_timestamp(
        self,
        traces: List[Dict[str, Any]],
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sort traces by timestamp.

        Args:
            traces: List of traces
            descending: Sort in descending order (newest first)

        Returns:
            Sorted traces
        """
        try:
            return sorted(
                traces,
                key=lambda t: datetime.fromisoformat(t.get("timestamp", "")),
                reverse=descending
            )
        except Exception as e:
            print(f"Error sorting traces: {e}")
            return traces

    def calculate_stage_duration(self, trace: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate duration for each stage in a trace.

        Args:
            trace: Trace dictionary

        Returns:
            Dictionary mapping stage name to duration in ms
        """
        result = {}
        stages = trace.get("stages", [])

        for stage in stages:
            name = stage.get("name", "")
            start_ms = stage.get("start_ms", 0)
            end_ms = stage.get("end_ms", 0)

            if name:
                result[name] = end_ms - start_ms

        return result

    def get_trace_detail(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific trace.

        Args:
            trace_id: Trace ID to retrieve

        Returns:
            Trace detail or None if not found
        """
        traces = self.load_traces()

        for trace in traces:
            if trace.get("trace_id") == trace_id:
                return trace

        return None

    def get_ingestion_traces(self) -> List[Dict[str, Any]]:
        """
        Get all ingestion traces sorted by timestamp (newest first).

        Returns:
            List of ingestion traces
        """
        traces = self.load_traces()
        ingestion_traces = self.filter_by_type(traces, "ingestion")
        return self.sort_by_timestamp(ingestion_traces, descending=True)

    def get_query_traces(self) -> List[Dict[str, Any]]:
        """
        Get all query traces sorted by timestamp (newest first).

        Returns:
            List of query traces
        """
        traces = self.load_traces()
        query_traces = self.filter_by_type(traces, "query")
        return self.sort_by_timestamp(query_traces, descending=True)

    def search_traces_by_document(
        self,
        document_path: str,
        trace_type: str = "ingestion"
    ) -> List[Dict[str, Any]]:
        """
        Search for traces related to a specific document.

        Args:
            document_path: Source path to search for
            trace_type: Type of trace to search

        Returns:
            List of matching traces
        """
        traces = self.load_traces()
        filtered = self.filter_by_type(traces, trace_type)

        # Filter by document in metadata
        results = []
        for trace in filtered:
            metadata = trace.get("metadata", {})
            if metadata.get("source_path") == document_path:
                results.append(trace)

        return self.sort_by_timestamp(results, descending=True)

    def calculate_total_duration(self, trace: Dict[str, Any]) -> int:
        """
        Calculate total duration for a trace.

        Args:
            trace: Trace dictionary

        Returns:
            Total duration in ms
        """
        stages = trace.get("stages", [])
        if not stages:
            return 0

        # Get last stage's end time
        last_stage = stages[-1]
        return last_stage.get("end_ms", 0)
