"""
Trace collector for collecting and persisting traces.
"""

from typing import Optional, List, Dict, Any
from .trace_context import TraceContext


class TraceCollector:
    """Collects and manages traces."""

    def __init__(self):
        """Initialize trace collector."""
        self._traces: List[Dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        """
        Collect a trace.

        Args:
            trace: TraceContext instance to collect
        """
        trace_dict = trace.to_dict()
        self._traces.append(trace_dict)

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all collected traces."""
        return self._traces.copy()

    def get_traces_by_type(self, trace_type: str) -> List[Dict[str, Any]]:
        """Get traces by type."""
        return [t for t in self._traces if t.get("trace_type") == trace_type]

    def clear(self) -> None:
        """Clear all collected traces."""
        self._traces.clear()
