"""
Trace context for tracking request/ingestion flow.
Minimal implementation for Phase C; will be enhanced in Phase F.
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class StageRecord:
    """Record for a single stage in the trace."""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TraceContext:
    """Context for tracing a request or ingestion flow."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_type: str = "query"  # "query" or "ingestion"
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None
    stages: List[StageRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_stage(
        self,
        stage_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Record a stage in the trace.

        Args:
            stage_name: Name of the stage
            start_time: Start timestamp (optional, uses current time if None)
            end_time: End timestamp (optional, uses current time if None)
            metadata: Optional metadata for the stage
            **kwargs: Additional metadata to merge into metadata dict
        """
        # If timestamps not provided, use current time
        if start_time is None:
            start_time = datetime.now().timestamp()
        if end_time is None:
            end_time = datetime.now().timestamp()

        duration_ms = (end_time - start_time) * 1000

        # Merge kwargs into metadata
        merged_metadata = metadata.copy() if metadata else {}
        merged_metadata.update(kwargs)

        stage = StageRecord(
            stage_name=stage_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            metadata=merged_metadata
        )
        self.stages.append(stage)

    def add_stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a stage with current timestamp (convenience method).

        Args:
            stage_name: Name of the stage
            metadata: Optional metadata for the stage
        """
        current_time = datetime.now().timestamp()
        self.record_stage(stage_name, current_time, current_time, metadata)

    def finish(self) -> None:
        """Mark trace as finished."""
        self.end_time = datetime.now().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata
        }
