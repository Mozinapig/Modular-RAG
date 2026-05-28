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
        start_time: float,
        end_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a stage in the trace.

        Args:
            stage_name: Name of the stage
            start_time: Start timestamp
            end_time: End timestamp
            metadata: Optional metadata for the stage
        """
        duration_ms = (end_time - start_time) * 1000
        stage = StageRecord(
            stage_name=stage_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        self.stages.append(stage)

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
