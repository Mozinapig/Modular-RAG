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

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        Get elapsed time in milliseconds.

        Args:
            stage_name: If provided, return elapsed time for that specific stage.
                       If None, return total elapsed time.

        Returns:
            Elapsed time in milliseconds
        """
        if stage_name:
            # Find the stage and return its duration
            for stage in self.stages:
                if stage.stage_name == stage_name:
                    return stage.duration_ms if stage.duration_ms is not None else 0
            return 0

        # Return total elapsed time
        if self.stages:
            # Calculate from start_time to the end of last stage
            last_stage = max(self.stages, key=lambda s: s.end_time if s.end_time else 0)
            end_time = last_stage.end_time if last_stage.end_time else datetime.now().timestamp()
            return (end_time - self.start_time) * 1000

        if self.end_time is None:
            # If not finished, calculate from now
            current_time = datetime.now().timestamp()
            return (current_time - self.start_time) * 1000

        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Calculate total elapsed time based on stages if available
        if self.stages:
            last_stage = max(self.stages, key=lambda s: s.end_time if s.end_time else 0)
            end_time = last_stage.end_time if last_stage.end_time else self.end_time
            total_elapsed_ms = (end_time - self.start_time) * 1000
        elif self.end_time:
            total_elapsed_ms = (self.end_time - self.start_time) * 1000
        else:
            total_elapsed_ms = (datetime.now().timestamp() - self.start_time) * 1000

        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.start_time,
            "finished_at": self.end_time,
            "total_elapsed_ms": total_elapsed_ms,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata
        }
