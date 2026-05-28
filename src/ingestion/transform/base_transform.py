"""
Abstract base class for chunk transformation/enrichment.
Defines the contract for all transform implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract base class for chunk transformations."""

    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform a list of chunks.

        Args:
            chunks: List of chunks to transform
            trace: Optional trace context for tracking

        Returns:
            List of transformed chunks

        Raises:
            ValueError: If transformation fails
        """
        pass
