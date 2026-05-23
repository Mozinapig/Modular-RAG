"""Base Evaluator interface for metrics calculation abstraction."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseEvaluator(ABC):
    """Abstract base class for Evaluator providers."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against golden standard.

        Args:
            query: The query string
            retrieved_ids: List of IDs retrieved by the system
            golden_ids: List of IDs that are golden standard (expected correct)
            trace: Optional TraceContext for tracking

        Returns:
            Dictionary of metrics (e.g., {"hit_rate": 0.8, "mrr": 0.5})

        Raises:
            ValueError: If input validation fails
            RuntimeError: If evaluation fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass
