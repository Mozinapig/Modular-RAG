"""Base Reranker interface for provider abstraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseReranker(ABC):
    """Abstract base class for Reranker providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate results based on query.

        Args:
            query: The query string
            candidates: List of candidate dicts with at least 'id' and 'score' fields
            trace: Optional TraceContext for tracking

        Returns:
            Reranked list of candidates, same structure as input

        Raises:
            ValueError: If input validation fails
            RuntimeError: If reranking fails
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
