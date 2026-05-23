"""Base Embedding interface for provider abstraction."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseEmbedding(ABC):
    """Abstract base class for Embedding providers."""

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            trace: Optional TraceContext for tracking

        Returns:
            List of embedding vectors (each vector is List[float])

        Raises:
            ValueError: If input validation fails
            RuntimeError: If embedding service fails
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
