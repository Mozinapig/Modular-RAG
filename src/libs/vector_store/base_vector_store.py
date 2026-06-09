"""Base VectorStore interface for provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class VectorRecord:
    """Represents a vector record stored in the vector store."""
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None  # Similarity/relevance score from query


class BaseVectorStore(ABC):
    """Abstract base class for VectorStore providers."""

    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
    ) -> None:
        """
        Upsert (insert or update) vector records.

        Args:
            records: List of VectorRecord objects to store
            trace: Optional TraceContext for tracking

        Raises:
            ValueError: If input validation fails
            RuntimeError: If storage operation fails
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        """
        Query similar vectors.

        Args:
            vector: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters (dict of field:value pairs)
            trace: Optional TraceContext for tracking

        Returns:
            List of similar VectorRecord objects, sorted by similarity

        Raises:
            ValueError: If input validation fails
            RuntimeError: If query fails
        """
        pass

    @abstractmethod
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        """
        Get records by IDs.

        Args:
            ids: List of record IDs to retrieve
            trace: Optional TraceContext for tracking

        Returns:
            List of VectorRecord objects matching the IDs (in request order, missing IDs skipped)

        Raises:
            ValueError: If input validation fails
            RuntimeError: If retrieval fails
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
