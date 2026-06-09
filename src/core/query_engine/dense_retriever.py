"""Dense retriever implementation."""
from typing import Optional, Dict, Any, List
from src.core.types import RetrievalResult
from src.libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """Retriever using dense vector embeddings."""

    def __init__(
        self,
        embedding_client=None,
        vector_store: Optional[BaseVectorStore] = None,
        settings=None,
    ):
        """
        Initialize DenseRetriever.

        Args:
            embedding_client: Embedding client for generating query vectors
            vector_store: VectorStore instance for querying
            settings: Settings object (for lazy initialization if needed)
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.settings = settings

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using dense vector search.

        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters
            trace: Optional TraceContext for tracking

        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            return []

        try:
            # Generate query embedding
            query_vector = self.embedding_client.embed([query])[0]

            # Query vector store
            records = self.vector_store.query(
                vector=query_vector,
                top_k=top_k,
                filters=filters,
                trace=trace,
            )

            # Convert VectorRecord to RetrievalResult
            results = []
            for record in records:
                result = RetrievalResult(
                    chunk_id=record.id,
                    score=record.score if record.score is not None else 1.0,
                    text=record.text,
                    metadata=record.metadata or {},
                )
                results.append(result)

            if trace:
                trace.record_stage(
                    "dense_retrieval",
                    method="embedding",
                    results_count=len(results),
                )

            return results

        except Exception as e:
            raise RuntimeError(f"Dense retrieval failed: {str(e)}")
