"""SparseRetriever: BM25-based retrieval using inverted index."""

from typing import List, Dict, Optional, Any

from src.core.types import RetrievalResult
from src.core.settings import Settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """
    Sparse (BM25-based) retrieval engine.

    Workflow:
    1. Extract keywords from query
    2. Query BM25 index for each keyword
    3. Aggregate and rank results
    4. Fetch full text/metadata from vector store
    5. Return RetrievalResult list
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        Initialize SparseRetriever.

        Args:
            settings: Settings object for configuration
            bm25_indexer: BM25Indexer instance (optional for dependency injection in tests)
            vector_store: VectorStore instance (optional for dependency injection in tests)
        """
        self.settings = settings
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using BM25-based sparse search.

        Args:
            keywords: List of keywords to search
            top_k: Maximum number of results to return
            trace: Optional TraceContext for tracking

        Returns:
            List of RetrievalResult objects sorted by relevance score

        Raises:
            RuntimeError: If retrieval fails
        """
        if not keywords:
            return []

        # Collect results from all keywords
        results_by_chunk_id: Dict[str, Dict] = {}

        for keyword in keywords:
            if not self.bm25_indexer:
                continue

            # Query BM25 index
            bm25_results = self.bm25_indexer.query(
                term=keyword,
                top_k=top_k,
                trace=trace,
            )

            # Aggregate scores (use max score for same chunk from different keywords)
            for result in bm25_results:
                chunk_id = result["chunk_id"]
                score = result["score"]

                if chunk_id not in results_by_chunk_id:
                    results_by_chunk_id[chunk_id] = {
                        "chunk_id": chunk_id,
                        "score": score,
                    }
                else:
                    # Keep the highest score if same chunk appears in multiple keywords
                    results_by_chunk_id[chunk_id]["score"] = max(
                        results_by_chunk_id[chunk_id]["score"],
                        score,
                    )

        if not results_by_chunk_id:
            return []

        # Sort by score (descending) and take top_k
        sorted_results = sorted(
            results_by_chunk_id.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        # Extract chunk IDs in order
        chunk_ids = [r["chunk_id"] for r in sorted_results]
        score_map = {r["chunk_id"]: r["score"] for r in sorted_results}

        # Fetch full text and metadata from vector store
        if not self.vector_store:
            return []

        vector_records = self.vector_store.get_by_ids(
            ids=chunk_ids,
            trace=trace,
        )

        # Build RetrievalResult list maintaining order
        retrieval_results = []
        for record in vector_records:
            result = RetrievalResult(
                chunk_id=record.id,
                score=score_map.get(record.id, 0.0),
                text=record.text,
                metadata=record.metadata or {},
            )
            retrieval_results.append(result)

        return retrieval_results
