"""
HybridSearch: Combined dense and sparse retrieval with RRF fusion.

Workflow:
1. Process query to extract keywords and filters
2. Run dense and sparse retrieval in parallel
3. Fuse results using Reciprocal Rank Fusion
4. Apply metadata filters as post-processing
5. Return top-K results
"""

from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.types import RetrievalResult
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion


class HybridSearch:
    """
    Hybrid search combining dense (semantic) and sparse (keyword) retrieval.

    Uses RRF to merge rankings from both sources, preferring items that
    appear in both rankings.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion: RRFFusion,
    ):
        """
        Initialize HybridSearch.

        Args:
            query_processor: QueryProcessor for parsing queries
            dense_retriever: DenseRetriever for semantic search
            sparse_retriever: SparseRetriever for keyword search
            fusion: RRFFusion for combining results
        """
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query text
            top_k: Maximum results to return
            filters: Optional metadata filters to apply post-fusion
            trace: Optional TraceContext for tracking

        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        if not query or not query.strip():
            return []

        # Process query to extract keywords and filters
        processed_query = self.query_processor.process(query)

        # Run dense and sparse retrieval in parallel
        dense_results = []
        sparse_results = []

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(
                    self.dense_retriever.retrieve,
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    trace=trace,
                )
                sparse_future = executor.submit(
                    self.sparse_retriever.retrieve,
                    keywords=processed_query.keywords,
                    top_k=top_k,
                    trace=trace,
                )

                # Collect results with fallback handling
                try:
                    dense_results = dense_future.result()
                except Exception:
                    # Dense retrieval failed, fall back to sparse
                    dense_results = []

                try:
                    sparse_results = sparse_future.result()
                except Exception:
                    # Sparse retrieval failed, fall back to dense
                    sparse_results = []

        except Exception:
            # If parallel execution fails, try sequential as fallback
            try:
                dense_results = self.dense_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    trace=trace,
                )
            except Exception:
                dense_results = []

            try:
                sparse_results = self.sparse_retriever.retrieve(
                    keywords=processed_query.keywords,
                    top_k=top_k,
                    trace=trace,
                )
            except Exception:
                sparse_results = []

        # Fuse results from both sources
        fused_results = self.fusion.fuse(dense_results, sparse_results)

        # Apply metadata filters if provided
        if filters:
            fused_results = self._apply_metadata_filters(fused_results, filters)

        # Return top-K results
        return fused_results[:top_k]

    def _apply_metadata_filters(
        self,
        candidates: List[RetrievalResult],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """
        Apply metadata filtering to retrieval results.

        Filters results based on metadata field values. All filter conditions
        must be satisfied for a result to be included (AND logic).

        Args:
            candidates: List of RetrievalResult to filter
            filters: Dict of metadata field:value pairs to match

        Returns:
            Filtered list of RetrievalResult objects
        """
        if not filters:
            return candidates

        filtered = []
        for result in candidates:
            # Check if all filters match this result's metadata
            match = True
            for filter_key, filter_value in filters.items():
                if result.metadata.get(filter_key) != filter_value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered
