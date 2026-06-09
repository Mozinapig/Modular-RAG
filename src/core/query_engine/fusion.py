"""
RRF (Reciprocal Rank Fusion) implementation for ranking fusion.

Combines dense and sparse retrieval rankings using the RRF formula:
    score = sum(1 / (k + rank))

where k is a configurable parameter (default 60) and rank is 1-indexed position.
"""

from typing import List
from src.core.types import RetrievalResult


class RRFFusion:
    """RRF-based fusion of dense and sparse rankings.

    Merges two ranked lists (dense and sparse retrieval results) into a single
    ranked list using Reciprocal Rank Fusion formula.

    Attributes:
        k: Configurable parameter for RRF formula (default 60, commonly used value)
    """

    def __init__(self, k: int = 60):
        """Initialize RRF fusion.

        Args:
            k: Parameter for RRF formula (default 60). Prevents score explosion
               for top-ranked items and provides smoother score distribution.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.k = k

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Fuse dense and sparse retrieval results using RRF.

        Combines rankings from two sources and produces a unified ranking
        where items appearing in both sources rank higher.

        Args:
            dense_results: Ranked list from dense (semantic) retrieval
            sparse_results: Ranked list from sparse (keyword) retrieval

        Returns:
            Merged and re-ranked list of RetrievalResult objects, sorted by
            RRF score in descending order.
        """
        # Handle edge cases
        if not dense_results and not sparse_results:
            return []

        if not dense_results:
            return sparse_results

        if not sparse_results:
            return dense_results

        # Calculate RRF scores for each chunk
        rrf_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            score = 1.0 / (self.k + rank)
            if result.chunk_id not in rrf_scores:
                rrf_scores[result.chunk_id] = {
                    "score": 0.0,
                    "result": result,
                }
            rrf_scores[result.chunk_id]["score"] += score

        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            score = 1.0 / (self.k + rank)
            if result.chunk_id not in rrf_scores:
                rrf_scores[result.chunk_id] = {
                    "score": 0.0,
                    "result": result,
                }
            rrf_scores[result.chunk_id]["score"] += score

        # Create fused results with updated scores
        fused_results = []
        for chunk_id in sorted(rrf_scores.keys(),
                             key=lambda cid: rrf_scores[cid]["score"],
                             reverse=True):
            original_result = rrf_scores[chunk_id]["result"]
            fused_result = RetrievalResult(
                chunk_id=original_result.chunk_id,
                score=rrf_scores[chunk_id]["score"],
                text=original_result.text,
                metadata=original_result.metadata,
            )
            fused_results.append(fused_result)

        return fused_results
