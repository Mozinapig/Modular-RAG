"""
Core-layer Reranker orchestrator with fallback logic.

Handles:
1. Reranking through pluggable libs.reranker backends
2. Graceful fallback to fusion results on failure
3. Trace context propagation
4. Top-M candidate limiting
"""

from typing import List, Optional, Any

from src.core.types import RetrievalResult
from src.libs.reranker.reranker_factory import RerankerFactory


class CoreReranker:
    """Core layer Reranker with graceful fallback mechanism.

    Orchestrates the reranking phase of the query pipeline:
    - Receives candidate results from fusion stage
    - Applies optional reranking via configurable backend
    - Falls back to original results on backend failure
    - Propagates trace context through the pipeline

    Attributes:
        settings: System settings containing rerank configuration
        reranker: Initialized reranker backend (if applicable)
        fallback_enabled: Whether fallback should be used on errors
    """

    def __init__(self, settings):
        """Initialize Core Reranker.

        Args:
            settings: Settings object with rerank configuration
                - backend: "none" | "cross_encoder" | "llm"
                - top_m: Number of top candidates to rerank (optional)
                - timeout: Timeout in seconds (optional)
                - Other backend-specific configs

        Raises:
            ValueError: If settings validation fails
        """
        self.settings = settings
        self.reranker = None
        self.fallback_enabled = True
        self._initialization_error = None

        # Initialize backend if not "none"
        if self.settings.rerank.backend != "none":
            try:
                self.reranker = RerankerFactory().create(self.settings.rerank)
            except Exception as e:
                # Log initialization error but continue (fallback will be used)
                self._initialization_error = str(e)
                self.reranker = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidates using configured backend with fallback.

        Args:
            query: User query string
            candidates: List of RetrievalResult from fusion stage
            trace: Optional TraceContext for tracking

        Returns:
            Reranked list of RetrievalResult objects (or original if backend unavailable)
        """
        # Handle empty input
        if not candidates:
            return []

        # If backend is "none" or not initialized, pass through or fallback
        if self.settings.rerank.backend == "none":
            return candidates

        # If reranker not available (initialization failed), fallback
        if self.reranker is None:
            if self._initialization_error:
                return self._apply_fallback_metadata(
                    candidates,
                    f"Reranker initialization failed: {self._initialization_error}"
                )
            else:
                # No reranker backend configured
                return candidates

        # Prepare candidates for reranking (limit to top_m if configured)
        candidates_to_rerank = self._prepare_candidates(candidates)

        # Track which candidates were sent for reranking
        top_m = getattr(self.settings.rerank, "top_m", None)
        should_limit = isinstance(top_m, int) and top_m > 0

        # Attempt reranking with fallback on failure
        try:
            # Convert RetrievalResult to dict for backend
            candidates_dict = self._retrieval_results_to_dict(candidates_to_rerank)

            # Call backend reranker
            reranked_dict = self.reranker.rerank(
                query=query,
                candidates=candidates_dict,
                trace=trace,
            )

            # Convert back to RetrievalResult
            reranked_results = self._dict_to_retrieval_results(reranked_dict)

            # If we limited to top_m, return only reranked results
            # Otherwise, append any candidates not in the reranked set
            if should_limit:
                return reranked_results
            else:
                # Keep any candidates not reranked
                reranked_ids = {r.chunk_id for r in reranked_results}
                other_candidates = [c for c in candidates if c.chunk_id not in reranked_ids]
                return reranked_results + other_candidates

        except Exception as e:
            # Fallback to original results on any error
            if self.fallback_enabled:
                return self._apply_fallback_metadata(candidates, str(e))
            else:
                raise

    def _prepare_candidates(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Limit candidates to top_m if configured.

        Args:
            candidates: Full candidate list

        Returns:
            Top_m candidates if configured, otherwise full list
        """
        top_m = getattr(self.settings.rerank, "top_m", None)
        # Handle cases where top_m might be a Mock or invalid value
        if top_m is None:
            return candidates
        # Ensure top_m is an integer
        if not isinstance(top_m, int) or top_m <= 0:
            return candidates
        if len(candidates) > top_m:
            return candidates[:top_m]
        return candidates

    def _retrieval_results_to_dict(self, results: List[RetrievalResult]) -> List[dict]:
        """Convert RetrievalResult to dict for backend compatibility.

        Args:
            results: List of RetrievalResult

        Returns:
            List of dicts with required fields
        """
        return [
            {
                "id": r.chunk_id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in results
        ]

    def _dict_to_retrieval_results(self, results_dict: List[dict]) -> List[RetrievalResult]:
        """Convert backend results (dict format) back to RetrievalResult.

        Args:
            results_dict: List of dicts from backend

        Returns:
            List of RetrievalResult objects
        """
        results = []
        for item in results_dict:
            # Use cross_encoder_score if available (from reranker), otherwise use original score
            if "cross_encoder_score" in item:
                score = item["cross_encoder_score"]
            else:
                score = item.get("score", 0.0)

            result = RetrievalResult(
                chunk_id=item.get("id") or item.get("chunk_id"),
                score=score,
                text=item.get("text", ""),
                metadata=item.get("metadata", {}),
            )
            results.append(result)
        return results

    def _apply_fallback_metadata(
        self,
        candidates: List[RetrievalResult],
        error_msg: str,
    ) -> List[RetrievalResult]:
        """Apply fallback metadata to candidates.

        Args:
            candidates: Original candidates to mark as fallback
            error_msg: Error message explaining the fallback

        Returns:
            Candidates with fallback metadata added
        """
        fallback_results = []
        for candidate in candidates:
            # Create new result with fallback metadata
            updated_metadata = candidate.metadata.copy()
            updated_metadata["fallback"] = True
            updated_metadata["fallback_reason"] = f"{self.settings.rerank.backend} backend error: {error_msg}"

            fallback_result = RetrievalResult(
                chunk_id=candidate.chunk_id,
                score=candidate.score,
                text=candidate.text,
                metadata=updated_metadata,
            )
            fallback_results.append(fallback_result)

        return fallback_results
