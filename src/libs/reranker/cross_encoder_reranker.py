"""Cross-Encoder based Reranker implementation."""

from typing import List, Dict, Optional, Any

from src.libs.reranker.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Reranker using Cross-Encoder model to score query-candidate pairs."""

    def __init__(self, settings):
        """
        Initialize CrossEncoderReranker.

        Args:
            settings: Settings object with reranker configuration including model
        """
        self.settings = settings
        self.scorer = None  # Lazy load scorer on first use

    def validate_config(self) -> None:
        """Validate Cross-Encoder configuration."""
        # Get reranker settings (handle both full Settings and RerankerSettings)
        reranker_config = None

        if hasattr(self.settings, "rerank"):
            # This is a full Settings object
            reranker_config = self.settings.rerank
        elif hasattr(self.settings, "model"):
            # This is already RerankerSettings-like object
            reranker_config = self.settings

        if not reranker_config:
            raise ValueError("Reranker configuration is required")

        # Check for model field
        model = None
        if isinstance(reranker_config, dict):
            model = reranker_config.get("model")
        else:
            model = getattr(reranker_config, "model", None)

        if not model:
            raise ValueError("Cross-Encoder model configuration is required")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Cross-Encoder model.

        Args:
            query: The query string
            candidates: List of candidate dicts with at least 'id' and 'text' fields
            trace: Optional TraceContext for tracking

        Returns:
            Reranked list of candidates in order of relevance

        Raises:
            ValueError: If input validation fails or scorer output is invalid
            RuntimeError: If reranking fails
        """
        # Validate inputs
        self._validate_inputs(query, candidates)

        # Return empty list if no candidates
        if not candidates:
            return []

        try:
            # Extract texts from candidates for scoring
            texts = [candidate.get("text", "") for candidate in candidates]

            # Call scorer to get relevance scores
            scores = self._call_scorer(query, texts)

            # Validate scorer output
            self._validate_scores(scores, candidates)

            # Reorder candidates based on scores
            result = self._reorder_candidates(scores, candidates)

            # Record trace if provided
            if trace:
                trace.record_stage(
                    "cross_encoder_rerank",
                    method="cross_encoder",
                    count=len(result),
                    model=self._get_model_name(),
                )

            return result

        except ValueError:
            raise
        except TimeoutError:
            raise RuntimeError("Cross-Encoder reranking timeout - fallback to original ranking")
        except Exception as e:
            raise RuntimeError(f"reranking failed: {str(e)}")

    def _get_model_name(self) -> str:
        """Extract model name from settings."""
        if isinstance(self.settings.rerank, dict):
            return self.settings.rerank.get("model", "unknown")
        return getattr(self.settings.rerank, "model", "unknown")

    def _validate_inputs(self, query: str, candidates: List[Dict[str, Any]]) -> None:
        """Validate input query and candidates."""
        if not query:
            raise ValueError("Query cannot be empty")

        # Check each candidate has required fields
        seen_ids = set()
        for i, candidate in enumerate(candidates):
            if "id" not in candidate:
                raise ValueError(f"Candidate at index {i} is missing required 'id' field")

            candidate_id = candidate["id"]
            if candidate_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate ID: {candidate_id}. "
                    "All candidate IDs must be unique."
                )
            seen_ids.add(candidate_id)

    def _call_scorer(self, query: str, texts: List[str]) -> List[float]:
        """Call Cross-Encoder scorer to get relevance scores."""
        try:
            # In real usage, load the cross-encoder model and score pairs
            # For now, this would be implemented with:
            # from sentence_transformers import CrossEncoder
            # if self.scorer is None:
            #     self.scorer = CrossEncoder(self.model_name)
            # scores = self.scorer.predict([[query, text] for text in texts])

            # This is a placeholder that will be mocked in tests
            # and would use actual scorer in production
            raise NotImplementedError(
                "Cross-Encoder scorer not initialized. "
                "In tests, patch _call_scorer with mock scores."
            )
        except TimeoutError:
            raise
        except NotImplementedError:
            raise

    def _validate_scores(
        self, scores: List[float], candidates: List[Dict[str, Any]]
    ) -> None:
        """Validate that scorer returned correct number of scores."""
        if len(scores) != len(candidates):
            raise ValueError(
                f"Cross-Encoder returned {len(scores)} scores but expected {len(candidates)}. "
                f"count mismatch indicates invalid output."
            )

        # Validate each score is numeric
        for i, score in enumerate(scores):
            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"Cross-Encoder score at index {i} is not numeric: {type(score).__name__}"
                )

    def _reorder_candidates(
        self, scores: List[float], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reorder candidates according to scores (highest first)."""
        # Pair each candidate with its score
        scored_candidates = [
            {**candidate, "cross_encoder_score": score}
            for candidate, score in zip(candidates, scores)
        ]

        # Sort by score in descending order
        reordered = sorted(scored_candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

        return reordered
