"""Cross-Encoder based Reranker implementation."""

import json
from typing import List, Dict, Optional, Any

import requests

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
        """Call Cross-Encoder scorer API to get relevance scores.

        Supports BGE Reranker and OpenAI-compatible Rerank APIs.

        Raises:
            ValueError: If configuration is missing or invalid
            TimeoutError: If API call times out
            RuntimeError: If API call fails
        """
        # Get reranker config
        reranker_config = self._get_reranker_config()

        # Check for required API fields
        api_key = reranker_config.get("api_key") if isinstance(reranker_config, dict) else getattr(reranker_config, "api_key", None)
        base_url = reranker_config.get("base_url") if isinstance(reranker_config, dict) else getattr(reranker_config, "base_url", None)

        if not api_key or not base_url:
            raise ValueError(
                "Cross-Encoder API requires 'api_key' and 'base_url' configuration. "
                "Set these in your settings.yaml or environment variables."
            )

        # Prepare request data for BGE Reranker API
        # Typical format: {"query": "...", "passages": ["...", "..."]}
        request_data = {
            "query": query,
            "passages": texts,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Call the reranker API
            url = f"{base_url.rstrip('/')}/v1/rerank"
            response = requests.post(url, json=request_data, headers=headers, timeout=30)

            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                raise RuntimeError(f"Reranker API error: {error_msg}")

            # Parse response and extract scores
            result = response.json()

            # Expected format: {"results": [{"index": 0, "score": 0.9}, ...]}
            if "results" not in result:
                raise ValueError(
                    f"Unexpected API response format. Expected 'results' field. Got: {result}"
                )

            # Extract scores maintaining original order
            scores_by_index = {}
            for item in result["results"]:
                idx = item.get("index")
                score = item.get("score")
                if idx is None or score is None:
                    raise ValueError(
                        f"Invalid result item format. Expected 'index' and 'score'. Got: {item}"
                    )
                scores_by_index[idx] = float(score)

            # Return scores in original order
            scores = [scores_by_index.get(i, 0.0) for i in range(len(texts))]
            return scores

        except requests.exceptions.Timeout:
            raise TimeoutError("Reranker API request timeout")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Reranker API request failed: {str(e)}")
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse reranker API response: {str(e)}")

    def _get_reranker_config(self) -> Dict[str, Any]:
        """Extract reranker config from settings."""
        if hasattr(self.settings, "rerank"):
            # Full Settings object
            config = self.settings.rerank
        else:
            # Direct config object
            config = self.settings

        if isinstance(config, dict):
            return config

        # Convert object to dict for easier access
        return {
            "backend": getattr(config, "backend", None),
            "model": getattr(config, "model", None),
            "api_key": getattr(config, "api_key", None),
            "base_url": getattr(config, "base_url", None),
        }

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
