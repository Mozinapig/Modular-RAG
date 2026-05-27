"""LLM-based Reranker implementation."""

import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import ChatMessage


class LLMReranker(BaseReranker):
    """Reranker using LLM to rank candidates based on query relevance."""

    def __init__(self, settings):
        """
        Initialize LLMReranker.

        Args:
            settings: Settings object with llm and reranker configuration
        """
        self.settings = settings
        self.llm_client = None
        self.prompt_template = None
        self._load_prompt()

    def _get_prompt_path(self) -> str:
        """Get the path to the rerank prompt file."""
        return "config/prompts/rerank.txt"

    def _load_prompt(self) -> None:
        """Load rerank prompt from config file, with fallback to default."""
        prompt_path = Path(self._get_prompt_path())

        # Try to load from file
        if prompt_path.exists():
            try:
                self.prompt_template = prompt_path.read_text()
                return
            except Exception as e:
                print(f"Warning: Failed to load prompt from {prompt_path}: {e}")

        # Fallback to default prompt template
        self.prompt_template = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default reranking prompt template."""
        return """You are an intelligent document ranking assistant. Your task is to rerank a list of documents based on their relevance to the given query.

Query: {query}

Documents:
{candidates}

Instructions:
1. Analyze each document's relevance to the query
2. Consider the semantic similarity and topical match
3. Return a JSON object with the 'ranked_ids' field containing the document IDs in order of relevance (most relevant first)
4. Return ONLY valid JSON, no additional text

Response format:
{{"ranked_ids": ["doc_id_1", "doc_id_2", ...]}}"""

    def _ensure_llm_client(self) -> None:
        """Lazy initialize the LLM client."""
        if self.llm_client is None:
            factory = LLMFactory()
            if not hasattr(self.settings, "llm") or not self.settings.llm:
                raise ValueError("LLM provider is required for LLMReranker")

            # Use LLMSettings directly if available
            if hasattr(self.settings.llm, "provider"):
                # Already an LLMSettings object
                self.llm_client = factory.create(self.settings.llm)
            else:
                # It's a dict, convert to LLMSettings
                from src.core.settings import LLMSettings

                llm_settings = LLMSettings(**self.settings.llm)
                self.llm_client = factory.create(llm_settings)

    def validate_config(self) -> None:
        """Validate LLM Reranker configuration."""
        # Handle both Settings object (with .llm) and RerankerSettings object
        llm_config = None

        if hasattr(self.settings, "llm"):
            # This is a full Settings object
            llm_config = self.settings.llm
        elif hasattr(self.settings, "provider"):
            # This is a RerankerSettings-like object, but Reranker doesn't have LLM
            # We need to look elsewhere for LLM config
            pass

        if not llm_config:
            raise ValueError("LLM provider is required for LLMReranker")

        # Validate that we can create LLM client
        try:
            self._ensure_llm_client()
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM for reranking: {str(e)}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using LLM.

        Args:
            query: The query string
            candidates: List of candidate dicts with at least 'id' and 'score' fields
            trace: Optional TraceContext for tracking

        Returns:
            Reranked list of candidates in order of relevance

        Raises:
            ValueError: If input validation fails or LLM output is invalid
            RuntimeError: If reranking fails
        """
        # Validate inputs
        self._validate_inputs(query, candidates)

        # Return empty list if no candidates
        if not candidates:
            return []

        # Single candidate needs no reranking
        if len(candidates) == 1:
            return candidates

        try:
            # Build prompt with query and candidates
            prompt = self._build_prompt(query, candidates)

            # Call LLM to get ranked IDs
            ranked_ids = self._call_llm(prompt)

            # Validate LLM output
            self._validate_llm_output(ranked_ids, candidates)

            # Reorder candidates based on ranked IDs
            result = self._reorder_candidates(ranked_ids, candidates)

            # Record trace if provided
            if trace:
                trace.record_stage(
                    "llm_rerank",
                    method="llm",
                    count=len(result),
                    provider=self.settings.llm.get("provider", "unknown"),
                )

            return result

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to rerank candidates: {str(e)}")

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

    def _build_prompt(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Build the prompt for LLM reranking."""
        # Format candidates for the prompt
        candidates_text = self._format_candidates(candidates)

        # Build prompt - use simple string replacement to avoid KeyError with {}
        prompt = self.prompt_template
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{candidates}", candidates_text)

        return prompt

    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates for inclusion in the prompt."""
        lines = []
        for i, candidate in enumerate(candidates, 1):
            candidate_id = candidate.get("id", f"doc_{i}")
            text = candidate.get("text", "")
            score = candidate.get("score", 0)

            # Include text if available, otherwise just ID and score
            if text:
                lines.append(
                    f"{i}. ID: {candidate_id} (score: {score:.3f})\n   Text: {text[:100]}..."
                )
            else:
                lines.append(f"{i}. ID: {candidate_id} (score: {score:.3f})")

        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> List[str]:
        """Call LLM to get ranked IDs."""
        self._ensure_llm_client()

        # Prepare message for LLM
        messages = [ChatMessage(role="user", content=prompt)]

        # Call LLM
        response = self.llm_client.chat(messages)

        # Extract and parse the response
        llm_output = response.content

        # Parse JSON from response
        ranked_ids = self._parse_llm_response(llm_output)

        return ranked_ids

    def _parse_llm_response(self, llm_output: str) -> List[str]:
        """Parse LLM response to extract ranked IDs."""
        # Try to find JSON in the output
        json_match = re.search(r"\{.*?\}", llm_output, re.DOTALL)

        if not json_match:
            raise ValueError(
                f"Failed to parse LLM response. No JSON found in: {llm_output}"
            )

        try:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in LLM response: {str(e)}. "
                f"Response: {llm_output[:200]}"
            )

        # Extract ranked_ids from JSON
        if "ranked_ids" not in parsed:
            raise ValueError(
                f"LLM response missing 'ranked_ids' field. Got: {parsed}"
            )

        ranked_ids = parsed["ranked_ids"]

        if not isinstance(ranked_ids, list):
            raise ValueError(
                f"'ranked_ids' must be a list, got {type(ranked_ids).__name__}"
            )

        return ranked_ids

    def _validate_llm_output(
        self, ranked_ids: List[str], candidates: List[Dict[str, Any]]
    ) -> None:
        """Validate that LLM output contains all candidate IDs."""
        candidate_ids = {c["id"] for c in candidates}

        # Check that we got the same number of IDs
        if len(ranked_ids) != len(candidate_ids):
            raise ValueError(
                f"LLM returned {len(ranked_ids)} IDs but expected {len(candidate_ids)}. "
                f"Count mismatch indicates invalid output."
            )

        # Check that all returned IDs are in the candidates
        ranked_set = set(ranked_ids)
        for ranked_id in ranked_ids:
            if ranked_id not in candidate_ids:
                raise ValueError(
                    f"Unknown ID in LLM output: {ranked_id}. "
                    f"Valid IDs: {candidate_ids}"
                )

        # Check for duplicates in returned IDs
        if len(ranked_set) != len(ranked_ids):
            raise ValueError(
                "LLM output contains duplicate IDs. "
                "Each ID should appear exactly once."
            )

    def _reorder_candidates(
        self, ranked_ids: List[str], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reorder candidates according to ranked IDs."""
        # Create a mapping from ID to candidate
        candidate_map = {c["id"]: c for c in candidates}

        # Reorder according to ranked_ids
        reordered = []
        for ranked_id in ranked_ids:
            reordered.append(candidate_map[ranked_id])

        return reordered
