"""
MetadataEnricher - rule-based and LLM-enhanced metadata enrichment.
Generates semantic metadata (title, summary, tags) for chunks with graceful degradation.
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """Enriches chunks with semantic metadata (title, summary, tags)."""

    def __init__(self, settings: Any, llm=None, prompt_path: Optional[str] = None):
        """
        Initialize MetadataEnricher.

        Args:
            settings: Settings object with use_llm flag
            llm: Optional LLM instance (if None, will be created from settings)
            prompt_path: Optional path to prompt template
        """
        self.settings = settings
        self.use_llm = getattr(settings, 'use_llm', False)
        self.llm = llm
        self.prompt = self._load_prompt(prompt_path)

        # Try to create LLM if enabled and not provided
        if self.use_llm and self.llm is None:
            try:
                from src.libs.llm.llm_factory import LLMFactory
                factory = LLMFactory()
                self.llm = factory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create LLM: {e}. Will fall back to rule-based enrichment.")
                self.use_llm = False

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform chunks through metadata enrichment.

        Args:
            chunks: List of chunks to enrich
            trace: Optional trace context

        Returns:
            List of enriched chunks
        """
        if not chunks:
            return []

        enriched_chunks = []
        start_time = datetime.now().timestamp()

        for chunk in chunks:
            try:
                enriched_chunk = self._enrich_chunk(chunk)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.warning(f"Error enriching chunk {chunk.id}: {e}. Preserving original.")
                # Preserve original chunk on error
                enriched_chunks.append(chunk)

        # Record in trace if provided
        if trace:
            end_time = datetime.now().timestamp()
            trace.record_stage(
                "metadata_enricher",
                start_time,
                end_time,
                {"chunks_processed": len(enriched_chunks)}
            )

        return enriched_chunks

    def _enrich_chunk(self, chunk: Chunk) -> Chunk:
        """
        Enrich a single chunk with metadata.

        Args:
            chunk: Chunk to enrich

        Returns:
            Enriched chunk
        """
        # First apply rule-based enrichment
        rule_metadata = self._rule_based_enrich(chunk.text)

        # Then optionally apply LLM enrichment
        if self.use_llm and self.llm:
            llm_metadata = self._llm_enrich(chunk.text)
            if llm_metadata is not None:
                rule_metadata.update(llm_metadata)
                chunk.metadata["enriched_by"] = "llm"
            else:
                # LLM failed, keep rule-based result
                chunk.metadata["enriched_by"] = "rule"
                chunk.metadata["fallback_reason"] = "llm_failed"
        else:
            chunk.metadata["enriched_by"] = "rule"

        # Merge enriched metadata
        chunk.metadata.update(rule_metadata)

        # Create new chunk with enriched metadata
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset
        )

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """
        Apply rule-based enrichment to generate title, summary, and tags.

        Args:
            text: Text to enrich

        Returns:
            Dictionary with title, summary, and tags
        """
        if not text or not text.strip():
            return {
                "title": "Untitled",
                "summary": "No content",
                "tags": []
            }

        # Generate title from first sentence or first N words
        title = self._extract_title(text)

        # Generate summary from first 1-2 sentences or first N words
        summary = self._extract_summary(text)

        # Extract tags from keywords
        tags = self._extract_tags(text)

        return {
            "title": title,
            "summary": summary,
            "tags": tags
        }

    def _extract_title(self, text: str) -> str:
        """
        Extract title from text.

        Args:
            text: Text to extract title from

        Returns:
            Title string
        """
        # Try to get first sentence
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0].strip():
            title = sentences[0].strip()
            # Limit to 100 characters
            if len(title) > 100:
                title = title[:97] + "..."
            return title

        # Fallback: first N words
        words = text.split()[:10]
        return " ".join(words)

    def _extract_summary(self, text: str) -> str:
        """
        Extract summary from text.

        Args:
            text: Text to extract summary from

        Returns:
            Summary string
        """
        # Get first 1-2 sentences
        sentences = re.split(r'[.!?]\s+', text)
        summary_sentences = []
        char_count = 0
        max_chars = 200

        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_with_period = sentence.strip() + "."
            if char_count + len(sentence_with_period) <= max_chars:
                summary_sentences.append(sentence_with_period)
                char_count += len(sentence_with_period)
            else:
                break

        if summary_sentences:
            summary = " ".join(summary_sentences)
            return summary.strip()

        # Fallback: first N words
        words = text.split()[:30]
        return " ".join(words)

    def _extract_tags(self, text: str) -> List[str]:
        """
        Extract tags from text.

        Args:
            text: Text to extract tags from

        Returns:
            List of tags
        """
        tags = []

        # Extract common keywords (words that appear frequently or are capitalized)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top frequent words as tags
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:5]:
            if freq >= 1 and word.lower() not in ['the', 'a', 'an', 'is', 'are']:
                tags.append(word.lower())

        # If no tags found, extract from lowercase words
        if not tags:
            words = re.findall(r'\b[a-z]+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only words longer than 3 chars
                    word_freq[word] = word_freq.get(word, 0) + 1

            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:5]:
                if freq >= 2:  # Only if appears at least twice
                    tags.append(word)

        # Ensure at least one tag
        if not tags:
            # Extract first meaningful word
            words = re.findall(r'\b[a-z]+\b', text.lower())
            for word in words:
                if len(word) > 3:
                    tags.append(word)
                    break

        return tags[:5]  # Limit to 5 tags

    def _llm_enrich(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Apply LLM-based enrichment.

        Args:
            text: Text to enrich

        Returns:
            Dictionary with title, summary, tags or None if LLM call fails
        """
        if not self.llm or not self.prompt:
            return None

        try:
            # Format prompt with text
            prompt = self.prompt.format(text=text)

            # Call LLM
            response = self.llm.generate(prompt)

            if response:
                # Parse JSON response
                response_text = response.strip()
                metadata = json.loads(response_text)

                # Validate response structure
                if isinstance(metadata, dict):
                    result = {}
                    if "title" in metadata and isinstance(metadata["title"], str):
                        result["title"] = metadata["title"]
                    if "summary" in metadata and isinstance(metadata["summary"], str):
                        result["summary"] = metadata["summary"]
                    if "tags" in metadata and isinstance(metadata["tags"], list):
                        result["tags"] = [str(t) for t in metadata["tags"]]

                    if result:
                        return result

            return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            return None

    def _load_prompt(self, prompt_path: Optional[str]) -> Optional[str]:
        """
        Load prompt template from file.

        Args:
            prompt_path: Path to prompt file

        Returns:
            Prompt template or None
        """
        if prompt_path:
            try:
                with open(prompt_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load prompt from {prompt_path}: {e}")

        # Try default location
        default_path = "config/prompts/metadata_enrichment.txt"
        try:
            with open(default_path, 'r') as f:
                return f.read()
        except Exception:
            logger.debug(f"No prompt found at {default_path}")

        return None
