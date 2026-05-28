"""
ChunkRefiner - rule-based and LLM-enhanced chunk refinement.
Performs noise removal and optional LLM-based enhancement with graceful degradation.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)


class ChunkRefiner(BaseTransform):
    """Refines chunks through rule-based and optional LLM-based enhancement."""

    def __init__(self, settings: Any, llm=None, prompt_path: Optional[str] = None):
        """
        Initialize ChunkRefiner.

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
                logger.warning(f"Failed to create LLM: {e}. Will fall back to rule-based refinement.")
                self.use_llm = False

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform chunks through refinement.

        Args:
            chunks: List of chunks to refine
            trace: Optional trace context

        Returns:
            List of refined chunks
        """
        if not chunks:
            return []

        refined_chunks = []
        start_time = datetime.now().timestamp()

        for chunk in chunks:
            try:
                refined_chunk = self._refine_chunk(chunk)
                refined_chunks.append(refined_chunk)
            except Exception as e:
                logger.warning(f"Error refining chunk {chunk.id}: {e}. Preserving original.")
                # Preserve original chunk on error
                refined_chunks.append(chunk)

        # Record in trace if provided
        if trace:
            end_time = datetime.now().timestamp()
            trace.record_stage(
                "chunk_refiner",
                start_time,
                end_time,
                {"chunks_processed": len(refined_chunks)}
            )

        return refined_chunks

    def _refine_chunk(self, chunk: Chunk) -> Chunk:
        """
        Refine a single chunk.

        Args:
            chunk: Chunk to refine

        Returns:
            Refined chunk
        """
        # First apply rule-based refinement
        refined_text = self._rule_based_refine(chunk.text)

        # Then optionally apply LLM refinement
        if self.use_llm and self.llm:
            llm_result = self._llm_refine(refined_text)
            if llm_result is not None:
                refined_text = llm_result
                chunk.metadata["refined_by"] = "llm"
            else:
                # LLM failed, keep rule-based result
                chunk.metadata["refined_by"] = "rule"
                chunk.metadata["fallback_reason"] = "llm_failed"
        else:
            chunk.metadata["refined_by"] = "rule"

        # Create new chunk with refined text
        return Chunk(
            id=chunk.id,
            text=refined_text,
            metadata=chunk.metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset
        )

    def _rule_based_refine(self, text: str) -> str:
        """
        Apply rule-based refinement to remove common noise.

        Handles:
        - Excessive whitespace (multiple consecutive newlines)
        - Leading/trailing whitespace
        - Multiple spaces within lines
        - HTML comments
        - Preserves code blocks and Markdown formatting

        Args:
            text: Text to refine

        Returns:
            Refined text
        """
        if not text:
            return text

        # Preserve code blocks
        code_blocks = []
        code_pattern = r'```[\s\S]*?```'
        for match in re.finditer(code_pattern, text):
            code_blocks.append(match.group())

        # Replace code blocks with placeholders
        temp_text = text
        for i, block in enumerate(code_blocks):
            temp_text = temp_text.replace(block, f"__CODE_BLOCK_{i}__", 1)

        # Remove HTML comments
        temp_text = re.sub(r'<!--[\s\S]*?-->', '', temp_text)

        # Remove HTML tags but preserve content
        temp_text = re.sub(r'<[^>]+>', '', temp_text)

        # Reduce multiple consecutive newlines to max 2
        temp_text = re.sub(r'\n\n\n+', '\n\n', temp_text)

        # Reduce multiple spaces to single space (but preserve indentation structure)
        lines = temp_text.split('\n')
        processed_lines = []
        for line in lines:
            # Preserve leading whitespace for indentation
            leading_ws = len(line) - len(line.lstrip())
            content = line.lstrip()
            # Reduce multiple spaces in content
            content = re.sub(r' {2,}', ' ', content)
            processed_lines.append(' ' * leading_ws + content)

        temp_text = '\n'.join(processed_lines)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            temp_text = temp_text.replace(f"__CODE_BLOCK_{i}__", block)

        # Strip leading/trailing whitespace
        temp_text = temp_text.strip()

        return temp_text

    def _llm_refine(self, text: str) -> Optional[str]:
        """
        Apply LLM-based refinement.

        Args:
            text: Text to refine

        Returns:
            Refined text or None if LLM call fails
        """
        if not self.llm or not self.prompt:
            return None

        try:
            # Format prompt with text
            prompt = self.prompt.format(text=text)

            # Call LLM
            response = self.llm.generate(prompt)

            if response:
                return response.strip()
            return None

        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
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
        default_path = "config/prompts/chunk_refinement.txt"
        try:
            with open(default_path, 'r') as f:
                return f.read()
        except Exception:
            logger.debug(f"No prompt found at {default_path}")

        return None
