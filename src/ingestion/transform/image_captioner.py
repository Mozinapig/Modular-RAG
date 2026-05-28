"""
ImageCaptioner: Generate captions for images in chunks using Vision LLM.
Supports graceful degradation when Vision LLM is disabled or fails.
"""

import re
import logging
from typing import List, Optional, Dict
from abc import ABC

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.core.settings import Settings
from src.libs.llm.llm_factory import LLMFactory


logger = logging.getLogger(__name__)


class ImageCaptioner(BaseTransform):
    """
    Transform stage that generates captions for images referenced in chunks.

    Behavior:
    - When disabled: Marks chunks with images as 'has_unprocessed_images: True'
    - When enabled: Calls Vision LLM to generate captions
    - On LLM failure: Gracefully falls back, marks chunk as unprocessed, never blocks pipeline
    """

    def __init__(self, settings: Settings):
        """Initialize ImageCaptioner with settings."""
        self.settings = settings
        # Get use_llm from settings - supports both direct attribute and nested structure
        self.use_llm = self._get_use_llm(settings)
        self.vision_llm = None

        if self.use_llm:
            try:
                self.vision_llm = LLMFactory().create_vision_llm(settings)
            except Exception as e:
                logger.warning(f"Failed to initialize Vision LLM: {e}")
                self.vision_llm = None

    def _get_use_llm(self, settings: Settings) -> bool:
        """Extract use_llm flag from settings, supporting multiple formats."""
        # Try direct attribute first (for mocked/mock-like settings)
        if hasattr(settings, 'ingestion') and hasattr(settings.ingestion, 'image_captioner'):
            if hasattr(settings.ingestion.image_captioner, 'use_llm'):
                return settings.ingestion.image_captioner.use_llm
            if hasattr(settings.ingestion.image_captioner, 'enabled'):
                return settings.ingestion.image_captioner.enabled
        # Try in extra dict (for parsed YAML settings)
        if hasattr(settings, 'extra') and isinstance(settings.extra, dict):
            ingestion_config = settings.extra.get('ingestion', {})
            if isinstance(ingestion_config, dict):
                image_config = ingestion_config.get('image_captioner', {})
                if isinstance(image_config, dict):
                    return image_config.get('enabled', False)
        # Default fallback
        return False

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Process chunks and generate image captions.

        Args:
            chunks: List of chunks to process
            trace: Optional trace context for recording

        Returns:
            List of processed chunks with image captions or unprocessed markers
        """
        result = []
        for chunk in chunks:
            try:
                processed_chunk = self._process_chunk(chunk, trace)
                result.append(processed_chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}")
                # Never block pipeline - add error metadata and continue
                chunk.metadata["has_unprocessed_images"] = True
                chunk.metadata["fallback_reason"] = str(e)
                result.append(chunk)

        return result

    def _process_chunk(self, chunk: Chunk, trace: Optional[TraceContext] = None) -> Chunk:
        """Process a single chunk."""
        # Extract image references
        image_refs = chunk.metadata.get("images", [])
        if not image_refs:
            return chunk

        if not self.use_llm:
            # Disabled mode: mark as unprocessed
            chunk.metadata["has_unprocessed_images"] = True
            return chunk

        # Enabled mode: try to generate captions
        if not self.vision_llm:
            chunk.metadata["has_unprocessed_images"] = True
            chunk.metadata["fallback_reason"] = "Vision LLM not available"
            return chunk

        # Generate captions for each image
        captions = {}
        failed_count = 0

        for image_ref in image_refs:
            image_id = image_ref.get("id")
            image_path = image_ref.get("path")

            if not image_id or not image_path:
                failed_count += 1
                continue

            try:
                caption = self.vision_llm.chat_with_image(
                    text="Describe this image briefly.",
                    image_path=image_path,
                    trace=trace
                )
                captions[image_id] = caption
            except Exception as e:
                logger.warning(f"Failed to caption image {image_id}: {e}")
                failed_count += 1

        # Check if all or any captions failed
        if failed_count == len(image_refs):
            # All failed - mark as unprocessed
            chunk.metadata["has_unprocessed_images"] = True
            chunk.metadata["fallback_reason"] = "Vision LLM failed to process all images"
            return chunk

        if failed_count > 0:
            # Partial failure - mark as unprocessed
            chunk.metadata["has_unprocessed_images"] = True
            chunk.metadata["fallback_reason"] = f"Vision LLM failed for {failed_count}/{len(image_refs)} images"

        # Add captions to metadata
        if captions:
            chunk.metadata["image_captions"] = captions
            chunk.metadata["captions_enriched_by"] = "vision_llm"

        return chunk
