"""
Multimodal Assembler - E6 任务实现
Assembles multimodal content (text + images) for MCP responses
"""
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)


class MultimodalAssembler:
    """Assembles multimodal content for MCP responses"""

    def assemble(self, retrieval_results: List[Any]) -> List[Dict[str, Any]]:
        """
        Assemble multimodal content from retrieval results

        Args:
            retrieval_results: List of retrieval results that may have image_refs

        Returns:
            List of content items (text and image types)
        """
        content = []

        try:
            for result in retrieval_results:
                # Add text content
                text_content = {
                    'type': 'text',
                    'text': getattr(result, 'content', '')
                }
                content.append(text_content)

                # Add image content if present
                image_refs = getattr(result, 'image_refs', [])

                for image_ref in image_refs:
                    image_content = self._create_image_content(image_ref)
                    if image_content:
                        content.append(image_content)

            logger.info(f"Assembled {len(content)} content items")

        except Exception as e:
            logger.error(f"Error assembling multimodal content: {e}", exc_info=True)

        return content

    def _create_image_content(self, image_ref: str) -> Optional[Dict[str, Any]]:
        """
        Create image content from image reference

        Args:
            image_ref: Image reference (file path or ID)

        Returns:
            Image content dict with base64 data or None if not found
        """
        try:
            # Try to find image file
            image_paths = [
                Path(f'data/images/{image_ref}'),
                Path(f'data/images/') / image_ref,
                Path(image_ref)
            ]

            image_path = None
            for path in image_paths:
                if path.exists():
                    image_path = path
                    break

            if not image_path:
                logger.warning(f"Image not found: {image_ref}")
                return None

            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(suffix, 'application/octet-stream')

            # Encode to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')

            image_content = {
                'type': 'image',
                'mimeType': mime_type,
                'data': base64_data
            }

            logger.debug(f"Created image content: {image_ref}")

            return image_content

        except Exception as e:
            logger.error(f"Error creating image content for {image_ref}: {e}")
            return None
