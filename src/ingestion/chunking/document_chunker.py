"""
DocumentChunker - adapter layer for converting Document to Chunks.
Integrates libs.splitter and adds business logic for metadata inheritance,
chunk ID generation, and image reference distribution.
"""

import hashlib
import logging
import re
from typing import Dict, List, Any, Optional

from src.core.types import Document, Chunk, ImageRef
from src.libs.splitter.splitter_factory import SplitterFactory

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Converts Document objects to Chunks with business logic."""

    def __init__(self, settings: Any):
        """
        Initialize DocumentChunker.

        Args:
            settings: Settings object containing splitter configuration
                     Should have provider, chunk_size, chunk_overlap attributes
        """
        self.settings = settings
        self.splitter_factory = SplitterFactory()
        self.splitter = self.splitter_factory.create(settings)

    def split_document(self, document: Document) -> List[Chunk]:
        """
        Convert a Document into a list of Chunks.

        Performs:
        1. Text splitting using configured splitter
        2. Chunk ID generation (deterministic format)
        3. Metadata inheritance from document
        4. Image reference distribution per chunk
        5. Type conversion to Chunk objects

        Args:
            document: Document object to split

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If document is invalid
        """
        if not document or not document.text:
            return []

        # Split text into chunks using configured splitter
        text_chunks = self.splitter.split_text(document.text)

        chunks = []
        current_offset = 0

        for index, chunk_text in enumerate(text_chunks):
            # Generate deterministic chunk ID
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)

            # Inherit and enrich metadata
            metadata = self._inherit_metadata(document, index, chunk_text)

            # Create Chunk object
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=current_offset,
                end_offset=current_offset + len(chunk_text)
            )

            chunks.append(chunk)
            current_offset += len(chunk_text)

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        Generate deterministic chunk ID.

        Format: {doc_id}_{index:04d}_{hash_8chars}

        Args:
            doc_id: Parent document ID
            index: Chunk index (0-based)
            text: Chunk text content

        Returns:
            Deterministic chunk ID
        """
        # Compute hash of chunk text (first 8 chars)
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

        chunk_id = f"{doc_id}_{index:04d}_{text_hash}"
        return chunk_id

    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str
    ) -> Dict[str, Any]:
        """
        Inherit metadata from document and add chunk-specific fields.

        Performs:
        1. Copy document metadata to chunk
        2. Add chunk_index field
        3. Add source_ref pointing to parent document
        4. Distribute image references based on placeholders in chunk text
        5. Add image_refs list

        Args:
            document: Parent document
            chunk_index: Index of this chunk in document
            chunk_text: Text content of this chunk

        Returns:
            Chunk metadata dict
        """
        # Start with copy of document metadata
        metadata = document.metadata.copy()

        # Add chunk-specific fields
        metadata["chunk_index"] = chunk_index
        metadata["source_ref"] = document.id

        # Handle image distribution
        self._distribute_images(document, chunk_text, metadata)

        return metadata

    def _distribute_images(
        self, document: Document, chunk_text: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Distribute image references to chunk based on placeholders in text.

        Scans chunk text for [IMAGE: id] placeholders and extracts corresponding
        ImageRef objects from document metadata. Only includes images that are
        actually referenced in this chunk.

        Args:
            document: Parent document
            chunk_text: Chunk text to scan for placeholders
            metadata: Chunk metadata dict to populate (modified in place)
        """
        # Find all image placeholders in chunk text
        placeholder_pattern = r'\[IMAGE: (\w+)\]'
        image_ids_in_chunk = re.findall(placeholder_pattern, chunk_text)

        if not image_ids_in_chunk:
            # No images in this chunk
            return

        # Get document's images if they exist
        document_images = document.metadata.get("images", [])
        if not document_images:
            return

        # Build a map of image ID to ImageRef
        image_map = {}
        for img in document_images:
            if isinstance(img, ImageRef):
                image_map[img.id] = img

        # Extract only the images referenced in this chunk
        chunk_images = []
        for img_id in image_ids_in_chunk:
            if img_id in image_map:
                chunk_images.append(image_map[img_id])

        # Add to metadata if we found any images
        if chunk_images:
            metadata["images"] = chunk_images
            metadata["image_refs"] = image_ids_in_chunk
