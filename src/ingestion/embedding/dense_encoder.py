"""
DenseEncoder: Batch embed chunks using BaseEmbedding.
Fuses image captions into text for comprehensive semantic representation.
"""

import logging
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.core.settings import Settings
from src.libs.embedding.embedding_factory import EmbeddingFactory


logger = logging.getLogger(__name__)


class DenseEncoder:
    """
    Encode chunks into dense embeddings using configured embedding service.

    Features:
    - Batch processing: efficiently embed multiple chunks at once
    - Caption fusion: optionally fuse image captions into text for better semantic coverage
    - Metadata preservation: keeps all chunk metadata and adds dense_embedding field
    """

    def __init__(self, settings: Settings, embedding_client=None):
        """
        Initialize DenseEncoder.

        Args:
            settings: Settings object with embedding configuration
            embedding_client: Optional embedding client for dependency injection (testing)
        """
        self.settings = settings
        self.embedding_client = embedding_client
        self.embedding_dim = settings.embedding.dimensions

        if self.embedding_client is None:
            try:
                factory = EmbeddingFactory()
                self.embedding_client = factory.create(settings.embedding)
            except Exception as e:
                logger.error(f"Failed to initialize embedding client: {e}")
                raise

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Encode chunks into dense embeddings.

        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for tracking

        Returns:
            List of chunks with dense_embedding added to metadata
        """
        if not chunks:
            return []

        # Prepare texts for embedding (with caption fusion)
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]

        # Embed texts
        embeddings = self.embedding_client.embed(texts, trace=trace)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata["dense_embedding"] = embedding

        return chunks

    def _prepare_text_for_embedding(self, chunk: Chunk) -> str:
        """
        Prepare chunk text for embedding, optionally fusing image captions.

        Args:
            chunk: Chunk to prepare

        Returns:
            Text ready for embedding (with captions fused if available)
        """
        text = chunk.text

        # Fuse image captions if available
        if "image_captions" in chunk.metadata:
            captions = chunk.metadata["image_captions"]
            caption_text = " ".join(captions.values())
            text = f"{text} {caption_text}"

        return text
