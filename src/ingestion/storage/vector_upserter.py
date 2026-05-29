"""
VectorUpserter: Generate stable chunk IDs and perform idempotent vector storage writes.
Ensures same content produces same ID, enabling deduplication.
"""

import logging
import hashlib
from typing import List, Optional, Dict

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


logger = logging.getLogger(__name__)


class VectorUpserter:
    """
    Write dense embeddings to vector store with idempotency guarantee.

    Features:
    - Deterministic ID generation: hash(source_path + chunk_index + content_hash[:8])
    - Idempotent upsert: same content always produces same ID
    - Metadata preservation: keeps all chunk metadata for retrieval context
    - Batch operations: efficient bulk writes to vector store
    """

    def __init__(self, vector_store):
        """
        Initialize VectorUpserter.

        Args:
            vector_store: BaseVectorStore instance for writing embeddings
        """
        self.vector_store = vector_store

    def upsert(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Upsert chunks with dense embeddings to vector store.

        Args:
            chunks: List of chunks with dense_embedding in metadata
            trace: Optional trace context for tracking
        """
        if not chunks:
            return

        # Generate stable IDs for all chunks
        ids = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            # Generate deterministic ID
            chunk_id = self._generate_chunk_id(chunk)
            ids.append(chunk_id)

            # Extract dense embedding
            dense_embedding = chunk.metadata.get("dense_embedding")
            if dense_embedding is not None:
                embeddings.append(dense_embedding)
            else:
                # Skip chunks without embeddings
                embeddings.append([])

            # Prepare metadata for storage
            metadata = self._prepare_metadata(chunk)
            metadatas.append(metadata)

        # Write to vector store
        if embeddings:
            self.vector_store.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Upserted {len(chunks)} chunks to vector store")

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """
        Generate stable chunk ID from metadata.

        Formula: hash(source_path + chunk_index + content_hash[:8])

        Args:
            chunk: Chunk to generate ID for

        Returns:
            Hexadecimal hash string (deterministic per chunk content)
        """
        source_path = chunk.metadata.get("source_path", "")
        chunk_index = chunk.metadata.get("chunk_index", 0)
        content_hash = chunk.metadata.get("content_hash", "")

        # Use only first 8 characters of content_hash
        content_hash_prefix = content_hash[:8] if content_hash else ""

        # Combine inputs
        combined = f"{source_path}#{chunk_index}#{content_hash_prefix}"

        # Generate stable hash
        hash_obj = hashlib.sha256(combined.encode())
        return hash_obj.hexdigest()

    def _prepare_metadata(self, chunk: Chunk) -> Dict:
        """
        Prepare chunk metadata for vector store storage.

        Args:
            chunk: Chunk with metadata

        Returns:
            Dictionary with metadata suitable for vector store
        """
        # Preserve key metadata fields
        metadata = {
            "source_path": chunk.metadata.get("source_path", ""),
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "text": chunk.text,
        }

        # Add optional fields if present
        if "title" in chunk.metadata:
            metadata["title"] = chunk.metadata["title"]
        if "summary" in chunk.metadata:
            metadata["summary"] = chunk.metadata["summary"]
        if "tags" in chunk.metadata:
            metadata["tags"] = chunk.metadata["tags"]
        if "image_refs" in chunk.metadata:
            metadata["image_refs"] = str(chunk.metadata["image_refs"])

        return metadata
