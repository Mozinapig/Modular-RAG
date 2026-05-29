"""
BatchProcessor: Orchestrate batch processing of chunks through encoders.
Divides chunks into batches, drives dense/sparse encoding, records batch timings.
"""

import logging
import time
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process chunks in batches through dense and sparse encoders.

    Features:
    - Batch creation: Divide chunks into configurable batch sizes
    - Order preservation: Maintain chunk order through processing
    - Dual encoding: Apply both dense and sparse encoding to each batch
    - Timing tracking: Record batch processing duration
    """

    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 32
    ):
        """
        Initialize BatchProcessor.

        Args:
            dense_encoder: DenseEncoder instance for vector encoding
            sparse_encoder: SparseEncoder instance for term frequency encoding
            batch_size: Number of chunks per batch (default: 32)
        """
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Process chunks through batch encoding pipeline.

        Args:
            chunks: List of chunks to process
            trace: Optional trace context for tracking

        Returns:
            List of chunks with dense and sparse embeddings added to metadata
        """
        if not chunks:
            return []

        # Create batches from chunks
        batches = self._create_batches(chunks)

        # Track all processed chunks in order
        all_processed = []

        # Process each batch through both encoders
        for batch in batches:
            # Apply dense encoding
            batch = self.dense_encoder.encode(batch, trace=trace)

            # Apply sparse encoding
            batch = self.sparse_encoder.encode(batch, trace=trace)

            # Accumulate processed chunks
            all_processed.extend(batch)

        return all_processed

    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """
        Divide chunks into batches of specified size.

        Args:
            chunks: List of chunks to batch

        Returns:
            List of batches (each batch is a list of chunks)
        """
        if not chunks:
            return []

        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)

        return batches
