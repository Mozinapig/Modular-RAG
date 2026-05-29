"""
IngestionPipeline: Orchestrate the complete ingestion flow.
Chains Loader → Chunker → Transform → Encoder → Upserter with trace tracking.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from src.core.types import Document, Chunk, IngestionResult
from src.core.trace.trace_context import TraceContext
from src.core.settings import Settings, load_settings
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Complete ingestion pipeline for RAG system.

    Features:
    - Serial execution: Load → Split → Transform → Encode → Store
    - Trace context propagation throughout all stages
    - Progress callbacks for real-time monitoring
    - Error recovery: Continue on individual document failures
    - Comprehensive statistics tracking
    """

    def __init__(
        self,
        base_path: str = "data",
        images_dir: str = "data/images",
        db_path: str = "data/db/image_index.db",
        settings: Optional[Settings] = None
    ):
        """
        Initialize IngestionPipeline with all required components.

        Args:
            base_path: Base directory for outputs
            images_dir: Directory for storing extracted images
            db_path: Path to image index database
            settings: Settings object (uses defaults if not provided)
        """
        self.base_path = Path(base_path)
        self.images_dir = Path(images_dir)
        self.db_path = Path(db_path)

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use provided settings or load from config
        if settings is None:
            settings = load_settings("config/settings.yaml")
        self.settings = settings

        # Initialize components
        self.loader = PdfLoader(image_output_dir=str(self.images_dir))

        # Create a settings-like object for splitter with required attributes
        splitter_settings = type('SplitterSettings', (), {
            'provider': self.settings.extra.get('splitter', {}).get('provider', 'recursive'),
            'chunk_size': self.settings.extra.get('splitter', {}).get('chunk_size', 1024),
            'chunk_overlap': self.settings.extra.get('splitter', {}).get('chunk_overlap', 100)
        })()

        self.chunker = DocumentChunker(splitter_settings)
        self.vector_store = VectorStoreFactory().create(self.settings)
        self.bm25_indexer = BM25Indexer()
        self.image_storage = ImageStorage(
            base_path=str(self.images_dir),
            db_path=str(self.db_path)
        )

        # Initialize encoders
        self.dense_encoder = DenseEncoder(self.settings)
        self.sparse_encoder = SparseEncoder(self.bm25_indexer)
        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=32
        )

        # Initialize transformers
        self.chunk_refiner = ChunkRefiner(self.settings)
        self.image_captioner = ImageCaptioner(self.settings)
        self.metadata_enricher = MetadataEnricher(self.settings)

        # Initialize storage writer
        self.vector_upserter = VectorUpserter(self.vector_store)

    def run(
        self,
        source_path: str,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None
    ) -> IngestionResult:
        """
        Execute complete ingestion pipeline for a document.

        Pipeline stages:
        1. Load: Extract text and images from PDF
        2. Split: Chunk document into semantically meaningful pieces
        3. Transform: Enrich chunks with metadata and image descriptions
        4. Encode: Generate dense and sparse embeddings
        5. Store: Persist to vector DB, BM25 index, and image storage

        Args:
            source_path: Path to source PDF file
            collection: Collection name for grouping documents
            trace: Optional trace context for observability
            on_progress: Optional callback for progress updates

        Returns:
            IngestionResult with statistics and any errors
        """
        start_time = time.time()
        result = IngestionResult(success=False, collection=collection)

        try:
            # Load document
            if on_progress:
                on_progress("load", 0, 1)

            try:
                document = self._load_document(source_path, trace=trace)
            except Exception as e:
                error_msg = f"Failed to load document: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.elapsed_ms = (time.time() - start_time) * 1000
                return result

            if on_progress:
                on_progress("load", 1, 1)

            result.document_count = 1

            # Split document into chunks
            if on_progress:
                on_progress("split", 0, 1)

            try:
                chunks = self._chunk_document(document, trace=trace)
            except Exception as e:
                error_msg = f"Failed to chunk document: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.elapsed_ms = (time.time() - start_time) * 1000
                return result

            if on_progress:
                on_progress("split", 1, 1)

            result.chunk_count = len(chunks)

            # Transform chunks
            if on_progress:
                on_progress("transform", 0, 1)

            try:
                chunks = self._transform_chunks(chunks, trace=trace)
            except Exception as e:
                error_msg = f"Failed to transform chunks: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.elapsed_ms = (time.time() - start_time) * 1000
                return result

            if on_progress:
                on_progress("transform", 1, 1)

            # Encode chunks
            if on_progress:
                on_progress("encode", 0, 1)

            try:
                chunks = self._encode_chunks(chunks, trace=trace)
            except Exception as e:
                error_msg = f"Failed to encode chunks: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.elapsed_ms = (time.time() - start_time) * 1000
                return result

            if on_progress:
                on_progress("encode", 1, 1)

            # Store chunks and metadata
            if on_progress:
                on_progress("store", 0, 1)

            try:
                self._store_chunks(chunks, collection, trace=trace)
                # Store images if any
                image_count = self._store_images(document, collection)
                result.image_count = image_count
            except Exception as e:
                error_msg = f"Failed to store data: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.elapsed_ms = (time.time() - start_time) * 1000
                return result

            if on_progress:
                on_progress("store", 1, 1)

            result.success = True
            result.elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Successfully ingested document from {source_path}: "
                f"{result.document_count} docs, {result.chunk_count} chunks, "
                f"{result.image_count} images in {result.elapsed_ms:.0f}ms"
            )

            return result

        except Exception as e:
            error_msg = f"Unexpected pipeline error: {str(e)}"
            logger.exception(error_msg)
            result.errors.append(error_msg)
            result.elapsed_ms = (time.time() - start_time) * 1000
            return result

    def _load_document(
        self,
        source_path: str,
        trace: Optional[TraceContext] = None
    ) -> Document:
        """
        Load document from file using appropriate loader.

        Args:
            source_path: Path to source file
            trace: Optional trace context

        Returns:
            Document object with extracted content
        """
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Currently only PDF loader is implemented
        document = self.loader.load(source_path)

        if trace:
            trace.add_stage("load", {"source": source_path})

        logger.debug(f"Loaded document from {source_path}: {len(document.text)} chars")

        return document

    def _chunk_document(
        self,
        document: Document,
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Split document into chunks.

        Args:
            document: Document to split
            trace: Optional trace context

        Returns:
            List of Chunk objects
        """
        chunks = self.chunker.split_document(document)

        if trace:
            trace.add_stage("split", {"chunk_count": len(chunks)})

        logger.debug(f"Split document into {len(chunks)} chunks")

        return chunks

    def _transform_chunks(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Apply transformation and enrichment to chunks.

        Applies:
        1. ChunkRefiner: Remove noise, merge incomplete chunks
        2. MetadataEnricher: Add semantic metadata (title, summary, tags)
        3. ImageCaptioner: Generate descriptions for referenced images

        Args:
            chunks: List of chunks to transform
            trace: Optional trace context

        Returns:
            Transformed list of chunks
        """
        if not chunks:
            return []

        # Apply chunk refining (noise removal, merging)
        chunks = self.chunk_refiner.transform(chunks, trace=trace)

        # Enrich metadata
        chunks = self.metadata_enricher.transform(chunks, trace=trace)

        # Generate image captions
        chunks = self.image_captioner.transform(chunks, trace=trace)

        if trace:
            trace.add_stage("transform", {"chunk_count": len(chunks)})

        logger.debug(f"Transformed {len(chunks)} chunks")

        return chunks

    def _encode_chunks(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Generate embeddings (dense and sparse) for chunks.

        Args:
            chunks: List of chunks to encode
            trace: Optional trace context

        Returns:
            Chunks with dense_embedding and sparse_embedding in metadata
        """
        if not chunks:
            return []

        # Process through batch processor
        chunks = self.batch_processor.process(chunks, trace=trace)

        if trace:
            trace.add_stage("encode", {"chunk_count": len(chunks)})

        logger.debug(f"Encoded {len(chunks)} chunks with embeddings")

        return chunks

    def _store_chunks(
        self,
        chunks: List[Chunk],
        collection: str,
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Store chunks in vector database and sparse index.

        Performs:
        1. Vector upsert: Store embeddings to Chroma
        2. BM25 upsert: Update sparse index
        3. BM25 persistence: Save index to disk

        Args:
            chunks: List of encoded chunks to store
            collection: Collection name for grouping
            trace: Optional trace context
        """
        if not chunks:
            return

        # Add collection to chunk metadata
        for chunk in chunks:
            chunk.metadata["collection"] = collection

        # Upsert to vector store
        self.vector_upserter.upsert(chunks, trace=trace)

        # Build/update BM25 index
        self.bm25_indexer.build(chunks, trace=trace)

        # Save BM25 index to disk
        bm25_dir = Path(self.base_path) / "db" / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = bm25_dir / f"{collection}.json"
        self.bm25_indexer.save(str(bm25_path))

        if trace:
            trace.add_stage("store", {"chunks_stored": len(chunks)})

        logger.debug(f"Stored {len(chunks)} chunks to vector DB and BM25 index at {bm25_path}")

    def _store_images(
        self,
        document: Document,
        collection: str
    ) -> int:
        """
        Extract and store images from document.

        Args:
            document: Document with image references
            collection: Collection name for grouping

        Returns:
            Number of images stored
        """
        images = document.metadata.get("images", [])
        if not images:
            return 0

        stored_count = 0
        for image_ref in images:
            try:
                # Read image file and store
                if hasattr(image_ref, 'path'):
                    image_path = image_ref.path
                    if Path(image_path).exists():
                        with open(image_path, 'rb') as f:
                            image_data = f.read()

                        image_id = self.image_storage.save(
                            image_data=image_data,
                            collection=collection,
                            doc_hash=document.id,
                            page_num=getattr(image_ref, 'page', None)
                        )

                        stored_count += 1
                        logger.debug(f"Stored image {image_id}")
            except Exception as e:
                logger.warning(f"Failed to store image: {str(e)}")
                continue

        return stored_count
