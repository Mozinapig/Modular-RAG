#!/usr/bin/env python3
"""
Pipeline extraction script for complex_technical_doc.pdf.
Uses IngestionPipeline to extract, chunk, transform, encode and store document.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ingestion.ingestion_pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(stage: str, current: int, total: int):
    """Display progress of pipeline stages."""
    if total > 0:
        percent = (current / total) * 100
        status = "✓" if current == total else "→"
        logger.info(f"  {status} {stage.capitalize()}: {current}/{total} ({percent:.0f}%)")
    else:
        logger.info(f"  → {stage.capitalize()}")


def main():
    """Execute pipeline extraction."""
    # Define paths
    pdf_path = str(project_root / "tests/fixtures/sample_documents/complex_technical_doc.pdf")
    collection_name = "complex_technical"

    logger.info(f"Starting pipeline extraction for: {pdf_path}")

    # Initialize pipeline
    pipeline = IngestionPipeline(
        base_path="data",
        images_dir="data/images",
        db_path="data/db/image_index.db"
    )

    # Execute pipeline
    logger.info("Running ingestion pipeline...")
    result = pipeline.run(
        source_path=pdf_path,
        collection=collection_name,
        on_progress=progress_callback
    )

    # Report results
    logger.info("\n" + "="*60)
    logger.info("Pipeline Execution Summary:")
    logger.info("="*60)
    logger.info(f"Status:        {'SUCCESS' if result.success else 'FAILED'}")
    logger.info(f"Documents:     {result.document_count}")
    logger.info(f"Chunks:        {result.chunk_count}")
    logger.info(f"Images:        {result.image_count}")
    logger.info(f"Elapsed time:  {result.elapsed_ms:.2f}ms")

    if result.errors:
        logger.warning("Errors encountered:")
        for error in result.errors:
            logger.warning(f"  - {error}")

    logger.info("="*60 + "\n")

    # Output file locations
    logger.info("Generated files:")
    logger.info(f"  - Vector embeddings: data/db/chroma/")
    logger.info(f"  - BM25 index: data/db/bm25/{collection_name}.json")
    logger.info(f"  - Images: data/images/")
    logger.info(f"  - Image index: data/db/image_index.db")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
