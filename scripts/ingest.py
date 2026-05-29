#!/usr/bin/env python3
"""
Offline data ingestion script (C15).

Entry point for ingesting documents into the knowledge hub.
Supports PDF documents, chunk splitting, transformation, embedding, and storage.

Usage:
    python scripts/ingest.py --path <file_path> [--collection <name>] [--force]

Examples:
    # Ingest single PDF
    python scripts/ingest.py --path data/documents/sample.pdf --collection my_docs

    # Force re-ingestion (skip file integrity check)
    python scripts/ingest.py --path data/documents/sample.pdf --collection my_docs --force

    # Use default collection name
    python scripts/ingest.py --path data/documents/sample.pdf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure src package is in path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.ingestion_pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the ingestion script."""
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for ingest.py."""
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Ingest documents into the knowledge hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the document file (PDF)",
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="default",
        help="Collection name for grouping documents (default: 'default')",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-ingestion even if file unchanged (skip integrity check)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for data storage (default: 'data')",
    )

    return parser


def validate_input_file(file_path: str) -> Path:
    """Validate that input file exists and is readable.

    Args:
        file_path: Path to input file

    Returns:
        Absolute Path object

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file is not readable
    """
    path = Path(file_path).absolute()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {file_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Only PDF files are supported. Got: {path.suffix}")

    return path


def print_progress(stage: str, current: int, total: int) -> None:
    """Print progress update for pipeline stage.

    Args:
        stage: Stage name (load, split, transform, encode, store)
        current: Current progress count
        total: Total expected count
    """
    percentage = (current / total * 100) if total > 0 else 0
    print(f"  [{stage:12s}] {current}/{total} ({percentage:5.1f}%)")


def print_result_summary(result) -> None:
    """Print summary of ingestion result.

    Args:
        result: IngestionResult object
    """
    print("\n" + "=" * 60)
    print("INGESTION RESULT")
    print("=" * 60)

    print(f"Status:        {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"Documents:     {result.document_count}")
    print(f"Chunks:        {result.chunk_count}")
    print(f"Images:        {result.image_count}")
    print(f"Elapsed time:  {result.elapsed_ms:.1f}ms")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60 + "\n")


def main(args: Optional[list] = None) -> int:
    """Main entry point for ingest.py script.

    Args:
        args: Command-line arguments (for testing). If None, uses sys.argv

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(verbose=parsed_args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting data ingestion...")

    # Validate input file
    try:
        file_path = validate_input_file(parsed_args.path)
        logger.info(f"Input file: {file_path}")
    except (FileNotFoundError, ValueError, PermissionError) as e:
        logger.error(f"Input validation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load configuration
    try:
        settings = load_settings("config/settings.yaml")
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Failed to load settings - {e}", file=sys.stderr)
        return 1

    # Initialize pipeline
    try:
        data_dir = Path(parsed_args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        pipeline = IngestionPipeline(
            base_path=str(data_dir),
            images_dir=str(data_dir / "images"),
            db_path=str(data_dir / "db" / "image_index.db"),
            settings=settings,
        )
        logger.info("Ingestion pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        print(f"Error: Failed to initialize pipeline - {e}", file=sys.stderr)
        return 1

    # Create trace context
    trace = TraceContext(trace_type="ingestion")

    # Execute pipeline
    print(f"\nIngesting: {file_path.name}")
    print(f"Collection: {parsed_args.collection}")
    print(f"Force: {parsed_args.force}")
    print("\nProgress:")

    try:
        result = pipeline.run(
            source_path=str(file_path),
            collection=parsed_args.collection,
            trace=trace,
            on_progress=print_progress,
        )

        # Print result summary
        print_result_summary(result)

        # Check for errors
        if not result.success or result.errors:
            logger.error(f"Ingestion failed: {result.errors}")
            return 1

        logger.info(
            f"Ingestion completed successfully: "
            f"{result.document_count} documents, "
            f"{result.chunk_count} chunks"
        )

        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"Error: Pipeline execution failed - {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
