#!/usr/bin/env python3
"""
Online query script (D7).

Entry point for querying the knowledge hub using HybridSearch + Reranker.
Supports dense and sparse retrieval with optional reranking.

Usage:
    python scripts/query.py --query "问题" [--top-k 10] [--collection xxx] [--verbose] [--no-rerank]

Examples:
    # Simple query
    python scripts/query.py --query "如何配置 Azure？"

    # With custom top-k
    python scripts/query.py --query "Python最佳实践" --top-k 20

    # Verbose mode to see intermediate results
    python scripts/query.py --query "什么是RAG" --verbose

    # Skip reranking
    python scripts/query.py --query "文件上传" --no-rerank
"""

import argparse
import logging
import sys
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure src package is in path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import CoreReranker
from src.core.types import RetrievalResult
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for query.py."""
    parser = argparse.ArgumentParser(
        prog="query",
        description="Query the knowledge hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Query text (required)",
    )

    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)",
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=None,
        help="Collection name to search in (optional, searches all if not specified)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show intermediate results from each stage (Dense/Sparse/Fusion/Rerank)",
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking stage",
    )

    return parser


def load_bm25_indexer() -> Optional[BM25Indexer]:
    """Load BM25 indexer for sparse retrieval."""
    try:
        indexer = BM25Indexer()
        indexer.load("data/db/bm25")
        # indexer.load(r"D:\pythonProject\Modular-RAG\data\db\bm25")
        return indexer
    except Exception as e:
        logger.warning(f"Failed to load BM25 indexer: {e}")
        return None


def initialize_components(settings):
    """Initialize all required components for query processing."""
    try:
        embedding_factory = EmbeddingFactory()
        embedding_client = embedding_factory.create(settings.embedding)

        vector_store_factory = VectorStoreFactory()
        vector_store = vector_store_factory.create(settings)

        bm25_indexer = load_bm25_indexer()

        return {
            "embedding_client": embedding_client,
            "vector_store": vector_store,
            "bm25_indexer": bm25_indexer,
        }
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


def create_retrievers(settings, components):
    """Create retriever instances."""
    query_processor = QueryProcessor()
    dense_retriever = DenseRetriever(
        embedding_client=components["embedding_client"],
        vector_store=components["vector_store"],
        settings=settings
    )
    sparse_retriever = SparseRetriever(
        settings=settings,
        bm25_indexer=components["bm25_indexer"],
        vector_store=components["vector_store"]
    )
    fusion = RRFFusion(k=60)

    return {
        "query_processor": query_processor,
        "dense_retriever": dense_retriever,
        "sparse_retriever": sparse_retriever,
        "fusion": fusion,
    }


def format_result(result: RetrievalResult, index: int) -> str:
    """Format a single retrieval result for display."""
    # Truncate text to first 100 chars for display
    text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
    source = result.metadata.get("source_path", "unknown")
    page = result.metadata.get("page", "")

    page_str = f" [p{page}]" if page else ""
    return f"{index}. [{result.score:.3f}] {text_preview}\n   Source: {source}{page_str}"


def print_results(
    results: List[RetrievalResult],
    verbose: bool = False,
    stage_name: str = "Final"
) -> None:
    """Print results in formatted output."""
    if not results:
        print(f"({stage_name}) No results found")
        return

    print(f"\n({stage_name}) Found {len(results)} results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(format_result(result, i))
    print("-" * 80)


def handle_no_data() -> None:
    """Handle case when no data is available."""
    print("❌ No retrieval results found.")
    print("\nPossible reasons:")
    print("  1. No documents have been ingested yet")
    print("  2. Query doesn't match any documents")
    print("  3. Vector store or BM25 index is not initialized")
    print("\nNext steps:")
    print("  - Run: python scripts/ingest.py --path <file_path> --collection <name>")
    print("  - Try a different query")


def main():
    """Main entry point for query.py script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.top_k <= 0:
        logger.error("--top-k must be positive")
        sys.exit(1)

    try:
        # Load settings
        logger.info("Loading settings...")
        settings = load_settings("config/settings.yaml")

        # Initialize components
        logger.info("Initializing components...")
        components = initialize_components(settings)

        # Create retrievers
        logger.info("Creating retrievers...")
        retrievers = create_retrievers(settings, components)

        # Create HybridSearch
        hybrid_search = HybridSearch(
            query_processor=retrievers["query_processor"],
            dense_retriever=retrievers["dense_retriever"],
            sparse_retriever=retrievers["sparse_retriever"],
            fusion=retrievers["fusion"]
        )

        # Create trace context
        trace = TraceContext(trace_type="query")

        # Process query
        logger.info(f"Processing query: {args.query}")

        # Build filters if collection specified
        filters = {"collection": args.collection} if args.collection else None

        # Perform hybrid search
        results = hybrid_search.search(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            trace=trace
        )

        # Show results
        if args.verbose:
            print_results(results, verbose=True, stage_name="HybridSearch")

        # Apply reranking if enabled (unless explicitly disabled or backend is "none")
        should_rerank = (
            not args.no_rerank  # User didn't explicitly disable reranking
            and results  # Have results to rerank
            and settings.rerank.backend != "none"  # Config says to enable reranking
        )

        if should_rerank:
            logger.info(f"Applying reranker (backend: {settings.rerank.backend})...")
            reranker = CoreReranker(settings)
            results = reranker.rerank(
                query=args.query,
                candidates=results,
                trace=trace
            )
            if args.verbose:
                print_results(results, verbose=True, stage_name="Reranker")

        # Print final results
        if results:
            print_results(results, verbose=args.verbose, stage_name="Final Results")

            # Show summary stats
            total_score = sum(r.score for r in results)
            avg_score = total_score / len(results) if results else 0
            print(f"\n📊 Summary: {len(results)} results (avg score: {avg_score:.3f})")
        else:
            handle_no_data()

        trace.finish()
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error during query processing: {e}", exc_info=True)
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
