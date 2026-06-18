"""
query_knowledge_hub Tool - E3 任务实现
Main tool for querying the knowledge hub with hybrid search
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.reranker import CoreReranker
from src.core.response.response_builder import ResponseBuilder
from src.core.response.citation_generator import CitationGenerator
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

logger = get_logger(__name__)


def _get_settings_path() -> str:
    """Get the path to settings.yaml relative to project root."""
    # Try multiple common locations
    candidates = [
        Path(__file__).parent.parent.parent.parent / 'config' / 'settings.yaml',  # src/../config
        Path('config') / 'settings.yaml',
        Path.cwd() / 'config' / 'settings.yaml',
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("Could not find config/settings.yaml in any expected location")


def query_knowledge_hub(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Query the knowledge hub with hybrid search and optional reranking

    Args:
        query: Search query string
        top_k: Number of results to return
        collection: Collection to search in (optional)
        **kwargs: Additional parameters

    Returns:
        MCP tool result with markdown content and citations
    """
    try:
        logger.info(f"Processing query: {query[:100]}")

        # Load settings from default location
        settings_path = _get_settings_path()
        settings = load_settings(settings_path)

        # Create trace context
        trace = TraceContext(trace_type="query")

        # Initialize search engine
        hybrid_search = HybridSearch(settings=settings)

        # Perform hybrid search
        retrieval_results = hybrid_search.search(
            query=query,
            top_k=top_k,
            collection=collection,
            trace=trace
        )

        logger.info(f"Retrieved {len(retrieval_results)} results")

        # Optional: Apply reranker if configured
        if settings.rerank.enabled:
            reranker = CoreReranker(settings=settings)
            retrieval_results = reranker.rerank(
                query=query,
                candidates=retrieval_results,
                trace=trace
            )
            logger.info("Applied reranking")

        # Build response
        response_builder = ResponseBuilder()
        markdown_response = response_builder.build(retrieval_results, query)

        # Generate citations
        citation_generator = CitationGenerator()
        citations = citation_generator.generate(retrieval_results)

        # Finish trace
        trace.finish()

        return {
            'content': [
                {
                    'type': 'text',
                    'text': markdown_response
                }
            ],
            'structuredContent': {
                'citations': citations,
                'query': query,
                'retrieved_count': len(retrieval_results)
            }
        }

    except Exception as e:
        logger.error(f"Error in query_knowledge_hub: {e}", exc_info=True)
        return {
            'content': [
                {
                    'type': 'text',
                    'text': f'Error processing query: {str(e)}'
                }
            ],
            'error': True
        }
