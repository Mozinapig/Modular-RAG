"""
Response Builder - E3 任务实现
Builds MCP-formatted responses with markdown and citations
"""
import logging
from typing import Any, Dict, List

from src.observability.logger import get_logger

logger = get_logger(__name__)


class ResponseBuilder:
    """Builds MCP-formatted responses from retrieval results"""

    def build(self, retrieval_results: List[Any], query: str) -> str:
        """
        Build markdown response with citation markers

        Args:
            retrieval_results: List of retrieval results
            query: Original query string

        Returns:
            Markdown-formatted response with [1], [2] citation markers
        """
        if not retrieval_results:
            return f"No results found for query: '{query}'"

        try:
            # Build markdown with citations
            lines = []
            lines.append(f"# Results for: {query}\n")

            for idx, result in enumerate(retrieval_results, 1):
                # Extract content
                content = result.content if hasattr(result, 'content') else str(result)

                # Add citation marker
                lines.append(f"{content} [{idx}]\n")

            markdown = "\n".join(lines)

            logger.info(f"Built response with {len(retrieval_results)} citations")

            return markdown

        except Exception as e:
            logger.error(f"Error building response: {e}", exc_info=True)
            return f"Error building response: {str(e)}"

    def build_error(self, error_message: str) -> str:
        """
        Build error response

        Args:
            error_message: Error message

        Returns:
            Formatted error message
        """
        return f"## Error\n\n{error_message}"
