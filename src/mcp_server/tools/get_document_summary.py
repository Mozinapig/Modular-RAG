"""
get_document_summary Tool - E5 任务实现
Gets summary information for a document
"""
import logging
from typing import Any, Dict, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)


def get_document_summary(doc_id: str, **kwargs) -> Dict[str, Any]:
    """
    Get summary information for a document

    Args:
        doc_id: Document ID

    Returns:
        MCP tool result with document summary
    """
    try:
        logger.info(f"Getting summary for document: {doc_id}")

        # Try to load document from ChromaStore
        from src.core.settings import load_settings
        from src.libs.vector_store.chroma_store import ChromaStore

        settings = load_settings()
        vector_store = ChromaStore(settings=settings)

        # Query document by ID
        try:
            results = vector_store.query_by_metadata(
                collection_name=None,
                where={'doc_id': doc_id},
                limit=1
            )
        except Exception as e:
            # Fallback to searching by source_path as doc_id
            logger.debug(f"Trying fallback search: {e}")
            results = vector_store.query_by_metadata(
                collection_name=None,
                where={'source': doc_id},
                limit=1
            )

        if not results:
            logger.warning(f"Document not found: {doc_id}")
            return {
                'content': [
                    {
                        'type': 'text',
                        'text': f'Document "{doc_id}" not found.'
                    }
                ],
                'error': True
            }

        # Extract summary info from first result
        result = results[0]
        metadata = result.get('metadata', {})

        summary = {
            'doc_id': doc_id,
            'title': metadata.get('title', 'Untitled'),
            'summary': metadata.get('summary', 'No summary available'),
            'tags': metadata.get('tags', []),
            'source': metadata.get('source', 'Unknown source'),
            'chunk_count': metadata.get('chunk_count', 'Unknown'),
            'created_at': metadata.get('created_at', 'Unknown')
        }

        # Format response
        text = f"""
Document Summary
================
**Title**: {summary['title']}
**Source**: {summary['source']}

**Summary**: {summary['summary']}

**Tags**: {', '.join(summary['tags']) if summary['tags'] else 'None'}

**Metadata**:
- Chunks: {summary['chunk_count']}
- Created: {summary['created_at']}
""".strip()

        return {
            'content': [
                {
                    'type': 'text',
                    'text': text
                }
            ],
            'summary': summary
        }

    except Exception as e:
        logger.error(f"Error in get_document_summary: {e}", exc_info=True)
        return {
            'content': [
                {
                    'type': 'text',
                    'text': f'Error getting document summary: {str(e)}'
                }
            ],
            'error': True
        }
