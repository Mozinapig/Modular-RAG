"""
list_collections Tool - E4 任务实现
Lists all available document collections
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.observability.logger import get_logger

logger = get_logger(__name__)


def list_collections(**kwargs) -> Dict[str, Any]:
    """
    List all available document collections

    Returns:
        MCP tool result with list of collections
    """
    try:
        logger.info("Listing collections")

        # Find collections in data/documents directory
        data_dir = Path('data/documents')

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return {
                'content': [
                    {
                        'type': 'text',
                        'text': 'No collections found. Data directory does not exist.'
                    }
                ],
                'collections': []
            }

        # Get all subdirectories as collections
        collections = []

        try:
            for item in data_dir.iterdir():
                if item.is_dir():
                    # Count files in collection
                    file_count = len(list(item.glob('*')))

                    collections.append({
                        'name': item.name,
                        'path': str(item),
                        'file_count': file_count
                    })

        except Exception as e:
            logger.error(f"Error iterating collections: {e}")

        logger.info(f"Found {len(collections)} collections")

        # Format response
        if collections:
            collection_names = [c['name'] for c in collections]
            text = f"Found {len(collections)} collection(s):\n" + \
                   "\n".join(f"- {c['name']} ({c['file_count']} files)" for c in collections)
        else:
            text = "No collections found."

        return {
            'content': [
                {
                    'type': 'text',
                    'text': text
                }
            ],
            'collections': collections
        }

    except Exception as e:
        logger.error(f"Error in list_collections: {e}", exc_info=True)
        return {
            'content': [
                {
                    'type': 'text',
                    'text': f'Error listing collections: {str(e)}'
                }
            ],
            'error': True
        }
