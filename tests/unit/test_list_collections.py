"""
Unit tests for list_collections tool (E4) - Simplified
"""
import pytest


class TestListCollections:
    """Test list_collections tool"""

    def test_list_collections_function_exists(self):
        """Test list_collections function can be imported"""
        from src.mcp_server.tools.list_collections import list_collections
        assert callable(list_collections)

    def test_list_collections_returns_dict(self):
        """Test list_collections returns dictionary"""
        from src.mcp_server.tools.list_collections import list_collections

        result = list_collections()

        assert isinstance(result, dict)
        assert 'content' in result
        assert 'collections' in result

    def test_list_collections_returns_content_list(self):
        """Test result has content field with text"""
        from src.mcp_server.tools.list_collections import list_collections

        result = list_collections()

        assert isinstance(result['content'], list)
        assert len(result['content']) > 0
        assert result['content'][0]['type'] == 'text'

    def test_list_collections_returns_collections_array(self):
        """Test result has collections array"""
        from src.mcp_server.tools.list_collections import list_collections

        result = list_collections()

        assert isinstance(result['collections'], list)

    def test_list_collections_has_readable_response(self):
        """Test response has readable text"""
        from src.mcp_server.tools.list_collections import list_collections

        result = list_collections()

        text = result['content'][0]['text']
        assert len(text) > 0
        # Should say either "No collections" or "Found N collection(s)"
        assert 'collection' in text.lower()
