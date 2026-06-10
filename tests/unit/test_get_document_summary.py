"""
Unit tests for get_document_summary tool (E5) - Simplified
"""
import pytest


class TestGetDocumentSummary:
    """Test get_document_summary tool"""

    def test_get_document_summary_function_exists(self):
        """Test get_document_summary function can be imported"""
        from src.mcp_server.tools.get_document_summary import get_document_summary
        assert callable(get_document_summary)

    def test_get_document_summary_returns_dict(self):
        """Test get_document_summary returns dictionary"""
        from src.mcp_server.tools.get_document_summary import get_document_summary

        # Call with non-existent doc
        result = get_document_summary('nonexistent_id_xyz')

        assert isinstance(result, dict)
        assert 'content' in result
        assert 'error' in result or 'content' in result

    def test_get_document_summary_error_response_structure(self):
        """Test error response has proper structure"""
        from src.mcp_server.tools.get_document_summary import get_document_summary

        result = get_document_summary('nonexistent_doc')

        # Should have content field
        assert 'content' in result
        assert isinstance(result['content'], list)
        assert len(result['content']) > 0

        # First content item should be text
        if result['content']:
            assert result['content'][0]['type'] == 'text'

    def test_get_document_summary_returns_summary_on_not_found(self):
        """Test returns meaningful message when not found"""
        from src.mcp_server.tools.get_document_summary import get_document_summary

        result = get_document_summary('abc_xyz_123')

        # Should indicate not found
        text = result['content'][0]['text']
        assert 'not found' in text.lower() or 'error' in text.lower()
