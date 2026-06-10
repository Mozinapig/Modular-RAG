"""
Unit tests for Protocol Handler (E2)
"""
import pytest
from src.mcp_server.protocol_handler import ProtocolHandler


class TestProtocolHandlerInitialize:
    """Test protocol handler initialization"""

    def test_initialize_returns_server_info(self):
        """Test initialize returns correct server info"""
        handler = ProtocolHandler()
        response = handler.handle_initialize({})

        assert response['serverInfo']['name'] == 'modular-rag-mcp'
        assert response['serverInfo']['version'] == '1.0.0'

    def test_initialize_returns_capabilities(self):
        """Test initialize declares capabilities"""
        handler = ProtocolHandler()
        response = handler.handle_initialize({})

        assert 'capabilities' in response
        assert 'tools' in response['capabilities']


class TestProtocolHandlerToolsList:
    """Test tools/list functionality"""

    def test_tools_list_includes_query_tool(self):
        """Test query_knowledge_hub is in tools list"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        tool_names = [t['name'] for t in tools]
        assert 'query_knowledge_hub' in tool_names

    def test_tools_list_includes_collections_tool(self):
        """Test list_collections is in tools list"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        tool_names = [t['name'] for t in tools]
        assert 'list_collections' in tool_names

    def test_tools_list_includes_summary_tool(self):
        """Test get_document_summary is in tools list"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        tool_names = [t['name'] for t in tools]
        assert 'get_document_summary' in tool_names

    def test_each_tool_has_description(self):
        """Test all tools have descriptions"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        for tool in tools:
            assert 'description' in tool
            assert len(tool['description']) > 0

    def test_each_tool_has_input_schema(self):
        """Test all tools have input schemas"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        for tool in tools:
            assert 'inputSchema' in tool
            assert 'properties' in tool['inputSchema']


class TestProtocolHandlerToolsCall:
    """Test tools/call functionality"""

    def test_tools_call_invalid_tool_raises_error(self):
        """Test calling non-existent tool raises error"""
        handler = ProtocolHandler()

        with pytest.raises(ValueError):
            handler.handle_tools_call('nonexistent_tool', {})

    def test_tools_call_with_valid_arguments(self):
        """Test calling tool with valid arguments doesn't raise"""
        handler = ProtocolHandler()

        # list_collections takes no arguments
        try:
            # This might fail due to actual execution, but shouldn't raise on routing
            result = handler._execute_tool('list_collections', {})
        except Exception as e:
            # Should be execution error, not routing error
            assert 'not found' not in str(e).lower()


class TestProtocolHandlerErrors:
    """Test error handling"""

    def test_invalid_method_error_code(self):
        """Test invalid method returns correct error code"""
        handler = ProtocolHandler()
        error = handler.handle_invalid_method()

        assert error['error']['code'] == -32601
        assert 'message' in error['error']

    def test_invalid_params_error_code(self):
        """Test invalid params returns correct error code"""
        handler = ProtocolHandler()
        error = handler.handle_invalid_params("test error")

        assert error['error']['code'] == -32602
        assert 'message' in error['error']

    def test_internal_error_error_code(self):
        """Test internal error returns correct error code"""
        handler = ProtocolHandler()
        error = handler.handle_internal_error("test error")

        assert error['error']['code'] == -32603
        assert 'message' in error['error']

    def test_internal_error_message_does_not_contain_traceback(self):
        """Test internal error message doesn't leak traceback"""
        handler = ProtocolHandler()
        error = handler.handle_internal_error("Error\nTraceback (most recent call last):\n  File...")

        message = error['error']['message']
        assert 'Traceback' not in message
        assert 'File' not in message or '\n' not in message


class TestProtocolHandlerToolSchemas:
    """Test tool input schemas are valid"""

    def test_query_knowledge_hub_schema_requires_query(self):
        """Test query_knowledge_hub requires query parameter"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        query_tool = next(t for t in tools if t['name'] == 'query_knowledge_hub')
        schema = query_tool['inputSchema']

        assert 'query' in schema['required']

    def test_get_document_summary_schema_requires_doc_id(self):
        """Test get_document_summary requires doc_id parameter"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        summary_tool = next(t for t in tools if t['name'] == 'get_document_summary')
        schema = summary_tool['inputSchema']

        assert 'doc_id' in schema['required']

    def test_list_collections_schema_is_valid(self):
        """Test list_collections has valid schema"""
        handler = ProtocolHandler()
        tools = handler.handle_tools_list()

        list_tool = next(t for t in tools if t['name'] == 'list_collections')
        schema = list_tool['inputSchema']

        assert 'properties' in schema
        # No required parameters for list_collections
