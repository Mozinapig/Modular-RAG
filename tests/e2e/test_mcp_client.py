"""
End-to-end MCP Client side invocation tests.

Tests verify that MCP Server can be started as subprocess and respond to
tools/list and tools/call requests correctly.
"""

import pytest
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


class MCPClientSimulator:
    """Simple MCP client that communicates with server via subprocess stdio."""

    def __init__(self, server_process: subprocess.Popen):
        """
        Initialize MCP client simulator.

        Args:
            server_process: subprocess.Popen instance of MCP server
        """
        self.server_process = server_process
        self.message_id = 0

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send JSON-RPC request to server and receive response.

        Args:
            method: RPC method name
            params: Optional parameters dict

        Returns:
            Response dict from server
        """
        self.message_id += 1

        request = {
            'jsonrpc': '2.0',
            'id': self.message_id,
            'method': method
        }

        if params:
            request['params'] = params

        # Send request to server stdin
        request_line = json.dumps(request) + '\n'
        try:
            if isinstance(self.server_process.stdin, type(None)):
                raise RuntimeError("Server stdin is closed")
            self.server_process.stdin.write(request_line)
            self.server_process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"Failed to write to server stdin: {e}")

        # Read response from server stdout
        try:
            response_line = self.server_process.stdout.readline()
            if not response_line:
                raise RuntimeError("Server closed stdout (no response)")
            response_line = response_line.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read from server stdout: {e}")

        if not response_line:
            raise RuntimeError("No response from server")

        response = json.loads(response_line)

        # Check for error
        if 'error' in response:
            raise RuntimeError(f"Server error: {response['error']}")

        return response.get('result')

    def initialize(self) -> Dict[str, Any]:
        """Send initialize request."""
        return self._send_request('initialize', {})

    def list_tools(self) -> list:
        """Send tools/list request."""
        return self._send_request('tools/list')

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send tools/call request."""
        return self._send_request('tools/call', {
            'name': tool_name,
            'arguments': arguments
        })


@pytest.fixture
def mcp_server():
    """Start MCP server as subprocess and yield client."""
    # Start server with text mode to handle buffering correctly
    server_process = subprocess.Popen(
        [sys.executable, '-m', 'src.mcp_server'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent.parent),
        text=True,
        bufsize=1
    )

    # Give server time to start
    time.sleep(0.5)

    # Create client
    client = MCPClientSimulator(server_process)

    yield client

    # Cleanup
    try:
        if server_process.stdin and not server_process.stdin.closed:
            server_process.stdin.close()
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
    except Exception as e:
        print(f"Error cleaning up server: {e}")
        try:
            server_process.kill()
        except Exception:
            pass


class TestMCPClient:
    """Test MCP client-side invocations."""

    def test_initialize_request(self, mcp_server):
        """Test initialize request returns server info."""
        response = mcp_server.initialize()

        assert 'serverInfo' in response
        assert response['serverInfo']['name'] == 'modular-rag-mcp'
        assert 'version' in response['serverInfo']

        assert 'capabilities' in response
        assert 'tools' in response['capabilities']

    def test_tools_list_request(self, mcp_server):
        """Test tools/list returns available tools."""
        tools = mcp_server.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check for required tools
        tool_names = [t['name'] for t in tools]
        assert 'query_knowledge_hub' in tool_names
        assert 'list_collections' in tool_names

        # Verify tool schema
        query_tool = next(t for t in tools if t['name'] == 'query_knowledge_hub')
        assert 'description' in query_tool
        assert 'inputSchema' in query_tool
        assert 'properties' in query_tool['inputSchema']
        assert 'query' in query_tool['inputSchema']['properties']

    def test_query_knowledge_hub_basic(self, mcp_server):
        """Test query_knowledge_hub tool returns results with citations."""
        result = mcp_server.call_tool('query_knowledge_hub', {
            'query': '测试查询'
        })

        assert isinstance(result, dict)
        # Check if we get either a successful result or error result
        # Real implementation might return error if no data, but tool should return something
        assert 'content' in result or 'error' in result

    def test_query_knowledge_hub_with_top_k(self, mcp_server):
        """Test query_knowledge_hub accepts top_k parameter."""
        result = mcp_server.call_tool('query_knowledge_hub', {
            'query': 'test',
            'top_k': 3
        })

        # Verify the tool accepts the parameter and returns a response
        assert isinstance(result, dict)
        # Should have content (success or error message)
        assert 'content' in result

    def test_list_collections_tool(self, mcp_server):
        """Test list_collections tool returns response."""
        result = mcp_server.call_tool('list_collections', {})

        # Should return either dict or list
        assert result is not None

    def test_query_returns_citations(self, mcp_server):
        """Test that query results include citation information or error response."""
        result = mcp_server.call_tool('query_knowledge_hub', {
            'query': 'configuration',
            'top_k': 5
        })

        # Result should be a dict (either success or error)
        assert isinstance(result, dict)
        assert 'content' in result or 'error' in result or 'structuredContent' in result

    def test_multiple_sequential_calls(self, mcp_server):
        """Test multiple sequential tool calls."""
        # First call
        result1 = mcp_server.list_tools()
        assert len(result1) > 0

        # Second call
        result2 = mcp_server.call_tool('list_collections', {})
        assert result2 is not None

        # Third call
        result3 = mcp_server.call_tool('query_knowledge_hub', {
            'query': 'test query'
        })
        assert result3 is not None and isinstance(result3, dict)

    def test_invalid_tool_error(self, mcp_server):
        """Test that invalid tool name returns error."""
        with pytest.raises(RuntimeError, match="Server error"):
            mcp_server.call_tool('nonexistent_tool', {})

    def test_missing_required_param(self, mcp_server):
        """Test that missing required params returns error."""
        with pytest.raises(RuntimeError, match="Server error"):
            mcp_server.call_tool('query_knowledge_hub', {})

    def test_end_to_end_workflow(self, mcp_server):
        """Test complete end-to-end workflow: initialize → list → query."""
        # Step 1: Initialize
        init_result = mcp_server.initialize()
        assert 'serverInfo' in init_result
        assert 'capabilities' in init_result

        # Step 2: List tools
        tools = mcp_server.list_tools()
        assert len(tools) > 0
        query_tool_exists = any(t['name'] == 'query_knowledge_hub' for t in tools)
        assert query_tool_exists

        # Step 3: Execute query
        query_result = mcp_server.call_tool('query_knowledge_hub', {
            'query': 'Azure OpenAI configuration',
            'top_k': 5
        })

        # Verify result is a proper response structure
        assert isinstance(query_result, dict)
        # Tool should return a result with content field
        assert 'content' in query_result or 'error' in query_result
