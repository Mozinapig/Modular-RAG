"""
Protocol Handler - E2 任务实现
Handles JSON-RPC 2.0 protocol parsing and tool routing
"""
import logging
from typing import Any, Dict, List, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)


class ProtocolHandler:
    """Handles MCP protocol parsing and tool routing"""

    def __init__(self):
        """Initialize protocol handler with registered tools"""
        self.server_name = 'modular-rag-mcp'
        self.server_version = '1.0.0'
        self._register_tools()

    def _register_tools(self):
        """Register all available MCP tools"""
        self.tools_registry = {
            'query_knowledge_hub': {
                'name': 'query_knowledge_hub',
                'description': 'Query the knowledge hub with hybrid search (dense + sparse) and optional reranking',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'Number of results to return',
                            'default': 5
                        },
                        'collection': {
                            'type': 'string',
                            'description': 'Collection to search in (optional)'
                        }
                    },
                    'required': ['query']
                }
            },
            'list_collections': {
                'name': 'list_collections',
                'description': 'List all available document collections',
                'inputSchema': {
                    'type': 'object',
                    'properties': {}
                }
            },
            'get_document_summary': {
                'name': 'get_document_summary',
                'description': 'Get summary of a document by ID',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'doc_id': {
                            'type': 'string',
                            'description': 'Document ID'
                        }
                    },
                    'required': ['doc_id']
                }
            }
        }

    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request - respond with server capabilities

        Args:
            params: Initialize request parameters

        Returns:
            Server info and capabilities
        """
        logger.info("Handling initialize request")

        return {
            'serverInfo': {
                'name': self.server_name,
                'version': self.server_version
            },
            'capabilities': {
                'tools': {
                    'list': {},
                    'call': {}
                }
            }
        }

    def handle_tools_list(self) -> List[Dict[str, Any]]:
        """
        Handle tools/list request - return all registered tool schemas

        Returns:
            List of tool schemas
        """
        logger.debug("Handling tools/list request")

        tools = []
        for tool_name, tool_schema in self.tools_registry.items():
            tools.append({
                'name': tool_schema['name'],
                'description': tool_schema['description'],
                'inputSchema': tool_schema['inputSchema']
            })

        return tools

    def handle_tools_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tools/call request - route to specific tool implementation

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        logger.info(f"Handling tools/call: {tool_name}")

        if tool_name not in self.tools_registry:
            logger.error(f"Tool not found: {tool_name}")
            raise ValueError(f"Tool '{tool_name}' not found")

        # Route to tool implementation
        result = self._execute_tool(tool_name, arguments)

        return result

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specific tool by name

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        try:
            if tool_name == 'query_knowledge_hub':
                from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub
                return query_knowledge_hub(**arguments)

            elif tool_name == 'list_collections':
                from src.mcp_server.tools.list_collections import list_collections
                return list_collections()

            elif tool_name == 'get_document_summary':
                from src.mcp_server.tools.get_document_summary import get_document_summary
                return get_document_summary(**arguments)

            else:
                raise ValueError(f"Tool '{tool_name}' not implemented")

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            raise

    def handle_invalid_method(self) -> Dict[str, Any]:
        """
        Handle invalid method error (JSON-RPC -32601)

        Returns:
            Error response
        """
        return {
            'jsonrpc': '2.0',
            'error': {
                'code': -32601,
                'message': 'Method not found'
            }
        }

    def handle_invalid_params(self, error_msg: str) -> Dict[str, Any]:
        """
        Handle invalid params error (JSON-RPC -32602)

        Args:
            error_msg: Error message

        Returns:
            Error response
        """
        return {
            'jsonrpc': '2.0',
            'error': {
                'code': -32602,
                'message': f'Invalid params: {error_msg}'
            }
        }

    def handle_internal_error(self, error_msg: str) -> Dict[str, Any]:
        """
        Handle internal error (JSON-RPC -32603)
        Never leaks stack traces to client

        Args:
            error_msg: Error message (stripped of traceback)

        Returns:
            Error response
        """
        # Clean up error message to not leak traceback
        clean_msg = error_msg.split('\n')[0] if '\n' in error_msg else error_msg

        return {
            'jsonrpc': '2.0',
            'error': {
                'code': -32603,
                'message': f'Internal error: {clean_msg}'
            }
        }
