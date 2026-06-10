"""
MCP Server 入口 - E1 任务实现
Implements stdio-based MCP server with clean stdout/stderr separation
"""
import json
import logging
import sys
from typing import Any, Dict, Optional

from src.mcp_server.protocol_handler import ProtocolHandler
from src.observability.logger import get_logger

logger = get_logger(__name__)


class MCPServer:
    """MCP Server implementing stdio transport with JSON-RPC 2.0"""

    def __init__(self):
        """Initialize MCP Server"""
        self.protocol_handler = ProtocolHandler()
        self.transport_type = "stdio"
        logger.info("MCP Server initialized")

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP message (JSON-RPC 2.0)

        Args:
            message: Parsed JSON-RPC request

        Returns:
            JSON-RPC response dict
        """
        method = message.get('method')
        params = message.get('params', {})
        message_id = message.get('id')

        logger.debug(f"Received message: method={method}, id={message_id}")

        try:
            # Route to appropriate handler
            if method == 'initialize':
                result = self.protocol_handler.handle_initialize(params)
            elif method == 'tools/list':
                result = self.protocol_handler.handle_tools_list()
            elif method == 'tools/call':
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                result = self.protocol_handler.handle_tools_call(tool_name, arguments)
            else:
                return self.protocol_handler.handle_invalid_method()

            response = {
                'jsonrpc': '2.0',
                'id': message_id,
                'result': result
            }

        except ValueError as e:
            logger.error(f"Invalid params: {e}")
            response = self.protocol_handler.handle_invalid_params(str(e))
            response['id'] = message_id

        except Exception as e:
            logger.error(f"Internal error: {e}", exc_info=True)
            response = self.protocol_handler.handle_internal_error(str(e))
            response['id'] = message_id

        return response

    def run(self):
        """
        Main server loop reading from stdin and writing to stdout.
        All logs go to stderr, only JSON-RPC messages go to stdout.
        """
        logger.info("MCP Server starting stdio loop")

        while True:
            try:
                line = sys.stdin.readline()

                if not line:  # EOF
                    logger.info("EOF received, shutting down")
                    break

                # Parse JSON-RPC message
                try:
                    message = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    # Send error response to stdout
                    error_response = {
                        'jsonrpc': '2.0',
                        'error': {
                            'code': -32700,
                            'message': 'Parse error'
                        }
                    }
                    sys.stdout.write(json.dumps(error_response) + '\n')
                    sys.stdout.flush()
                    continue

                # Handle message and get response
                response = self.handle_message(message)

                # Write JSON-RPC response to stdout (only)
                sys.stdout.write(json.dumps(response) + '\n')
                sys.stdout.flush()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                # Send error response
                error_response = {
                    'jsonrpc': '2.0',
                    'error': {
                        'code': -32603,
                        'message': 'Internal error'
                    }
                }
                sys.stdout.write(json.dumps(error_response) + '\n')
                sys.stdout.flush()

        logger.info("MCP Server shutting down")
