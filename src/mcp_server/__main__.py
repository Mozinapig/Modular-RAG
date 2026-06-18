"""
MCP Server entry point for module execution.
Allows running: python -m src.mcp_server
"""

from src.mcp_server.server import MCPServer

if __name__ == '__main__':
    server = MCPServer()
    server.run()
