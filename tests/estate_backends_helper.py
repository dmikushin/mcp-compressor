"""A tiny MCP server runnable over any transport, for estate integration tests.

    python estate_backends_helper.py <stdio|http|sse> <port> <tool_name>

Serves one tool named <tool_name> that echoes its argument, so a test can prove
the estate reached this specific backend over this specific transport.
"""

import sys

from fastmcp import FastMCP


def main() -> None:
    transport = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tool_name = sys.argv[3] if len(sys.argv) > 3 else "echo"

    mcp: FastMCP = FastMCP(f"backend-{tool_name}")

    @mcp.tool(name=tool_name)
    def _echo(value: str) -> str:
        return f"{tool_name}:{value}"

    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
