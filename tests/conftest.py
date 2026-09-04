from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import FastMCPProxy, ProxyClient

from mcp_compressor.tools import CompressedTools, ReloadableClientManager
from mcp_compressor.types import CompressionLevel


@asynccontextmanager
async def _compressed_proxy(
    compression_level: CompressionLevel,
    server_name: str,
    toonify: bool = False,
):
    """Build a CompressedTools-wrapped proxy over the stdio test backend.

    Mirrors the estate's own composition (estate_server.EstateBackend.tools):
    a ReloadableClientManager over a StdioTransport, a FastMCPProxy, and
    CompressedTools on top. There is no separate 'wrap' code path any more —
    the daemon is the only product surface — so tests assemble the pieces the
    daemon uses directly.
    """
    server_path = Path(__file__).parent / "mcp_server.py"
    transport = StdioTransport(command="python", args=[str(server_path)])

    async def connect() -> ProxyClient:
        client = ProxyClient(transport=transport, init_timeout=None)
        await client.__aenter__()
        return client

    manager = ReloadableClientManager(connect=connect)
    await manager.start()
    try:
        proxy = FastMCPProxy(client_factory=manager.get_client, name="MCP Compressor Proxy")
        compressed = CompressedTools(
            proxy,
            compression_level=compression_level,
            server_name=server_name,
            toonify=toonify,
            client_manager=manager,
        )
        await compressed.configure_server()
        yield proxy
    finally:
        await manager.stop()


@pytest.fixture
async def proxy_mcp_client(request) -> AsyncGenerator[Client, None]:
    """A FastMCP client connected to a CompressedTools-wrapped test backend."""
    compression_level = getattr(request, "param", None) or CompressionLevel.LOW
    async with (
        _compressed_proxy(compression_level=compression_level, server_name="test_server") as mcp,
        Client(mcp) as client,
    ):
        yield client


@pytest.fixture
async def proxy_mcp_client_toonify() -> AsyncGenerator[Client, None]:
    """A FastMCP client connected to a toonified CompressedTools-wrapped test backend."""
    async with (
        _compressed_proxy(
            compression_level=CompressionLevel.LOW, server_name="test_server", toonify=True
        ) as mcp,
        Client(mcp) as client,
    ):
        yield client


@pytest.fixture
async def backend_mcp_client() -> AsyncGenerator[Client, None]:
    """A FastMCP client connected directly to the backend MCP server."""
    server_path = Path(__file__).parent / "mcp_server.py"
    async with Client(server_path) as client:
        yield client
