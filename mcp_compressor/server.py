"""Dynamic backend pool for MCP Compressor server mode.

When mcp-compressor runs as ``--server``, it starts this pool manager instead
of a single fixed backend.  Clients POST their backend command to ``/spawn``;
the pool starts the backend on a dynamic port (if not already cached) and
returns the port number.  The client then connects directly to that port for
MCP operations.

Multiple clients with the same backend command share one backend process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import socket
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastmcp import FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import FastMCPProxy, ProxyClient
from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .tools import CompressedTools, ReloadableClientManager
from .types import CompressionLevel


def _make_backend_key(command: str, args: list[str]) -> str:
    """Return a short deterministic key for a backend command."""
    payload = command + "|" + "|".join(args)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _find_free_port() -> int:
    """Find a free port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class BackendInstance:
    """A running backend with its compressed proxy."""

    port: int
    client_manager: ReloadableClientManager
    compressed_tools: CompressedTools
    mcp: FastMCP
    server: uvicorn.Server
    serve_task: asyncio.Task[None] | None = None


class BackendPool:
    """Pool of running backend MCP servers, keyed by deterministic backend key.

    Each backend gets its own uvicorn server on a random free port on localhost.
    The spawn endpoint returns the port so clients can connect directly.
    """

    def __init__(self, log_level: str = "warning") -> None:
        self._backends: dict[str, BackendInstance] = {}
        self._lock = asyncio.Lock()
        self._log_level = log_level

    async def spawn(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        server_name: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a backend (or return existing) and return its key + port."""
        key = _make_backend_key(command, args)

        async with self._lock:
            if key in self._backends:
                logger.info(f"Backend {key!r} already running on port {self._backends[key].port}; reusing")
                inst = self._backends[key]
            else:
                port = _find_free_port()
                logger.info(f"Spawning backend {key!r} on port {port}: {command} {' '.join(args)}")

                transport = StdioTransport(command=command, args=args, env=env, cwd=cwd)

                async def connect() -> ProxyClient:
                    client = ProxyClient(transport=transport, init_timeout=None)
                    await client.__aenter__()
                    return client

                client_manager = ReloadableClientManager(connect=connect)
                await client_manager.start()

                mcp = FastMCPProxy(client_factory=client_manager.get_client, name="MCP Compressor Proxy")

                compressed_tools = CompressedTools(
                    mcp,
                    compression_level=compression_level,
                    server_name=server_name,
                    toonify=False,
                    client_manager=client_manager,
                )
                await compressed_tools.configure_server()

                # Start uvicorn for this backend's MCP endpoint
                mcp_app = mcp.http_app(path="/mcp", transport="streamable-http")
                config = uvicorn.Config(
                    mcp_app,
                    host="127.0.0.1",
                    port=port,
                    log_level=self._log_level,
                )
                uvicorn_server = uvicorn.Server(config)
                serve_task = asyncio.create_task(uvicorn_server.serve())

                inst = BackendInstance(
                    port=port,
                    client_manager=client_manager,
                    compressed_tools=compressed_tools,
                    mcp=mcp,
                    server=uvicorn_server,
                    serve_task=serve_task,
                )
                self._backends[key] = inst

                # Wait for the server to actually start listening
                for _ in range(50):
                    await asyncio.sleep(0.05)
                    if uvicorn_server.started:
                        break

            stats = await inst.compressed_tools.get_compression_stats()
            return {
                "backend_key": key,
                "port": inst.port,
                "tool_count": stats["original_tool_count"],
            }

    async def close(self) -> None:
        """Stop all running backends."""
        async with self._lock:
            for key, instance in list(self._backends.items()):
                logger.info(f"Stopping backend {key!r} on port {instance.port}")
                instance.server.should_exit = True
                if instance.serve_task:
                    instance.serve_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await instance.serve_task
                with contextlib.suppress(Exception):
                    await instance.client_manager.stop()
            self._backends.clear()


def create_pool_app(pool: BackendPool) -> Starlette:
    """Build the Starlette app that fronts the backend pool."""

    async def spawn(request: Request) -> JSONResponse:
        body = await request.json()
        if "command" not in body:
            return JSONResponse({"error": "Missing required field: command"}, status_code=400)
        command = body["command"]
        args = body.get("args", [])
        env = body.get("env")
        cwd = body.get("cwd")
        compression = body.get("compression_level", "medium")
        server_name = body.get("server_name")

        try:
            comp_level = CompressionLevel(compression.lower())
        except ValueError:
            comp_level = CompressionLevel.MEDIUM

        result = await pool.spawn(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            compression_level=comp_level,
            server_name=server_name,
        )
        return JSONResponse(result)

    async def health(request: Request) -> Response:
        return Response("ok", media_type="text/plain")

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/spawn", spawn, methods=["POST"]),
        ]
    )
