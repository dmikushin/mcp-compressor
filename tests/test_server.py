"""Tests for mcp_compressor/server.py — the daemon's HTTP surface."""

from __future__ import annotations

import asyncio
import contextlib
import socket


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestCreateDaemonApp:
    """The daemon serves the estate at /mcp alongside /health.

    This drives a real uvicorn on a free port so the FastMCP streamable-HTTP
    lifespan (its session-manager task group) actually runs — a TestClient
    would not exercise it, and a dropped lifespan is exactly the failure this
    endpoint is prone to.
    """

    async def test_estate_four_tools_over_http_plus_health(self, tmp_path, monkeypatch):
        import uvicorn
        from fastmcp.client import Client

        from mcp_compressor import catalog_cache as cc
        from mcp_compressor.estate import ServerSpec
        from mcp_compressor.estate_server import Estate, build_estate_server
        from mcp_compressor.server import create_daemon_app

        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        # A backend with a warm catalog so `catalog` answers without spawning.
        estate = Estate([ServerSpec(name="demo", command="some-backend")])
        cc.save(estate._backend("demo").cache_key, [{"name": "a"}, {"name": "b"}], server="demo")

        estate_app = build_estate_server(estate).http_app(path="/mcp", transport="streamable-http")
        app = create_daemon_app(estate_app)

        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        serve_task = asyncio.create_task(server.serve())
        try:
            for _ in range(100):
                await asyncio.sleep(0.02)
                if server.started:
                    break
            else:
                raise RuntimeError("daemon app did not start")

            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                tools = await client.list_tools()
                assert {t.name for t in tools} == {
                    "catalog",
                    "get_tool_schema",
                    "invoke_tool",
                    "reload",
                }
                result = await client.call_tool("catalog", {})
                assert "demo (2): a, b" in result.content[0].text

            import httpx

            async with httpx.AsyncClient() as http:
                health = await http.get(f"http://127.0.0.1:{port}/health")
                assert health.status_code == 200
                assert health.text == "ok"
        finally:
            server.should_exit = True
            with contextlib.suppress(BaseException):
                await serve_task
            await estate.close()
