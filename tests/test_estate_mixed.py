"""Integration tests: the estate fronts stdio, http and sse backends alike.

Each remote backend is a real subprocess serving MCP over that transport on a
free port; readiness is polled with a bounded retry (never an open-ended wait),
so a server that fails to come up fails the test quickly instead of hanging.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastmcp import Client

from mcp_compressor import catalog_cache as cc
from mcp_compressor.estate import ServerSpec
from mcp_compressor.estate_server import Estate

HELPER = str(Path(__file__).parent / "estate_backends_helper.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_ready(url: str, *, tries: int = 60, delay: float = 0.1) -> None:
    """Poll the MCP endpoint until it answers list_tools, with a hard cap."""
    last: Exception | None = None
    for _ in range(tries):
        try:
            async with Client(url) as client:
                await client.list_tools()
                return
        except Exception as exc:  # not ready yet
            last = exc
            await asyncio.sleep(delay)
    raise RuntimeError(f"backend at {url} never became ready: {last}")


class _Remote:
    def __init__(self, transport: str, tool_name: str) -> None:
        self.transport = transport
        self.tool_name = tool_name
        self.port = _free_port()
        path = "/mcp" if transport == "http" else "/sse"
        self.url = f"http://127.0.0.1:{self.port}{path}"
        self._proc = subprocess.Popen(  # noqa: S603 - launching our own test helper with sys.executable
            [sys.executable, HELPER, transport, str(self.port), tool_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


@pytest.fixture(scope="module")
def http_backend() -> Iterator[_Remote]:
    r = _Remote("http", "http_echo")
    try:
        asyncio.run(_wait_ready(r.url))
        yield r
    finally:
        r.stop()


@pytest.fixture(scope="module")
def sse_backend() -> Iterator[_Remote]:
    r = _Remote("sse", "sse_echo")
    try:
        asyncio.run(_wait_ready(r.url))
        yield r
    finally:
        r.stop()


def _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend) -> Estate:
    monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
    return Estate(
        [
            ServerSpec(name="local", command=sys.executable, args=[HELPER, "stdio", "0", "stdio_echo"]),
            ServerSpec(name="remote", transport="http", url=http_backend.url),
            ServerSpec(name="events", transport="sse", url=sse_backend.url),
        ]
    )


class TestMixedEstate:
    async def test_catalog_lists_all_three_transports(
        self, tmp_path, monkeypatch, http_backend, sse_backend
    ) -> None:
        estate = _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend)
        try:
            # Naming a server indexes it; then the whole-estate catalog names all three.
            for name in ("local", "remote", "events"):
                await estate.catalog(name)
            out = await estate.catalog()
            assert "stdio_echo" in out
            assert "http_echo" in out
            assert "sse_echo" in out
            assert "3 tools across 3 servers" in out
        finally:
            await estate.close()

    async def test_invoke_reaches_each_transport(
        self, tmp_path, monkeypatch, http_backend, sse_backend
    ) -> None:
        estate = _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend)
        try:
            for name, tool in (("local", "stdio_echo"), ("remote", "http_echo"), ("events", "sse_echo")):
                result = await estate.invoke_tool(name, tool, {"value": "hi"})
                assert f"{tool}:hi" in result.content[0].text
        finally:
            await estate.close()

    async def test_concurrent_first_calls_connect_once(
        self, tmp_path, monkeypatch, http_backend, sse_backend
    ) -> None:
        estate = _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend)
        backend = estate._backend("remote")
        calls = 0
        real = backend._make_transport

        def counting():
            nonlocal calls
            calls += 1
            return real()

        monkeypatch.setattr(backend, "_make_transport", counting)
        try:
            await asyncio.gather(
                estate.invoke_tool("remote", "http_echo", {"value": "a"}),
                estate.invoke_tool("remote", "http_echo", {"value": "b"}),
            )
            assert calls == 1  # the per-backend lock made the first start singular
        finally:
            await estate.close()

    async def test_stop_then_invoke_reconnects(
        self, tmp_path, monkeypatch, http_backend, sse_backend
    ) -> None:
        estate = _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend)
        backend = estate._backend("remote")
        built = 0
        real = backend._make_transport

        def counting():
            nonlocal built
            built += 1
            return real()

        monkeypatch.setattr(backend, "_make_transport", counting)
        try:
            await estate.invoke_tool("remote", "http_echo", {"value": "1"})
            await backend.stop()
            assert not backend.is_started
            result = await estate.invoke_tool("remote", "http_echo", {"value": "2"})
            assert "http_echo:2" in result.content[0].text
            assert built == 2  # a real reconnect, not a reused dead session
        finally:
            await estate.close()

    async def test_reload_reports_reconnect_for_remote_and_pid_for_stdio(
        self, tmp_path, monkeypatch, http_backend, sse_backend
    ) -> None:
        estate = _mixed_estate(tmp_path, monkeypatch, http_backend, sse_backend)
        try:
            http_msg = await estate.reload("remote")
            assert "remote" in http_msg
            assert "tools available" in http_msg
            assert "pid" not in http_msg  # a remote has no process to name

            sse_msg = await estate.reload("events")
            assert "events" in sse_msg
            assert "pid" not in sse_msg

            stdio_msg = await estate.reload("local")
            assert "pid" in stdio_msg  # a subprocess restart names the pid change
        finally:
            await estate.close()
