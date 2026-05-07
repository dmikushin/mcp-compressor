"""Tests for mcp_compressor/server.py — BackendPool and HTTP API."""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from mcp_compressor.server import (
    BackendInstance,
    BackendPool,
    _find_free_port,
    _make_backend_key,
    create_pool_app,
)
from mcp_compressor.types import CompressionLevel


class FakeTask:
    """A mock asyncio.Task that is awaitable and supports .cancel() / .done()."""
    def __init__(self):
        self._done = False
    def cancel(self):
        self._done = True
    def done(self):
        return self._done
    def __await__(self):
        async def _noop():
            pass
        yield from _noop().__await__()


class TestMakeBackendKey:
    def test_same_command_same_key(self):
        k1 = _make_backend_key("uvx", ["mcp-server-fetch"])
        k2 = _make_backend_key("uvx", ["mcp-server-fetch"])
        assert k1 == k2

    def test_different_command_different_key(self):
        k1 = _make_backend_key("uvx", ["mcp-a"])
        k2 = _make_backend_key("uvx", ["mcp-b"])
        assert k1 != k2

    def test_key_is_12_hex_chars(self):
        key = _make_backend_key("npx", ["-y", "@anthropic/mcp-github"])
        assert len(key) == 12
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_env_different_key(self):
        """Regression: backend key must include env (otherwise GITHUB_TOKEN=A and =B collide)."""
        k1 = _make_backend_key("uvx", ["mcp-x"], env={"TOKEN": "A"})
        k2 = _make_backend_key("uvx", ["mcp-x"], env={"TOKEN": "B"})
        assert k1 != k2

    def test_same_env_different_order_same_key(self):
        """Env dict ordering must not affect the key — sorted internally."""
        k1 = _make_backend_key("uvx", ["mcp-x"], env={"A": "1", "B": "2"})
        k2 = _make_backend_key("uvx", ["mcp-x"], env={"B": "2", "A": "1"})
        assert k1 == k2

    def test_different_cwd_different_key(self):
        """Regression: backend key must include cwd — same command in different cwd is a different backend."""
        k1 = _make_backend_key("uvx", ["mcp-x"], cwd="/tmp/proj-a")
        k2 = _make_backend_key("uvx", ["mcp-x"], cwd="/tmp/proj-b")
        assert k1 != k2

    def test_different_server_name_different_key(self):
        k1 = _make_backend_key("uvx", ["mcp-x"], server_name="foo")
        k2 = _make_backend_key("uvx", ["mcp-x"], server_name="bar")
        assert k1 != k2

    def test_no_env_vs_empty_env_same_key(self):
        """An empty env dict should hash the same as no env (falsy)."""
        k1 = _make_backend_key("uvx", ["mcp-x"])
        k2 = _make_backend_key("uvx", ["mcp-x"], env={})
        assert k1 == k2


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535


class TestBackendPool:
    @pytest.fixture
    def pool(self):
        return BackendPool(log_level="error")

    @pytest.fixture
    def mock_connect(self):
        """Return a mock ProxyClient that can be used as __aenter__/__aexit__."""
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    @pytest.fixture
    def mock_compressed_tools(self):
        tools = MagicMock()
        tools.configure_server = AsyncMock()
        tools.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 9,
            "compressed_tool_count": 9,
            "original_schema_size": 5000,
            "compressed_schema_sizes": {"low": 2500, "medium": 1500, "high": 600, "max": 450},
        })
        return tools

    def _mock_spawn_deps(self, mock_connect, mock_compressed_tools):
        """Patch out all heavy dependencies for spawn()."""
        return (
            patch("mcp_compressor.server.StdioTransport", return_value=MagicMock()),
            patch("mcp_compressor.server.ProxyClient", return_value=mock_connect),
            patch("mcp_compressor.server.FastMCPProxy", return_value=MagicMock()),
            patch("mcp_compressor.server.CompressedTools", return_value=mock_compressed_tools),
            patch("mcp_compressor.server.uvicorn.Config"),
            patch("mcp_compressor.server.uvicorn.Server", return_value=MagicMock()),
            patch("mcp_compressor.server.asyncio.create_task", return_value=FakeTask()),
            patch("mcp_compressor.server.asyncio.sleep", AsyncMock()),  # skip startup wait
        )

    @pytest.mark.asyncio
    async def test_spawn_creates_backend(self, pool, mock_connect, mock_compressed_tools):
        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            result = await pool.spawn("uvx", ["mcp-server-fetch"])

        assert "backend_key" in result
        assert "port" in result
        assert result["tool_count"] == 9

    @pytest.mark.asyncio
    async def test_spawn_same_command_returns_cached(self, pool, mock_connect, mock_compressed_tools):
        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            r1 = await pool.spawn("uvx", ["mcp-server-fetch"])
            r2 = await pool.spawn("uvx", ["mcp-server-fetch"])

        assert r1["backend_key"] == r2["backend_key"]
        assert r1["port"] == r2["port"]

    @pytest.mark.asyncio
    async def test_spawn_different_commands_different_key(self, pool, mock_connect, mock_compressed_tools):
        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            r1 = await pool.spawn("uvx", ["mcp-a"])
            r2 = await pool.spawn("uvx", ["mcp-b"])

        assert r1["backend_key"] != r2["backend_key"]

    @pytest.mark.asyncio
    async def test_spawn_multiple_cached_only_one_client_start(self, pool, mock_connect, mock_compressed_tools):
        """First spawn creates; second reuses cached — ReloadableClientManager.start called once."""
        start_count = 0
        orig_start = MagicMock()

        async def tracked_start(self):
            nonlocal start_count
            start_count += 1

        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        # Override ReloadableClientManager.start to track calls
        manager_patch = patch(
            "mcp_compressor.server.ReloadableClientManager.start",
            new=tracked_start,
        )
        patches = list(patches) + [manager_patch]

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            await pool.spawn("uvx", ["mcp-a"])
            await pool.spawn("uvx", ["mcp-a"])

        assert start_count == 1

    @pytest.mark.asyncio
    async def test_close_stops_all_backends(self, pool, mock_connect, mock_compressed_tools):
        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            await pool.spawn("uvx", ["mcp-a"])
            await pool.spawn("uvx", ["mcp-b"])

        await pool.close()
        # After close, spawn should create fresh
        assert len(pool._backends) == 0

    @pytest.mark.asyncio
    async def test_compression_level_passed(self, pool, mock_connect, mock_compressed_tools):
        patches = list(self._mock_spawn_deps(mock_connect, mock_compressed_tools))
        # Capture CompressedTools call args — need an awaitable configure_server on the instance
        mock_ct_instance = MagicMock()
        mock_ct_instance.configure_server = AsyncMock()
        mock_ct_instance.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 9,
            "original_schema_size": 5000,
            "compressed_schema_sizes": {"high": 600},
        })
        ct_patch = patch("mcp_compressor.server.CompressedTools", return_value=mock_ct_instance)
        patches = [p for p in patches if p.attribute != "CompressedTools"] + [ct_patch]

        with contextlib.ExitStack() as stack:
            ctx_mocks = {p.attribute: stack.enter_context(p) for p in patches}
            mock_ct_class = ctx_mocks.get("CompressedTools")

            await pool.spawn("uvx", ["mcp-a"], compression_level=CompressionLevel.HIGH)

        # CompressedTools was called with compression_level=CompressionLevel.HIGH
        call_kwargs = mock_ct_class.call_args[1]
        assert call_kwargs["compression_level"] == CompressionLevel.HIGH

    @pytest.mark.asyncio
    async def test_spawn_failure_does_not_register_backend(self, pool, mock_connect):
        """Regression: if configure_server() fails, the partial backend must not be left in the pool."""
        failing_tools = MagicMock()
        failing_tools.configure_server = AsyncMock(side_effect=RuntimeError("boom"))

        patches = self._mock_spawn_deps(mock_connect, failing_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            with pytest.raises(RuntimeError, match="boom"):
                await pool.spawn("uvx", ["mcp-broken"])

        # Pool must be empty — a retry should be possible.
        assert len(pool._backends) == 0

    @pytest.mark.asyncio
    async def test_spawn_stats_failure_evicts_backend(self, pool, mock_connect):
        """Regression: if get_compression_stats() fails (outside the lock), the backend is evicted."""
        flaky_tools = MagicMock()
        flaky_tools.configure_server = AsyncMock()
        flaky_tools.get_compression_stats = AsyncMock(side_effect=RuntimeError("stats failed"))

        patches = self._mock_spawn_deps(mock_connect, flaky_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            with pytest.raises(RuntimeError, match="stats failed"):
                await pool.spawn("uvx", ["mcp-x"])

        assert len(pool._backends) == 0

    @pytest.mark.asyncio
    async def test_spawn_startup_timeout_raises_no_orphan(self, pool, mock_connect, mock_compressed_tools):
        """Regression: if uvicorn never reports started, raise — do not register a dead backend."""
        patches = list(self._mock_spawn_deps(mock_connect, mock_compressed_tools))
        # Override uvicorn.Server with one whose .started stays False.
        dead_server = MagicMock()
        dead_server.started = False
        # Override the create_task too: serve_task must not be done() (otherwise we'd re-raise its result)
        dead_task = FakeTask()
        # Strip prior uvicorn.Server / create_task patches and inject our own.
        patches = [
            p for p in patches
            if p.attribute not in ("Server", "create_task")
        ]
        patches.append(patch("mcp_compressor.server.uvicorn.Server", return_value=dead_server))
        patches.append(patch("mcp_compressor.server.asyncio.create_task", return_value=dead_task))

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            with pytest.raises(RuntimeError, match="failed to start"):
                await pool.spawn("uvx", ["mcp-stuck"])

        assert len(pool._backends) == 0

    @pytest.mark.asyncio
    async def test_spawn_different_cwd_different_backend(self, pool, mock_connect, mock_compressed_tools):
        """Regression: cwd is part of the backend identity — same command in different cwd → different backend."""
        patches = self._mock_spawn_deps(mock_connect, mock_compressed_tools)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            r1 = await pool.spawn("uvx", ["mcp-x"], cwd="/tmp/a")
            r2 = await pool.spawn("uvx", ["mcp-x"], cwd="/tmp/b")

        assert r1["backend_key"] != r2["backend_key"]
        assert len(pool._backends) == 2


class TestCreatePoolApp:
    @pytest.fixture
    def app(self):
        pool = BackendPool(log_level="error")
        return create_pool_app(pool)

    def test_health_returns_ok(self, app):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_spawn_requires_command(self, app):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/spawn", json={})
        assert resp.status_code == 400

    def test_spawn_creates_backend(self, app):
        """End-to-end spawn via HTTP — mocks spawn internals."""
        mock_tools = MagicMock()
        mock_tools.configure_server = AsyncMock()
        mock_tools.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 5,
            "compressed_tool_count": 5,
            "original_schema_size": 1000,
            "compressed_schema_sizes": {"low": 500, "medium": 300, "high": 100, "max": 50},
        })

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_connect)
        mock_connect.__aexit__ = AsyncMock(return_value=None)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("mcp_compressor.server.StdioTransport", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.ProxyClient", return_value=mock_connect))
            stack.enter_context(patch("mcp_compressor.server.FastMCPProxy", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.CompressedTools", return_value=mock_tools))
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Config"))
            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.started = True
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Server", return_value=mock_uvicorn_server))
            stack.enter_context(patch("mcp_compressor.server.asyncio.create_task", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.asyncio.sleep", AsyncMock()))

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/spawn", json={"command": "uvx", "args": ["mcp-fetch"]})

        assert resp.status_code == 200
        data = resp.json()
        assert "backend_key" in data
        assert "port" in data
        assert data["tool_count"] == 5

    def test_spawn_with_env(self, app):
        """Spawn with environment variables."""
        mock_tools = MagicMock()
        mock_tools.configure_server = AsyncMock()
        mock_tools.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 1,
            "original_schema_size": 100,
            "compressed_schema_sizes": {"max": 10},
        })

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_connect)
        mock_connect.__aexit__ = AsyncMock(return_value=None)

        with contextlib.ExitStack() as stack:
            mock_transport = stack.enter_context(patch("mcp_compressor.server.StdioTransport"))
            stack.enter_context(patch("mcp_compressor.server.ProxyClient", return_value=mock_connect))
            stack.enter_context(patch("mcp_compressor.server.FastMCPProxy", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.CompressedTools", return_value=mock_tools))
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Config"))
            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.started = True
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Server", return_value=mock_uvicorn_server))
            stack.enter_context(patch("mcp_compressor.server.asyncio.create_task", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.asyncio.sleep", AsyncMock()))

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/spawn", json={
                "command": "uvx", "args": ["mcp-fetch"],
                "env": {"NODE_ENV": "production"},
            })

        assert resp.status_code == 200
        # Verify env was passed to StdioTransport
        call_kwargs = mock_transport.call_args[1]
        assert call_kwargs["env"] == {"NODE_ENV": "production"}

    def test_spawn_with_server_name(self, app):
        """Spawn with custom server name."""
        mock_tools = MagicMock()
        mock_tools.configure_server = AsyncMock()
        mock_tools.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 1,
            "original_schema_size": 100,
            "compressed_schema_sizes": {"medium": 30},
        })

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_connect)
        mock_connect.__aexit__ = AsyncMock(return_value=None)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("mcp_compressor.server.StdioTransport", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.ProxyClient", return_value=mock_connect))
            stack.enter_context(patch("mcp_compressor.server.FastMCPProxy", return_value=MagicMock()))
            ct_patch = stack.enter_context(patch("mcp_compressor.server.CompressedTools", return_value=mock_tools))
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Config"))
            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.started = True
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Server", return_value=mock_uvicorn_server))
            stack.enter_context(patch("mcp_compressor.server.asyncio.create_task", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.asyncio.sleep", AsyncMock()))

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/spawn", json={
                "command": "uvx", "args": ["mcp-fetch"],
                "server_name": "my-server",
            })

        assert resp.status_code == 200
        # Verify server_name was passed to CompressedTools
        call_kwargs = ct_patch.call_args[1]
        assert call_kwargs["server_name"] == "my-server"

    def test_spawn_invalid_compression_falls_back(self, app):
        """Invalid compression level falls back to MEDIUM."""
        mock_tools = MagicMock()
        mock_tools.configure_server = AsyncMock()
        mock_tools.get_compression_stats = AsyncMock(return_value={
            "original_tool_count": 1,
            "original_schema_size": 100,
            "compressed_schema_sizes": {"medium": 30},
        })

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_connect)
        mock_connect.__aexit__ = AsyncMock(return_value=None)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("mcp_compressor.server.StdioTransport", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.ProxyClient", return_value=mock_connect))
            stack.enter_context(patch("mcp_compressor.server.FastMCPProxy", return_value=MagicMock()))
            ct_patch = stack.enter_context(patch("mcp_compressor.server.CompressedTools", return_value=mock_tools))
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Config"))
            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.started = True
            stack.enter_context(patch("mcp_compressor.server.uvicorn.Server", return_value=mock_uvicorn_server))
            stack.enter_context(patch("mcp_compressor.server.asyncio.create_task", return_value=MagicMock()))
            stack.enter_context(patch("mcp_compressor.server.asyncio.sleep", AsyncMock()))

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/spawn", json={
                "command": "uvx", "args": ["mcp-fetch"],
                "compression_level": "garbage",
            })

        assert resp.status_code == 200
        call_kwargs = ct_patch.call_args[1]
        assert call_kwargs["compression_level"] == CompressionLevel.MEDIUM
