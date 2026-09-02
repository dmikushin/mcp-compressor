"""One compressed MCP surface over every server a client has configured.

The arrangement this replaces gave each backend its own compressor process, so a
client wrapping twenty servers declared eighty wrapper tools and had to be told,
somewhere in its prompt, which tools existed behind them. Whatever form that
announcement took, it was assembled from whichever backends had finished
connecting at that moment and it was paid for on every request.

Here the client declares four tools and is told nothing. It asks:

    catalog()                       names, grouped by server
    get_tool_schema(server, tool)   the real API of one tool
    invoke_tool(server, tool, ...)  call it
    reload(server)                  restart one backend

Each step costs what it costs and happens because a model asked for it, so
nothing about the estate is injected into a prompt and nothing about the estate
sits in front of a conversation where a change to it would invalidate the whole
prefix. The measured difference on a real estate: 598 tools are 2,977 tokens as
names and 154,769 as schemas.

Backends are started when a tool is actually invoked, never to answer `catalog`
— that reads the catalogs ``--lazy`` already wrote. A server nobody calls never
runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import psutil
from fastmcp import FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import FastMCPProxy, ProxyClient
from fastmcp.tools.tool import ToolResult
from loguru import logger

from . import catalog_cache as _catalog_cache
from .estate import ConfigError, ServerSpec, load_servers
from .tools import CompressedTools, ReloadableClientManager
from .types import CompressionLevel


def _own_child_pids() -> set[int]:
    """PIDs of our direct children.

    Backends are stdio subprocesses of this process, so a diff across a
    restart says which one died and which one replaced it. Other backends are
    untouched and cancel out of the diff.
    """
    try:
        return {child.pid for child in psutil.Process().children()}
    except Exception:  # pragma: no cover - psutil failing is not a reload error
        return set()


class UnknownServer(ValueError):
    """Names the servers that do exist, because a model that guessed a name
    needs the list more than it needs to be told it was wrong."""

    def __init__(self, name: str, known: list[str]) -> None:
        super().__init__(f"No server named {name!r}. Configured servers: {', '.join(known)}")


class EstateBackend:
    """One configured server, connected only if something calls it."""

    def __init__(self, spec: ServerSpec, compression_level: CompressionLevel) -> None:
        self.spec = spec
        self._compression_level = compression_level
        self._tools: CompressedTools | None = None
        self._manager: ReloadableClientManager | None = None
        self._transport: StdioTransport | None = None
        self._lock = asyncio.Lock()

    @property
    def cache_key(self) -> str:
        return _catalog_cache.make_cache_key(" ".join(self.spec.argv), server_name=self.spec.name)

    @property
    def is_started(self) -> bool:
        return self._tools is not None

    async def tools(self) -> CompressedTools:
        """The backend's compressed interface, building it on first use.

        The lock matters: two concurrent invocations of different tools on the
        same server would otherwise each start a subprocess, and the second
        would replace the first with no one closing it.
        """
        if self._tools is not None:
            return self._tools
        async with self._lock:
            if self._tools is not None:
                return self._tools
            logger.info(f"Starting backend {self.spec.name!r}: {' '.join(self.spec.argv)}")
            transport = StdioTransport(
                command=self.spec.command,
                args=self.spec.args,
                env=self.spec.env or None,
                cwd=self.spec.cwd,
            )

            async def connect() -> ProxyClient:
                client = ProxyClient(transport=transport, init_timeout=None)
                await client.__aenter__()
                return client

            manager = ReloadableClientManager(connect=connect)
            proxy = FastMCPProxy(client_factory=manager.get_client, name=f"compressor:{self.spec.name}")
            compressed = CompressedTools(
                proxy,
                compression_level=self._compression_level,
                server_name=self.spec.name,
                client_manager=manager,
                catalog_cache_key=self.cache_key,
                catalog_source=" ".join(self.spec.argv),
            )
            try:
                await compressed.configure_server()
            except BaseException:
                with contextlib.suppress(Exception):
                    await manager.stop()
                raise
            self._manager = manager
            self._tools = compressed
            self._transport = transport
            return compressed

    async def stop(self) -> None:
        if self._manager is not None:
            with contextlib.suppress(Exception):
                await self._manager.stop()
        # Closing the session is not enough. StdioTransport defaults to
        # keep_alive=True, which deliberately leaves the subprocess running
        # between connections, and a later connect() on the same transport
        # returns early because _connect_task is still set - so the process is
        # neither replaced nor reused-from-scratch. disconnect() is the only
        # thing that actually takes it down.
        if self._transport is not None:
            with contextlib.suppress(Exception):
                await self._transport.disconnect()
        self._manager = None
        self._tools = None
        self._transport = None

    async def restart(self) -> CompressedTools:
        """Take the backend down and bring it back up, for real.

        stop() alone is not enough to make the next call use new code: the
        estate is lazy, and a warm catalog cache lets tools() rebuild the
        wrapper without ever connecting to anything. Reload is asked for
        precisely when the code on disk has changed, so the connection is
        forced here instead of being deferred to whenever something next
        happens to call this server.

        Forcing the connection is still not enough on its own. tools() will
        have filled the wrapper from the disk cache and connecting does not
        replace what it filled, so the catalog would remain the one the old
        code wrote — a restart nobody could observe, on top of a stale file
        that would go on lying to the next start. refresh_catalog() is what
        asks the restarted backend what it actually has.
        """
        await self.stop()
        tools = await self.tools()
        await tools.refresh_catalog()
        return tools


class Estate:
    """Every configured server, and the four things a client can ask of them."""

    def __init__(
        self,
        specs: list[ServerSpec],
        compression_level: CompressionLevel = CompressionLevel.MAX,
        config_path: str | Path | None = None,
    ) -> None:
        self._backends = {s.name: EstateBackend(s, compression_level) for s in specs}
        self._compression_level = compression_level
        self._config_path = config_path

    @property
    def names(self) -> list[str]:
        return sorted(self._backends)

    def _backend(self, server: str) -> EstateBackend:
        try:
            return self._backends[server]
        except KeyError:
            raise UnknownServer(server, self.names) from None

    async def rescan(self) -> list[str]:
        """Bring the estate back in line with the configuration file.

        Adding a server to the configuration used to require restarting the
        client, because the estate was built once at startup and never looked
        at the file again. Nothing about a backend is expensive until it is
        called, so there is no reason to make anyone pay a restart to learn a
        name: a spec is a few strings, and the subprocess still waits for a
        deliberate first use.

        Backends already running are left strictly alone unless their spec
        actually changed, since restarting them would drop live connections and
        discard their catalogs for no gain.
        """
        if self._config_path is None:
            return []

        wanted = {s.name: s for s in load_servers(self._config_path)}
        changed: list[str] = []

        for name in sorted(set(self._backends) - set(wanted)):
            await self._backends.pop(name).stop()
            changed.append(f"removed {name}")

        for name, spec in wanted.items():
            current = self._backends.get(name)
            if current is None:
                self._backends[name] = EstateBackend(spec, self._compression_level)
                changed.append(f"added {name}")
            elif current.spec != spec:
                # The command itself changed, so a running child is now the
                # wrong program; drop it and let the next call start the right one.
                await current.stop()
                self._backends[name] = EstateBackend(spec, self._compression_level)
                changed.append(f"reconfigured {name}")

        if changed:
            logger.info(f"Rescan of {self._config_path}: {', '.join(changed)}")
        return changed

    async def catalog(self, server: str | None = None) -> str:
        """Tool names, grouped by server.

        Every call re-reads the configuration first. Discovery is exactly the
        moment a name that was added since startup has to become visible, and
        the read costs one small file, so there is nothing to be gained by
        making the caller ask for it separately.

        Asked for the whole estate, this starts nothing: it reads the catalogs
        ``--lazy`` already wrote. Servers with no catalog are named as such
        rather than omitted — for as long as that was silent, four broken
        servers on this machine were invisible to everyone, including their
        owner.

        Asked for ONE server that has never been indexed, it starts that server
        once and indexes it. Without this the estate deadlocks: a backend is
        started on first invocation, an invocation needs a tool name, and a tool
        name comes from a catalog the backend has not written yet. So a server
        added to the configuration could never become visible. Naming a server
        is a deliberate act, which is what makes it a safe place to spend a
        subprocess; asking for everything is not, and does not.
        """
        stale = ""
        try:
            await self.rescan()
        except ConfigError as exc:
            # A half-written config must not blind the estate to the servers it
            # already knows; say what happened and answer from what we have.
            stale = f"Configuration could not be re-read, so this list may be stale: {exc}"

        if server is not None and server not in self._backends:
            raise UnknownServer(server, self.names)
        wanted = [server] if server is not None else self.names

        if server is not None and _catalog_cache.load_entry(
            self._backends[server].cache_key
        ) is None:
            logger.info(f"Catalog requested for unindexed server {server!r}; indexing it.")
            try:
                await self._backends[server].tools()
            except Exception as exc:
                return (
                    f"{server} could not be indexed: {exc}\n"
                    "The server is configured but does not start, so its tools "
                    "are unknown. Fix the server, then ask for it again."
                )

        lines: list[str] = []
        unindexed: list[str] = []
        total = 0
        for name in wanted:
            entry = _catalog_cache.load_entry(self._backends[name].cache_key)
            if entry is None:
                unindexed.append(name)
                continue
            names = entry.tool_names
            total += len(names)
            lines.append(f"{name} ({len(names)}): {', '.join(names)}")

        out: list[str] = []
        if stale:
            out.append(stale)
        if lines:
            out.append(
                f"{total} tools across {len(lines)} servers. Names only — call "
                f"get_tool_schema(server, tool) for the one you intend to use, "
                f"then invoke_tool(server, tool, tool_input)."
            )
            out.extend(lines)
        if unindexed:
            out.append(
                "Configured but never indexed — ask for one by name to index it "
                f"(catalog with server=<name>): {', '.join(unindexed)}"
            )
        if not out:
            return "No servers configured."
        return "\n".join(out)

    async def get_tool_schema(self, server: str, tool: str) -> str:
        backend = self._backend(server)
        return await (await backend.tools()).get_tool_schema(tool)

    async def invoke_tool(
        self,
        server: str,
        tool: str,
        tool_input: str | dict[str, Any] | None = None,
    ) -> ToolResult:
        backend = self._backend(server)
        return await (await backend.tools()).invoke_tool(tool, tool_input)

    async def reload(self, server: str) -> str:
        """Restart one backend against the current configuration.

        The configuration is re-read first. Reload is asked for after something
        on disk changed, and the change is as likely to be the command in the
        configuration as the code it points at; restarting a backend against
        the spec this process happened to read at startup would relaunch the
        very program the caller is trying to stop using.

        A backend that was never started is started rather than refused. The
        estate is lazy, so under normal use almost nothing is running, and a
        reload that declines to act whenever it finds nothing running is a
        reload that declines to act nearly always — while the caller, who asked
        because something changed, is told a no-op was correct and moves on.
        """
        try:
            await self.rescan()
        except ConfigError as exc:
            # Reloading against a known-stale spec is worse than not reloading:
            # it would look like it worked.
            raise RuntimeError(
                f"Configuration could not be re-read, so {server!r} was not "
                f"reloaded: {exc}"
            ) from exc

        backend = self._backend(server)
        was_started = backend.is_started

        # Recycling the client session is not a restart: the transport keeps
        # the subprocess alive, so the old code stays in memory and the caller
        # is told it was reloaded. Tear the backend down and build it again,
        # then prove the process changed instead of asserting that it did.
        before = _own_child_pids()
        tools = await backend.restart()
        after = _own_child_pids()

        died, born = sorted(before - after), sorted(after - before)
        if not born:
            raise RuntimeError(
                f"Reload of {server!r} did not start a new backend process "
                f"(children before: {sorted(before)}, after: {sorted(after)}). "
                "The backend was left stopped."
            )
        catalog = await tools.get_backend_tools()
        verb = "Restarted" if was_started else "Started"
        transition = f"pid {died[0]} -> {born[0]}" if died else f"pid {born[0]}"
        return f"{verb} {server!r}: {transition}, {len(catalog)} tools available."

    async def close(self) -> None:
        for backend in self._backends.values():
            await backend.stop()


def build_estate_server(estate: Estate, name: str = "MCP Compressor Estate") -> FastMCP:
    """The four-tool MCP surface a client connects to."""
    mcp: FastMCP = FastMCP(name=name)

    @mcp.tool()
    async def catalog(server: str | None = None) -> str:
        """List the tool names of every configured MCP server, or of one server.

        Names only, no descriptions and no schemas. Start here: find the tool you
        need, then call get_tool_schema for its real API.

        The configuration is re-read on every call, so a server added to it
        since this process started is listed here without restarting anything.

        Asking for everything starts nothing. Asking for ONE server that has
        never been indexed starts it once, so that a server newly added to the
        configuration can become visible at all.
        """
        return await estate.catalog(server)

    @mcp.tool()
    async def get_tool_schema(server: str, tool: str) -> str:
        """Return the real input schema of one tool on one server.

        The name from `catalog` is a handle; this is the API behind it. Call this
        before invoking a tool you have not invoked before.
        """
        return await estate.get_tool_schema(server, tool)

    @mcp.tool()
    async def invoke_tool(
        server: str,
        tool: str,
        tool_input: str | dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a tool on a server, with arguments matching its real schema.

        `tool_input` is the tool's own argument object, as returned by
        get_tool_schema — not a flattened argument list.
        """
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)
        return await estate.invoke_tool(server, tool, tool_input)

    @mcp.tool()
    async def reload(server: str) -> str:
        """Restart one backend and refresh its tool catalog.

        For a server whose code changed on disk, or whose command changed in the
        configuration, or one that has stopped answering. The configuration is
        re-read first, and a backend that was not running is started, so this
        acts whatever state the estate happened to be in.

        The reply names the old and new process ids, because a restart that
        silently did not happen is the failure this tool is most prone to.
        """
        return await estate.reload(server)

    return mcp
