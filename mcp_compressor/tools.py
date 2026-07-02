"""Tool compression helpers built on FastMCP v3 transforms.

This module provides a transform-first implementation that replaces the visible tool
catalog with a compressed wrapper interface while keeping backend tools available for
passthrough access.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import difflib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import toons
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.exceptions import ToolError
from fastmcp.resources import Resource
from fastmcp.server.context import Context
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers.proxy import ProxyClient
from fastmcp.server.transforms import GetResourceNext, GetToolNext
from fastmcp.server.transforms.catalog import CatalogTransform
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from loguru import logger
from mcp.types import CallToolRequestParams, ContentBlock, TextContent
from pydantic import ValidationError

from . import catalog_cache as _catalog_cache
from .cli_script import find_script_dir
from .cli_tools import build_help_tool_description
from .types import CompressionLevel

# Minimum output length before quiet mode truncation applies
QUIET_MODE_THRESHOLD = 1000


@dataclasses.dataclass
class CachedTool:
    """Lightweight tool metadata stub used in lazy-loading mode.

    Holds only the fields needed for list/schema operations (``name``,
    ``description``, ``parameters``).  When the backend is finally connected the
    stub is replaced by a real ``ProxyTool`` that can be invoked.
    """

    name: str
    description: str | None
    parameters: dict[str, Any]

    @classmethod
    def from_mcp_dict(cls, data: dict[str, Any]) -> CachedTool:
        """Reconstruct a stub from the JSON dict produced by ``tool.to_mcp_tool().model_dump()``."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            parameters=data.get("inputSchema", {}),
        )


class ToolNotFoundError(ValueError):
    """Exception raised when a requested tool is not found in the backend MCP server."""

    def __init__(self, tool_name: str, available_tools: Sequence[str]) -> None:
        self.tool_name = tool_name
        self.available_tools = tuple(available_tools)
        suggestions = difflib.get_close_matches(tool_name, self.available_tools, n=3, cutoff=0.4)
        parts = [f"Tool '{tool_name}' not found."]
        if suggestions:
            parts.append(f"Did you mean: {', '.join(suggestions)}?")
        available_tools_text = ", ".join(self.available_tools) if self.available_tools else "(none)"
        parts.append(f"Available tools: {available_tools_text}")
        super().__init__(" ".join(parts))


class ReloadableClientManager:
    """Manages a reloadable MCP ProxyClient session.

    The wrapped MCP server can be restarted in-place (e.g. to pick up a new backend
    subprocess, or to recover from a hung connection) without restarting mcp-compressor
    itself.  Exposes a ``get_client`` callable suitable for use as the FastMCPProxy
    ``client_factory``.
    """

    def __init__(
        self,
        connect: Callable[[], Awaitable[Client]],
    ) -> None:
        """Create a manager.

        Args:
            connect: An async callable that creates a *connected* Client (i.e. already
                inside its ``async with`` context).  The manager takes ownership of its
                lifecycle; it will call ``__aexit__`` on the returned client on stop
                and reload.  This callable is invoked once at start and again on each
                reload.
        """
        self._connect = connect
        self._current_client: Client | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Connect the initial client session."""
        if self._current_client is not None:
            return
        self._current_client = await self._connect()

    async def stop(self) -> None:
        """Close the active client session, if any."""
        async with self._lock:
            await self._close_current()

    async def reload(self) -> None:
        """Close the current session and establish a new one."""
        async with self._lock:
            logger.info("Reloading wrapped MCP server backend")
            await self._close_current()
            self._current_client = await self._connect()
            logger.info("Wrapped MCP server backend reloaded")

    async def _close_current(self) -> None:
        if self._current_client is None:
            return
        client = self._current_client
        self._current_client = None
        with contextlib.suppress(Exception):
            await client.__aexit__(None, None, None)

    @property
    def is_connected(self) -> bool:
        """Return whether a live backend session exists."""
        return self._current_client is not None

    async def ensure_connected(self) -> None:
        """Connect if not already connected (used by lazy-loading mode)."""
        if self._current_client is not None:
            return
        async with self._lock:
            if self._current_client is None:
                logger.info("Lazy-connecting wrapped MCP server backend on first use")
                self._current_client = await self._connect()

    def get_client(self) -> Client:
        """Return the currently active client (factory for FastMCPProxy)."""
        if self._current_client is None:
            raise RuntimeError("ReloadableClientManager is not started")
        return self._current_client


class InvokeToolCompatibilityMiddleware(Middleware):
    """Small compatibility shim for flattened invoke_tool arguments and direct toonify."""

    def __init__(self, compressed_tools: CompressedTools) -> None:
        self._compressed_tools = compressed_tools

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        tool_args = context.message.arguments or {}
        if (
            tool_name in self._compressed_tools.invoke_tool_names
            and tool_args
            and ("tool_input" not in tool_args or tool_args["tool_input"] is None)
        ):
            flat_input = {k: v for k, v in tool_args.items() if k not in {"tool_name", "quiet"}}
            if flat_input and "tool_name" in tool_args:
                return await self._compressed_tools.invoke_tool(
                    tool_name=tool_args["tool_name"],
                    tool_input=flat_input,
                    quiet=tool_args.get("quiet", False),
                )

        result = await call_next(context)
        if self._compressed_tools.should_toonify_tool(tool_name):
            return self._compressed_tools._toonify_tool_result(result)
        return result


class CompressedTools(CatalogTransform):
    """Transform that replaces the tool catalog with compressed wrapper tools.

    In normal mode it exposes two or three public wrapper tools:
    - get_tool_schema: Retrieves the full schema for a specific tool
    - invoke_tool: Executes a tool with the provided arguments
    - list_tools: (optional) Lists all available tools with brief descriptions (only if compression level is MAX)

    It also exposes a resource (``compressor://uncompressed-tools``) that returns the upstream server's original
    list_tools payload in machine-readable JSON form.

    In CLI mode it exposes a single help tool (<server_name>_help) instead of the wrapper tool catalog.
    """

    def __init__(
        self,
        proxy_server: FastMCP,
        compression_level: CompressionLevel,
        server_name: str | None = None,
        toonify: bool = False,
        cli_mode: bool = False,
        cli_name: str | None = None,
        include_tools: Sequence[str] | None = None,
        exclude_tools: Sequence[str] | None = None,
        client_manager: ReloadableClientManager | None = None,
        catalog_cache_key: str | None = None,
    ) -> None:
        super().__init__()
        self._proxy_server = proxy_server
        self._compression_level = compression_level
        self._tool_name_prefix = f"{server_name}_" if server_name else ""
        self._server_description = f"the {server_name} toolset" if server_name else "this toolset"
        self._toonify = toonify
        self._cli_mode = cli_mode
        self._cli_name = cli_name or (server_name or "mcp")
        self._include_tools = set(include_tools or [])
        self._exclude_tools = set(exclude_tools or [])
        self._client_manager = client_manager
        self._catalog_cache_key = catalog_cache_key
        self._cached_backend_tools: dict[str, Tool | CachedTool] | None = None
        self._tool_cache_lock: asyncio.Lock = asyncio.Lock()
        self._help_tool_name = sanitize_tool_name(f"{server_name}_help" if server_name else "help")
        self._get_schema_tool_name = sanitize_tool_name(f"{self._tool_name_prefix}get_tool_schema")
        self._invoke_tool_name = sanitize_tool_name(f"{self._tool_name_prefix}invoke_tool")
        self._invoke_tool_alias_name = sanitize_tool_name("invoke_tool")
        self._list_tools_name = sanitize_tool_name(f"{self._tool_name_prefix}list_tools")
        self._reload_tool_name = sanitize_tool_name(f"{self._tool_name_prefix}reload")
        self._uncompressed_tools_resource_uri = "compressor://uncompressed-tools"

    @property
    def invoke_tool_names(self) -> set[str]:
        """All invoke_tool wrapper names, including the hidden alias."""
        return {self._invoke_tool_name, self._invoke_tool_alias_name}

    def should_toonify_tool(self, tool_name: str) -> bool:
        """Return whether direct calls to a tool should be toonified."""
        if not self._toonify:
            return False
        return tool_name not in self._wrapper_tool_names()

    def _wrapper_tool_names(self) -> set[str]:
        if self._cli_mode:
            tool_names = {self._help_tool_name}
        else:
            tool_names = {self._get_schema_tool_name, self._invoke_tool_name, self._invoke_tool_alias_name}
            if self._compression_level == CompressionLevel.MAX:
                tool_names.add(self._list_tools_name)
        if self._client_manager is not None:
            tool_names.add(self._reload_tool_name)
        return tool_names

    async def configure_server(self) -> None:
        """Attach the transform and any small compatibility middleware to the server."""
        await self._configure_backend_tool_visibility()
        self._proxy_server.add_transform(self)
        if not self._cli_mode:
            # Surface the anti-bypass guardrail once via server instructions
            # instead of repeating it in every wrapper tool description. Preserve
            # any backend-provided instructions by appending them.
            base = self._wrapper_instructions()
            existing = getattr(self._proxy_server, "instructions", None)
            self._proxy_server.instructions = f"{base}\n\n{existing}" if existing else base
            self._proxy_server.add_middleware(InvokeToolCompatibilityMiddleware(self))

    async def _configure_backend_tool_visibility(self) -> None:
        """Populate the tool cache, connecting to the backend or loading from disk cache.

        In lazy mode (client_manager not connected + catalog_cache_key provided):
        - If a disk cache exists: populate ``_cached_backend_tools`` with ``CachedTool``
          stubs and skip the backend connection entirely.
        - If no disk cache: fall through to the eager path which connects, fetches the
          catalog, and saves it to disk for next time.

        In normal (eager) mode: connect immediately and fetch the catalog from the backend.
        """
        lazy = (
            self._catalog_cache_key is not None
            and self._client_manager is not None
            and not self._client_manager.is_connected
        )
        if lazy:
            cached = _catalog_cache.load(self._catalog_cache_key)  # type: ignore[arg-type]
            if cached is not None:
                logger.info(
                    f"Lazy mode: loaded {len(cached)} tool(s) from disk cache "
                    f"(key={self._catalog_cache_key!r}); backend not started."
                )
                self._cached_backend_tools = {
                    entry["name"]: CachedTool.from_mcp_dict(entry) for entry in cached
                }
                return
            # Cache miss in lazy mode → fall through and connect eagerly this time
            logger.info("Lazy mode: no disk cache found; connecting backend to build initial catalog.")
            await self._client_manager.ensure_connected()  # type: ignore[union-attr]

        all_tools = await self._proxy_server.list_tools(run_middleware=False)
        filters_applied = False
        if self._include_tools:
            all_tool_names = {tool.name for tool in all_tools}
            names_to_disable = all_tool_names - self._include_tools
            if names_to_disable:
                self._proxy_server.disable(names=names_to_disable, components={"tool"})
                filters_applied = True
        if self._exclude_tools:
            self._proxy_server.disable(names=self._exclude_tools, components={"tool"})
            filters_applied = True
        # Warm the tool cache after visibility rules are applied so the cache
        # reflects the filtered tool set that clients will actually see.
        # Re-fetch only when filters changed the visible set; otherwise reuse the
        # list we already have (avoids a redundant backend round-trip).
        if filters_applied:
            visible_tools = await self._proxy_server.list_tools(run_middleware=False)
        else:
            visible_tools = all_tools
        self._cached_backend_tools = {tool.name: tool for tool in visible_tools}
        # Persist the catalog to disk if a cache key is configured.
        if self._catalog_cache_key:
            self._save_catalog_cache(list(visible_tools))

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        """Replace the visible tool catalog with compressed wrapper tools.

        Prefers ``_cached_backend_tools`` over the ``tools`` parameter passed by FastMCP.
        This allows lazy mode to serve the cached catalog even when the backend is not
        yet connected (in which case ``tools`` would be empty).
        """
        effective = (
            list(self._cached_backend_tools.values())
            if self._cached_backend_tools is not None
            else list(tools)
        )
        if self._cli_mode:
            visible_tools = [self._make_help_tool(await self._build_cli_description_from(effective))]
        else:
            visible_tools = [
                self._make_get_schema_tool(await self._get_tool_descriptions_from(effective, self._compression_level)),
                self._make_invoke_tool(self._invoke_tool_name),
            ]
            if self._compression_level == CompressionLevel.MAX:
                visible_tools.append(self._make_list_tools_tool())
        if self._client_manager is not None:
            visible_tools.append(self._make_reload_tool())
        return visible_tools

    async def get_tool(self, name: str, call_next: GetToolNext, *, version: Any | None = None) -> Tool | None:
        """Return synthetic wrapper tools and delegate backend tool lookups unchanged."""
        if self._cli_mode and name == self._help_tool_name:
            return self._make_help_tool()
        if name == self._get_schema_tool_name:
            return self._make_get_schema_tool()
        if name in self.invoke_tool_names:
            return self._make_invoke_tool(name)
        if name == self._list_tools_name and self._compression_level == CompressionLevel.MAX:
            return self._make_list_tools_tool()
        if self._client_manager is not None and name == self._reload_tool_name:
            return self._make_reload_tool()
        return await call_next(name, version=version)

    async def transform_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]:
        """Append the synthetic uncompressed-tools resource in normal mode."""
        if self._cli_mode:
            return resources
        return [*resources, self._make_uncompressed_tools_resource()]

    async def get_resource(
        self, uri: str, call_next: GetResourceNext, *, version: Any | None = None
    ) -> Resource | None:
        """Return the synthetic resource when requested, else delegate."""
        if not self._cli_mode and uri == self._uncompressed_tools_resource_uri:
            return self._make_uncompressed_tools_resource()
        return await call_next(uri, version=version)

    async def list_tools_tool(self, ctx: Context = None) -> str:  # type: ignore[assignment]
        """List all available tools in {server_description}."""
        if ctx is None:
            async with Context(fastmcp=self._proxy_server) as active_ctx:
                return await self.list_tools_tool(active_ctx)
        backend_tools = await self._get_backend_tools(ctx)
        return await self._get_tool_descriptions_from(list(backend_tools.values()), CompressionLevel.MEDIUM)

    async def reload_backend(self) -> str:
        """Reload the wrapped MCP server by reconnecting its backend session.

        Closes the current backend connection (which for stdio servers terminates the
        subprocess) and establishes a fresh one, then re-fetches the tool catalog so
        that subsequent calls see the restarted backend.
        """
        if self._client_manager is None:
            raise ToolError("This compressor instance is not configured with a reloadable client manager.")
        await self._client_manager.reload()
        self.invalidate_tool_cache()
        try:
            backend_tools = await self.get_backend_tools()
        except Exception as exc:
            raise ToolError(f"Backend reloaded but tool catalog refresh failed: {exc}") from exc
        await self._configure_backend_tool_visibility_post_reload()
        return (
            f"Reloaded wrapped MCP server for {self._server_description}. "
            f"{len(backend_tools)} tools available."
        )

    async def _configure_backend_tool_visibility_post_reload(self) -> None:
        """Re-apply include/exclude visibility filters after a reload and update the disk cache."""
        # Use get_tool_catalog() to bypass the CatalogTransform that is already
        # attached to the proxy server.  list_tools(run_middleware=False) only
        # skips middleware, NOT transforms — so after the first configure_server()
        # call it would return the compressed wrapper tools instead of the raw
        # backend catalog.
        async with Context(fastmcp=self._proxy_server) as ctx:
            all_tools = list(await self.get_tool_catalog(ctx, run_middleware=False))
        if self._include_tools:
            all_tool_names = {tool.name for tool in all_tools}
            names_to_disable = all_tool_names - self._include_tools
            if names_to_disable:
                self._proxy_server.disable(names=names_to_disable, components={"tool"})
        if self._exclude_tools:
            self._proxy_server.disable(names=self._exclude_tools, components={"tool"})
        # Re-warm cache with filtered tool set
        if self._include_tools or self._exclude_tools:
            async with Context(fastmcp=self._proxy_server) as ctx:
                visible_tools = list(await self.get_tool_catalog(ctx, run_middleware=False))
        else:
            visible_tools = all_tools
        self._cached_backend_tools = {tool.name: tool for tool in visible_tools}
        if self._catalog_cache_key:
            self._save_catalog_cache(list(visible_tools))

    async def _lazy_connect_and_populate(self) -> None:
        """Connect the backend on first use and replace CachedTool stubs with real ProxyTools.

        Called transparently from ``invoke_tool`` when the backend has not been started yet
        (lazy mode with a cache hit at startup).  Uses ``_tool_cache_lock`` to prevent
        multiple concurrent invocations from all trying to connect simultaneously.
        """
        if self._client_manager is None:
            return
        async with self._tool_cache_lock:
            # Check again inside the lock — another coroutine may have connected already.
            if self._cached_backend_tools and not any(
                isinstance(t, CachedTool) for t in self._cached_backend_tools.values()
            ):
                return
            logger.info("Lazy mode: first invoke_tool — connecting backend and populating real tool catalog.")
            await self._client_manager.ensure_connected()
            await self._configure_backend_tool_visibility_post_reload()

    def _save_catalog_cache(self, tools: list[Tool]) -> None:
        """Persist the tool catalog to disk for future lazy-loading starts."""
        try:
            data = [t.to_mcp_tool().model_dump(mode="json") for t in tools]
            _catalog_cache.save(self._catalog_cache_key, data)  # type: ignore[arg-type]
            logger.debug(f"Saved {len(data)} tool(s) to catalog cache (key={self._catalog_cache_key!r})")
        except Exception as exc:
            logger.warning(f"Failed to save catalog cache: {exc}")

    async def get_tool_schema(self, tool_name: str, ctx: Context = None) -> str:  # type: ignore[assignment]
        """Get the input schema for a specific tool from {server_description}."""
        if ctx is None:
            async with Context(fastmcp=self._proxy_server) as active_ctx:
                return await self.get_tool_schema(tool_name, active_ctx)
        tool = await self._get_backend_tool(ctx, tool_name)
        tool_description = self._format_tool_description(tool, CompressionLevel.LOW)
        return tool_description + "\n\n" + json.dumps(tool.parameters, indent=2)

    async def invoke_tool(
        self,
        tool_name: str,
        tool_input: str | dict[str, Any] | None = None,
        quiet: bool = False,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> ToolResult:
        """Invoke a backend tool from the compressed catalog."""
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)
        if ctx is None:
            async with Context(fastmcp=self._proxy_server) as active_ctx:
                return await self.invoke_tool(tool_name, tool_input, quiet, active_ctx)
        tool = await self._get_backend_tool(ctx, tool_name)
        # Lazy mode: if the cached tool is a stub (no .run()), connect backend now.
        if isinstance(tool, CachedTool):
            await self._lazy_connect_and_populate()
            tool = await self._get_backend_tool(ctx, tool_name)
        if tool_input:
            tool_input = self._autocorrect_param_names(tool, tool_input)
            tool_input = self._autocorrect_enum_values(tool, tool_input)
        try:
            tool_result = await tool.run(tool_input or {})
        except ValidationError as exc:
            raise ToolError(
                await self._format_validation_error(ctx, tool, tool_input, str(exc))
            ) from exc
        except ToolError as exc:
            if self._is_validation_error_message(str(exc)):
                raise ToolError(
                    await self._format_validation_error(ctx, tool, tool_input, str(exc))
                ) from exc
            raise
        if self._toonify:
            tool_result = self._toonify_tool_result(tool_result)
        if not quiet:
            return tool_result
        if len(tool_result.content) == 1 and isinstance(tool_result.content[0], TextContent):
            return_text = tool_result.content[0].text
            if len(return_text) < QUIET_MODE_THRESHOLD:
                return tool_result
            preview_length = QUIET_MODE_THRESHOLD // 2
            return_text = (
                return_text[:preview_length]
                + "\n...\n(truncated due to quiet mode)\n...\n"
                + return_text[-preview_length:]
            )
        else:
            return_text = f"Successfully executed tool '{tool.name}' without output."
        return ToolResult(content=[TextContent(type="text", text=return_text)])

    async def list_uncompressed_tools(self, ctx: Context = None) -> str:  # type: ignore[assignment]
        """Return the upstream server's original list_tools payload as JSON."""
        if ctx is None:
            async with Context(fastmcp=self._proxy_server) as active_ctx:
                return await self.list_uncompressed_tools(active_ctx)
        backend_tools = await self._get_backend_tools(ctx)
        return json.dumps(
            [
                tool.to_mcp_tool().model_dump(mode="json")  # type: ignore[union-attr]
                if not isinstance(tool, CachedTool)
                else {"name": tool.name, "description": tool.description, "inputSchema": tool.parameters}
                for tool in backend_tools.values()
            ],
            indent=2,
        )

    async def get_backend_tools(self) -> dict[str, Tool]:
        """Return the current backend tool catalog keyed by name."""
        async with Context(fastmcp=self._proxy_server) as ctx:
            return await self._get_backend_tools(ctx)

    async def get_compression_stats(self) -> dict[str, Any]:
        """Get statistics about the compression of tool descriptions."""
        backend_tools = await self.get_backend_tools()
        original_tool_count = len(backend_tools)
        original_schema_size = sum(
            len(json.dumps(tool.parameters)) + len(json.dumps(getattr(tool, 'output_schema', None))) + len(tool.description or "")
            for tool in backend_tools.values()
        )
        compressed_schema_sizes: dict[CompressionLevel | str, int] = {}
        for compression_level in [
            CompressionLevel.LOW,
            CompressionLevel.MEDIUM,
            CompressionLevel.HIGH,
            CompressionLevel.MAX,
        ]:
            compressed_schema_sizes[compression_level] = sum(
                len(self._format_tool_description(tool, compression_level)) for tool in backend_tools.values()
            )
        compressed_schema_sizes["cli"] = len(await self._build_cli_description())
        return {
            "original_tool_count": original_tool_count,
            "compressed_tool_count": original_tool_count,
            "original_schema_size": original_schema_size,
            "compressed_schema_sizes": compressed_schema_sizes,
        }

    async def _build_cli_description(self) -> str:
        """Build the full help description for CLI mode."""
        backend_tools = await self.get_backend_tools()
        return await self._build_cli_description_from(list(backend_tools.values()))

    async def _build_cli_description_from(self, tools: Sequence[Tool]) -> str:
        _, on_path = find_script_dir()
        return build_help_tool_description(self._cli_name, self._server_description, list(tools), on_path=on_path)

    async def _get_tool_descriptions_from(self, tools: Sequence[Tool], compression_level: CompressionLevel) -> str:
        """Generate formatted tool descriptions for a set of tools."""
        if compression_level == CompressionLevel.MAX:
            return ""
        return "\n".join(self._format_tool_description(tool, compression_level) for tool in tools)

    async def _get_backend_tools(self, ctx: Context) -> dict[str, Tool]:
        """Retrieve backend tools from cache, fetching from backend on first call.

        The tool catalog is cached on first access (normally at startup via ``configure_server()``) so subsequent
        operations — invoke_tool, get_tool_schema, list_uncompressed_tools, etc. — do not make a live backend call every
        time.  Use ``invalidate_tool_cache()`` to force a refresh if the backend tool catalog changes at runtime.
        """
        if self._cached_backend_tools is not None:
            return self._cached_backend_tools
        async with self._tool_cache_lock:
            # Double-checked locking: another coroutine may have filled the cache
            # while we waited for the lock.
            if self._cached_backend_tools is not None:
                return self._cached_backend_tools
            logger.debug("Tool cache is empty; fetching backend tool catalog.")
            self._cached_backend_tools = {
                tool.name: tool for tool in await self.get_tool_catalog(ctx, run_middleware=False)
            }
        return self._cached_backend_tools

    def invalidate_tool_cache(self) -> None:
        """Invalidate the cached backend tool catalog.

        The next call to any method that needs the backend tool list will
        re-fetch it from the backend server.
        """
        self._cached_backend_tools = None

    async def _get_backend_tool(self, ctx: Context, tool_name: str) -> Tool:
        """Retrieve a specific backend tool from the proxy server."""
        backend_tools = await self._get_backend_tools(ctx)
        tool = backend_tools.get(tool_name)
        if tool is None:
            available_tools = tuple(sorted(backend_tools))
            logger.error(f"Tool '{tool_name}' not found in backend tools. Available tools: {available_tools}")
            raise ToolNotFoundError(tool_name, available_tools)
        return tool

    async def _format_validation_error(
        self, ctx: Context, tool: Tool | str, tool_input: dict[str, Any] | None, error_message: str
    ) -> str:
        """Format a validation failure with the tool schema and parameter suggestions."""
        tool_name = tool.name if isinstance(tool, Tool) else tool
        tool_schema = await self.get_tool_schema(tool_name, ctx)
        parts = [f"Tool '{tool_name}' input validation failed: {error_message}"]
        if isinstance(tool, Tool) and tool_input:
            suggestions = self._suggest_unknown_params(tool, tool_input)
            if suggestions:
                parts.append("Did you mean: " + ", ".join(f"'{k}' -> '{v}'" for k, v in suggestions.items()) + "?")
        parts.append(f"Here is the result of get_tool_schema('{tool_name}'):\n{tool_schema}")
        return "\n\n".join(parts)

    def _suggest_unknown_params(self, tool: Tool, tool_input: dict[str, Any]) -> dict[str, str]:
        """Return a mapping of unknown param names to their closest schema matches."""
        properties = tool.parameters.get("properties", {})
        if not properties:
            return {}
        known_names = list(properties)
        suggestions: dict[str, str] = {}
        for key in tool_input:
            if key not in properties:
                matches = difflib.get_close_matches(key, known_names, n=1, cutoff=0.4)
                if matches:
                    suggestions[key] = matches[0]
        return suggestions

    def _wrapper_instructions(self) -> str:
        """Server-level guardrail, stated ONCE (surfaced via InitializeResult
        instructions) instead of being repeated in every wrapper tool's
        description. Clients that inject server instructions into the prompt
        (e.g. Claude Code) then pay for this text once per server rather than
        once per wrapper tool — a large token saving when many servers are wrapped.
        """
        discover = (
            self._list_tools_name
            if self._compression_level == CompressionLevel.MAX
            else self._get_schema_tool_name
        )
        return (
            f"Every capability of {self._server_description} is reachable ONLY through the "
            f"`{self._invoke_tool_name}` wrapper — there is no direct REST/CLI/HTTP path to "
            f"the backend. Discover tools with `{discover}` and `{self._get_schema_tool_name}`, "
            f"then call `{self._invoke_tool_name}` with `tool_name` and `tool_input`. Do NOT "
            f"bypass this wrapper by calling backend APIs directly, spawning curl/HTTP "
            f"requests, or asking the user for tokens."
        )

    def _make_help_tool(self, description: str | None = None) -> Tool:
        async def help_tool() -> str:
            return await self._build_cli_description()

        return Tool.from_function(
            help_tool,
            name=self._help_tool_name,
            description=description or f"Get help for the '{self._cli_name}' CLI. Lists all available subcommands.",
        )

    def _make_get_schema_tool(self, tool_descriptions: str | None = None) -> Tool:
        # The shared anti-bypass guardrail lives once in the server-level
        # `instructions` (see _wrapper_instructions), not repeated per tool.
        if tool_descriptions:
            catalog = f"\n\nAvailable tools are:\n{tool_descriptions}"
        elif self._compression_level == CompressionLevel.MAX:
            # At MAX the catalog is intentionally omitted from tool descriptions
            # to save tokens; point the model at list_tools instead of leaking a
            # literal "{tool_descriptions}" placeholder.
            catalog = f"\n\nCall `{self._list_tools_name}` to see the available tool names."
        else:
            catalog = ""
        description = (
            f"Get the input schema for a specific tool from {self._server_description}."
            f"{catalog}\n\n"
            f"Then call `{self._invoke_tool_name}` with `tool_name` and `tool_input`."
        )
        return Tool.from_function(self.get_tool_schema, name=self._get_schema_tool_name, description=description)

    def _make_invoke_tool(self, tool_name: str) -> Tool:
        # The shared anti-bypass guardrail lives once in the server-level
        # `instructions` (see _wrapper_instructions), not repeated per tool.
        description = (
            f"Execute a tool of {self._server_description} by name. First call "
            f"`{self._get_schema_tool_name}` to discover the available tool names and "
            f"their input schemas, then call this with `tool_name` and `tool_input`."
        )
        return Tool.from_function(self.invoke_tool, name=tool_name, description=description)

    def _make_list_tools_tool(self) -> Tool:
        # The shared anti-bypass guardrail lives once in the server-level
        # `instructions` (see _wrapper_instructions), not repeated per tool.
        description = (
            f"List all available tools in {self._server_description} with brief descriptions."
        )
        return Tool.from_function(self.list_tools_tool, name=self._list_tools_name, description=description)

    def _make_reload_tool(self) -> Tool:
        description = (
            f"Reload the wrapped MCP server backing {self._server_description}. "
            "Closes the current backend session and establishes a fresh one, then "
            "re-fetches the tool catalog. Useful when the backend server process needs "
            "to be restarted (e.g. to pick up new code) without restarting the agent."
        )
        return Tool.from_function(self.reload_backend, name=self._reload_tool_name, description=description)

    def _make_uncompressed_tools_resource(self) -> Resource:
        return Resource.from_function(
            self.list_uncompressed_tools,
            uri=self._uncompressed_tools_resource_uri,
            description="The upstream server's original uncompressed tool list as JSON.",
            mime_type="application/json",
        )

    def _format_tool_description(self, tool: Tool, compression_level: CompressionLevel) -> str:
        """Format a single tool's description based on the compression level."""
        tool_name = tool.name
        if compression_level == CompressionLevel.MAX:
            return f"<tool>{tool_name}</tool>"
        required_params = set(tool.parameters.get("required", []))
        tool_arg_names = [
            f"{name} [REQUIRED]" if name in required_params else name
            for name in tool.parameters.get("properties", {})
        ]
        tool_description = (tool.description or "").strip()
        if compression_level == CompressionLevel.HIGH:
            tool_description = ""
        elif tool_description and compression_level == CompressionLevel.MEDIUM:
            tool_description = tool_description.splitlines()[0].split(".")[0]
        tool_description = ": " + tool_description if tool_description else ""
        return f"<tool>{tool_name}({', '.join(tool_arg_names)}){tool_description}</tool>"

    def _autocorrect_param_names(self, tool: Tool, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Auto-correct unknown parameter names by fuzzy matching against the tool schema.

        When a parameter name doesn't exist in the schema (e.g. "pull_number" instead of
        "pullNumber"), this method attempts to find a close match and renames the key.
        If no close match is found, the unknown key is left as-is for downstream
        validation to report.
        """
        properties = tool.parameters.get("properties", {})
        if not properties:
            return tool_input
        known_names = set(properties)
        unknown_keys = [k for k in tool_input if k not in known_names]
        if not unknown_keys:
            return tool_input
        corrected = dict(tool_input)
        for key in unknown_keys:
            matches = difflib.get_close_matches(key, known_names, n=1, cutoff=0.6)
            if matches and matches[0] not in corrected:
                logger.debug(f"Auto-corrected parameter name '{key}' -> '{matches[0]}' for tool '{tool.name}'")
                corrected[matches[0]] = corrected.pop(key)
        return corrected

    def _autocorrect_enum_values(self, tool: Tool, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Auto-correct enum parameter values by case-insensitive matching.

        When a tool schema defines enum constraints for string parameters,
        this method fixes mismatched casing (e.g. "GET" -> "get") so that
        minor LLM mistakes don't cause validation errors.
        """
        properties = tool.parameters.get("properties", {})
        corrected = dict(tool_input)
        for key, value in corrected.items():
            if not isinstance(value, str):
                continue
            prop_schema = properties.get(key)
            if prop_schema is None:
                continue
            enum_values = self._extract_enum_values(prop_schema)
            if not enum_values:
                continue
            if value in enum_values:
                continue
            lower_map = {ev.lower(): ev for ev in enum_values if isinstance(ev, str)}
            corrected_value = lower_map.get(value.lower())
            if corrected_value is not None:
                logger.debug(f"Auto-corrected enum value '{value}' -> '{corrected_value}' for parameter '{key}'")
                corrected[key] = corrected_value
        return corrected

    @staticmethod
    def _extract_enum_values(schema: dict[str, Any]) -> list[Any] | None:
        """Extract enum values from a property schema, handling anyOf/oneOf."""
        if "enum" in schema:
            return schema["enum"]
        for combiner in ("anyOf", "oneOf"):
            if combiner in schema:
                for sub_schema in schema[combiner]:
                    if "enum" in sub_schema:
                        return sub_schema["enum"]
        return None

    def _is_validation_error_message(self, error_message: str) -> bool:
        """Return whether a tool error message appears to be an input validation failure."""
        lowered_message = error_message.lower()
        return (
            "validation error" in lowered_message
            or "missing required argument" in lowered_message
            or "missing required parameter" in lowered_message
        )

    def _toonify_tool_result(self, tool_result: ToolResult) -> ToolResult:
        """Convert JSON text content blocks in a tool result to TOON format."""
        converted_content: list[ContentBlock] = []
        content_changed = False
        for content_block in tool_result.content:
            if isinstance(content_block, TextContent):
                converted_text = self._toonify_json_text(content_block.text)
                if converted_text != content_block.text:
                    content_changed = True
                    converted_content.append(TextContent(type="text", text=converted_text))
                    continue
            converted_content.append(content_block)
        if not content_changed:
            return tool_result
        return ToolResult(
            content=converted_content,
            structured_content=tool_result.structured_content,
            meta=tool_result.meta,
        )

    def _toonify_json_text(self, text: str) -> str:
        """Convert a JSON object/array string to TOON; pass through other text unchanged."""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(parsed, dict | list):
            return text
        return toons.dumps(parsed)


def sanitize_tool_name(name: str) -> str:
    """Sanitize a tool name to conform to MCP tool name specifications."""
    sanitized = re.sub(r"[^A-Za-z0-9_\-.]", "_", name).lower()
    if not sanitized:
        raise ValueError("Tool name must contain at least one valid character after sanitization.")
    return sanitized[:128]
