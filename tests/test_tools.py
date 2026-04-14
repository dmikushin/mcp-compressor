"""Tests for mcp_compressor/tools.py."""

import pytest
import toons
from fastmcp.tools import Tool

from mcp_compressor.tools import (
    CachedTool,
    CompressedTools,
    ReloadableClientManager,
    ToolNotFoundError,
    sanitize_tool_name,
)
from mcp_compressor.types import CompressionLevel


@pytest.mark.parametrize(
    "input_name,expected",
    [
        # Valid characters pass through unchanged
        ("my_tool", "my_tool"),
        ("my-tool", "my-tool"),
        ("my.tool", "my.tool"),
        ("MyTool123", "mytool123"),
        # Invalid characters are replaced with underscores
        ("my tool", "my_tool"),
        ("my!tool", "my_tool"),
        ("my@tool#name", "my_tool_name"),
        ("tool with spaces!", "tool_with_spaces_"),
        # Mixed valid and invalid
        ("github_get-schema.v1!", "github_get-schema.v1_"),
    ],
)
def test_sanitize_tool_name(input_name: str, expected: str) -> None:
    """Test that invalid characters are replaced with underscores."""
    assert sanitize_tool_name(input_name) == expected


def test_sanitize_tool_name_truncates_long_names() -> None:
    """Test that names longer than 128 characters are truncated."""
    long_name = "a" * 150
    result = sanitize_tool_name(long_name)
    assert len(result) == 128
    assert result == "a" * 128


def test_sanitize_tool_name_all_invalid_chars_become_underscores() -> None:
    """Test that all-invalid input becomes underscores."""
    assert sanitize_tool_name("!!!") == "___"


class TestCompressedTools:
    """Tests for the CompressedTools class."""

    @pytest.fixture
    def compressed_tools(self) -> CompressedTools:
        """Create a CompressedTools instance for testing."""
        # We don't need a real proxy server for these tests
        return CompressedTools(None, CompressionLevel.LOW, server_name=None)  # type: ignore[arg-type]

    @pytest.fixture
    def sample_tool(self) -> Tool:
        """Create a sample tool for testing."""

        def dummy_fn(param1: str, param2: int) -> str:
            """First sentence of description. Second sentence here.

            More details on another line.
            """
            return ""

        return Tool.from_function(dummy_fn)

    @pytest.mark.parametrize(
        "compression_level,expected_in_result",
        [
            # LOW keeps full first line of description
            (CompressionLevel.LOW, ": First sentence of description. Second sentence here."),
            # MEDIUM takes only up to first period
            (CompressionLevel.MEDIUM, ": First sentence of description"),
            # HIGH removes description entirely
            (CompressionLevel.HIGH, "dummy_fn(param1 [REQUIRED], param2 [REQUIRED])</tool>"),
        ],
    )
    def test_compression_levels(
        self,
        compressed_tools: CompressedTools,
        sample_tool: Tool,
        compression_level: CompressionLevel,
        expected_in_result: str,
    ) -> None:
        """Test that different compression levels produce appropriate output."""
        result = compressed_tools._format_tool_description(sample_tool, compression_level)
        assert expected_in_result in result
        assert result.startswith("<tool>dummy_fn(param1 [REQUIRED], param2 [REQUIRED])")
        assert result.endswith("</tool>")

    def test_tool_with_no_description(self, compressed_tools: CompressedTools) -> None:
        """Test formatting a tool with no description."""

        def no_desc_tool(arg: str) -> str:
            return arg

        tool = Tool.from_function(no_desc_tool)
        tool.description = None
        result = compressed_tools._format_tool_description(tool, CompressionLevel.LOW)
        assert result == "<tool>no_desc_tool(arg [REQUIRED])</tool>"

    def test_tool_with_no_parameters(self, compressed_tools: CompressedTools) -> None:
        """Test formatting a tool with no parameters."""

        def empty_tool() -> None:
            """A tool with no params."""
            pass

        tool = Tool.from_function(empty_tool)
        result = compressed_tools._format_tool_description(tool, CompressionLevel.LOW)
        assert result == "<tool>empty_tool(): A tool with no params.</tool>"

    def test_toonify_json_text_converts_objects_and_arrays(self, compressed_tools: CompressedTools) -> None:
        """Test that toonify converts JSON object/array strings to TOON."""
        assert compressed_tools._toonify_json_text('{"name":"Alice","age":30}') == toons.dumps({
            "name": "Alice",
            "age": 30,
        })
        assert compressed_tools._toonify_json_text('[{"id":1},{"id":2}]') == toons.dumps([{"id": 1}, {"id": 2}])

    def test_toonify_json_text_leaves_non_json_text_unchanged(self, compressed_tools: CompressedTools) -> None:
        """Test that toonify leaves non-JSON text unchanged."""
        assert compressed_tools._toonify_json_text("plain text") == "plain text"
        assert compressed_tools._toonify_json_text("123") == "123"


async def test_configure_server_applies_visibility_filters_for_backend_tools() -> None:
    """Test that include/exclude filters are translated into FastMCP visibility rules."""

    class FakeProxyServer:
        def __init__(self) -> None:
            self.enabled_calls: list[dict] = []
            self.disabled_calls: list[dict] = []
            self.middleware: list[object] = []
            self.transforms: list[object] = []
            self.tools = [
                Tool.from_function(lambda a, b: a + b, name="add"),
                Tool.from_function(lambda arg: arg, name="do_nothing"),
                Tool.from_function(lambda: None, name="empty_tool"),
            ]

        def enable(self, **kwargs):
            self.enabled_calls.append(kwargs)
            return self

        def disable(self, **kwargs):
            self.disabled_calls.append(kwargs)
            return self

        def add_middleware(self, middleware) -> None:
            self.middleware.append(middleware)

        def add_transform(self, transform) -> None:
            self.transforms.append(transform)

        async def list_tools(self, *, run_middleware: bool = True):
            return self.tools

    proxy_server = FakeProxyServer()
    compressed_tools = CompressedTools(
        proxy_server,  # type: ignore[arg-type]
        CompressionLevel.LOW,
        server_name="test_server",
        include_tools=["add", "do_nothing"],
        exclude_tools=["do_nothing"],
    )

    await compressed_tools.configure_server()

    assert proxy_server.enabled_calls == []
    assert proxy_server.disabled_calls == [
        {
            "names": {"empty_tool"},
            "components": {"tool"},
        },
        {
            "names": {"do_nothing"},
            "components": {"tool"},
        },
    ]
    assert proxy_server.transforms == [compressed_tools]
    assert len(proxy_server.middleware) == 1


class FakeProxyServer:
    """Minimal proxy server fake for caching tests."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools
        self.list_tools_call_count = 0
        self.disabled_calls: list[dict] = []
        self.middleware: list[object] = []
        self.transforms: list[object] = []

    def disable(self, **kwargs):
        self.disabled_calls.append(kwargs)
        return self

    def add_middleware(self, middleware) -> None:
        self.middleware.append(middleware)

    def add_transform(self, transform) -> None:
        self.transforms.append(transform)

    async def list_tools(self, *, run_middleware: bool = True):
        self.list_tools_call_count += 1
        return self.tools


async def test_tool_cache_warmed_at_configure_server() -> None:
    """Tool catalog should be cached after configure_server() so no extra backend calls are made."""
    tools = [
        Tool.from_function(lambda a, b: a + b, name="add"),
        Tool.from_function(lambda arg: arg, name="echo"),
    ]
    proxy_server = FakeProxyServer(tools)
    compressed_tools = CompressedTools(
        proxy_server,  # type: ignore[arg-type]
        CompressionLevel.LOW,
        server_name="test",
    )

    await compressed_tools.configure_server()

    # Cache should be populated after configure_server
    assert compressed_tools._cached_backend_tools is not None
    assert set(compressed_tools._cached_backend_tools.keys()) == {"add", "echo"}
    # No include/exclude filters, so list_tools is called exactly once and the
    # result is reused directly for the cache (no redundant second fetch).
    assert proxy_server.list_tools_call_count == 1


async def test_get_backend_tools_uses_cache_after_configure_server() -> None:
    """_get_backend_tools() should not call the backend after the cache is warmed."""
    from fastmcp import FastMCP
    from fastmcp.server import create_proxy
    from fastmcp.server.context import Context

    backend = FastMCP(name="backend")

    @backend.tool()
    def my_tool() -> str:
        """A test tool."""
        return "result"

    proxy_server = create_proxy(backend, name="proxy")
    compressed_tools = CompressedTools(
        proxy_server,
        CompressionLevel.LOW,
        server_name="test",
    )

    await compressed_tools.configure_server()

    # Cache should be warm — record what's in it
    assert compressed_tools._cached_backend_tools is not None
    assert "my_tool" in compressed_tools._cached_backend_tools

    # Patch out the backend to confirm no further fetches happen
    original_cache = compressed_tools._cached_backend_tools

    async with Context(fastmcp=proxy_server) as ctx:
        result1 = await compressed_tools._get_backend_tools(ctx)
        result2 = await compressed_tools._get_backend_tools(ctx)
        result3 = await compressed_tools._get_backend_tools(ctx)

    # All calls should return the same cached dict object (identity check)
    assert result1 is original_cache
    assert result2 is original_cache
    assert result3 is original_cache


async def test_invalidate_tool_cache_forces_refetch() -> None:
    """invalidate_tool_cache() should clear the cache so the next call re-fetches from backend."""
    from fastmcp import FastMCP
    from fastmcp.server import create_proxy
    from fastmcp.server.context import Context

    backend = FastMCP(name="backend")

    @backend.tool()
    def my_tool() -> str:
        """A test tool."""
        return "result"

    proxy_server = create_proxy(backend, name="proxy")
    compressed_tools = CompressedTools(
        proxy_server,
        CompressionLevel.LOW,
        server_name="test",
    )

    await compressed_tools.configure_server()
    original_cache = compressed_tools._cached_backend_tools

    # Invalidate the cache
    compressed_tools.invalidate_tool_cache()
    assert compressed_tools._cached_backend_tools is None

    # Next call should re-fetch and produce a new cache object
    async with Context(fastmcp=proxy_server) as ctx:
        result = await compressed_tools._get_backend_tools(ctx)

    assert set(result.keys()) == {"my_tool"}
    assert compressed_tools._cached_backend_tools is not None
    # A fresh dict was created (different object from original)
    assert compressed_tools._cached_backend_tools is not original_cache


async def test_get_backend_tools_lazy_fetch_when_cache_cold() -> None:
    """_get_backend_tools() should fetch from backend if cache is cold (configure_server not called)."""
    from fastmcp import FastMCP
    from fastmcp.server import create_proxy
    from fastmcp.server.context import Context

    backend = FastMCP(name="backend")

    @backend.tool()
    def lazy_tool() -> str:
        """A lazy test tool."""
        return "result"

    proxy_server = create_proxy(backend, name="proxy")
    compressed_tools = CompressedTools(
        proxy_server,
        CompressionLevel.LOW,
        server_name="test",
    )

    # Cache is cold — configure_server was not called
    assert compressed_tools._cached_backend_tools is None

    async with Context(fastmcp=proxy_server) as ctx:
        result = await compressed_tools._get_backend_tools(ctx)

    assert set(result.keys()) == {"lazy_tool"}
    # Cache should now be populated
    assert compressed_tools._cached_backend_tools is not None


class TestAutocorrectEnumValues:
    """Tests for enum auto-correction in invoke_tool."""

    @pytest.fixture
    def compressed_tools(self) -> CompressedTools:
        return CompressedTools(None, CompressionLevel.LOW, server_name=None)  # type: ignore[arg-type]

    @staticmethod
    def _make_tool_with_schema(parameters: dict) -> Tool:
        """Create a Tool with a custom parameters schema."""

        def stub(x: str = "") -> str:
            return "ok"

        tool = Tool.from_function(stub, name="test_tool")
        tool.parameters = parameters
        return tool

    def _make_tool_with_enum(self, enum_values: list[str], prop_name: str = "method") -> Tool:
        return self._make_tool_with_schema({
            "type": "object",
            "properties": {
                prop_name: {"type": "string", "enum": enum_values},
            },
        })

    def test_exact_match_unchanged(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff"])
        result = compressed_tools._autocorrect_enum_values(tool, {"method": "get"})
        assert result == {"method": "get"}

    def test_uppercase_corrected(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff", "get_status"])
        result = compressed_tools._autocorrect_enum_values(tool, {"method": "GET"})
        assert result == {"method": "get"}

    def test_mixed_case_corrected(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff", "get_status"])
        result = compressed_tools._autocorrect_enum_values(tool, {"method": "Get_Diff"})
        assert result == {"method": "get_diff"}

    def test_no_match_left_unchanged(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff"])
        result = compressed_tools._autocorrect_enum_values(tool, {"method": "nonexistent"})
        assert result == {"method": "nonexistent"}

    def test_non_string_values_skipped(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff"])
        result = compressed_tools._autocorrect_enum_values(tool, {"method": 42})
        assert result == {"method": 42}

    def test_non_enum_property_skipped(self, compressed_tools: CompressedTools) -> None:
        def dummy(name: str) -> str:
            return name

        tool = Tool.from_function(dummy, name="test_tool")
        result = compressed_tools._autocorrect_enum_values(tool, {"name": "HELLO"})
        assert result == {"name": "HELLO"}

    def test_anyof_enum_corrected(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {
                "anchor": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string", "enum": ["newest", "oldest", "first_unread"]},
                    ]
                }
            },
        })
        result = compressed_tools._autocorrect_enum_values(tool, {"anchor": "NEWEST"})
        assert result == {"anchor": "newest"}

    def test_multiple_params_corrected(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["get", "post"]},
                "format": {"type": "string", "enum": ["json", "xml"]},
            },
        })
        result = compressed_tools._autocorrect_enum_values(tool, {"method": "GET", "format": "JSON"})
        assert result == {"method": "get", "format": "json"}

    def test_original_input_not_mutated(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_enum(["get", "get_diff"])
        original = {"method": "GET"}
        result = compressed_tools._autocorrect_enum_values(tool, original)
        assert original == {"method": "GET"}
        assert result == {"method": "get"}


class TestAutocorrectParamNames:
    """Tests for parameter name auto-correction in invoke_tool."""

    @pytest.fixture
    def compressed_tools(self) -> CompressedTools:
        return CompressedTools(None, CompressionLevel.LOW, server_name=None)  # type: ignore[arg-type]

    @staticmethod
    def _make_tool_with_schema(parameters: dict) -> Tool:
        def stub(x: str = "") -> str:
            return "ok"

        tool = Tool.from_function(stub, name="test_tool")
        tool.parameters = parameters
        return tool

    def test_known_params_unchanged(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"pullNumber": {"type": "integer"}, "owner": {"type": "string"}},
        })
        result = compressed_tools._autocorrect_param_names(tool, {"pullNumber": 1, "owner": "x"})
        assert result == {"pullNumber": 1, "owner": "x"}

    def test_snake_to_camel_corrected(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"pullNumber": {"type": "integer"}, "owner": {"type": "string"}},
        })
        result = compressed_tools._autocorrect_param_names(tool, {"pull_number": 1, "owner": "x"})
        assert result == {"pullNumber": 1, "owner": "x"}

    def test_no_correction_when_no_close_match(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"pullNumber": {"type": "integer"}},
        })
        result = compressed_tools._autocorrect_param_names(tool, {"zzzzz": 1})
        assert result == {"zzzzz": 1}

    def test_no_overwrite_existing_param(self, compressed_tools: CompressedTools) -> None:
        """Don't rename if the target name is already present in tool_input."""
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"pullNumber": {"type": "integer"}},
        })
        result = compressed_tools._autocorrect_param_names(tool, {"pull_number": 1, "pullNumber": 2})
        assert result == {"pull_number": 1, "pullNumber": 2}


class TestSuggestUnknownParams:
    """Tests for parameter name suggestions in validation errors."""

    @pytest.fixture
    def compressed_tools(self) -> CompressedTools:
        return CompressedTools(None, CompressionLevel.LOW, server_name=None)  # type: ignore[arg-type]

    @staticmethod
    def _make_tool_with_schema(parameters: dict) -> Tool:
        def stub(x: str = "") -> str:
            return "ok"

        tool = Tool.from_function(stub, name="test_tool")
        tool.parameters = parameters
        return tool

    def test_suggests_close_match(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"pullNumber": {"type": "integer"}, "method": {"type": "string"}},
        })
        suggestions = compressed_tools._suggest_unknown_params(tool, {"pull_number": 1, "method": "get"})
        assert suggestions == {"pull_number": "pullNumber"}

    def test_no_suggestion_for_known_params(self, compressed_tools: CompressedTools) -> None:
        tool = self._make_tool_with_schema({
            "type": "object",
            "properties": {"method": {"type": "string"}},
        })
        suggestions = compressed_tools._suggest_unknown_params(tool, {"method": "get"})
        assert suggestions == {}


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_error_message_contains_tool_name_and_available_tools(self) -> None:
        """Test that the error message includes the tool name and available tools."""
        error = ToolNotFoundError("missing_tool", ["add", "do_nothing"])
        assert "missing_tool" in str(error)
        assert "Available tools: add, do_nothing" in str(error)
        assert error.tool_name == "missing_tool"
        assert error.available_tools == ("add", "do_nothing")

    def test_error_message_suggests_similar_tool_names(self) -> None:
        """Test that the error message suggests similar tool names."""
        error = ToolNotFoundError("get_pull_request", ["pull_request_read", "create_pull_request", "list_pull_requests"])
        msg = str(error)
        assert "Did you mean:" in msg
        assert "pull_request_read" in msg or "create_pull_request" in msg

    def test_error_message_no_suggestions_when_no_match(self) -> None:
        """Test that no suggestions are shown when nothing is close."""
        error = ToolNotFoundError("zzzzzzzzz", ["add", "do_nothing"])
        assert "Did you mean:" not in str(error)


class TestReloadableClientManager:
    """Tests for the ReloadableClientManager lifecycle."""

    async def test_start_calls_connect_once(self) -> None:
        """start() should create the initial client by calling connect() exactly once."""
        call_count = 0

        class FakeClient:
            async def __aexit__(self, *args) -> None:
                pass

        async def connect() -> FakeClient:
            nonlocal call_count
            call_count += 1
            return FakeClient()

        manager = ReloadableClientManager(connect=connect)
        await manager.start()
        assert call_count == 1
        assert manager.get_client() is not None

        # Second start() is idempotent
        await manager.start()
        assert call_count == 1

        await manager.stop()

    async def test_reload_closes_old_and_creates_new_client(self) -> None:
        """reload() should __aexit__ the old client and create a fresh one."""
        closed_clients = []
        created_clients = []

        class FakeClient:
            def __init__(self, label: str) -> None:
                self.label = label

            async def __aexit__(self, *args) -> None:
                closed_clients.append(self.label)

        labels = iter(["client_1", "client_2", "client_3"])

        async def connect() -> FakeClient:
            client = FakeClient(next(labels))
            created_clients.append(client.label)
            return client

        manager = ReloadableClientManager(connect=connect)
        await manager.start()
        assert created_clients == ["client_1"]
        assert manager.get_client().label == "client_1"  # type: ignore[attr-defined]

        await manager.reload()
        assert closed_clients == ["client_1"]
        assert created_clients == ["client_1", "client_2"]
        assert manager.get_client().label == "client_2"  # type: ignore[attr-defined]

        await manager.reload()
        assert closed_clients == ["client_1", "client_2"]
        assert created_clients == ["client_1", "client_2", "client_3"]
        assert manager.get_client().label == "client_3"  # type: ignore[attr-defined]

        await manager.stop()
        assert closed_clients == ["client_1", "client_2", "client_3"]

    async def test_get_client_raises_before_start(self) -> None:
        """get_client() before start() should raise RuntimeError."""

        async def connect():
            raise AssertionError("should not be called")

        manager = ReloadableClientManager(connect=connect)
        with pytest.raises(RuntimeError, match="not started"):
            manager.get_client()

    async def test_reload_close_errors_are_suppressed(self) -> None:
        """Errors during __aexit__ of the old client should not prevent reload."""

        class FakeClient:
            def __init__(self, fail_on_close: bool) -> None:
                self.fail_on_close = fail_on_close

            async def __aexit__(self, *args) -> None:
                if self.fail_on_close:
                    raise RuntimeError("simulated close failure")

        call_count = 0

        async def connect() -> FakeClient:
            nonlocal call_count
            call_count += 1
            # First client fails on close; subsequent clients are clean
            return FakeClient(fail_on_close=(call_count == 1))

        manager = ReloadableClientManager(connect=connect)
        await manager.start()
        # Should not raise even though old client __aexit__ fails
        await manager.reload()
        assert call_count == 2
        await manager.stop()


class TestReloadToolExposure:
    """Tests that CompressedTools exposes the reload tool iff a client manager is provided."""

    def test_reload_tool_not_in_wrapper_set_without_manager(self) -> None:
        compressed_tools = CompressedTools(None, CompressionLevel.LOW, server_name="test")  # type: ignore[arg-type]
        assert compressed_tools._reload_tool_name not in compressed_tools._wrapper_tool_names()

    def test_reload_tool_in_wrapper_set_with_manager(self) -> None:
        class StubManager:
            async def start(self) -> None: ...
            async def stop(self) -> None: ...
            async def reload(self) -> None: ...
            def get_client(self): ...

        compressed_tools = CompressedTools(
            None,  # type: ignore[arg-type]
            CompressionLevel.LOW,
            server_name="test",
            client_manager=StubManager(),  # type: ignore[arg-type]
        )
        assert "test_reload" in compressed_tools._wrapper_tool_names()

    def test_reload_tool_name_honors_server_prefix(self) -> None:
        class StubManager:
            async def reload(self) -> None: ...

        compressed_tools = CompressedTools(
            None,  # type: ignore[arg-type]
            CompressionLevel.LOW,
            server_name="github",
            client_manager=StubManager(),  # type: ignore[arg-type]
        )
        assert compressed_tools._reload_tool_name == "github_reload"


class TestCachedTool:
    """Tests for the CachedTool lazy-loading stub."""

    def test_from_mcp_dict_round_trip(self) -> None:
        data = {
            "name": "pull_request_read",
            "description": "Read a pull request.",
            "inputSchema": {
                "type": "object",
                "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}},
                "required": ["owner", "repo"],
            },
        }
        tool = CachedTool.from_mcp_dict(data)
        assert tool.name == "pull_request_read"
        assert tool.description == "Read a pull request."
        assert tool.parameters == data["inputSchema"]

    def test_from_mcp_dict_missing_description(self) -> None:
        data = {"name": "empty_tool", "inputSchema": {}}
        tool = CachedTool.from_mcp_dict(data)
        assert tool.description is None

    def test_format_description_works_with_cached_tool(self) -> None:
        """CompressedTools._format_tool_description must accept CachedTool stubs."""
        ct = CompressedTools(None, CompressionLevel.LOW, server_name="test")  # type: ignore[arg-type]
        stub = CachedTool(name="my_tool", description="Does stuff.", parameters={"properties": {"x": {}}})
        result = ct._format_tool_description(stub, CompressionLevel.HIGH)  # type: ignore[arg-type]
        assert "<tool>my_tool" in result
        assert "</tool>" in result


class TestEnsureConnected:
    """Tests for ReloadableClientManager.ensure_connected (lazy-mode helper)."""

    async def test_ensure_connected_when_not_connected(self) -> None:
        """ensure_connected() should call connect() if not yet connected."""
        call_count = 0

        class FakeClient:
            async def __aexit__(self, *args) -> None:
                pass

        async def connect() -> FakeClient:
            nonlocal call_count
            call_count += 1
            return FakeClient()

        manager = ReloadableClientManager(connect=connect)
        assert not manager.is_connected
        await manager.ensure_connected()
        assert call_count == 1
        assert manager.is_connected
        await manager.stop()

    async def test_ensure_connected_idempotent(self) -> None:
        """Multiple ensure_connected() calls must not reconnect."""
        call_count = 0

        class FakeClient:
            async def __aexit__(self, *args) -> None:
                pass

        async def connect() -> FakeClient:
            nonlocal call_count
            call_count += 1
            return FakeClient()

        manager = ReloadableClientManager(connect=connect)
        await manager.ensure_connected()
        await manager.ensure_connected()
        await manager.ensure_connected()
        assert call_count == 1
        await manager.stop()

    async def test_is_connected_false_before_start(self) -> None:
        manager = ReloadableClientManager(connect=lambda: None)  # type: ignore[arg-type]
        assert not manager.is_connected

    async def test_is_connected_true_after_start(self) -> None:
        class FakeClient:
            async def __aexit__(self, *args) -> None:
                pass

        async def connect() -> FakeClient:
            return FakeClient()

        manager = ReloadableClientManager(connect=connect)
        await manager.start()
        assert manager.is_connected
        await manager.stop()
        assert not manager.is_connected


class TestCatalogCache:
    """Tests for the catalog_cache module."""

    def test_load_returns_none_on_cache_miss(self, tmp_path, monkeypatch) -> None:
        import mcp_compressor.catalog_cache as cc

        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        assert cc.load("nonexistent") is None

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch) -> None:
        import mcp_compressor.catalog_cache as cc

        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        tools = [{"name": "foo", "description": "Foo tool.", "inputSchema": {"properties": {}}}]
        cc.save("mykey", tools)
        loaded = cc.load("mykey")
        assert loaded == tools

    def test_clear_removes_cache(self, tmp_path, monkeypatch) -> None:
        import mcp_compressor.catalog_cache as cc

        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cc.save("mykey", [{"name": "x", "inputSchema": {}}])
        assert cc.load("mykey") is not None
        cc.clear("mykey")
        assert cc.load("mykey") is None

    def test_make_cache_key_stable(self) -> None:
        import mcp_compressor.catalog_cache as cc

        k1 = cc.make_cache_key("uvx mcp-github")
        k2 = cc.make_cache_key("uvx mcp-github")
        assert k1 == k2

    def test_make_cache_key_differs_with_filters(self) -> None:
        import mcp_compressor.catalog_cache as cc

        k1 = cc.make_cache_key("uvx mcp-github")
        k2 = cc.make_cache_key("uvx mcp-github", include_tools=["foo"])
        assert k1 != k2


async def test_on_call_tool_extracts_flat_args_as_tool_input(proxy_mcp_client) -> None:
    """Test that invoke_tool creates tool_input from flat args when tool_input is not provided."""
    # Call invoke_tool with flat args (no tool_input wrapper)
    # This simulates how some LLMs call tools with args flattened
    result = await proxy_mcp_client.call_tool(
        "test_server_invoke_tool",
        {"tool_name": "add", "a": 5, "b": 3},
    )
    assert result.content
    assert result.content[0].text == "8"
