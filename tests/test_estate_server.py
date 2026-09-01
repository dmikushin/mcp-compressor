"""Tests for mcp_compressor/estate_server.py."""

import json

import pytest

from mcp_compressor import catalog_cache as cc
from mcp_compressor.estate import ServerSpec
from mcp_compressor.estate_server import Estate, UnknownServer, build_estate_server
from mcp_compressor.types import CompressionLevel


def spec(name: str, command: str = "some-backend") -> ServerSpec:
    return ServerSpec(name=name, command=command)


class TestCatalog:
    async def test_lists_names_grouped_by_server(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github"), spec("zulip")])
        for name, tools in (("github", ["create_issue", "list_prs"]), ("zulip", ["send_message"])):
            key = estate._backend(name).cache_key
            cc.save(key, [{"name": t} for t in tools], server=name)

        out = await estate.catalog()
        assert "github (2): create_issue, list_prs" in out
        assert "zulip (1): send_message" in out
        assert "3 tools across 2 servers" in out

    async def test_says_nothing_about_schemas(self, tmp_path, monkeypatch) -> None:
        # The whole saving is that descriptions and schemas stay out of it.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        cc.save(
            estate._backend("github").cache_key,
            [{"name": "create_issue", "description": "SHOULD NOT APPEAR", "inputSchema": {"a": 1}}],
            server="github",
        )
        out = await estate.catalog()
        assert "create_issue" in out
        assert "SHOULD NOT APPEAR" not in out
        assert "inputSchema" not in out

    async def test_reports_configured_servers_that_never_indexed(self, tmp_path, monkeypatch) -> None:
        # Four servers on this machine were broken and invisible for exactly as
        # long as this was silent.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("working"), spec("broken")])
        cc.save(estate._backend("working").cache_key, [{"name": "t"}], server="working")

        out = await estate.catalog()
        assert "working (1): t" in out
        assert "Configured but never indexed" in out
        assert "broken" in out.split("Configured but never indexed")[1]

    async def test_starts_no_backend(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        cc.save(estate._backend("github").cache_key, [{"name": "t"}], server="github")
        await estate.catalog()
        assert not estate._backend("github").is_started, "catalog must never launch a subprocess"

    async def test_one_server_can_be_asked_for(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("a"), spec("b")])
        cc.save(estate._backend("a").cache_key, [{"name": "ta"}], server="a")
        cc.save(estate._backend("b").cache_key, [{"name": "tb"}], server="b")
        out = await estate.catalog("a")
        assert "ta" in out
        assert "tb" not in out

    async def test_an_empty_estate_says_so(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        assert await Estate([]).catalog() == "No servers configured."


class TestIndexingDeadlock:
    """Backends start on first invocation; an invocation needs a tool name; a
    tool name comes from a catalog the backend writes when it starts. A server
    newly added to the configuration is outside that loop and would never become
    visible. Naming one is the way in."""

    async def test_asking_for_one_unindexed_server_indexes_it(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("fresh")])
        started: list[str] = []

        async def fake_tools():
            started.append("fresh")
            cc.save(
                estate._backend("fresh").cache_key,
                [{"name": "newly_found"}],
                server="fresh",
            )

        monkeypatch.setattr(estate._backend("fresh"), "tools", fake_tools)
        out = await estate.catalog("fresh")
        assert started == ["fresh"], "the named server must be started once"
        assert "newly_found" in out

    async def test_asking_for_everything_still_starts_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("a"), spec("b")])

        async def explode():
            raise AssertionError("a whole-estate catalog must not start anything")

        for name in ("a", "b"):
            monkeypatch.setattr(estate._backend(name), "tools", explode)
        out = await estate.catalog()
        assert "Configured but never indexed" in out

    async def test_a_server_that_cannot_start_says_why(
        self, tmp_path, monkeypatch
    ) -> None:
        # Reporting the failure is the point: a silent empty catalog sends the
        # model looking for tools that will never appear.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("broken")])

        async def fail():
            raise RuntimeError("no such binary")

        monkeypatch.setattr(estate._backend("broken"), "tools", fail)
        out = await estate.catalog("broken")
        assert "could not be indexed" in out
        assert "no such binary" in out


class TestUnknownServer:
    """A model that guessed a name needs the list of real ones more than it
    needs to be told it was wrong."""

    async def test_catalog_of_an_unknown_server_lists_the_known_ones(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github"), spec("zulip")])
        with pytest.raises(UnknownServer, match="github, zulip"):
            await estate.catalog("guthib")

    async def test_get_tool_schema_of_an_unknown_server(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        with pytest.raises(UnknownServer, match="github"):
            await estate.get_tool_schema("nope", "t")

    async def test_invoke_of_an_unknown_server(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        with pytest.raises(UnknownServer, match="github"):
            await estate.invoke_tool("nope", "t", {})


class TestReload:
    async def test_reloading_an_unstarted_server_does_not_start_it(self, tmp_path, monkeypatch) -> None:
        # Starting a subprocess as a side effect of "reload" is an action
        # dressed as a no-op.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        message = await estate.reload("github")
        assert "not running" in message
        assert not estate._backend("github").is_started


class TestCacheKeyCompatibility:
    """The estate must reuse the catalogs the per-server arrangement already
    wrote, or every server would be re-indexed on the first run after switching."""

    def test_key_matches_the_per_server_form(self) -> None:
        s = ServerSpec(name="github", command="docker", args=["run", "-i", "ghcr.io/x"])
        estate = Estate([s])
        expected = cc.make_cache_key("docker run -i ghcr.io/x", server_name="github")
        assert estate._backend("github").cache_key == expected


class TestServerSurface:
    """Four tools, and no more: the count is the point."""

    async def test_exposes_exactly_four_tools(self) -> None:
        from fastmcp import Client

        mcp = build_estate_server(Estate([]))
        async with Client(mcp) as client:
            names = sorted(t.name for t in await client.list_tools())
        assert names == ["catalog", "get_tool_schema", "invoke_tool", "reload"]

    async def test_catalog_is_reachable_through_the_server(self, tmp_path, monkeypatch) -> None:
        from fastmcp import Client

        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        cc.save(estate._backend("github").cache_key, [{"name": "create_issue"}], server="github")
        mcp = build_estate_server(estate)
        async with Client(mcp) as client:
            result = await client.call_tool("catalog", {})
        assert "create_issue" in json.dumps(
            [c.model_dump(mode="json") for c in result.content], default=str
        )


class TestCompressionLevel:
    def test_backends_are_built_at_the_level_the_estate_was_given(self) -> None:
        estate = Estate([spec("x")], compression_level=CompressionLevel.LOW)
        assert estate._backend("x")._compression_level is CompressionLevel.LOW


class TestRescan:
    """Adding a server to the configuration must not require a restart."""

    @staticmethod
    def write_config(path, **servers) -> str:
        path.write_text(json.dumps({"mcpServers": {
            name: {"command": command} for name, command in servers.items()
        }}))
        return str(path)

    async def test_a_server_added_after_startup_becomes_visible(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cfg = tmp_path / "claude.json"
        self.write_config(cfg, github="gh-backend")
        estate = Estate([spec("github", "gh-backend")], config_path=cfg)

        self.write_config(cfg, github="gh-backend", telegram="tg-backend")
        assert "telegram" not in estate.names
        await estate.catalog()
        assert "telegram" in estate.names

    async def test_a_server_removed_from_the_config_disappears(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cfg = tmp_path / "claude.json"
        self.write_config(cfg, github="gh-backend", zulip="zulip-backend")
        estate = Estate([spec("github", "gh-backend"), spec("zulip", "zulip-backend")], config_path=cfg)

        self.write_config(cfg, github="gh-backend")
        await estate.catalog()
        assert estate.names == ["github"]

    async def test_an_untouched_server_keeps_its_backend(self, tmp_path, monkeypatch) -> None:
        # Rescanning must not quietly restart backends: a replaced object is a
        # dropped subprocess and a discarded catalog.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cfg = tmp_path / "claude.json"
        self.write_config(cfg, github="gh-backend")
        estate = Estate([spec("github", "gh-backend")], config_path=cfg)
        before = estate._backend("github")

        self.write_config(cfg, github="gh-backend", telegram="tg-backend")
        await estate.rescan()
        assert estate._backend("github") is before

    async def test_a_changed_command_replaces_the_backend(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cfg = tmp_path / "claude.json"
        self.write_config(cfg, github="gh-backend")
        estate = Estate([spec("github", "gh-backend")], config_path=cfg)
        before = estate._backend("github")

        self.write_config(cfg, github="gh-backend-v2")
        assert await estate.rescan() == ["reconfigured github"]
        assert estate._backend("github") is not before
        assert estate._backend("github").spec.command == "gh-backend-v2"

    async def test_an_unreadable_config_keeps_the_known_servers(self, tmp_path, monkeypatch) -> None:
        # A config caught mid-write must not blind the estate to what it has.
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        cfg = tmp_path / "claude.json"
        self.write_config(cfg, github="gh-backend")
        estate = Estate([spec("github", "gh-backend")], config_path=cfg)
        cc.save(estate._backend("github").cache_key, [{"name": "create_issue"}], server="github")

        cfg.write_text("{ this is not json")
        out = await estate.catalog()
        assert "create_issue" in out
        assert "may be stale" in out

    async def test_without_a_config_path_nothing_is_rescanned(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cc, "_CACHE_DIR", tmp_path / "catalogs")
        estate = Estate([spec("github")])
        assert await estate.rescan() == []
        assert estate.names == ["github"]
