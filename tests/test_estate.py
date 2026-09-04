"""Tests for mcp_compressor/estate.py."""

import json

import pytest

from mcp_compressor.estate import ConfigError, ServerSpec, load_servers


def write(tmp_path, doc) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    return str(path)


class TestLoadServers:
    def test_reads_stdio_servers_from_the_top_level(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "github": {"command": "docker", "args": ["run", "-i"], "env": {"TOKEN": "x"}},
                }
            },
        )
        assert load_servers(path) == [
            ServerSpec(name="github", command="docker", args=["run", "-i"], env={"TOKEN": "x"})
        ]

    def test_order_comes_from_the_configuration_not_the_dict(self, tmp_path) -> None:
        # A catalog that reorders between runs cannot be cached by its consumers.
        path = write(
            tmp_path,
            {"mcpServers": {"zulip": {"command": "z"}, "apkext": {"command": "a"}, "gdb": {"command": "g"}}},
        )
        assert [s.name for s in load_servers(path)] == ["apkext", "gdb", "zulip"]

    def test_fronts_stdio_http_and_sse(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "remote": {"type": "http", "url": "https://x/mcp", "headers": {"A": "b"}},
                    "events": {"type": "sse", "url": "https://y/sse"},
                    "local": {"command": "l"},
                }
            },
        )
        specs = {s.name: s for s in load_servers(path)}
        assert sorted(specs) == ["events", "local", "remote"]
        assert specs["remote"].transport == "http"
        assert specs["remote"].url == "https://x/mcp"
        assert specs["remote"].headers == {"A": "b"}
        assert specs["remote"].source == "https://x/mcp"
        assert specs["events"].transport == "sse"
        assert specs["local"].transport == "stdio"
        assert specs["local"].source == "l"

    def test_streamable_http_is_a_synonym_for_http(self, tmp_path) -> None:
        path = write(tmp_path, {"mcpServers": {"r": {"type": "streamable-http", "url": "https://z"}}})
        assert [s.transport for s in load_servers(path)] == ["http"]

    def test_ignores_unknown_type_and_remote_without_url(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "weird": {"type": "carrier-pigeon", "url": "https://x"},
                    "nourl": {"type": "http"},
                    "ok": {"command": "c"},
                }
            },
        )
        assert [s.name for s in load_servers(path)] == ["ok"]

    def test_skips_entries_with_no_command(self, tmp_path) -> None:
        path = write(tmp_path, {"mcpServers": {"broken": {"args": ["x"]}, "ok": {"command": "c"}}})
        assert [s.name for s in load_servers(path)] == ["ok"]

    def test_reads_per_project_servers(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {"global": {"command": "g"}},
                "projects": {"/home/x/proj": {"mcpServers": {"local": {"command": "l"}}}},
            },
        )
        assert [s.name for s in load_servers(path)] == ["global", "local"]

    def test_a_named_project_narrows_the_selection(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "projects": {
                    "/a": {"mcpServers": {"only_a": {"command": "a"}}},
                    "/b": {"mcpServers": {"only_b": {"command": "b"}}},
                }
            },
        )
        assert [s.name for s in load_servers(path, project="/a")] == ["only_a"]


class TestSelfReference:
    """Once one compressor fronts the estate, the config lists the compressor.

    Reading that entry back and launching it would launch a process that reads
    the same file and launches itself again. The recursion has to be cut by
    construction, not by hoping a name never matches.
    """

    def test_the_estate_front_end_does_not_wrap_itself(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "compressor": {
                        "command": "mcp-compressor",
                        "args": ["--mcp-config", "/home/user/.claude.json"],
                    },
                    "real": {"command": "some-backend"},
                }
            },
        )
        assert [s.name for s in load_servers(path)] == ["real"]

    def test_self_reference_is_caught_by_an_absolute_path_too(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "compressor": {
                        "command": "/home/user/.local/bin/mcp-compressor",
                        "args": ["--mcp-config", "/home/user/.claude.json"],
                    }
                }
            },
        )
        assert load_servers(path) == []

    def test_a_bare_compressor_entry_is_not_mistaken_for_a_backend(self, tmp_path) -> None:
        path = write(tmp_path, {"mcpServers": {"c": {"command": "mcp-compressor", "args": []}}})
        assert load_servers(path) == []


class TestLegacyWrappers:
    """A configuration written for one-process-per-server still describes real
    backends. Losing them would mean the estate could not be read until the user
    rewrote the file by hand."""

    def test_unwraps_an_old_style_wrapper(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "gdb": {
                        "command": "mcp-compressor",
                        "args": ["--lazy", "-n", "gdb", "-c", "max", "--", "/usr/bin/gdb-mcp", "--flag"],
                    }
                }
            },
        )
        (spec,) = load_servers(path)
        assert spec.command == "/usr/bin/gdb-mcp"
        assert spec.args == ["--flag"]

    def test_env_inside_the_wrapper_is_recovered(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "forgejo": {
                        "command": "mcp-compressor",
                        "args": ["-n", "forgejo", "-c", "max", "-e", "TOKEN=secret", "--", "forgejo-mcp"],
                    }
                }
            },
        )
        (spec,) = load_servers(path)
        assert spec.command == "forgejo-mcp"
        assert spec.env == {"TOKEN": "secret"}

    def test_a_wrapper_without_the_double_dash_still_unwraps(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "orca": {
                        "command": "mcp-compressor",
                        "args": ["--lazy", "-n", "orca", "-c", "max", "/opt/orca-mcp"],
                    }
                }
            },
        )
        (spec,) = load_servers(path)
        assert spec.command == "/opt/orca-mcp"
        assert spec.args == []

    def test_the_configs_own_env_wins_over_the_wrappers(self, tmp_path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "x": {
                        "command": "mcp-compressor",
                        "args": ["-e", "TOKEN=old", "--", "backend"],
                        "env": {"TOKEN": "new"},
                    }
                }
            },
        )
        (spec,) = load_servers(path)
        assert spec.env == {"TOKEN": "new"}


class TestFailures:
    """A configuration that cannot be read must say which file and why. The
    estate is the whole tool surface; failing quietly here would present a model
    with an empty world and no reason for it."""

    def test_a_missing_file_names_itself(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match=r"nowhere\.json"):
            load_servers(tmp_path / "nowhere.json")

    def test_invalid_json_names_the_file(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ConfigError, match=r"bad\.json"):
            load_servers(path)

    def test_a_json_scalar_is_not_a_configuration(self, tmp_path) -> None:
        path = tmp_path / "scalar.json"
        path.write_text("42")
        with pytest.raises(ConfigError, match="JSON object"):
            load_servers(path)

    def test_a_config_with_no_servers_is_empty_not_an_error(self, tmp_path) -> None:
        assert load_servers(write(tmp_path, {"other": "data"})) == []
