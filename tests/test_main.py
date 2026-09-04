"""Tests for mcp_compressor/main.py — the daemon CLI.

mcp-compressor is a service: the only runnable form is
``mcp-compressor --server [--server-port N] [--mcp-config PATH]``. There is no
wrap mode, no client mode, no CLI mode — those were deleted by design when the
client (free-code) moved to connecting over the socket.
"""

from typer.testing import CliRunner

from mcp_compressor.main import app


def test_no_args_is_rejected_with_a_service_hint() -> None:
    """Running without --server must fail honestly, not silently start anything."""
    result = CliRunner().invoke(app, [])
    assert result.exit_code != 0
    assert "service" in result.output
    assert "--server" in result.output


def test_wrap_mode_argument_is_gone() -> None:
    """A positional COMMAND_OR_URL is no longer accepted."""
    result = CliRunner().invoke(app, ["uvx", "mcp-server-fetch"])
    assert result.exit_code != 0


def test_help_describes_the_daemon() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--server" in result.output
    assert "--server-port" in result.output
    assert "--mcp-config" in result.output
