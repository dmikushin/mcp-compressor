"""Entry point for the MCP Compressor daemon.

MCP Compressor is one service per user. It reads the client's MCP configuration,
fronts every server in it behind four tools (catalog, get_tool_schema,
invoke_tool, reload), and serves them over streamable HTTP at
``http://127.0.0.1:<port>/mcp``. A backend is started once, on first use, and
shared by every session that connects — nothing is spawned per client session.

The only way to run it:

    mcp-compressor --server [--server-port N] [--mcp-config PATH]

Clients connect to the socket; they do not launch it.
"""

import asyncio
import warnings
from typing import Annotated

import typer
import uvicorn
from loguru import logger

from .estate import DEFAULT_MCP_CONFIG, load_servers
from .estate_server import Estate, build_estate_server
from .logging import configure_logging
from .server import create_daemon_app
from .types import CompressionLevel, LogLevel

# uvicorn's websockets implementation uses WebSocketServerProtocol, deprecated in
# websockets 14.0; not actionable from this project.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")

app = typer.Typer(name="MCP Compressor", help="A per-user MCP service that fronts your estate behind four tools.")


@app.command()
def main(
    server_mode: Annotated[
        bool,
        typer.Option(
            ...,
            "--server",
            help="Run the daemon. Required — mcp-compressor is a service, not a per-session process.",
        ),
    ] = False,
    server_port: Annotated[
        int,
        typer.Option(..., "--server-port", help="Port for the daemon to listen on."),
    ] = 9020,
    mcp_config: Annotated[
        str | None,
        typer.Option(
            ...,
            "--mcp-config",
            help=f"Path to the MCP configuration to front. Defaults to {DEFAULT_MCP_CONFIG}.",
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option(..., "--log-level", "-l", help="The logging level.", case_sensitive=False),
    ] = LogLevel.WARNING,
):
    """Run the MCP Compressor daemon fronting the estate at /mcp."""
    configure_logging(log_level)

    if not server_mode:
        raise typer.BadParameter(
            "mcp-compressor is a service. Run it with --server (optionally "
            "--server-port and --mcp-config); clients then connect to "
            "http://127.0.0.1:9020/mcp.",
            param_hint="'--server'",
        )

    asyncio.run(_run_daemon(server_port=server_port, mcp_config=mcp_config, log_level=log_level))


async def _run_daemon(server_port: int, mcp_config: str | None, log_level: LogLevel) -> None:
    """Serve the estate over streamable HTTP, one estate for the whole daemon.

    Every session that connects to /mcp shares these backends: the point of the
    service is that a backend runs once per user, not once per session. Backends
    are still lazy — a spec is a few strings until a tool is invoked
    (estate_server.EstateBackend).
    """
    config_path = mcp_config or DEFAULT_MCP_CONFIG
    specs = load_servers(config_path)
    logger.info(f"Estate: {len(specs)} server(s) from {config_path}")

    estate = Estate(specs, compression_level=CompressionLevel.MAX, config_path=config_path)
    estate_app = build_estate_server(estate).http_app(path="/mcp", transport="streamable-http")
    app_ = create_daemon_app(estate_app)

    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app_, host="127.0.0.1", port=server_port, log_level=log_level.value)
    )
    logger.info(f"MCP Compressor daemon on http://127.0.0.1:{server_port} (estate at /mcp)")
    try:
        # Do NOT install our own signal handlers: uvicorn installs its own when
        # serve() starts, setting should_exit=True so serve() returns cleanly and
        # the finally below tears down the estate's stdio backends.
        await uvicorn_server.serve()
    finally:
        logger.info("Shutting down daemon")
        await estate.close()


def entrypoint() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    entrypoint()
