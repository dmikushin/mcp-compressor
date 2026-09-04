"""The daemon's HTTP surface.

When mcp-compressor runs as ``--server`` it is one service per user: it serves
the four-tool estate at ``/mcp`` (see ``estate_server.build_estate_server``) and
a ``/health`` probe, and nothing is spawned per client session. Clients connect
to ``http://127.0.0.1:<port>/mcp`` over streamable HTTP.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


async def _health(request: Request) -> Response:
    return Response("ok", media_type="text/plain")


def create_daemon_app(estate_app: Starlette) -> Starlette:
    """The daemon's full surface: the estate at /mcp, plus /health.

    ``estate_app`` is a FastMCP streamable-HTTP app — it already carries the
    /mcp route and, crucially, the lifespan that runs the MCP session manager's
    task group. The /health route is appended to it rather than the estate being
    mounted into a fresh Starlette, so that lifespan stays the app's own and is
    not silently dropped.
    """
    estate_app.router.routes.append(Route("/health", _health, methods=["GET"]))
    return estate_app
