#!/usr/bin/env python3
"""Drive the estate front-end the way a client would.

Starts `mcp-compressor --mcp-config <config>` over stdio and exercises the four
tools against real backends, asserting the properties the whole arrangement
exists for:

  four tools    the client's tool array is four entries, not four per server
  catalog       names only, no schemas, and no backend started to produce it
  schema        the real API of one tool, on demand
  invoke        the call actually reaches the backend
  unindexed     a configured server that cannot start is reported, not hidden

Usage:
    scripts/verify-estate.py [--config PATH]

With no --config it builds a throwaway one containing a working backend and a
deliberately broken one, so the check is self-contained.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402

WORKING_BACKEND = '''
import sys
from fastmcp import FastMCP

mcp = FastMCP(name="probe-backend")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool()
def shout(text: str) -> str:
    """Return the text uppercased."""
    return text.upper()


if __name__ == "__main__":
    mcp.run()
'''


class Failure(Exception):
    pass


def build_throwaway_config(workdir: Path) -> Path:
    backend = workdir / "probe_backend.py"
    backend.write_text(WORKING_BACKEND)
    config = workdir / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "probe": {"command": sys.executable, "args": [str(backend)]},
                    # Never starts. Must be reported, not silently omitted.
                    "broken": {"command": "/nonexistent/binary-that-cannot-run"},
                    # The front-end's own entry: reading this back must not
                    # make it wrap itself.
                    "estate": {
                        "command": "mcp-compressor",
                        "args": ["--mcp-config", str(config)],
                    },
                }
            },
            indent=2,
        )
    )
    return config


async def run(config: Path, cache_dir: Path) -> int:
    env = dict(os.environ)
    env["HOME"] = str(cache_dir)  # catalogs land in a throwaway ~/.cache
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "mcp_compressor.main", "--mcp-config", str(config), "-c", "max"],
        env=env,
        cwd=str(REPO),
    )

    async with Client(transport) as client:
        names = sorted(t.name for t in await client.list_tools())
        if names != ["catalog", "get_tool_schema", "invoke_tool", "reload"]:
            raise Failure(f"expected four tools, got {names}")
        print(f"  four tools: {', '.join(names)}")

        # First catalog: the working backend has never run, so it is unindexed
        # too. Invoking a tool is what indexes it.
        first = (await client.call_tool("catalog", {})).content[0].text
        if "probe" not in first or "broken" not in first:
            raise Failure(f"catalog does not mention both servers:\n{first}")
        if "Configured but never indexed" not in first:
            raise Failure(f"catalog hides servers that never started:\n{first}")
        print("  catalog reports servers that have never indexed")

        result = await client.call_tool(
            "invoke_tool", {"server": "probe", "tool": "add", "tool_input": {"a": 2, "b": 3}}
        )
        text = result.content[0].text
        if "5" not in text:
            raise Failure(f"invoke_tool did not reach the backend: {text!r}")
        print("  invoke_tool reaches the backend (2 + 3 = 5)")

        second = (await client.call_tool("catalog", {})).content[0].text
        if "add" not in second or "shout" not in second:
            raise Failure(f"catalog does not list the backend's tools:\n{second}")
        if "Add two integers" in second:
            raise Failure("catalog leaked descriptions; it must be names only")
        print("  catalog lists names only, no descriptions")
        if "broken" not in second.split("Configured but never indexed")[-1]:
            raise Failure("a server that cannot start stopped being reported")
        print("  a server that cannot start is still reported")

        schema = (await client.call_tool("get_tool_schema", {"server": "probe", "tool": "add"})).content[0].text
        if '"a"' not in schema or '"b"' not in schema:
            raise Failure(f"get_tool_schema did not return the real schema:\n{schema}")
        print("  get_tool_schema returns the real API")

        unknown = await client.call_tool(
            "catalog", {"server": "prob"}, raise_on_error=False
        )
        message = str(unknown.content[0].text if unknown.content else "")
        if "probe" not in message:
            raise Failure(f"an unknown server name must list the real ones, got: {message}")
        print("  an unknown server name is answered with the real ones")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    scratch = Path.home() / "scratch" / "mcp-compressor"
    scratch.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="estate-verify-", dir=scratch))
    try:
        config = Path(args.config) if args.config else build_throwaway_config(workdir)
        cache = workdir / "home"
        cache.mkdir()
        print(f"estate over {config}:")
        rc = asyncio.run(run(config, cache))
    except Failure as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        args.keep = True
        return 1
    else:
        print("\nPASS")
        return rc
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"workdir kept at {workdir}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
