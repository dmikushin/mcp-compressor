# mcp-compressor

A per-user MCP service that fronts your whole MCP estate behind four tools,
cutting the tokens that tool descriptions cost on every request.

## Why?

MCP servers are popular, but their tool descriptions consume significant tokens
in every LLM request. For example:

- The official **GitHub MCP server** exposes 94 tools consuming **~17,600 tokens**
- The official **Atlassian MCP server** consumes **~10,000 tokens**

With 30k+ tokens just for tool descriptions, costs can reach **1–10 cents per
request** depending on prompt caching. mcp-compressor replaces the whole estate
with four tools whose descriptions are tiny and never grow with the number of
backends:

- `catalog` — names only, of every tool across every configured server
- `get_tool_schema` — the full schema for one tool, when the model means to call it
- `invoke_tool` — call a tool by name with its input
- `reload` — re-read the configuration and refresh the catalog

Nothing about the estate sits in the prompt until the model asks for it.

## Design: one service per user

mcp-compressor is **one systemd user service**, not a per-session process. It
reads your MCP configuration, starts each backend once — on first use, shared by
every session — and serves the four tools over streamable HTTP at
`http://127.0.0.1:9020/mcp`. MCP clients **connect** to that URL; they do not
launch mcp-compressor.

```
client ──HTTP──▶ mcp-compressor --server :9020 /mcp
                    └─ backends (stdio), one per user, started on first use
```

## Installation

```bash
uv tool install mcp-compressor
# or
pipx install mcp-compressor
```

## Run it as a service

```bash
mkdir -p ~/.config/systemd/user
cp contrib/mcp-compressor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mcp-compressor.service
```

Check it:

```bash
curl http://127.0.0.1:9020/health   # -> ok
```

Or run it in the foreground:

```bash
mcp-compressor --server [--server-port 9020] [--mcp-config ~/.claude.json]
```

That is the only way to run it. There is no per-session wrap mode: point your
client at the socket instead.

## Connect a client

Add one HTTP server to your MCP host's configuration:

```json
{
  "mcpServers": {
    "tools": { "type": "http", "url": "http://127.0.0.1:9020/mcp" }
  }
}
```

The four tools appear as `catalog`, `get_tool_schema`, `invoke_tool`, `reload`.

## Configuration

By default the daemon fronts every server in `~/.claude.json`. Override the path
with `--mcp-config PATH`. Backends are started lazily — a server is a few strings
until one of its tools is invoked — and shared across every session connected to
the daemon.

## How it works

The daemon builds one estate over the configured servers. Each backend is a
stdio MCP server the daemon speaks to through a compressing proxy; the estate
exposes the four wrapper tools and, on `catalog`/`get_tool_schema`, serves names
and schemas from a per-server disk cache so the model can browse without every
backend being started. `reload` re-reads the configuration for the whole daemon.
