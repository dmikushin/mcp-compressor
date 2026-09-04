"""Read a client's MCP configuration and turn it into backend specs.

The compressor used to be attached to one backend, one process per server, with
the client launching all of them. Then only the client knew what the estate was:
the compressor could serve a catalog for a server that had run at least once and
was blind to the rest, so a server that failed to start was silently absent
rather than reported. Reading the client's own configuration puts that knowledge
where the estate now lives.

The shape read here is the one every MCP client writes — a ``mcpServers`` object
mapping a name to a launch spec. It is looked for at the top level and under
``projects.<path>``, because Claude Code keeps per-project servers there.

## Supported configurations

Every external transport a client can configure is fronted, so that the client
keeps exactly one external connection — to this estate — and dials nothing
itself:

- ``{"command", "args"?, "env"?, "cwd"?}`` or ``{"type": "stdio", ...}`` — a
  subprocess. ``command`` is required.
- ``{"type": "http", "url", "headers"?}`` (``"streamable-http"`` is accepted as
  a synonym) — a remote Streamable HTTP server. ``url`` is required.
- ``{"type": "sse", "url", "headers"?}`` — a remote SSE server. ``url`` is
  required.

An entry with an unknown ``type``, a stdio entry with no ``command``, or a
remote entry with no ``url`` is ignored — the estate fronts what it can launch
or dial and is silent about the rest.

## The self-reference trap

Once the estate is fronted by a single compressor, the client's configuration
lists *the compressor itself* as a server. Reading that file back, the compressor
finds its own entry and — without this — launches itself, which reads the file,
which launches itself. The recursion is not subtle when it happens and is
invisible when it does not, so entries whose command is the compressor are
dropped by construction rather than by hoping the name never matches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPRESSOR_COMMAND_NAMES = ("mcp-compressor",)

#: Where a client's MCP configuration lives unless told otherwise. The
#: compressor reads it itself rather than being handed a server list, so that
#: adding a server to the client's config is all it takes for the estate to
#: include it — no second place to keep in step.
DEFAULT_MCP_CONFIG = "~/.claude.json"


class ConfigError(RuntimeError):
    """The configuration could not be read. The message names the file."""


@dataclass(frozen=True)
class ServerSpec:
    """One backend the compressor is responsible for.

    A backend is one of three transports, discriminated by ``transport``:

    - ``"stdio"`` — a local subprocess. ``command`` is required; ``args``,
      ``env`` and ``cwd`` apply. This is the default when a configuration entry
      names no ``type``.
    - ``"http"`` — a remote Streamable HTTP server. ``url`` is required;
      ``headers`` apply. ``command``/``args``/``cwd`` do not.
    - ``"sse"`` — a remote SSE server. ``url`` is required; ``headers`` apply.

    ``source`` is the stable identity used for the catalog cache key and log
    lines: the argv for stdio, the URL for a remote. It must not change between
    runs for the same configured backend, or its cached catalog is orphaned.
    """

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_remote(self) -> bool:
        return self.transport in ("http", "sse")

    @property
    def argv(self) -> list[str]:
        return [self.command or "", *self.args]

    @property
    def source(self) -> str:
        """Stable identity for logs: the command line or the URL, no secrets."""
        return self.url if self.is_remote else " ".join(self.argv)

    @property
    def credentials_fingerprint(self) -> str:
        """A digest of ``env`` and ``headers`` — the part of a spec that
        changes a backend's tool list without changing its command.

        The catalog cache used to be keyed on ``source`` alone, so replacing a
        token in the configuration left the catalog the old token had written
        on disk, and ``catalog`` kept serving it until someone happened to
        ``reload`` that server. Measured on this machine: a GitHub PAT with
        org access lists 41 tools, one without lists 38. A digest, not the
        values: the key is a filename under ~/.cache and must not carry the
        secret it stands for. Empty when there is nothing to fingerprint, so
        specs without credentials keep the key they had.
        """
        if not self.env and not self.headers:
            return ""
        material = "\x00".join(
            f"{k}={v}" for k, v in sorted({**self.env, **{"h:" + k: v for k, v in self.headers.items()}}.items())
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]


def _is_compressor(command: str) -> bool:
    return Path(command).name in COMPRESSOR_COMMAND_NAMES


def _unwrap(command: str, args: list[str]) -> tuple[str, list[str], dict[str, str]] | None:
    """Return the backend a compressor invocation would have launched.

    A configuration written for the old one-process-per-server arrangement wraps
    each entry in ``mcp-compressor ... -- <backend>``. Those entries describe a
    real backend and must not be lost; unwrapping them is what lets a client
    migrate without rewriting its configuration first.

    ``None`` means the entry is the estate front-end itself — a compressor
    invocation with no backend behind it. That is the self-reference, and it is
    the expected content of a migrated configuration, not an error.
    """
    env: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--lazy", "--toonify"):
            i += 1
        elif arg in ("-n", "--server-name", "-c", "--compression-level", "-l", "--log-level"):
            i += 2
        elif arg in ("-e", "--env"):
            key, _, value = args[i + 1].partition("=")
            env[key] = value
            i += 2
        elif arg in ("--mcp-config", "--config"):
            # The front-end's own flag. Nothing is wrapped.
            return None
        elif arg == "--":
            i += 1
            break
        else:
            break
    rest = args[i:]
    if not rest:
        return None
    return rest[0], rest[1:], env


def _expand(value: str) -> str:
    return os.path.expandvars(value)


def _headers(raw: dict[str, Any]) -> dict[str, str]:
    return {str(k): _expand(str(v)) for k, v in (raw.get("headers") or {}).items()}


def _spec_from(name: str, raw: dict[str, Any]) -> ServerSpec | None:
    """Build a spec, or ``None`` when the entry is not a launchable backend.

    Three transports are recognised, by the ``type`` the client writes:

    - absent or ``"stdio"`` → a subprocess; requires ``command``.
    - ``"http"`` (or ``"streamable-http"``) → a remote; requires ``url``.
    - ``"sse"`` → a remote; requires ``url``.

    Anything else — an unknown ``type``, a stdio entry with no command, a remote
    entry with no url — is ignored (returns ``None``): the estate fronts what it
    can launch or dial and says nothing about what it cannot, exactly as it did
    when it only understood stdio.
    """
    kind = raw.get("type", "stdio")

    if kind in ("http", "streamable-http", "sse"):
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            return None
        transport = "sse" if kind == "sse" else "http"
        return ServerSpec(
            name=name,
            transport=transport,
            url=url,
            headers=_headers(raw),
        )

    if kind != "stdio":
        return None

    command = raw.get("command")
    if not isinstance(command, str) or not command:
        return None
    args = [str(a) for a in raw.get("args", [])]
    env = {str(k): _expand(str(v)) for k, v in (raw.get("env") or {}).items()}

    if _is_compressor(command):
        unwrapped = _unwrap(command, args)
        if unwrapped is None:
            # The estate front-end reading its own entry. See the module note.
            return None
        command, args, wrapped_env = unwrapped
        # The config's own env wins: it is the more specific statement.
        merged = dict(wrapped_env)
        merged.update(env)
        env = merged

    return ServerSpec(
        name=name, transport="stdio", command=command, args=args, env=env, cwd=raw.get("cwd")
    )


def _server_maps(doc: dict[str, Any], project: str | None) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    top = doc.get("mcpServers")
    if isinstance(top, dict):
        maps.append(top)
    projects = doc.get("projects")
    if isinstance(projects, dict):
        keys = [project] if project is not None else list(projects)
        for key in keys:
            entry = projects.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("mcpServers"), dict):
                maps.append(entry["mcpServers"])
    return maps


def load_servers(path: str | Path, project: str | None = None) -> list[ServerSpec]:
    """Every external server a client configuration declares, sorted by name.

    Sorted because the order of the estate must be a property of the
    configuration and not of dictionary iteration: a catalog that reorders
    between runs is a catalog whose consumers cannot cache it.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"No MCP configuration at {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"{path} does not contain a JSON object")

    specs: dict[str, ServerSpec] = {}
    for server_map in _server_maps(doc, project):
        for name, raw in server_map.items():
            if not isinstance(raw, dict):
                continue
            spec = _spec_from(name, raw)
            if spec is not None:
                specs.setdefault(name, spec)
    return [specs[name] for name in sorted(specs)]
