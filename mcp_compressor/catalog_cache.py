"""Disk-based catalog cache for lazy-loading mode.

When mcp-compressor runs with ``--lazy``, it stores the backend tool catalog
to disk after the first connection.  On subsequent starts the catalog is served
from this cache so the backend subprocess is not launched until an actual tool
call arrives.

Cache files live in ``~/.cache/mcp-compressor/catalogs/<key>.json``.
The cache key is a hex digest of the backend command/URL plus any include/exclude
filter strings, so different invocation parameters get independent cache entries.

Each file records which server it came from.  The key alone cannot answer that —
it is a digest — so without the name in the file a reader holding only the cache
directory can list tools but cannot say whose they are.  That is precisely what
a client needs in order to present one catalog across every wrapped server
without starting any of them, so the name travels with the data.

Files written by earlier versions are a bare JSON array with no envelope.  They
are still read, and reported with ``server=None`` rather than skipped: a catalog
that silently omitted servers would be worse than one that admits it cannot name
them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_DIR = Path.home() / ".cache" / "mcp-compressor" / "catalogs"


def make_cache_key(
    command_or_url: str,
    include_tools: list[str] | None = None,
    exclude_tools: list[str] | None = None,
    server_name: str | None = None,
    credentials_fingerprint: str = "",
) -> str:
    """Return a short stable hex key for the given backend configuration.

    ``credentials_fingerprint`` is a digest of the spec's env and headers
    (ServerSpec.credentials_fingerprint): a backend's tool list depends on
    what it is authorised to do, so a changed token is a different catalog
    and must be a different key. Empty keeps the key of a spec that has no
    credentials.
    """
    parts = [command_or_url]
    if server_name:
        parts.append("name:" + server_name)
    if credentials_fingerprint:
        parts.append("cred:" + credentials_fingerprint)
    if include_tools:
        parts.append("include:" + ",".join(sorted(include_tools)))
    if exclude_tools:
        parts.append("exclude:" + ",".join(sorted(exclude_tools)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class CatalogEntry:
    """One cached server catalog, as read off disk."""

    key: str
    server: str | None
    command: str | None
    tools: list[dict[str, Any]]

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools if isinstance(t, dict) and "name" in t]


def _parse(key: str, data: Any) -> CatalogEntry | None:
    if isinstance(data, list):
        # Legacy: a bare tool array, written before the envelope existed.
        return CatalogEntry(key=key, server=None, command=None, tools=data)
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        return CatalogEntry(
            key=key,
            server=data.get("server"),
            command=data.get("command"),
            tools=data["tools"],
        )
    return None


def load(cache_key: str) -> list[dict[str, Any]] | None:
    """Load cached tool schemas; return ``None`` if the cache does not exist or is corrupt."""
    entry = load_entry(cache_key)
    return entry.tools if entry else None


def load_entry(cache_key: str) -> CatalogEntry | None:
    """Load a cached catalog with its provenance, or ``None`` if absent or corrupt."""
    path = _CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        return _parse(cache_key, json.loads(path.read_text()))
    except Exception:
        return None


def entries() -> list[CatalogEntry]:
    """Every cached catalog on disk, sorted by server name so the order of a
    listing is a property of the configuration and not of the filesystem."""
    if not _CACHE_DIR.exists():
        return []
    found: list[CatalogEntry] = []
    for path in _CACHE_DIR.glob("*.json"):
        entry = load_entry(path.stem)
        if entry is not None:
            found.append(entry)
    return sorted(found, key=lambda e: (e.server is None, e.server or "", e.key))


def save(
    cache_key: str,
    tools: list[dict[str, Any]],
    *,
    server: str | None = None,
    command: str | None = None,
) -> None:
    """Persist tool schemas to disk, creating the cache directory if needed."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{cache_key}.json"
    payload = {"server": server, "command": command, "tools": tools}
    path.write_text(json.dumps(payload, indent=2))


def clear(cache_key: str) -> None:
    """Delete the cache file for the given key (no-op if it does not exist)."""
    path = _CACHE_DIR / f"{cache_key}.json"
    path.unlink(missing_ok=True)
