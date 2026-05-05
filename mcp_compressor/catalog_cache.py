"""Disk-based catalog cache for lazy-loading mode.

When mcp-compressor runs with ``--lazy``, it stores the backend tool catalog
to disk after the first connection.  On subsequent starts the catalog is served
from this cache so the backend subprocess is not launched until an actual tool
call arrives.

Cache files live in ``~/.cache/mcp-compressor/catalogs/<key>.json``.
The cache key is a hex digest of the backend command/URL plus any include/exclude
filter strings, so different invocation parameters get independent cache entries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CACHE_DIR = Path.home() / ".cache" / "mcp-compressor" / "catalogs"


def make_cache_key(
    command_or_url: str,
    include_tools: list[str] | None = None,
    exclude_tools: list[str] | None = None,
    server_name: str | None = None,
) -> str:
    """Return a short stable hex key for the given backend configuration."""
    parts = [command_or_url]
    if server_name:
        parts.append("name:" + server_name)
    if include_tools:
        parts.append("include:" + ",".join(sorted(include_tools)))
    if exclude_tools:
        parts.append("exclude:" + ",".join(sorted(exclude_tools)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def load(cache_key: str) -> list[dict[str, Any]] | None:
    """Load cached tool schemas; return ``None`` if the cache does not exist or is corrupt."""
    path = _CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return None


def save(cache_key: str, tools: list[dict[str, Any]]) -> None:
    """Persist tool schemas to disk, creating the cache directory if needed."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{cache_key}.json"
    path.write_text(json.dumps(tools, indent=2))


def clear(cache_key: str) -> None:
    """Delete the cache file for the given key (no-op if it does not exist)."""
    path = _CACHE_DIR / f"{cache_key}.json"
    path.unlink(missing_ok=True)
