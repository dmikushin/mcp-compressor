"""Logging setup for the MCP Compressor daemon.

Configures loguru as the logging backend and intercepts stdlib logging from
upstream libraries.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger
from loguru_logging_intercept import setup_loguru_logging_intercept

if TYPE_CHECKING:
    from mcp_compressor.types import LogLevel


__all__ = ["configure_logging"]


def configure_logging(log_level: LogLevel) -> None:
    """Configure loguru and intercept upstream stdlib loggers.

    Should be called once at startup before any I/O begins.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.value.upper())
    setup_loguru_logging_intercept(modules=("fastmcp",))
