"""
Lightweight logging configuration for the pipeline.

Uses only the standard-library ``logging`` module.  No external dependency
required.

Usage::

    from src.utils.logging import configure_logging, get_logger

    configure_logging(level="DEBUG")
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
#  Default log format
# ---------------------------------------------------------------------------
_DEFAULT_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def configure_logging(
    level: str | int = "INFO",
    log_format: Optional[str] = None,
) -> None:
    """Configure the root logger with a stream handler writing to stdout.

    Parameters
    ----------
    level : str or int
        Logging threshold (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``logging.DEBUG``).
    log_format : str, optional
        Override the default log format string.  The default is::

            %(asctime)s  %(levelname)-8s  %(name)s  %(message)s

    Notes
    -----
    Calling this function more than once **replaces** the existing handlers
    so that repeated calls do not produce duplicate log lines.
    """
    fmt = log_format or _DEFAULT_FORMAT

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with the given *name*.

    The returned logger inherits the configuration applied by
    :func:`configure_logging`.

    Example::

        logger = get_logger(__name__)
        logger.info("…")
    """
    return logging.getLogger(name)
