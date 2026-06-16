"""Centralized, pretty logging using Rich.

Every module should obtain its logger via :func:`get_logger`. Output goes to the
console with colors and to an optional rotating file under ``logs/``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str | None = "logs/jarvis.log") -> None:
    """Configure root logging once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    ]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
