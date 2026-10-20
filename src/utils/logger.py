"""Structured logging for the CFCL SEC Optimization pipeline.

Provides a preconfigured logger that writes:
- **JSON-formatted** log records to a rotating file (for machine parsing)
- **Human-readable** coloured records to the console (for development)

Usage
-----
::

    from src.utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Training started", extra={"model": "xgboost", "r2": 0.91})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR: Path = _PROJECT_ROOT / "outputs" / "logs"
_DEFAULT_LOG_FILE: str = "sec_pipeline.log"

_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT: int = 5               # Keep 5 rotated backups


# ---------------------------------------------------------------------------
# JSON Formatter (for file output)
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Each record includes:
    - ``timestamp`` (ISO 8601, UTC)
    - ``level``
    - ``logger``
    - ``message``
    - ``module``, ``function``, ``line``
    - Any extra fields passed via ``extra={}``
    """

    # Keys injected by the logging framework that we skip when
    # extracting user-supplied extras.
    _BUILTIN_ATTRS: frozenset[str] = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge user-supplied extras
        for key, value in record.__dict__.items():
            if key not in self._BUILTIN_ATTRS and key not in entry:
                entry[key] = value

        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Console Formatter (coloured, human-readable)
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """Coloured console formatter for development use.

    Uses ANSI escape codes when the stream supports it.
    """

    _COLORS: dict[int, str] = {
        logging.DEBUG:    "\033[36m",     # Cyan
        logging.INFO:     "\033[32m",     # Green
        logging.WARNING:  "\033[33m",     # Yellow
        logging.ERROR:    "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",   # Bold Red
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        name = record.name.rsplit(".", maxsplit=1)[-1]  # Short module name
        msg = record.getMessage()

        if self.use_color:
            color = self._COLORS.get(record.levelno, "")
            line = f"{color}{ts} | {level}{self._RESET} {name}: {msg}"
        else:
            line = f"{ts} | {level} {name}: {msg}"

        if record.exc_info and record.exc_info[0] is not None:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Create or retrieve a named pipeline logger.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : int
        Minimum logging level (default ``INFO``).
    log_dir : Path, optional
        Directory for the log file. Defaults to ``outputs/logs/``.
    log_file : str, optional
        Log filename. Defaults to ``sec_pipeline.log``.
    console : bool
        If ``True`` (default), attach a coloured console handler.
    file_logging : bool
        If ``True`` (default), attach a rotating file handler.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate output from root logger

    # --- Console handler ---
    if console:
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(ConsoleFormatter(use_color=use_color))
        logger.addHandler(console_handler)

    # --- Rotating file handler (JSON) ---
    if file_logging:
        target_dir = log_dir or _DEFAULT_LOG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / (log_file or _DEFAULT_LOG_FILE)

        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger


def get_null_logger(name: str = "null") -> logging.Logger:
    """Return a logger that discards all output (useful for tests)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    return logger
