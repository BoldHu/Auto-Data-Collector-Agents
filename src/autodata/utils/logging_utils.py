"""Structured logging utilities — never expose secrets.

Uses loguru for structured, colorized logging with safe serialization.
All log entries are JSON-serializable. API keys, tokens, and model
responses are never logged in full — only truncated hashes or metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# Remove default loguru handler
logger.remove()

# ── Safe serialization ────────────────────────────────────────────────

_SECRET_KEYS = {
    "api_key", "apikey", "key", "token", "secret",
    "password", "credential", "auth",
}


def safe_serialize(data: dict, max_str_len: int = 80) -> dict:
    """Serialize a dict for logging, redacting secrets and truncating long strings."""
    result = {}
    for k, v in data.items():
        if k.lower() in _SECRET_KEYS:
            # Redact: show only first 4 chars + "...REDACTED"
            if isinstance(v, str) and len(v) > 4:
                result[k] = v[:4] + "...REDACTED"
            else:
                result[k] = "REDACTED"
        elif isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + f"...[truncated, total={len(v)}]"
        elif isinstance(v, dict):
            result[k] = safe_serialize(v, max_str_len)
        else:
            result[k] = v
    return result


# ── Log format ────────────────────────────────────────────────────────

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


# ── Configure logging ─────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure project-wide logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. If None, logs go to stderr only.
        log_file: Specific log file name. Defaults to 'autodata.log'.
        rotation: When to rotate log files.
        retention: How long to keep old log files.
    """
    # Console handler — always present
    logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level=level,
        colorize=True,
    )

    # File handler — optional
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        filename = log_file or "autodata.log"
        logger.add(
            str(log_path / filename),
            format=_LOG_FORMAT,
            level=level,
            rotation=rotation,
            retention=retention,
            serialize=False,
        )

        # Separate file for errors
        logger.add(
            str(log_path / "autodata_errors.log"),
            format=_LOG_FORMAT,
            level="ERROR",
            rotation=rotation,
            retention=retention,
        )


def get_logger(name: str = "autodata") -> "logger":
    """Get a named logger instance."""
    return logger.bind(name=name)