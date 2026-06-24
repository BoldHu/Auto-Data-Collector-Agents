"""Safe I/O utilities — atomic writes, structured JSON/JSONL/YAML handling.

Provides:
- Atomic file writes (write to temp, then rename) to prevent corruption
- Safe JSON/JSONL/YAML read and write
- Path resolution relative to project root
- File existence checks and directory creation
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


# ── Project root ───────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Resolve the project root directory.

    Walks up from src/autodata/utils/io_utils.py to find the project root.
    """
    # This file is at: project_root/src/autodata/utils/io_utils.py
    return Path(__file__).resolve().parents[3]


# ── Atomic writes ──────────────────────────────────────────────────────

def atomic_write_text(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Write text content atomically (temp file + rename).

    Prevents partial writes on crash/interrupt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".tmp_",
        suffix=path.suffix,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | Path,
    data: Any,
    encoding: str = "utf-8",
    indent: int = 2,
) -> None:
    """Write JSON data atomically."""
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, content, encoding=encoding)


def atomic_write_jsonl(
    path: str | Path,
    records: list[dict],
    encoding: str = "utf-8",
) -> None:
    """Write JSONL data atomically (one JSON object per line)."""
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    content = "\n".join(lines) + "\n"
    atomic_write_text(path, content, encoding=encoding)


# ── Safe reads ────────────────────────────────────────────────────────

def safe_read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    """Read a JSON file safely. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def safe_read_jsonl(path: str | Path, encoding: str = "utf-8") -> list[dict]:
    """Read a JSONL file safely. Returns empty list if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines
    return records


def safe_read_yaml(path: str | Path, encoding: str = "utf-8") -> Any:
    """Read a YAML file safely. Returns None if file doesn't exist or yaml not available."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        import yaml
        with open(path, "r", encoding=encoding) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML is required for YAML file operations. Install it with: pip install pyyaml")


# ── Append to JSONL ───────────────────────────────────────────────────

def append_jsonl_record(
    path: str | Path,
    record: dict,
    encoding: str = "utf-8",
) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding=encoding) as f:
        f.write(line)


# ── Directory helpers ─────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path