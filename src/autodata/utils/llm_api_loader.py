"""LLM API loader for Phase 6.5.

Parses LLM_API/llm_api.txt which contains both Xiaomi and Doubao configs.
Supports KEY=VALUE, KEY: VALUE, and plain line formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_API_PATH = PROJECT_ROOT / "LLM_API" / "llm_api.txt"


@dataclass
class XiaomiAPIConfig:
    """Xiaomi API configuration."""
    api_key: str = ""
    openai_url: str = ""
    anthropic_url: str = ""


@dataclass
class DoubaoAPIConfig:
    """Doubao Token Plan API configuration."""
    api_key: str = ""
    openai_url: str = ""
    anthropic_url: str = ""


@dataclass
class UnifiedLLMAPIConfig:
    """Unified LLM API configuration."""
    xiaomi: XiaomiAPIConfig = None
    doubao: DoubaoAPIConfig = None
    xiaomi_found: bool = False
    doubao_found: bool = False

    def __post_init__(self):
        if self.xiaomi is None:
            self.xiaomi = XiaomiAPIConfig()
        if self.doubao is None:
            self.doubao = DoubaoAPIConfig()


def parse_api_file(path: Path = None) -> UnifiedLLMAPIConfig:
    """Parse llm_api.txt and return unified config.

    The file has 6 lines:
    - Lines 1-3: Xiaomi API key and URLs
    - Lines 4-6: Doubao API key and URLs

    Supports formats:
    - KEY=VALUE
    - KEY: VALUE
    - KEY VALUE (plain)
    """
    if path is None:
        path = DEFAULT_API_PATH

    if not path.exists():
        return UnifiedLLMAPIConfig()

    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)

    config = UnifiedLLMAPIConfig()

    # Parse lines into key-value pairs
    xiaomi_section = []
    doubao_section = []

    for line in lines:
        key, value = _parse_line(line)
        if not key:
            continue

        key_lower = key.lower()

        # Classify by key name
        if "xiaomi" in key_lower or "mimo" in key_lower:
            xiaomi_section.append((key_lower, value))
        elif "doubao" in key_lower or "ark" in key_lower:
            doubao_section.append((key_lower, value))
        else:
            # Guess by position: first 3 lines = Xiaomi, last 3 = Doubao
            idx = lines.index(line)
            if idx < 3:
                xiaomi_section.append((key_lower, value))
            else:
                doubao_section.append((key_lower, value))

    # Build Xiaomi config
    for key, value in xiaomi_section:
        if "api_key" in key or "api-key" in key or "apikey" in key:
            config.xiaomi.api_key = value
            config.xiaomi_found = True
        elif "openai" in key and "url" in key:
            config.xiaomi.openai_url = value
        elif "anthropic" in key and "url" in key:
            config.xiaomi.anthropic_url = value
        elif "url" in key and not config.xiaomi.openai_url:
            config.xiaomi.openai_url = value

    # Build Doubao config
    for key, value in doubao_section:
        if "api_key" in key or "api-key" in key or "apikey" in key:
            config.doubao.api_key = value
            config.doubao_found = True
        elif "openai" in key and "url" in key:
            config.doubao.openai_url = value
        elif "anthropic" in key and "url" in key:
            config.doubao.anthropic_url = value
        elif "url" in key and not config.doubao.openai_url:
            config.doubao.openai_url = value

    # Fallback: if no explicit keys found, try positional parsing
    if not config.xiaomi_found and len(lines) >= 3:
        for line in lines[:3]:
            key, value = _parse_line(line)
            if value and _looks_like_api_key(value):
                config.xiaomi.api_key = value
                config.xiaomi_found = True
            elif value and _looks_like_url(value):
                if not config.xiaomi.openai_url:
                    config.xiaomi.openai_url = value

    if not config.doubao_found and len(lines) >= 6:
        for line in lines[3:6]:
            key, value = _parse_line(line)
            if value and _looks_like_api_key(value):
                config.doubao.api_key = value
                config.doubao_found = True
            elif value and _looks_like_url(value):
                if not config.doubao.openai_url:
                    config.doubao.openai_url = value

    return config


def _parse_line(line: str) -> tuple[str, str]:
    """Parse a single line into (key, value)."""
    # KEY=VALUE
    if "=" in line:
        parts = line.split("=", 1)
        return parts[0].strip(), parts[1].strip()

    # KEY: VALUE
    if ":" in line:
        parts = line.split(":", 1)
        return parts[0].strip(), parts[1].strip()

    # Plain line - try to detect
    parts = line.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return "", line


def _looks_like_api_key(value: str) -> bool:
    """Check if value looks like an API key."""
    if not value:
        return False
    # API keys are typically alphanumeric with hyphens, 20+ chars
    if len(value) >= 15 and re.match(r'^[a-zA-Z0-9_-]+$', value):
        return True
    return False


def _looks_like_url(value: str) -> bool:
    """Check if value looks like a URL."""
    return value.startswith("http://") or value.startswith("https://")


def get_sanitized_status(config: UnifiedLLMAPIConfig) -> dict:
    """Get sanitized config status (no secrets)."""
    return {
        "xiaomi_config_found": config.xiaomi_found,
        "doubao_config_found": config.doubao_found,
        "xiaomi_url_found": bool(config.xiaomi.openai_url),
        "doubao_url_found": bool(config.doubao.openai_url),
        "xiaomi_has_key": bool(config.xiaomi.api_key),
        "doubao_has_key": bool(config.doubao.api_key),
        "note": "No secret values exposed",
    }


# Module-level singleton
_config: Optional[UnifiedLLMAPIConfig] = None


def get_llm_config() -> UnifiedLLMAPIConfig:
    """Get or create the unified LLM config singleton."""
    global _config
    if _config is None:
        _config = parse_api_file()
    return _config
