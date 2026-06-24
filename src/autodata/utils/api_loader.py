"""Safe API configuration loader.

Loads API keys and model endpoints from local files or environment variables.
Never logs, prints, or exposes secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Xiaomi MiMo API ──────────────────────────────────────────────────

@dataclass
class XiaomiConfig:
    """Xiaomi MiMo API configuration (secrets never stringified)."""
    api_key: str
    openai_base_url: str
    anthropic_base_url: str
    default_model: str = "mimo-v2.5-pro"


def load_xiaomi_config(
    config_path: Optional[str] = None,
    use_key2: bool = False,
) -> XiaomiConfig:
    """Load Xiaomi MiMo API configuration.

    Priority:
    1. Environment variables (MIMO_API_KEY, MIMO_OPENAI_URL, MIMO_ANTHROPIC_URL)
    2. Local config file (LLM_API/xiaomi_llm_api.txt)
    3. Raise if neither source provides required values.

    If use_key2=True, loads API_KEY2/OpenAI_URL2 instead.

    The config file format is KEY:VALUE per line, e.g.:
        API_KEY: sk-xxx
        OpenAI_URL: https://api.mimo-v2.com/v1
        Anthropic_URL: https://api.mimo-v2.com/anthropic
        API_KEY2: sk-yyy
        OpenAI_URL2: https://api2.mimo-v2.com/v1
    """
    # Try environment variables first
    api_key = os.environ.get("MIMO_API_KEY2" if use_key2 else "MIMO_API_KEY")
    openai_url = os.environ.get("MIMO_OPENAI_URL2" if use_key2 else "MIMO_OPENAI_URL")
    anthropic_url = os.environ.get("MIMO_ANTHROPIC_URL2" if use_key2 else "MIMO_ANTHROPIC_URL")

    # Fall back to file if env vars are missing
    if not api_key or not openai_url:
        if config_path is None:
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[3]
            # Try xiaomi_llm_api.txt first, then llm_api.txt
            xiaomi_path = project_root / "LLM_API" / "xiaomi_llm_api.txt"
            unified_path = project_root / "LLM_API" / "llm_api.txt"
            if xiaomi_path.exists():
                config_path = str(xiaomi_path)
            elif unified_path.exists():
                config_path = str(unified_path)
            else:
                config_path = str(xiaomi_path)  # Default for error message

        config_path = Path(config_path)
        if config_path.exists():
            file_vals = _parse_key_value_file(config_path)
            if use_key2:
                api_key = api_key or file_vals.get("API_KEY2")
                openai_url = openai_url or file_vals.get("OpenAI_URL2")
            else:
                # Try both with and without XIAOMI_ prefix
                api_key = (api_key or file_vals.get("API_KEY")
                           or file_vals.get("XIAOMI_API_KEY"))
                openai_url = (openai_url or file_vals.get("OpenAI_URL")
                              or file_vals.get("XIAOMI_OpenAI_URL"))
                anthropic_url = (anthropic_url or file_vals.get("Anthropic_URL")
                                 or file_vals.get("XIAOMI_Anthropic_URL"))

    if not api_key:
        raise ValueError(
            "Xiaomi API key not found. Set MIMO_API_KEY env var or "
            "provide LLM_API/xiaomi_llm_api.txt"
        )
    if not openai_url:
        openai_url = "https://api.mimo-v2.com/v1"
    if not anthropic_url:
        anthropic_url = "https://api.mimo-v2.com/anthropic"

    return XiaomiConfig(
        api_key=api_key,
        openai_base_url=openai_url,
        anthropic_base_url=anthropic_url,
    )


# ── Baseline Model APIs (env_llm.txt) ────────────────────────────────

@dataclass
class BaselineModelConfig:
    """Configuration for a single baseline model."""
    name: str
    display_name: str
    model: str
    api_key: str
    base_url: str
    use: str = "langchain_openai:ChatOpenAI"
    supports_thinking: bool = False


def load_baseline_configs(
    config_path: Optional[str] = None,
) -> list[BaselineModelConfig]:
    """Load baseline model configurations from env_llm.txt.

    The file format is YAML-like with entries like:
        - name: deepseek-v3.2
          display_name: DeepSeek V3.2
          use: langchain_openai:ChatOpenAI
          model: deepseek-v3.2
          api_key: sk-xxx
          base_url: https://cloud.infini-ai.com/maas/coding/v1

    Returns a list of BaselineModelConfig. Secrets are never logged.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parents[3]
        config_path = str(project_root / "LLM_API" / "env_llm.txt")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Baseline config not found: {config_path}")

    return _parse_env_llm_file(config_path)


# ── Internal parsers (never expose secrets) ──────────────────────────

def _parse_key_value_file(path: Path) -> dict[str, str]:
    """Parse a KEY:VALUE file into a dict. Values are stripped."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
    return result


def _parse_env_llm_file(path: Path) -> list[BaselineModelConfig]:
    """Parse the env_llm.txt YAML-like config.

    Uses a lightweight parser to avoid requiring PyYAML for this
    specific format (indented key-value pairs under dash-prefixed entries).
    """
    # Try yaml first for robustness
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Strip the comment header before parsing
        lines = content.split("\n")
        yaml_lines = [l for l in lines if not l.strip().startswith("#") or l.strip().startswith("-")]
        yaml_content = "\n".join(yaml_lines)
        entries = yaml.safe_load(yaml_content)
        if isinstance(entries, list):
            return [_yaml_entry_to_config(e) for e in entries if isinstance(e, dict)]
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: simple custom parser
    return _parse_env_llm_fallback(path)


def _yaml_entry_to_config(entry: dict) -> BaselineModelConfig:
    """Convert a YAML-parsed dict to BaselineModelConfig."""
    return BaselineModelConfig(
        name=entry.get("name", ""),
        display_name=entry.get("display_name", ""),
        model=entry.get("model", ""),
        api_key=entry.get("api_key", ""),
        base_url=entry.get("base_url", ""),
        use=entry.get("use", "langchain_openai:ChatOpenAI"),
        supports_thinking=entry.get("supports_thinking", False),
    )


def _parse_env_llm_fallback(path: Path) -> list[BaselineModelConfig]:
    """Fallback parser for env_llm.txt when PyYAML is unavailable."""
    configs = []
    current = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- name:"):
                if current:
                    configs.append(BaselineModelConfig(**current))
                current = {"name": stripped[len("- name:"):].strip()}
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Map YAML keys to BaselineModelConfig fields
                key_map = {
                    "name": "name",
                    "display_name": "display_name",
                    "use": "use",
                    "model": "model",
                    "api_key": "api_key",
                    "base_url": "base_url",
                    "supports_thinking": "supports_thinking",
                }
                if key in key_map:
                    bool_fields = {"supports_thinking"}
                    if key in bool_fields:
                        current[key_map[key]] = value.lower() == "true"
                    else:
                        current[key_map[key]] = value

    if current:
        configs.append(BaselineModelConfig(**current))

    return configs


# ── OpenAI-compatible client factory ─────────────────────────────────

def create_xiaomi_openai_client(config: Optional[XiaomiConfig] = None):
    """Create an OpenAI-compatible client for Xiaomi MiMo API.

    Returns an openai.OpenAI client instance.
    """
    from openai import OpenAI

    if config is None:
        config = load_xiaomi_config()

    return OpenAI(
        api_key=config.api_key,
        base_url=config.openai_base_url,
        timeout=120.0,
    )


def create_baseline_openai_client(model_config: BaselineModelConfig):
    """Create an OpenAI-compatible client for a baseline model.

    Returns an openai.OpenAI client instance.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        timeout=120.0,
    )
