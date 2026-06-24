"""Unit tests for API loader module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.autodata.utils.api_loader import (
    BaselineModelConfig,
    XiaomiConfig,
    load_baseline_configs,
    load_xiaomi_config,
    _parse_key_value_file,
    _parse_env_llm_fallback,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def xiaomi_config_file(tmp_path):
    """Create a temporary Xiaomi config file."""
    config = tmp_path / "xiaomi_llm_api.txt"
    config.write_text(
        "API_KEY: sk-test-key-123\n"
        "OpenAI_URL: https://api.test.com/v1\n"
        "Anthropic_URL: https://api.test.com/anthropic\n",
        encoding="utf-8",
    )
    return str(config)


@pytest.fixture
def env_llm_file(tmp_path):
    """Create a temporary env_llm.txt file."""
    content = (
        "# Baseline models\n"
        "- name: test-model-1\n"
        "  display_name: Test Model 1\n"
        "  use: langchain_openai:ChatOpenAI\n"
        "  model: test-v1\n"
        "  api_key: sk-test-key\n"
        "  base_url: https://api.test.com/v1\n"
        "\n"
        "- name: test-model-2\n"
        "  display_name: Test Model 2\n"
        "  use: langchain_openai:ChatOpenAI\n"
        "  model: test-v2\n"
        "  api_key: sk-test-key\n"
        "  base_url: https://api.test.com/v1\n"
        "  supports_thinking: true\n"
    )
    config = tmp_path / "env_llm.txt"
    config.write_text(content, encoding="utf-8")
    return str(config)


# ── Xiaomi config tests ───────────────────────────────────────────────

class TestXiaomiConfig:
    def test_load_from_file(self, xiaomi_config_file):
        cfg = load_xiaomi_config(config_path=xiaomi_config_file)
        assert isinstance(cfg, XiaomiConfig)
        assert cfg.api_key == "sk-test-key-123"
        assert cfg.openai_base_url == "https://api.test.com/v1"
        assert cfg.anthropic_base_url == "https://api.test.com/anthropic"
        assert cfg.default_model == "mimo-v2.5-pro"

    def test_load_from_env_vars(self):
        os.environ["MIMO_API_KEY"] = "sk-env-test-key"
        os.environ["MIMO_OPENAI_URL"] = "https://env.test.com/v1"
        os.environ["MIMO_ANTHROPIC_URL"] = "https://env.test.com/anthropic"
        cfg = load_xiaomi_config()
        assert cfg.api_key == "sk-env-test-key"
        assert cfg.openai_base_url == "https://env.test.com/v1"
        # Clean up
        del os.environ["MIMO_API_KEY"]
        del os.environ["MIMO_OPENAI_URL"]
        del os.environ["MIMO_ANTHROPIC_URL"]

    def test_default_urls_when_missing(self, tmp_path):
        config = tmp_path / "xiaomi_llm_api.txt"
        config.write_text("API_KEY: sk-minimal\n", encoding="utf-8")
        cfg = load_xiaomi_config(config_path=str(config))
        assert cfg.openai_base_url == "https://api.mimo-v2.com/v1"
        assert cfg.anthropic_base_url == "https://api.mimo-v2.com/anthropic"

    def test_missing_api_key_raises(self, tmp_path):
        config = tmp_path / "xiaomi_llm_api.txt"
        config.write_text("# empty\n", encoding="utf-8")
        # Remove env var if present
        os.environ.pop("MIMO_API_KEY", None)
        with pytest.raises(ValueError, match="Xiaomi API key not found"):
            load_xiaomi_config(config_path=str(config))

    def test_parse_key_value_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(
            "KEY1: value1\n"
            "KEY2: value with spaces\n"
            "# comment\n"
            "\n"
            "KEY3:value3\n",
            encoding="utf-8",
        )
        result = _parse_key_value_file(f)
        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value with spaces"
        assert result["KEY3"] == "value3"
        assert "KEY4" not in result


# ── Baseline config tests ─────────────────────────────────────────────

class TestBaselineConfig:
    def test_load_baseline_configs(self, env_llm_file):
        configs = load_baseline_configs(config_path=env_llm_file)
        assert isinstance(configs, list)
        assert len(configs) >= 2
        assert all(isinstance(c, BaselineModelConfig) for c in configs)

    def test_baseline_model_fields(self, env_llm_file):
        configs = load_baseline_configs(config_path=env_llm_file)
        c0 = configs[0]
        assert c0.name == "test-model-1"
        assert c0.display_name == "Test Model 1"
        assert c0.model == "test-v1"
        assert c0.api_key == "sk-test-key"
        assert c0.base_url == "https://api.test.com/v1"
        assert c0.use == "langchain_openai:ChatOpenAI"
        assert c0.supports_thinking == False

    def test_thinking_model_flag(self, env_llm_file):
        configs = load_baseline_configs(config_path=env_llm_file)
        # test-model-2 should have supports_thinking=True
        thinking_models = [c for c in configs if c.supports_thinking]
        assert len(thinking_models) >= 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline_configs(config_path=str(tmp_path / "nonexistent.txt"))

    def test_parse_env_llm_fallback(self, env_llm_file):
        configs = _parse_env_llm_fallback(Path(env_llm_file))
        assert len(configs) >= 2
        assert all(isinstance(c, BaselineModelConfig) for c in configs)