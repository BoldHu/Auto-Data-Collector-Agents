"""Unit tests for baseline model loader."""

import pytest

from src.autodata.utils.api_loader import BaselineModelConfig, load_baseline_configs
from src.autodata.utils.baseline_model_loader import (
    BaselineResponse,
    BaselineModelRunner,
    load_baseline_models,
)


# ── BaselineResponse tests ────────────────────────────────────────────

class TestBaselineResponse:
    def test_response_creation(self):
        resp = BaselineResponse(
            model_name="test-model",
            content="Test response",
            usage={"total_tokens": 100},
            latency_ms=150.0,
        )
        assert resp.model_name == "test-model"
        assert resp.content == "Test response"
        assert resp.total_tokens == 100
        assert resp.latency_ms == 150.0

    def test_response_defaults(self):
        resp = BaselineResponse(
            model_name="test",
            content="OK",
        )
        assert resp.reasoning is None
        assert resp.finish_reason is None
        assert resp.total_tokens == 0


# ── BaselineModelRunner tests ─────────────────────────────────────────

class TestBaselineModelRunner:
    def test_runner_creation(self):
        """Test creating a runner for a baseline model."""
        models = load_baseline_configs()
        if models:
            runner = BaselineModelRunner(models[0])
            assert runner.model_config.name == models[0].name
            assert runner.display_name == models[0].display_name

    def test_runner_supports_thinking_property(self):
        """Test supports_thinking property."""
        models = load_baseline_configs()
        for model in models:
            runner = BaselineModelRunner(model)
            assert runner.supports_thinking == model.supports_thinking

    def test_runner_stats(self):
        """Test initial stats."""
        models = load_baseline_configs()
        if models:
            runner = BaselineModelRunner(models[0])
            assert runner.call_count == 0
            assert runner.total_tokens_used == 0

    def test_thinking_raises_for_non_thinking_model(self):
        """Test that invoke with thinking=True raises for non-thinking models."""
        models = load_baseline_configs()
        non_thinking = [m for m in models if not m.supports_thinking]
        if non_thinking:
            runner = BaselineModelRunner(non_thinking[0])
            with pytest.raises(ValueError, match="does not support thinking"):
                runner.invoke("test prompt", thinking=True)


# ── Convenience function tests ────────────────────────────────────────

class TestConvenienceFunctions:
    def test_load_baseline_models(self):
        """Test that load_baseline_models returns configs."""
        models = load_baseline_models()
        assert isinstance(models, list)
        assert len(models) >= 5

    def test_create_runners(self):
        """Test creating runners for all baseline models."""
        models = load_baseline_models()
        runners = []
        for model in models:
            try:
                runner = BaselineModelRunner(model)
                runners.append(runner)
            except Exception:
                pass  # Some models might fail client creation (missing langchain)
        # At least some runners should be created
        assert len(runners) >= 1