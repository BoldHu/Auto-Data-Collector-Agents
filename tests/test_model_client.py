"""Unit tests for model client module."""

import json
import time

import pytest

from src.autodata.utils.model_client import ChatResponse, XiaomiModelClient


# ── ChatResponse tests ────────────────────────────────────────────────

class TestChatResponse:
    def test_response_creation(self):
        resp = ChatResponse(
            response_id="test_id",
            model="mimo-v2.5-pro",
            content="Hello, I am MiMo.",
            reasoning="User asked me to introduce myself.",
            usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        )
        assert resp.response_id == "test_id"
        assert resp.model == "mimo-v2.5-pro"
        assert resp.content == "Hello, I am MiMo."
        assert resp.reasoning == "User asked me to introduce myself."
        assert resp.total_tokens == 80
        assert resp.prompt_tokens == 50
        assert resp.completion_tokens == 30

    def test_response_defaults(self):
        resp = ChatResponse(
            response_id="test_id",
            model="mimo-v2.5-pro",
            content="Test",
        )
        assert resp.reasoning is None
        assert resp.tool_calls is None
        assert resp.finish_reason is None
        assert resp.total_tokens == 0


# ── XiaomiModelClient tests (mock-based, no real API calls) ──────────

class TestXiaomiModelClient:
    def test_client_creation(self):
        """Test that client can be created without errors."""
        # This uses the actual config file — should succeed if config is present
        client = XiaomiModelClient()
        assert client.default_model == "mimo-v2.5-pro"
        assert client.default_max_tokens == 4096
        assert client.default_temperature == 1.0
        assert client.default_top_p == 0.95

    def test_client_default_model_is_v25_pro(self):
        """Verify that the default model is mimo-v2.5-pro."""
        client = XiaomiModelClient()
        assert client.default_model == "mimo-v2.5-pro"

    def test_client_stats_tracking(self):
        """Test call count and token tracking."""
        client = XiaomiModelClient()
        assert client.call_count == 0
        assert client.total_tokens_used == 0
        client.reset_stats()
        assert client.call_count == 0
        assert client.total_tokens_used == 0

    def test_client_custom_model(self):
        """Test creating a client with a custom default model."""
        client = XiaomiModelClient(default_model="mimo-v2.5")
        assert client.default_model == "mimo-v2.5"

    def test_client_custom_params(self):
        """Test creating a client with custom parameters."""
        client = XiaomiModelClient(
            default_max_tokens=8192,
            default_temperature=0.7,
            default_top_p=0.9,
        )
        assert client.default_max_tokens == 8192
        assert client.default_temperature == 0.7
        assert client.default_top_p == 0.9