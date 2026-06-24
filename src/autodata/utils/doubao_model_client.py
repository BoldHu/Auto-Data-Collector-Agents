"""Doubao Token Plan model client for Phase 6.5.

Uses the Doubao API from llm_api.txt (OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import openai

from src.autodata.utils.llm_api_loader import get_llm_config


@dataclass
class DoubaoResponse:
    """Response from Doubao API."""
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    raw_response_metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class DoubaoModelClient:
    """Client for Doubao Token Plan models.

    Uses OpenAI-compatible API endpoint.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        default_model: str = "doubao-seed-2.0-lite",
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        config = get_llm_config()

        self.api_key = api_key or config.doubao.api_key
        # Doubao Coding Plan uses /api/coding/v3 for OpenAI-compatible API
        raw_url = config.doubao.openai_url or "https://ark.cn-beijing.volces.com/api/coding"
        if not raw_url.endswith("/v3"):
            raw_url = raw_url.rstrip("/") + "/v3"
        self.base_url = base_url or raw_url
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> DoubaoResponse:
        """Send a chat completion request.

        Args:
            messages: Chat messages
            model: Model name (uses default if None)
            max_tokens: Max output tokens
            temperature: Sampling temperature

        Returns:
            DoubaoResponse
        """
        model = model or self.default_model
        last_error = None

        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                latency = time.time() - start_time

                choice = response.choices[0] if response.choices else None
                content = choice.message.content if choice and choice.message else ""
                reasoning = getattr(choice.message, "reasoning_content", "") if choice and choice.message else ""
                tool_calls = []
                if choice and choice.message and choice.message.tool_calls:
                    tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                return DoubaoResponse(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    usage=usage,
                    latency_seconds=latency,
                    raw_response_metadata={"model": model, "attempt": attempt + 1},
                )

            except Exception as e:
                last_error = e
                latency = time.time() - start_time
                if attempt < self.max_retries - 1:
                    backoff = min(2 ** attempt * 2, 30)
                    time.sleep(backoff)

        return DoubaoResponse(
            content="",
            latency_seconds=latency if 'latency' in dir() else 0,
            raw_response_metadata={"error": str(last_error)[:200]},
        )

    def chat_with_thinking(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> DoubaoResponse:
        """Chat with thinking/reasoning mode enabled."""
        return self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
