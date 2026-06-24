"""Xiaomi MiMo LLM client — safe, retryable, structured.

Provides a unified interface for calling Xiaomi MiMo models via the
OpenAI-compatible API. Never logs or exposes API keys.

Usage:
    from src.autodata.utils.model_client import XiaomiModelClient

    client = XiaomiModelClient()
    response = client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="mimo-v2.5-pro",
    )
    print(response.content)
    print(response.reasoning)  # thinking/reasoning content if available
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.autodata.utils.api_loader import (
    XiaomiConfig,
    create_xiaomi_openai_client,
    load_xiaomi_config,
)


# ── Response dataclass ─────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """Structured response from Xiaomi MiMo LLM."""
    response_id: str
    model: str
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    finish_reason: Optional[str] = None
    usage: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


# ── Retryable errors ───────────────────────────────────────────────────

_RETRYABLE_ERRORS = (
    # 429 rate limit, 500 server error, 503 service unavailable
    # OpenAI SDK raises openai.RateLimitError, openai.APIServiceError etc.
)

try:
    import openai as _openai
    _RETRYABLE_ERRORS = (
        _openai.RateLimitError,
        _openai.APIStatusError,
        _openai.APIConnectionError,
    )
except ImportError:
    pass


# ── XiaomiModelClient ──────────────────────────────────────────────────

class XiaomiModelClient:
    """Unified client for Xiaomi MiMo models.

    Wraps the OpenAI-compatible API with:
    - Automatic retry with exponential backoff
    - Structured ChatResponse output
    - Thinking/reasoning content extraction
    - Token usage tracking
    - Never logs or exposes API keys
    """

    def __init__(
        self,
        config: Optional[XiaomiConfig] = None,
        default_model: Optional[str] = None,
        default_max_tokens: int = 4096,
        default_temperature: float = 1.0,
        default_top_p: float = 0.95,
        max_retries: int = 5,
        use_key2: bool = False,
    ) -> None:
        if config is None:
            config = load_xiaomi_config(use_key2=use_key2)
        self.config = config
        self.default_model = default_model or self.config.default_model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_retries = max_retries

        self._client = create_xiaomi_openai_client(self.config)
        self._call_count = 0
        self._total_tokens_used = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        stop: Optional[list[str]] = None,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with role/content.
            model: Model ID (defaults to default_model).
            max_completion_tokens: Max output tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stream: Whether to stream.
            tools: Tool/function definitions for tool calling.
            tool_choice: Tool choice strategy.
            stop: Stop sequences.
            frequency_penalty: Frequency penalty.
            presence_penalty: Presence penalty.

        Returns:
            ChatResponse with content, reasoning, usage, etc.
        """
        model = model or self.default_model
        max_completion_tokens = max_completion_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        top_p = top_p or self.default_top_p

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if stop is not None:
            params["stop"] = stop

        completion = self._client.chat.completions.create(**params)

        self._call_count += 1

        # Extract response fields
        choice = completion.choices[0]
        message = choice.message

        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        finish_reason = choice.finish_reason

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            self._total_tokens_used += completion.usage.total_tokens

        return ChatResponse(
            response_id=completion.id or uuid.uuid4().hex[:12],
            model=model,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def chat_with_thinking(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_completion_tokens: int = 16384,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request optimized for thinking/reasoning models.

        Uses higher max_completion_tokens to give the model space for
        internal reasoning. The reasoning_content is extracted from
        the response.
        """
        return self.chat(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            **kwargs,
        )

    def reset_stats(self) -> None:
        """Reset call count and token usage counters."""
        self._call_count = 0
        self._total_tokens_used = 0


# ── Convenience singleton ──────────────────────────────────────────────

_default_client: Optional[XiaomiModelClient] = None
_key2_client: Optional[XiaomiModelClient] = None


def get_default_client() -> XiaomiModelClient:
    """Get or create the default XiaomiModelClient singleton."""
    global _default_client
    if _default_client is None:
        _default_client = XiaomiModelClient()
    return _default_client


def get_key2_client(model: Optional[str] = None) -> XiaomiModelClient:
    """Get or create a XiaomiModelClient using API_KEY2.

    This client uses the second API key (unlimited quota) and
    defaults to mimo-v2.5 for faster throughput.
    """
    global _key2_client
    if _key2_client is None:
        _key2_client = XiaomiModelClient(
            use_key2=True,
            default_model=model or "mimo-v2.5",
        )
    return _key2_client