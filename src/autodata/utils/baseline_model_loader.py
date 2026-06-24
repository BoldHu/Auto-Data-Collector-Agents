"""Baseline model loader and runner — evaluation harness only.

Loads baseline model configs from LLM_API/env_llm.txt and provides
a runner that wraps langchain_openai:ChatOpenAI for each model.

IMPORTANT: Baseline models are used ONLY for evaluation and comparison,
never for routine data cleaning, annotation, or generation tasks.

Usage:
    from src.autodata.utils.baseline_model_loader import load_baseline_models, BaselineModelRunner

    models = load_baseline_models()
    runner = BaselineModelRunner(models[0])
    response = runner.invoke("What is carbon fiber?")
    print(response)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.autodata.utils.api_loader import (
    BaselineModelConfig,
    load_baseline_configs,
)


# ── Baseline response ──────────────────────────────────────────────────

@dataclass
class BaselineResponse:
    """Structured response from a baseline model."""
    model_name: str
    content: str
    reasoning: Optional[str] = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


# ── Retryable errors ───────────────────────────────────────────────────

try:
    import openai as _openai
    _RETRYABLE_BASELINE_ERRORS = (
        _openai.RateLimitError,
        _openai.APIStatusError,
        _openai.APIConnectionError,
    )
except ImportError:
    _RETRYABLE_BASELINE_ERRORS = ()


# ── BaselineModelRunner ────────────────────────────────────────────────

class BaselineModelRunner:
    """Runner for a single baseline model.

    Wraps langchain_openai:ChatOpenAI or direct OpenAI calls
    depending on the model config's 'use' field.

    Supports:
    - Standard chat completion
    - Thinking mode (for models with supports_thinking=True)
    - Token usage tracking
    - Automatic retry with exponential backoff
    """

    def __init__(
        self,
        model_config: BaselineModelConfig,
        max_retries: int = 5,
    ) -> None:
        self.model_config = model_config
        self.max_retries = max_retries
        self._call_count = 0
        self._total_tokens_used = 0

        # Initialize the underlying client based on 'use' field
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create the underlying LLM client from the 'use' specification."""
        use_spec = self.model_config.use
        parts = use_spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid use spec: '{use_spec}'. Expected 'module:Class'.")

        module_name, class_name = parts

        if module_name == "langchain_openai" and class_name == "ChatOpenAI":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.model_config.model,
                    api_key=self.model_config.api_key,
                    base_url=self.model_config.base_url,
                    temperature=1.0,
                    max_tokens=4096,
                )
            except ImportError:
                # Fallback to direct OpenAI client
                from openai import OpenAI
                return OpenAI(
                    api_key=self.model_config.api_key,
                    base_url=self.model_config.base_url,
                )
        else:
            raise ValueError(f"Unsupported use spec: '{use_spec}'. Only 'langchain_openai:ChatOpenAI' is currently supported.")

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_BASELINE_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking: bool = False,
    ) -> BaselineResponse:
        """Send a chat completion request to this baseline model.

        Args:
            prompt: User message content.
            system_prompt: Optional system message.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            thinking: Whether to use thinking mode (only for models with supports_thinking).

        Returns:
            BaselineResponse with content, usage, latency.
        """
        if thinking and not self.model_config.supports_thinking:
            raise ValueError(
                f"Model '{self.model_config.name}' does not support thinking mode."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        # Dispatch based on client type
        client_type = type(self._client).__name__
        if client_type == "ChatOpenAI":
            # LangChain ChatOpenAI: use invoke method
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_messages = []
            if system_prompt:
                lc_messages.append(SystemMessage(content=system_prompt))
            lc_messages.append(HumanMessage(content=prompt))

            # Update runtime params
            self._client.max_tokens = max_tokens
            self._client.temperature = temperature

            lc_response = self._client.invoke(lc_messages)
            content = lc_response.content or ""

            # Extract usage from response metadata
            usage = {}
            if hasattr(lc_response, "response_metadata"):
                meta = lc_response.response_metadata
                if "token_usage" in meta:
                    usage = {
                        "prompt_tokens": meta["token_usage"].get("prompt_tokens", 0),
                        "completion_tokens": meta["token_usage"].get("completion_tokens", 0),
                        "total_tokens": meta["token_usage"].get("total_tokens", 0),
                    }
        elif client_type == "OpenAI":
            # Direct OpenAI client
            completion = self._client.chat.completions.create(
                model=self.model_config.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None)

            usage = {}
            if completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            self._total_tokens_used += usage.get("total_tokens", 0)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

        latency_ms = (time.time() - start_time) * 1000
        self._call_count += 1
        self._total_tokens_used += usage.get("total_tokens", 0)

        return BaselineResponse(
            model_name=self.model_config.name,
            content=content,
            reasoning=None,
            usage=usage,
            latency_ms=latency_ms,
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    @property
    def supports_thinking(self) -> bool:
        return self.model_config.supports_thinking

    @property
    def display_name(self) -> str:
        return self.model_config.display_name


# ── Convenience functions ───────────────────────────────────────────────

def load_baseline_models() -> list[BaselineModelConfig]:
    """Load all baseline model configurations."""
    return load_baseline_configs()


def create_runners(
    models: Optional[list[BaselineModelConfig]] = None,
) -> list[BaselineModelRunner]:
    """Create BaselineModelRunner instances for all (or specified) models."""
    if models is None:
        models = load_baseline_models()
    return [BaselineModelRunner(m) for m in models]