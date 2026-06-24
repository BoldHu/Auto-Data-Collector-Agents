"""Unified model client for Phase 6.6 system ablation.

Wraps DoubaoModelClient for all system baselines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.autodata.utils.doubao_model_client import DoubaoModelClient, DoubaoResponse


@dataclass
class UnifiedResponse:
    """Standardized response from any model."""
    content: str = ""
    reasoning_content: str = ""
    usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    model_name: str = ""
    error: Optional[str] = None


class UnifiedModelClient:
    """Unified client wrapping DoubaoModelClient for Phase 6.6."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.client = DoubaoModelClient(
            default_model=model_name,
            max_retries=max_retries,
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> UnifiedResponse:
        """Send a chat request."""
        start = time.time()
        try:
            response: DoubaoResponse = self.client.chat(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return UnifiedResponse(
                content=response.content,
                reasoning_content=response.reasoning_content,
                usage=response.usage,
                latency_seconds=time.time() - start,
                model_name=self.model_name,
            )
        except Exception as e:
            return UnifiedResponse(
                content="",
                latency_seconds=time.time() - start,
                model_name=self.model_name,
                error=str(e)[:200],
            )
