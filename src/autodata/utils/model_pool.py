"""Model Pool — multi-API, round-robin, failover, retry, concurrency-limited.

Manages model clients across Xiaomi API endpoints:
- Xiaomi API (API_KEY + API_KEY2): mimo-v2-omni, mimo-v2.5, mimo-v2.5-pro

Provides:
- Round-robin across models with priority ordering (fastest first)
- Failover: if a model/API fails, automatically switch to next
- Concurrency tracking per API group (RPM limits)
- Latency stats, call counts, token usage per model
- Graceful degradation: fall back to quality model if fast model fails
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.autodata.utils.model_client import XiaomiModelClient, ChatResponse
from src.autodata.utils.api_loader import load_xiaomi_config
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("model_pool")


# ── Model endpoint definition ──────────────────────────────────────────

@dataclass
class ModelEndpoint:
    """A single model+API endpoint."""
    model_name: str
    api_group: str  # "xiaomi0", "xiaomi1", "infini", or custom
    priority: int  # lower = used first
    role: str = "general"  # "fast", "quality_fallback", "general"
    client: Any = None  # XiaomiModelClient or OpenAI client
    _call_count: int = 0
    _total_tokens: int = 0
    _total_latency_ms: float = 0.0
    _error_count: int = 0
    _last_error_time: float = 0.0
    _consecutive_errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / max(self._call_count, 1)

    def record_success(self, latency_ms: float, tokens: int) -> None:
        self._call_count += 1
        self._total_tokens += tokens
        self._total_latency_ms += latency_ms
        self._consecutive_errors = 0

    def record_error(self) -> None:
        self._error_count += 1
        self._consecutive_errors += 1
        self._last_error_time = time.time()


# ── Model Pool ──────────────────────────────────────────────────────────

class ModelPool:
    """Dual-API model pool with round-robin, failover, and concurrency control.

    Usage:
        pool = ModelPool()
        response = pool.chat(messages=[...])
        # Automatically picks fastest available model

    Configuration:
        - endpoints: list of ModelEndpoint with priority ordering
        - max_concurrent_per_api: RPM limit per API (default 80, below 100 RPM limit)
        - max_consecutive_errors: disable endpoint after N consecutive errors (default 5)
        - cooldown_seconds: wait before retrying a disabled endpoint (default 60)
    """

    # Default endpoint configuration — Xiaomi only, 4 models × 2 API keys = 8 endpoints
    # Format: (model_name, api_group, priority, role)
    DEFAULT_ENDPOINTS = [
        # Xiaomi fast models (lowest priority number = highest priority)
        ("mimo-v2-omni",   "xiaomi0", 1,  "fast"),
        ("mimo-v2-omni",   "xiaomi1", 2,  "fast"),
        ("mimo-v2.5",      "xiaomi0", 3,  "fast"),
        ("mimo-v2.5",      "xiaomi1", 4,  "fast"),
        ("mimo-v2-pro",    "xiaomi0", 5,  "fast"),
        ("mimo-v2-pro",    "xiaomi1", 6,  "fast"),
        # Xiaomi quality models (highest priority number = lowest priority for cleaning)
        ("mimo-v2.5-pro",  "xiaomi0", 7,  "quality_fallback"),
        ("mimo-v2.5-pro",  "xiaomi1", 8,  "quality_fallback"),
    ]

    # Multimodal endpoints — models that accept image input via content array
    # mimo-v2-omni and mimo-v2.5 are multimodal; mimo-v2-pro and mimo-v2.5-pro are text-only
    MULTIMODAL_ENDPOINTS = [
        ("mimo-v2-omni",   "xiaomi0", 1,  "multimodal"),
        ("mimo-v2-omni",   "xiaomi1", 2,  "multimodal"),
        ("mimo-v2.5",      "xiaomi0", 3,  "multimodal"),
        ("mimo-v2.5",      "xiaomi1", 4,  "multimodal"),
    ]

    # Single-key endpoints (only xiaomi0, no xiaomi1)
    SINGLE_KEY_ENDPOINTS = [
        ("mimo-v2-omni",   "xiaomi0", 1,  "fast"),
        ("mimo-v2.5",      "xiaomi0", 2,  "fast"),
        ("mimo-v2-pro",    "xiaomi0", 3,  "fast"),
        ("mimo-v2.5-pro",  "xiaomi0", 4,  "quality_fallback"),
    ]

    SINGLE_KEY_MULTIMODAL_ENDPOINTS = [
        ("mimo-v2-omni",   "xiaomi0", 1,  "multimodal"),
        ("mimo-v2.5",      "xiaomi0", 2,  "multimodal"),
    ]

    def __init__(
        self,
        endpoints: Optional[list[tuple[str, str, int, str]]] = None,
        max_concurrent_per_api: int = 150,
        max_consecutive_errors: int = 5,
        cooldown_seconds: float = 60.0,
        model_for_quality: str = "mimo-v2.5-pro",
        use_key2: bool = True,
    ) -> None:
        self.max_concurrent_per_api = max_concurrent_per_api
        self.max_consecutive_errors = max_consecutive_errors
        self.cooldown_seconds = cooldown_seconds
        self.model_for_quality = model_for_quality
        self.use_key2 = use_key2

        # Build endpoints: add multimodal entries alongside default entries
        # Each multimodal model (mimo-v2-omni, mimo-v2.5) gets an additional
        # endpoint with role="multimodal" for chat_multimodal() calls.
        # The same model+api_group also appears in DEFAULT_ENDPOINTS with
        # role="fast" for regular text calls.
        if endpoints is not None:
            ep_defs = endpoints
            multimodal_defs = self.MULTIMODAL_ENDPOINTS
        elif not use_key2:
            ep_defs = self.SINGLE_KEY_ENDPOINTS
            multimodal_defs = self.SINGLE_KEY_MULTIMODAL_ENDPOINTS
        else:
            ep_defs = self.DEFAULT_ENDPOINTS
            multimodal_defs = self.MULTIMODAL_ENDPOINTS

        self.endpoints: list[ModelEndpoint] = []
        # First add default endpoints
        for model_name, api_group, priority, role in ep_defs:
            if api_group.startswith("xiaomi"):
                use_key2 = api_group == "xiaomi1"
                client = XiaomiModelClient(
                    use_key2=use_key2,
                    default_model=model_name,
                )
            else:
                raise ValueError(f"Unknown api_group: {api_group}")
            ep = ModelEndpoint(
                model_name=model_name,
                api_group=api_group,
                priority=priority,
                role=role,
                client=client,
            )
            self.endpoints.append(ep)

        # Then add multimodal endpoints (separate entries, reuse client objects where possible)
        # Find the client for each multimodal model+api_group from existing endpoints
        client_lookup = {(ep.model_name, ep.api_group): ep.client for ep in self.endpoints}
        for model_name, api_group, priority, role in multimodal_defs:
            existing_client = client_lookup.get((model_name, api_group))
            if existing_client:
                client = existing_client
            else:
                # Fallback: create new client if not in default endpoints
                use_key2 = api_group == "xiaomi1"
                client = XiaomiModelClient(use_key2=use_key2, default_model=model_name)
            ep = ModelEndpoint(
                model_name=model_name,
                api_group=api_group,
                priority=priority,
                role=role,
                client=client,
            )
            self.endpoints.append(ep)

        # Sort by priority (fastest first)
        self.endpoints.sort(key=lambda ep: ep.priority)

        # Concurrency tracking per API group
        self._active_calls: dict[str, int] = {}
        for ep in self.endpoints:
            if ep.api_group not in self._active_calls:
                self._active_calls[ep.api_group] = 0
        self._lock = threading.Lock()
        self._round_robin_idx = 0

    def _select_endpoint(self, prefer_role: Optional[str] = None) -> Optional[ModelEndpoint]:
        """Select next available endpoint, preferring fast models for cleaning.

        Selection strategy:
        1. For regular chat() calls (no prefer_role): prefer fast/general endpoints
        2. For chat_quality() calls (prefer_role="quality_fallback"): use quality endpoints
        3. Within preferred role: round-robin
        4. Fallback to any eligible endpoint if preferred role is exhausted
        5. Skip disabled endpoints (consecutive errors + cooldown)
        6. Skip endpoints at concurrency limit
        """
        now = time.time()

        # Determine which roles to prefer
        if prefer_role == "quality_fallback":
            preferred_roles = ["quality_fallback"]
        elif prefer_role == "multimodal":
            preferred_roles = ["multimodal"]
        elif prefer_role:
            preferred_roles = [prefer_role]
        else:
            # Regular cleaning calls: prefer fast + multimodal models, exclude quality_fallback
            # Multimodal models (mimo-v2-omni, mimo-v2.5) can also handle text-only calls
            preferred_roles = ["fast", "multimodal", "general"]

        with self._lock:
            # Filter eligible endpoints within preferred roles first
            eligible_primary = []
            eligible_any = []

            for ep in self.endpoints:
                # Check consecutive errors
                if ep._consecutive_errors >= self.max_consecutive_errors:
                    if now - ep._last_error_time < self.cooldown_seconds:
                        continue
                    ep._consecutive_errors = 0

                # Check concurrency
                if self._active_calls[ep.api_group] >= self.max_concurrent_per_api:
                    continue

                if ep.role in preferred_roles:
                    eligible_primary.append(ep)
                else:
                    eligible_any.append(ep)

            # Use primary candidates if available, otherwise fall back to any
            eligible = eligible_primary if eligible_primary else eligible_any

            if not eligible:
                # Burst mode: allow slightly above concurrency limit
                for ep in self.endpoints:
                    if ep._consecutive_errors >= self.max_consecutive_errors:
                        if now - ep._last_error_time < self.cooldown_seconds:
                            continue
                        ep._consecutive_errors = 0
                    if self._active_calls[ep.api_group] < self.max_concurrent_per_api + 20:
                        eligible.append(ep)

                if not eligible:
                    logger.warning("All model endpoints unavailable")
                    return None

            # Round-robin selection within eligible
            self._round_robin_idx = self._round_robin_idx % len(eligible)
            selected = eligible[self._round_robin_idx]
            self._round_robin_idx += 1

            # Track concurrency
            self._active_calls[selected.api_group] += 1

            return selected

    def _release_endpoint(self, ep: ModelEndpoint) -> None:
        """Release concurrency slot after call completes."""
        with self._lock:
            self._active_calls[ep.api_group] -= 1

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        prefer_role: Optional[str] = None,
        max_retries: int = 5,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request using the best available model.

        Automatically selects endpoint, retries on failure with failover
        to next endpoint.

        Args:
            messages: Chat messages.
            model: Override model name (uses pool selection if None).
            max_completion_tokens: Max output tokens.
            temperature: Sampling temperature.
            prefer_role: Prefer endpoints with this role (e.g., "quality_fallback").
            max_retries: Max retries across endpoints before giving up.

        Returns:
            ChatResponse from whichever endpoint succeeded.
        """
        # If explicit model specified, find that endpoint
        if model:
            target_eps = [ep for ep in self.endpoints if ep.model_name == model]
            if target_eps:
                prefer_role = target_eps[0].role

        last_error = None
        for attempt in range(max_retries):
            ep = self._select_endpoint(prefer_role=prefer_role)
            if ep is None:
                # Brief wait before retry
                time.sleep(2)
                ep = self._select_endpoint(prefer_role=prefer_role)
                if ep is None:
                    last_error = RuntimeError("No model endpoints available")
                    continue

            try:
                start_time = time.time()
                response = ep.client.chat(
                    messages=messages,
                    model=model or ep.model_name,
                    max_completion_tokens=max_completion_tokens or 4096,
                    temperature=temperature or 1.0,
                    **kwargs,
                )
                latency_ms = (time.time() - start_time) * 1000
                ep.record_success(latency_ms, response.total_tokens)
                self._release_endpoint(ep)
                return response

            except Exception as e:
                ep.record_error()
                self._release_endpoint(ep)
                last_error = e
                logger.warning(
                    f"Endpoint {ep.model_name}/{ep.api_group} failed: "
                    f"{str(e)[:80]}, attempt {attempt+1}/{max_retries}"
                )
                # Exponential backoff: 3s, 6s, 12s, ...
                backoff = min(3 * (2 ** attempt), 30)
                time.sleep(backoff)
                continue

        raise RuntimeError(
            f"All retries exhausted. Last error: {str(last_error)[:100]}"
        )

    def chat_quality(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request using the quality fallback model.

        Always uses mimo-v2.5-pro for high-quality operations like
        quality verification and knowledge extraction.
        """
        return self.chat(
            messages=messages,
            model=self.model_for_quality,
            prefer_role="quality_fallback",
            max_completion_tokens=max_completion_tokens or 4096,
            **kwargs,
        )

    def chat_multimodal(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        max_retries: int = 5,
        **kwargs,
    ) -> ChatResponse:
        """Send a multimodal chat request (text + images) using multimodal-capable models.

        Only selects endpoints with role="multimodal" (mimo-v2-omni, mimo-v2.5).
        These models accept image input via OpenAI content array format with
        base64 inline encoding (data:image/jpeg;base64,...).

        Falls back to any endpoint if all multimodal endpoints are unavailable.

        Args:
            messages: Chat messages with content array format for images.
            model: Override model name (must be a multimodal model).
            max_completion_tokens: Max output tokens.
            max_retries: Max retries across endpoints.

        Returns:
            ChatResponse from whichever multimodal endpoint succeeded.
        """
        return self.chat(
            messages=messages,
            model=model,
            prefer_role="multimodal",
            max_completion_tokens=max_completion_tokens or 4096,
            max_retries=max_retries,
            **kwargs,
        )

    def stats(self) -> dict:
        """Return pool statistics."""
        with self._lock:
            return {
                "endpoints": [
                    {
                        "model": ep.model_name,
                        "api": ep.api_group,
                        "role": ep.role,
                        "calls": ep._call_count,
                        "tokens": ep._total_tokens,
                        "avg_latency_ms": round(ep.avg_latency_ms, 1),
                        "errors": ep._error_count,
                        "consecutive_errors": ep._consecutive_errors,
                    }
                    for ep in self.endpoints
                ],
                "active_calls": dict(self._active_calls),
                "total_calls": sum(ep._call_count for ep in self.endpoints),
                "total_tokens": sum(ep._total_tokens for ep in self.endpoints),
            }

    def reset_stats(self) -> None:
        """Reset all endpoint statistics."""
        with self._lock:
            for ep in self.endpoints:
                ep._call_count = 0
                ep._total_tokens = 0
                ep._total_latency_ms = 0.0
                ep._error_count = 0
                ep._consecutive_errors = 0


# ── Convenience singleton ──────────────────────────────────────────────

_pool: Optional[ModelPool] = None


def get_model_pool(use_key2: bool = True) -> ModelPool:
    """Get or create the default ModelPool singleton.

    Args:
        use_key2: If False, only use xiaomi0 (API_KEY1). Default True for backward compat.
    """
    global _pool
    if _pool is None:
        _pool = ModelPool(use_key2=use_key2)
    return _pool