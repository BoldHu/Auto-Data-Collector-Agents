"""Unified token accounting for Phase 6.7.

Provides consistent token estimation across all systems.
Uses a single method: len(text) // 4 for all estimates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses consistent len(text) // 4 approximation for all systems.
    This is ~4 chars per token for Chinese/mixed text.
    """
    if not text:
        return 0
    return len(text) // 4


@dataclass
class TokenBreakdown:
    """Detailed token breakdown for a single evaluation."""
    # Actual API tokens
    api_prompt_tokens: int = 0
    api_completion_tokens: int = 0

    # Component estimates (all using same estimation method)
    system_prompt_tokens: int = 0
    question_tokens: int = 0
    evidence_context_tokens: int = 0
    message_history_tokens: int = 0
    selected_context_tokens: int = 0
    broadcast_context_tokens: int = 0

    # Judge tokens
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    # Derived
    context_overhead_tokens: int = 0  # system_prompt + context
    pure_context_tokens: int = 0  # just the injected context

    def to_dict(self) -> dict:
        return {
            "api_prompt_tokens": self.api_prompt_tokens,
            "api_completion_tokens": self.api_completion_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "question_tokens": self.question_tokens,
            "evidence_context_tokens": self.evidence_context_tokens,
            "message_history_tokens": self.message_history_tokens,
            "selected_context_tokens": self.selected_context_tokens,
            "broadcast_context_tokens": self.broadcast_context_tokens,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "context_overhead_tokens": self.context_overhead_tokens,
            "pure_context_tokens": self.pure_context_tokens,
        }


def compute_token_breakdown(
    system_prompt: str,
    user_prompt: str,
    evidence_context: str = "",
    message_history: str = "",
    selected_context: str = "",
    broadcast_context: str = "",
    api_usage: Optional[dict] = None,
    judge_usage: Optional[dict] = None,
) -> TokenBreakdown:
    """Compute detailed token breakdown.

    Args:
        system_prompt: The system prompt text
        user_prompt: The user prompt text (includes question + template)
        evidence_context: Available evidence/context for the task
        message_history: Full message history (for broadcast)
        selected_context: DTCG-selected context
        broadcast_context: Full broadcast context estimate
        api_usage: Actual API token usage dict
        judge_usage: Judge model token usage dict

    Returns:
        TokenBreakdown with all fields populated
    """
    breakdown = TokenBreakdown()

    # Actual API tokens
    if api_usage:
        breakdown.api_prompt_tokens = api_usage.get("prompt_tokens", 0)
        breakdown.api_completion_tokens = api_usage.get("completion_tokens", 0)

    # Component estimates
    breakdown.system_prompt_tokens = estimate_tokens(system_prompt)
    breakdown.question_tokens = estimate_tokens(user_prompt)
    breakdown.evidence_context_tokens = estimate_tokens(evidence_context)
    breakdown.message_history_tokens = estimate_tokens(message_history)
    breakdown.selected_context_tokens = estimate_tokens(selected_context)
    breakdown.broadcast_context_tokens = estimate_tokens(broadcast_context)

    # Judge tokens
    if judge_usage:
        breakdown.judge_input_tokens = judge_usage.get("prompt_tokens", 0)
        breakdown.judge_output_tokens = judge_usage.get("completion_tokens", 0)

    # Derived
    breakdown.context_overhead_tokens = breakdown.system_prompt_tokens + breakdown.evidence_context_tokens
    breakdown.pure_context_tokens = breakdown.evidence_context_tokens

    return breakdown


def compute_context_saving(selected_tokens: int, broadcast_tokens: int) -> float:
    """Compute context saving ratio.

    Args:
        selected_tokens: DTCG selected context tokens
        broadcast_tokens: Full broadcast context tokens

    Returns:
        Saving ratio (0.0 to 1.0)
    """
    if broadcast_tokens <= 0:
        return 0.0
    return 1.0 - (selected_tokens / broadcast_tokens)


def compute_duplicate_ratio(context_tokens: int, unique_estimate: int) -> float:
    """Estimate duplicate context ratio.

    Args:
        context_tokens: Total context tokens
        unique_estimate: Estimated unique context tokens

    Returns:
        Duplicate ratio (0.0 to 1.0)
    """
    if context_tokens <= 0:
        return 0.0
    return 1.0 - (unique_estimate / context_tokens)
