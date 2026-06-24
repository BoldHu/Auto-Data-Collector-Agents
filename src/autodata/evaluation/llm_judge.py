"""LLM judge for Phase 6 evaluation.

Uses Xiaomi API_KEY1 (mimo-v2.5-pro) for judging.
"""

from __future__ import annotations

import json
from typing import Optional

from src.autodata.evaluation.judge_prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.autodata.evaluation.evaluation_schema import JudgeResult


def judge_response(pool, item: dict, model_answer: str) -> JudgeResult:
    """Use LLM judge to evaluate a model response.

    Args:
        pool: ModelPool instance (API_KEY1 only)
        item: Benchmark item dict
        model_answer: Model's response text

    Returns:
        JudgeResult with scores.
    """
    user_prompt = build_judge_prompt(item, model_answer)

    try:
        response = pool.chat_quality(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2048,
            temperature=0.3,
        )
        response_text = response.content
    except Exception:
        return JudgeResult(verdict="invalid", rationale="Judge call failed")

    return parse_judge_response(response_text)


def parse_judge_response(response_text: str) -> JudgeResult:
    """Parse LLM judge response into JudgeResult."""
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1

    if json_start >= 0 and json_end > json_start:
        try:
            data = json.loads(response_text[json_start:json_end])
            return JudgeResult(
                correctness=float(data.get("correctness", 0)),
                evidence_support=float(data.get("evidence_support", 0)),
                reasoning_quality=float(data.get("reasoning_quality", 0)),
                hallucination=float(data.get("hallucination", 0)),
                format_validity=float(data.get("format_validity", 0)),
                final_score=float(data.get("final_score", 0)),
                verdict=str(data.get("verdict", "incorrect")),
                rationale=str(data.get("rationale", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return JudgeResult(verdict="invalid", rationale="Failed to parse judge response")
