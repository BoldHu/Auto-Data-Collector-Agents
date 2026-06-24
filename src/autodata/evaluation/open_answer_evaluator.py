"""Open-ended answer evaluator for Phase 6.5.

Uses LLM judge for open-ended questions instead of strict exact match.
"""

from __future__ import annotations

import json
from typing import Optional

from src.autodata.evaluation.evaluation_schema import JudgeResult
from src.autodata.evaluation.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    AGENT_TASK_JUDGE_PROMPT,
)


# Task types that require LLM judge (not exact match)
OPEN_ENDED_TASKS = {
    "domain_knowledge_qa",
    "qa",
    "explanation",
    "extraction",
    "classification",
    "comparison",
    "source_grounded_reasoning",
    "process_reasoning",
    "causal_reasoning",
    "error_diagnosis",
    "agent_task",
    "visual_qa",
    "cross_modal_reasoning",
    "defect_diagnosis",
    "diagram_reasoning",
    "chart_reading",
    "ocr_reasoning",
}


def needs_llm_judge(item: dict, metrics: dict) -> bool:
    """Check if an item needs LLM judge evaluation."""
    task_type = item.get("task_type", "")

    # Always judge open-ended task types
    if task_type in OPEN_ENDED_TASKS:
        return True

    # Judge when exact_match is False for non-choice items
    if metrics.get("exact_match") is False and not item.get("options"):
        return True

    return False


def judge_open_answer(
    pool,
    item: dict,
    model_answer: str,
    judge_model: str = "mimo-v2.5-pro",
) -> JudgeResult:
    """Use LLM judge to evaluate an open-ended answer.

    Args:
        pool: ModelPool instance (API_KEY1 only)
        item: Benchmark item dict
        model_answer: Model's response text
        judge_model: Model to use for judging

    Returns:
        JudgeResult with scores.
    """
    # Build judge prompt
    if item.get("source_type") == "agent_task" or item.get("task_type") == "agent_task":
        user_prompt = AGENT_TASK_JUDGE_PROMPT.format(
            scenario=item.get("task_scenario", item.get("question", "")),
            constraints="\n".join(item.get("constraints", [])),
            rubric=item.get("scoring_rubric", ""),
            model_answer=model_answer,
        )
    else:
        evidence = "\n".join(item.get("evidence", [])[:3])
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
    except Exception as e:
        return JudgeResult(
            verdict="invalid",
            rationale=f"Judge call failed: {str(e)[:100]}",
        )

    return parse_judge_response(response_text)


def parse_judge_response(response_text: str) -> JudgeResult:
    """Parse LLM judge response into JudgeResult."""
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1

    if json_start >= 0 and json_end > json_start:
        try:
            data = json.loads(response_text[json_start:json_end])

            correctness = float(data.get("correctness", 0))
            semantic_eq = float(data.get("semantic_equivalence", data.get("evidence_support", 0)))
            evidence_support = float(data.get("evidence_support", 0))
            reasoning_quality = float(data.get("reasoning_quality", 0))
            hallucination = float(data.get("hallucination", 0))
            format_validity = float(data.get("format_validity", 0))
            final_score = float(data.get("final_score", 0))

            # Compute final_score if not provided
            if final_score == 0:
                final_score = (
                    correctness * 0.4 +
                    semantic_eq * 0.2 +
                    evidence_support * 0.2 +
                    reasoning_quality * 0.1 +
                    (1.0 - hallucination) * 0.05 +
                    format_validity * 0.05
                )

            # Determine verdict from final_score
            verdict = data.get("verdict", "")
            if not verdict:
                if final_score >= 0.75:
                    verdict = "correct"
                elif final_score >= 0.45:
                    verdict = "partially_correct"
                else:
                    verdict = "incorrect"

            return JudgeResult(
                correctness=correctness,
                evidence_support=evidence_support,
                reasoning_quality=reasoning_quality,
                hallucination=hallucination,
                format_validity=format_validity,
                final_score=final_score,
                verdict=verdict,
                rationale=str(data.get("rationale", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return JudgeResult(
        verdict="invalid",
        rationale="Failed to parse judge response",
    )


def is_judge_correct(judge_result: JudgeResult) -> bool:
    """Check if judge verdict indicates correctness."""
    return judge_result.verdict == "correct" or (
        judge_result.verdict == "partially_correct" and judge_result.final_score >= 0.6
    )
