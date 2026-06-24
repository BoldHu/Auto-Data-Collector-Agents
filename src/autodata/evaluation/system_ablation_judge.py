"""Judge for Phase 6.6 system ablation.

Uses deepseek-v4-flash as self-judge and doubao-seed-2.0-pro as cross-judge.
"""

from __future__ import annotations

import json
from typing import Optional

from src.autodata.evaluation.system_prompts import JUDGE_SYSTEM, JUDGE_USER
from src.autodata.evaluation.system_trace_schema import AblationTrace


def judge_response(
    client,
    item: dict,
    trace: AblationTrace,
) -> dict:
    """Judge a system response using LLM.

    Args:
        client: UnifiedModelClient for judging
        item: Original benchmark item
        trace: AblationTrace with model response

    Returns:
        Judge result dict
    """
    question = item.get("question", "")
    gold = item.get("answer", "")
    answer = trace.parsed_answer or trace.raw_answer[:500]
    system_type = trace.system_type

    user_prompt = JUDGE_USER.format(
        question=question[:500],
        gold_answer=str(gold)[:300],
        model_answer=answer[:500],
        system_type=system_type,
    )

    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return parse_judge_response(response.content)
    except Exception:
        return {"verdict": "judge_failed", "final_score": 0.0, "rationale": "Judge call failed"}


def parse_judge_response(text: str) -> dict:
    """Parse judge response JSON."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            # Compute final_score if not provided
            if data.get("final_score", 0) == 0:
                correctness = float(data.get("correctness", 0))
                evidence = float(data.get("evidence_support", 0))
                constraint = float(data.get("constraint_satisfaction", 0))
                planning = float(data.get("planning_quality", 0))
                hallucination = float(data.get("hallucination", 0))
                fmt = float(data.get("format_validity", 0))
                data["final_score"] = (
                    correctness * 0.4 + evidence * 0.2 + constraint * 0.15 +
                    planning * 0.1 + (1.0 - hallucination) * 0.1 + fmt * 0.05
                )
            # Derive verdict from score
            if not data.get("verdict"):
                score = data.get("final_score", 0)
                if score >= 0.75:
                    data["verdict"] = "correct"
                elif score >= 0.45:
                    data["verdict"] = "partially_correct"
                else:
                    data["verdict"] = "incorrect"
            return data
        except (json.JSONDecodeError, ValueError):
            pass
    return {"verdict": "judge_failed", "final_score": 0.0, "rationale": "Parse failed"}


def rule_based_check(item: dict, trace: AblationTrace) -> Optional[dict]:
    """Rule-based scoring for MC/TF tasks (skip LLM judge)."""
    task_type = item.get("task_type", "")
    options = item.get("options", [])
    gold = str(item.get("answer", "")).strip()
    answer = str(trace.parsed_answer or trace.raw_answer).strip()

    # Multiple choice: extract letter
    if options and len(options) >= 2:
        import re
        pred_match = re.search(r'([A-H])', answer.upper())
        gold_match = re.search(r'([A-H])', gold.upper())
        if pred_match and gold_match:
            correct = pred_match.group(1) == gold_match.group(1)
            return {"verdict": "correct" if correct else "incorrect", "final_score": 1.0 if correct else 0.0, "correctness": 1.0 if correct else 0.0}

    # True/false
    if task_type == "exam_true_false" or any(kw in item.get("question", "") for kw in ["判断", "是否"]):
        def norm_tf(s):
            s = s.strip().lower()
            if s in ("true", "correct", "对", "正确", "是"):
                return "正确"
            if s in ("false", "incorrect", "错", "错误", "否"):
                return "错误"
            return s
        correct = norm_tf(answer) == norm_tf(gold)
        return {"verdict": "correct" if correct else "incorrect", "final_score": 1.0 if correct else 0.0, "correctness": 1.0 if correct else 0.0}

    return None  # Need LLM judge
