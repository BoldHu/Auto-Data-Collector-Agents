"""Metric calculator for Phase 6 evaluation.

Computes accuracy, exact_match, F1, numeric tolerance.
"""

from __future__ import annotations

from src.autodata.evaluation.answer_normalizer import (
    normalize_answer,
    extract_mc_answer,
    normalize_tf_answer,
    parse_numeric_answer,
    check_numeric_tolerance,
    compute_token_f1,
)


def compute_metrics(item: dict, parsed_answer: str, raw_response: str) -> dict:
    """Compute all applicable metrics for an evaluation result.

    Returns:
        Dict with is_correct, exact_match, f1, numeric_score.
    """
    task_type = item.get("task_type", "")
    gold = item.get("answer", "")
    options = item.get("options", [])

    result = {
        "is_correct": None,
        "exact_match": None,
        "f1": None,
        "numeric_score": None,
    }

    # Multiple choice
    if options and len(options) >= 2:
        pred_letter = extract_mc_answer(parsed_answer)
        gold_letter = extract_mc_answer(gold)
        result["is_correct"] = pred_letter == gold_letter
        result["exact_match"] = result["is_correct"]
        return result

    # True/false
    if task_type == "exam_true_false" or any(kw in item.get("question", "") for kw in ["判断", "是否"]):
        pred_tf = normalize_tf_answer(parsed_answer)
        gold_tf = normalize_tf_answer(gold)
        result["is_correct"] = pred_tf == gold_tf
        result["exact_match"] = result["is_correct"]
        return result

    # Calculation / numeric
    if task_type == "exam_calculation" or "计算" in item.get("question", ""):
        pred_num = parse_numeric_answer(parsed_answer)
        gold_num = parse_numeric_answer(gold)
        if pred_num is not None and gold_num is not None:
            result["numeric_score"] = 1.0 if check_numeric_tolerance(pred_num, gold_num) else 0.0
            result["is_correct"] = result["numeric_score"] == 1.0
        else:
            result["is_correct"] = False
        return result

    # Short answer / fill blank / other
    norm_pred = normalize_answer(parsed_answer)
    norm_gold = normalize_answer(gold)

    # Exact match
    result["exact_match"] = norm_pred == norm_gold
    result["is_correct"] = result["exact_match"]

    # F1
    result["f1"] = compute_token_f1(norm_pred, norm_gold)

    return result
