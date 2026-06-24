"""Answer normalizer for Phase 6 evaluation.

Normalizes model answers for comparison with gold answers.
"""

from __future__ import annotations

import re
from typing import Optional


def normalize_answer(answer: str) -> str:
    """General answer normalization."""
    if not answer:
        return ""
    answer = answer.strip()
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def extract_mc_answer(response: str) -> str:
    """Extract multiple-choice answer from response.

    Looks for option letter patterns like A, B, C, D.
    """
    response = response.strip()

    # Direct single letter
    if len(response) == 1 and response.upper() in "ABCDEFGH":
        return response.upper()

    # "答案是A" or "答案: A" pattern
    match = re.search(r'答案[是为：:]\s*([A-H])', response)
    if match:
        return match.group(1).upper()

    # "A." or "A、" pattern
    match = re.search(r'^([A-H])[.、．)\s]', response)
    if match:
        return match.group(1).upper()

    # First letter that looks like an answer
    for char in response:
        if char.upper() in "ABCDEFGH":
            return char.upper()

    return response.strip()


def normalize_tf_answer(answer: str) -> str:
    """Normalize true/false answer."""
    answer = answer.strip().lower()
    if answer in ("true", "correct", "对", "正确", "是", "√", "✓"):
        return "正确"
    if answer in ("false", "incorrect", "错", "错误", "否", "×", "✗"):
        return "错误"
    if "正确" in answer:
        return "正确"
    if "错误" in answer or "不正确" in answer:
        return "错误"
    return answer


def parse_numeric_answer(answer: str) -> Optional[float]:
    """Parse numeric value from answer string."""
    answer = answer.strip()

    # Direct number
    try:
        return float(answer)
    except ValueError:
        pass

    # Scientific notation
    match = re.search(r'[-+]?\d*\.?\d+[eE][-+]?\d+', answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    # Number with unit
    match = re.search(r'([-+]?\d*\.?\d+)\s*[a-zA-Z%°]', answer)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Any number in the text
    match = re.search(r'[-+]?\d+\.?\d*', answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return None


def check_numeric_tolerance(predicted: float, gold: float, tolerance: float = 0.05) -> bool:
    """Check if predicted value is within tolerance of gold."""
    if gold == 0:
        return abs(predicted) < 0.001
    return abs(predicted - gold) / abs(gold) < tolerance


def compute_token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
