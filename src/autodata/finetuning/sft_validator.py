"""SFT sample validator.

Validates individual SFT samples for quality and correctness.
"""

from __future__ import annotations

import re


def validate_sample(sample: dict) -> dict:
    """Validate a single SFT sample.

    Returns:
        dict with 'valid' bool, 'issues' list, 'score' float.
    """
    issues = []
    score = 1.0

    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    input_text = sample.get("input", "")
    evidence = sample.get("evidence", [])

    # 1. Non-empty fields
    if not instruction or len(instruction.strip()) < 5:
        issues.append("empty_instruction")
        score -= 0.5

    if not output or len(output.strip()) < 2:
        issues.append("empty_output")
        score -= 0.5

    # 2. Reasonable length
    if len(output) > 5000:
        issues.append("output_too_long")
        score -= 0.2

    if len(instruction) > 2000:
        issues.append("instruction_too_long")
        score -= 0.1

    # 3. No hallucination markers
    hallucination_markers = ["我不知道", "无法回答", "抱歉", "sorry", "I don't know"]
    for marker in hallucination_markers:
        if marker in output:
            issues.append(f"hallucination_marker:{marker}")
            score -= 0.3

    # 4. Domain relevance
    domain_terms = ["碳纤维", "复合材料", "CFRP", "碳化", "纤维", "树脂", "基体",
                    "carbon", "fiber", "composite", "PAN", "预浸料", "层压"]
    text = instruction + output
    if not any(term in text for term in domain_terms):
        issues.append("no_domain_relevance")
        score -= 0.3

    # 5. Evidence support check
    if evidence:
        evidence_text = " ".join(str(e) for e in evidence[:3])
        output_words = set(re.findall(r'[\w\u4e00-\u9fff]+', output[:200]))
        evidence_words = set(re.findall(r'[\w\u4e00-\u9fff]+', evidence_text[:500]))
        overlap = len(output_words & evidence_words)
        if overlap < 2 and len(output_words) > 5:
            issues.append("low_evidence_support")
            score -= 0.2

    # 6. No benchmark leakage markers
    if sample.get("benchmark_id", ""):
        issues.append("has_benchmark_id")
        score -= 0.5

    # 7. Format validity
    if output and len(re.sub(r'[^\w\s]', '', output)) < len(output) * 0.3:
        issues.append("output_mostly_noise")
        score -= 0.4

    score = max(0.0, min(1.0, score))

    return {
        "valid": score >= 0.5 and len(issues) == 0,
        "score": round(score, 3),
        "issues": issues,
    }


def validate_batch(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate a batch of samples.

    Returns:
        (passed_samples, rejected_samples)
    """
    passed = []
    rejected = []

    for sample in samples:
        result = validate_sample(sample)
        sample["_validation_score"] = result["score"]
        sample["_validation_issues"] = result["issues"]

        if result["valid"]:
            passed.append(sample)
        else:
            rejected.append(sample)

    return passed, rejected
