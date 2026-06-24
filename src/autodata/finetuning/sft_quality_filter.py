"""Quality filtering for SFT samples.

Rule-based validation for instruction clarity, output correctness,
format validity, and domain relevance.
"""

from __future__ import annotations

import re
from typing import Any


def check_sample_quality(sample: dict) -> dict:
    """Check quality of a single SFT sample.

    Returns:
        dict with 'passed' bool, 'score' float, 'issues' list
    """
    issues = []
    score = 1.0

    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    evidence = sample.get("evidence", [])

    # 1. No empty instruction
    if not instruction or len(instruction.strip()) < 5:
        issues.append("empty_or_short_instruction")
        score -= 0.5

    # 2. No empty output
    if not output or len(output.strip()) < 2:
        issues.append("empty_output")
        score -= 0.5

    # 3. No overly long noisy OCR fragments
    if len(output) > 5000:
        issues.append("output_too_long")
        score -= 0.2

    # 4. No hallucinated markers
    hallucination_markers = ["我不知道", "无法回答", "抱歉", "sorry", "I don't know"]
    for marker in hallucination_markers:
        if marker in output:
            issues.append(f"hallucination_marker: {marker}")
            score -= 0.3

    # 5. Format validity - output should not be pure noise
    if output and len(re.sub(r'[^\w\s]', '', output)) < len(output) * 0.3:
        issues.append("output_mostly_noise")
        score -= 0.4

    # 6. Domain relevance - should mention carbon fiber related terms
    domain_terms = ["碳纤维", "复合材料", "CFRP", "碳化", "纤维", "树脂", "基体",
                    "carbon", "fiber", "composite", "PAN", "预浸料", "层压"]
    text = instruction + output
    has_domain = any(term in text for term in domain_terms)
    if not has_domain:
        issues.append("no_domain_relevance")
        score -= 0.3

    # 7. Evidence support - if evidence exists, output should relate to it
    if evidence:
        evidence_text = " ".join(str(e) for e in evidence[:3])
        # Simple word overlap check
        output_words = set(re.findall(r'[\w\u4e00-\u9fff]+', output[:200]))
        evidence_words = set(re.findall(r'[\w\u4e00-\u9fff]+', evidence_text[:500]))
        overlap = len(output_words & evidence_words)
        if overlap < 2 and len(output_words) > 5:
            issues.append("low_evidence_support")
            score -= 0.2

    # 8. No duplicate content in instruction and output
    if instruction and output:
        sim = len(set(instruction) & set(output)) / max(len(set(instruction)), 1)
        if sim > 0.9 and len(instruction) > 50:
            issues.append("instruction_output_too_similar")
            score -= 0.2

    score = max(0.0, min(1.0, score))

    return {
        "passed": score >= 0.5 and len(issues) == 0,
        "score": round(score, 3),
        "issues": issues,
    }


def filter_samples(samples: list[dict], min_score: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Filter samples by quality.

    Returns:
        (passed_samples, rejected_samples)
    """
    passed = []
    rejected = []
    for sample in samples:
        result = check_sample_quality(sample)
        sample["_quality_score"] = result["score"]
        sample["_quality_issues"] = result["issues"]
        if result["passed"] and result["score"] >= min_score:
            passed.append(sample)
        else:
            rejected.append(sample)
    return passed, rejected


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """Remove exact and near-duplicate samples."""
    seen_hashes = set()
    unique = []
    for sample in samples:
        # Use instruction + output as dedup key
        key = (sample.get("instruction", "")[:100] + "||" + sample.get("output", "")[:100]).strip()
        h = hash(key)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(sample)
    return unique
