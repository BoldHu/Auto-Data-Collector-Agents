"""Error analyzer for Phase 6.

Classifies errors into 14 categories.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

ERROR_CATEGORIES = [
    "domain_knowledge_error",
    "visual_perception_error",
    "chart_reading_error",
    "ocr_reasoning_error",
    "calculation_error",
    "reasoning_error",
    "hallucination",
    "evidence_mismatch",
    "option_parsing_error",
    "format_error",
    "refusal_or_empty_answer",
    "multimodal_input_failure",
    "agent_planning_failure",
    "context_selection_failure",
]


def analyze_errors(results: list[dict]) -> dict:
    """Analyze evaluation results and classify errors.

    Args:
        results: List of evaluation result dicts

    Returns:
        Error analysis report.
    """
    errors = [r for r in results if r.get("is_correct") is False]
    total_errors = len(errors)

    # Classify errors
    error_counts = Counter()
    error_cases = []

    for r in errors:
        category = _classify_error(r)
        error_counts[category] += 1
        error_cases.append({
            "benchmark_id": r.get("benchmark_id"),
            "model_name": r.get("model_name"),
            "task_type": r.get("task_type"),
            "error_category": category,
            "gold_answer": r.get("gold_answer", "")[:100],
            "model_answer": r.get("parsed_answer", "")[:100],
        })

    # Hallucination cases
    hallucination_cases = [r for r in results if r.get("hallucination_flag")]

    # Format errors
    format_errors = [r for r in results if not r.get("format_valid", True)]

    report = {
        "total_evaluated": len(results),
        "total_errors": total_errors,
        "error_rate": total_errors / len(results) if results else 0,
        "error_distribution": dict(error_counts.most_common()),
        "top_error_categories": [cat for cat, _ in error_counts.most_common(5)],
        "hallucination_count": len(hallucination_cases),
        "format_error_count": len(format_errors),
    }

    return report


def _classify_error(result: dict) -> str:
    """Classify a single error into a category."""
    task_type = result.get("task_type", "")
    error = result.get("error", "")
    raw_response = result.get("raw_response", "")

    if error:
        if "multimodal" in error.lower() or "image" in error.lower():
            return "multimodal_input_failure"
        if "format" in error.lower() or "parse" in error.lower():
            return "format_error"
        if "timeout" in error.lower() or "rate" in error.lower():
            return "refusal_or_empty_answer"

    if not raw_response or len(raw_response.strip()) < 2:
        return "refusal_or_empty_answer"

    if task_type in ("exam_calculation",):
        return "calculation_error"

    if task_type in ("visual_qa", "defect_diagnosis", "diagram_reasoning"):
        return "visual_perception_error"

    if task_type in ("chart_reading",):
        return "chart_reading_error"

    if task_type in ("ocr_reasoning",):
        return "ocr_reasoning_error"

    if task_type in ("agent_task",):
        return "agent_planning_failure"

    if task_type in ("exam_single_choice", "exam_multiple_choice"):
        return "option_parsing_error"

    return "reasoning_error"


def save_error_analysis(report: dict, error_cases: list[dict]) -> tuple[Path, Path]:
    """Save error analysis."""
    error_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6" / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)

    summary_path = error_dir / "error_taxonomy_summary.json"
    cases_path = error_dir / "error_cases.jsonl"

    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(cases_path, "w") as f:
        for case in error_cases[:1000]:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    return summary_path, cases_path
