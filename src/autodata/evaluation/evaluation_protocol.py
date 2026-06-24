"""Evaluation protocol for Phase 6.

Defines metrics, answer formats, and evaluation rules per subset.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

EVALUATION_PROTOCOL = {
    "version": "1.0",
    "phase": "phase_6_baseline_evaluation",

    "subsets": {
        "CFBench-Text": {
            "description": "Text-only benchmark for text-only LLMs",
            "evaluated_by": "text_only_models",
            "metrics": ["accuracy", "exact_match", "f1", "llm_judge"],
            "answer_format": "short_text",
        },
        "CFBench-MM": {
            "description": "Multimodal benchmark requiring image understanding",
            "evaluated_by": "multimodal_models",
            "metrics": ["accuracy", "exact_match", "f1", "llm_judge"],
            "answer_format": "short_text",
        },
        "CFBench-Exam": {
            "description": "Exam-derived benchmark with explicit answers",
            "evaluated_by": "all_models",
            "metrics": ["accuracy", "exact_match"],
            "answer_format": "multiple_choice_or_short_text",
        },
        "CFBench-Hard": {
            "description": "Hard subset for showing performance gaps",
            "evaluated_by": "all_models",
            "metrics": ["accuracy", "exact_match", "f1", "llm_judge"],
            "answer_format": "mixed",
        },
        "CFBench-AgentTask": {
            "description": "Agent task evaluation for data construction capabilities",
            "evaluated_by": "all_models",
            "metrics": ["rubric_score", "llm_judge"],
            "answer_format": "structured_json",
        },
        "CFBench-Core": {
            "description": "Balanced official test set for main paper results",
            "evaluated_by": "all_models",
            "metrics": ["accuracy", "exact_match", "f1", "llm_judge"],
            "answer_format": "mixed",
        },
        "CFBench-Full": {
            "description": "All validated items",
            "evaluated_by": "all_models",
            "metrics": ["accuracy", "exact_match", "f1", "llm_judge"],
            "answer_format": "mixed",
        },
    },

    "metrics": {
        "accuracy": {
            "description": "Exact match accuracy",
            "formula": "correct / total",
            "applies_to": ["multiple_choice", "true_false", "short_text"],
        },
        "exact_match": {
            "description": "Exact string match after normalization",
            "formula": "normalized_match / total",
            "applies_to": ["short_text", "fill_blank"],
        },
        "f1": {
            "description": "Token-level F1 score",
            "formula": "2 * precision * recall / (precision + recall)",
            "applies_to": ["short_text", "long_text"],
        },
        "llm_judge": {
            "description": "LLM-based judgment using rubric",
            "formula": "judge_score / max_score",
            "applies_to": ["all"],
        },
        "numeric_tolerance": {
            "description": "Numeric answer within tolerance",
            "formula": "|predicted - gold| / |gold| < 0.05",
            "applies_to": ["calculation"],
        },
        "rubric_score": {
            "description": "Scoring rubric-based evaluation",
            "formula": "rubric_score / max_rubric_score",
            "applies_to": ["agent_task"],
        },
    },

    "answer_formats": {
        "multiple_choice": "Select from A/B/C/D options",
        "short_text": "Short text answer (1-50 chars)",
        "long_text": "Long text answer (50+ chars)",
        "numeric": "Numeric value with optional unit",
        "structured_json": "JSON object with required fields",
    },

    "evaluation_rules": [
        "Text-only models evaluate only CFBench-Text, CFBench-Exam, CFBench-Hard (text items), CFBench-Core (text items)",
        "Multimodal models evaluate all subsets",
        "For models without image support, skip multimodal items and report text-only accuracy separately",
        "Report both overall accuracy and per-task-type accuracy",
        "Report both overall accuracy and per-difficulty accuracy",
        "For agent-task items, use LLM judge with scoring rubric",
        "For calculation items, use numeric tolerance (5%)",
        "Normalize answers before comparison (strip whitespace, lowercase for English)",
        "Report token cost and latency alongside accuracy",
        "Report hallucination rate for open-ended questions",
    ],

    "recommended_tables": [
        "Table 1: Overall CFBench-Core results (all models)",
        "Table 2: CFBench-Text results (text-only models)",
        "Table 3: CFBench-MM results (multimodal models)",
        "Table 4: CFBench-Hard results (performance gap analysis)",
        "Table 5: CFBench-AgentTask results (agent capability)",
        "Table 6: Cost-performance comparison",
        "Table 7: Per-task-type breakdown",
        "Table 8: Per-difficulty breakdown",
    ],
}


def save_evaluation_protocol() -> tuple[Path, Path]:
    """Save evaluation protocol as JSON and MD."""
    benchmark_dir = PROJECT_ROOT / "data" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    json_path = benchmark_dir / "EVALUATION_PROTOCOL.json"
    md_path = benchmark_dir / "EVALUATION_PROTOCOL.md"

    with open(json_path, "w") as f:
        json.dump(EVALUATION_PROTOCOL, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# CARBON FIBER BENCHMARK EVALUATION PROTOCOL\n\n")

        f.write("## 1. Subset Evaluation Rules\n\n")
        for subset, rules in EVALUATION_PROTOCOL["subsets"].items():
            f.write(f"### {subset}\n\n")
            f.write(f"- Description: {rules['description']}\n")
            f.write(f"- Evaluated by: {rules['evaluated_by']}\n")
            f.write(f"- Metrics: {', '.join(rules['metrics'])}\n")
            f.write(f"- Answer format: {rules['answer_format']}\n\n")

        f.write("## 2. Metrics\n\n")
        for metric, details in EVALUATION_PROTOCOL["metrics"].items():
            f.write(f"### {metric}\n\n")
            f.write(f"- Description: {details['description']}\n")
            f.write(f"- Formula: {details['formula']}\n")
            f.write(f"- Applies to: {', '.join(details['applies_to'])}\n\n")

        f.write("## 3. Evaluation Rules\n\n")
        for i, rule in enumerate(EVALUATION_PROTOCOL["evaluation_rules"], 1):
            f.write(f"{i}. {rule}\n")

        f.write("\n## 4. Recommended Paper Tables\n\n")
        for table in EVALUATION_PROTOCOL["recommended_tables"]:
            f.write(f"- {table}\n")

    return json_path, md_path
