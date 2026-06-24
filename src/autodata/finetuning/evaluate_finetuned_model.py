"""Post-finetuning evaluation script.

Evaluates base vs fine-tuned model on CFBench subsets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def evaluate_model_on_benchmark(
    model_client,
    benchmark_items: list[dict],
    judge_client=None,
    max_items: int = 0,
) -> dict:
    """Evaluate a model on benchmark items.

    Args:
        model_client: Model client for inference
        benchmark_items: List of benchmark items
        judge_client: Optional judge client for scoring
        max_items: Max items to evaluate (0 = all)

    Returns:
        dict with evaluation results
    """
    from src.autodata.evaluation.system_ablation_judge import judge_response, rule_based_check
    from src.autodata.evaluation.system_trace_schema import AblationTrace

    if max_items > 0:
        benchmark_items = benchmark_items[:max_items]

    results = []
    correct = 0
    total = 0
    errors = 0

    for item in benchmark_items:
        try:
            question = item.get("question", "")
            evidence_parts = item.get("evidence", [])
            evidence = "\n".join(str(e) for e in evidence_parts[:3]) if evidence_parts else ""

            # Simple QA prompt
            prompt = f"问题：{question}"
            if evidence:
                prompt += f"\n\n证据：{evidence}"
            prompt += "\n\n请直接回答。"

            response = model_client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )

            trace = AblationTrace(
                benchmark_id=item.get("benchmark_id", ""),
                raw_answer=response.content,
                parsed_answer=response.content.strip()[:200],
                gold_answer=str(item.get("answer", "")),
                system_type="finetuned_eval",
            )

            # Judge
            rule_result = rule_based_check(item, trace)
            if rule_result:
                trace.judge_score = rule_result.get("final_score")
                trace.is_correct = rule_result.get("verdict") == "correct"
            elif judge_client:
                judge_result = judge_response(judge_client, item, trace)
                trace.judge_score = judge_result.get("final_score")
                trace.is_correct = judge_result.get("verdict") == "correct"

            if trace.is_correct:
                correct += 1
            total += 1

            results.append({
                "benchmark_id": item.get("benchmark_id", ""),
                "task_type": item.get("task_type", ""),
                "is_correct": trace.is_correct,
                "judge_score": trace.judge_score,
            })

        except Exception as e:
            errors += 1
            results.append({
                "benchmark_id": item.get("benchmark_id", ""),
                "error": str(e)[:100],
            })

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "errors": errors,
        "results": results,
    }


def load_benchmark_subset(project_root: Path, subset_name: str) -> list[dict]:
    """Load a benchmark subset."""
    subset_dir = project_root / "data" / "benchmark" / "subsets"

    # Try different naming conventions
    candidates = [
        subset_dir / f"{subset_name}.jsonl",
        subset_dir / f"cfbench_{subset_name}.jsonl",
        subset_dir / f"carbon_fiber_benchmark_{subset_name}.jsonl",
    ]

    for path in candidates:
        if path.exists():
            items = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            return items

    return []
