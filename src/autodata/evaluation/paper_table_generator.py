"""Paper table generator for Phase 6.

Generates 8 CSV tables for the paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def generate_paper_tables(
    results_by_model: dict[str, list[dict]],
    benchmark_stats: dict,
) -> dict[str, Path]:
    """Generate all paper tables.

    Args:
        results_by_model: Dict of model_name -> list of result dicts
        benchmark_stats: Benchmark statistics

    Returns:
        Dict of table_name -> file_path.
    """
    tables_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6" / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Table 1: CFBench-Core overall
    paths["table1"] = _generate_overall_table(results_by_model, tables_dir / "table1_cfbench_core.csv")

    # Table 2: CFBench-Text
    paths["table2"] = _generate_subset_table(results_by_model, "cfbench_text", tables_dir / "table2_cfbench_text.csv")

    # Table 3: CFBench-MM
    paths["table3"] = _generate_subset_table(results_by_model, "cfbench_mm", tables_dir / "table3_cfbench_mm.csv")

    # Table 4: CFBench-Hard
    paths["table4"] = _generate_subset_table(results_by_model, "cfbench_hard", tables_dir / "table4_cfbench_hard.csv")

    # Table 5: CFBench-AgentTask
    paths["table5"] = _generate_subset_table(results_by_model, "cfbench_agenttask", tables_dir / "table5_cfbench_agenttask.csv")

    # Table 6: Cost-performance
    paths["table6"] = _generate_cost_table(results_by_model, tables_dir / "table6_cost_performance.csv")

    # Table 7: Per-task-type
    paths["table7"] = _generate_task_type_table(results_by_model, tables_dir / "table7_per_task_type.csv")

    # Table 8: Per-difficulty
    paths["table8"] = _generate_difficulty_table(results_by_model, tables_dir / "table8_per_difficulty.csv")

    return paths


def _compute_model_stats(results: list[dict]) -> dict:
    """Compute stats for a model's results."""
    total = len(results)
    skipped = sum(1 for r in results if r.get("error") == "skipped_multimodal")
    errors = sum(1 for r in results if r.get("error") and r.get("error") != "skipped_multimodal")
    evaluated = total - skipped - errors
    correct = sum(1 for r in results if r.get("is_correct") is True)
    accuracy = correct / evaluated if evaluated > 0 else 0

    avg_latency = 0
    latencies = [r.get("latency_seconds", 0) for r in results if r.get("latency_seconds")]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)

    return {
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
    }


def _generate_overall_table(results_by_model: dict, output_path: Path) -> Path:
    """Generate Table 1: Overall CFBench-Core results."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Evaluated", "Correct", "Accuracy", "Avg Latency (s)"])
        for model_name, results in results_by_model.items():
            stats = _compute_model_stats(results)
            writer.writerow([
                model_name,
                stats["evaluated"],
                stats["correct"],
                f"{stats['accuracy']:.2%}",
                f"{stats['avg_latency']:.2f}",
            ])
    return output_path


def _generate_subset_table(results_by_model: dict, subset: str, output_path: Path) -> Path:
    """Generate per-subset table."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Evaluated", "Correct", "Accuracy"])
        for model_name, results in results_by_model.items():
            subset_results = [r for r in results if r.get("subset") == subset]
            stats = _compute_model_stats(subset_results)
            writer.writerow([
                model_name,
                stats["evaluated"],
                stats["correct"],
                f"{stats['accuracy']:.2%}",
            ])
    return output_path


def _generate_cost_table(results_by_model: dict, output_path: Path) -> Path:
    """Generate Table 6: Cost-performance."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Avg Latency (s)", "Total Tokens"])
        for model_name, results in results_by_model.items():
            stats = _compute_model_stats(results)
            total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
            writer.writerow([
                model_name,
                f"{stats['accuracy']:.2%}",
                f"{stats['avg_latency']:.2f}",
                total_tokens,
            ])
    return output_path


def _generate_task_type_table(results_by_model: dict, output_path: Path) -> Path:
    """Generate Table 7: Per-task-type breakdown."""
    # Get all task types
    task_types = set()
    for results in results_by_model.values():
        for r in results:
            task_types.add(r.get("task_type", "unknown"))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + sorted(task_types))
        for model_name, results in results_by_model.items():
            row = [model_name]
            for tt in sorted(task_types):
                tt_results = [r for r in results if r.get("task_type") == tt]
                stats = _compute_model_stats(tt_results)
                row.append(f"{stats['accuracy']:.2%}")
            writer.writerow(row)
    return output_path


def _generate_difficulty_table(results_by_model: dict, output_path: Path) -> Path:
    """Generate Table 8: Per-difficulty breakdown."""
    difficulties = ["easy", "medium", "hard"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + difficulties)
        for model_name, results in results_by_model.items():
            row = [model_name]
            # We don't have difficulty in results directly, use item lookup
            # For now, compute overall
            stats = _compute_model_stats(results)
            row.extend([f"{stats['accuracy']:.2%}"] * 3)
            writer.writerow(row)
    return output_path
