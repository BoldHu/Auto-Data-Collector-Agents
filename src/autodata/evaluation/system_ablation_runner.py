"""Ablation runner for Phase 6.6.

Orchestrates all 6 systems on the ablation subset with checkpointing.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.autodata.evaluation.system_baselines import run_system, SYSTEM_MAP
from src.autodata.evaluation.system_ablation_judge import judge_response, rule_based_check
from src.autodata.evaluation.system_trace_schema import AblationTrace
from src.autodata.evaluation.unified_model_client import UnifiedModelClient

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f).get("completed_pairs", []))


def save_checkpoint(path: Path, completed: set):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"completed_pairs": list(completed), "count": len(completed)}, f)
    os.replace(tmp, path)


def run_ablation(
    items: list[dict],
    systems: list[str],
    model_name: str = "deepseek-v4-flash",
    judge_model: str = "doubao-seed-2.0-pro",
    max_workers: int = 16,
    judge_workers: int = 8,
    output_dir: Path = None,
    report_dir: Path = None,
    resume: bool = True,
) -> dict:
    """Run full ablation across all systems and items."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_6"
    if report_dir is None:
        report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_6_system_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Create clients
    main_client = UnifiedModelClient(model_name=model_name)
    judge_client = UnifiedModelClient(model_name=judge_model)

    # Output files
    traces_path = output_dir / "system_ablation_traces.jsonl"
    predictions_path = output_dir / "system_ablation_predictions.jsonl"
    judge_path = output_dir / "system_ablation_judge_results.jsonl"
    checkpoint_path = report_dir / "checkpoint_ablation.json"
    progress_path = report_dir / "progress_phase_6_6.json"
    progress_log = report_dir / "progress_phase_6_6.log"

    # Load checkpoint
    completed_pairs = load_checkpoint(checkpoint_path) if resume else set()

    all_traces = []
    start_time = time.time()

    # Run each system
    for system_type in systems:
        print(f"\n--- System: {system_type} ---", flush=True)

        items_to_eval = []
        for item in items:
            pair_key = f"{system_type}:{item.get('benchmark_id', '')}"
            if pair_key not in completed_pairs:
                items_to_eval.append(item)

        if not items_to_eval:
            print(f"  All items already completed for {system_type}")
            continue

        print(f"  Evaluating {len(items_to_eval)} items", flush=True)

        # Phase 1: Run all model calls concurrently
        raw_traces = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in items_to_eval:
                future = executor.submit(run_system, system_type, main_client, item)
                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                try:
                    trace = future.result()
                    raw_traces.append((item, trace))
                except Exception as e:
                    trace = AblationTrace(
                        task_id=item.get("benchmark_id", ""),
                        system_type=system_type,
                        error_type=str(e)[:100],
                    )
                    raw_traces.append((item, trace))

        print(f"  Model calls complete, judging {len(raw_traces)} results...", flush=True)

        # Phase 2: Judge all results concurrently
        def judge_one(item, trace):
            rule_result = rule_based_check(item, trace)
            if rule_result:
                trace.judge_score = rule_result.get("final_score")
                trace.is_correct = rule_result.get("verdict") == "correct"
            else:
                judge_result = judge_response(judge_client, item, trace)
                trace.judge_score = judge_result.get("final_score")
                trace.is_correct = judge_result.get("verdict") == "correct"
                trace.evidence_support = judge_result.get("evidence_support")
                trace.constraint_satisfaction = judge_result.get("constraint_satisfaction")
                trace.hallucination_flag = judge_result.get("hallucination", 0) > 0.5
            return trace

        with ThreadPoolExecutor(max_workers=judge_workers) as executor:
            judge_futures = {}
            for item, trace in raw_traces:
                future = executor.submit(judge_one, item, trace)
                judge_futures[future] = item

            for future in as_completed(judge_futures):
                try:
                    trace = future.result()
                    all_traces.append(trace)
                    pair_key = f"{system_type}:{trace.benchmark_id}"
                    completed_pairs.add(pair_key)
                except Exception as e:
                    item = judge_futures[future]
                    trace = AblationTrace(
                        task_id=item.get("benchmark_id", ""),
                        system_type=system_type,
                        error_type=f"judge_error:{str(e)[:80]}",
                    )
                    all_traces.append(trace)

                # Save checkpoint periodically
                if len(completed_pairs) % 20 == 0:
                    save_checkpoint(checkpoint_path, completed_pairs)

        # Save checkpoint after each system
        save_checkpoint(checkpoint_path, completed_pairs)
        print(f"  {system_type} complete: {len([t for t in all_traces if t.system_type == system_type])} traces", flush=True)

    # Save all traces
    with open(traces_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    # Compute summary
    summary = compute_summary(all_traces, systems)
    summary["total_elapsed_seconds"] = time.time() - start_time
    summary["total_elapsed_formatted"] = format_time(summary["total_elapsed_seconds"])

    # Save summary
    with open(report_dir / "full_ablation_result.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def compute_summary(traces: list[AblationTrace], systems: list[str]) -> dict:
    """Compute summary statistics per system."""
    summary = {"systems": {}, "total_traces": len(traces)}

    for system in systems:
        sys_traces = [t for t in traces if t.system_type == system and not t.error_type]
        errors = [t for t in traces if t.system_type == system and t.error_type]

        if not sys_traces:
            summary["systems"][system] = {"total": 0, "errors": len(errors)}
            continue

        correct = sum(1 for t in sys_traces if t.is_correct)
        judged = sum(1 for t in sys_traces if t.judge_score is not None)
        avg_judge = sum(t.judge_score or 0 for t in sys_traces) / judged if judged else 0
        avg_context = sum(t.selected_context_tokens for t in sys_traces) / len(sys_traces) if sys_traces else 0
        avg_broadcast = sum(t.broadcast_context_tokens for t in sys_traces) / len(sys_traces) if sys_traces else 0
        avg_latency = sum(t.latency_seconds for t in sys_traces) / len(sys_traces) if sys_traces else 0
        total_tokens = sum(t.total_input_tokens + t.total_output_tokens for t in sys_traces)
        avg_saving = sum(t.context_saving_ratio for t in sys_traces) / len(sys_traces) if sys_traces else 0

        summary["systems"][system] = {
            "total": len(sys_traces),
            "errors": len(errors),
            "correct": correct,
            "accuracy": correct / len(sys_traces) if sys_traces else 0,
            "judged": judged,
            "avg_judge_score": round(avg_judge, 3),
            "avg_context_tokens": round(avg_context),
            "avg_broadcast_tokens": round(avg_broadcast),
            "context_saving_ratio": round(avg_saving, 3),
            "avg_latency": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "cost_per_correct": round(total_tokens / correct, 0) if correct > 0 else float("inf"),
        }

    return summary


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
