"""Phase 6.9: Targeted rerun after DTCG repair.

Reruns 120 tasks with repaired DTCG and baselines.

Usage:
    python scripts/run_phase_6_9_targeted_rerun.py \
        --model deepseek-v4-flash \
        --judge_model doubao-seed-2.0-pro \
        --max_workers 8 \
        --judge_workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 6.9 targeted rerun")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--judge_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=50)
    args = parser.parse_args()

    from src.autodata.evaluation.unified_model_client import UnifiedModelClient
    from src.autodata.evaluation.system_baselines import SYSTEM_MAP
    from src.autodata.evaluation.system_ablation_judge import judge_response, rule_based_check

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_9"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_9_dtcg_diagnosis"
    report_dir.mkdir(parents=True, exist_ok=True)

    progress_file = report_dir / "progress_phase_6_9.json"
    log_file = report_dir / "progress_phase_6_9.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    # Load items
    lc_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "long_context_expanded.jsonl"
    stress_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "ablation_subset_stress.jsonl"

    all_items = []
    if lc_path.exists():
        lc = load_jsonl(lc_path)
        all_items.extend(("long_context", it) for it in lc[:args.max_items])
    if stress_path.exists():
        stress = load_jsonl(stress_path)
        all_items.extend(("stress", it) for it in stress[:args.max_items])

    all_items = all_items[:args.max_items * 2]
    log(f"Loaded {len(all_items)} items")

    # Systems to evaluate
    systems = ["single_react", "broadcast", "static_router", "dtcg"]

    # Create clients
    main_client = UnifiedModelClient(model_name=args.model)
    judge_client = UnifiedModelClient(model_name=args.judge_model)

    # Run ablation
    all_traces = []
    completed = 0
    errors = 0

    def run_one(args_tuple):
        subset, item, system = args_tuple
        try:
            func = SYSTEM_MAP[system]
            trace = func(main_client, item)
            # Judge
            rule_result = rule_based_check(item, trace)
            if rule_result:
                trace.judge_score = rule_result.get("final_score")
                trace.is_correct = rule_result.get("verdict") == "correct"
            else:
                judge_result = judge_response(judge_client, item, trace)
                trace.judge_score = judge_result.get("final_score")
                trace.is_correct = judge_result.get("verdict") == "correct"
            return trace.to_dict(), None
        except Exception as e:
            return None, str(e)

    # Build task list
    tasks = []
    for subset, item in all_items:
        for system in systems:
            tasks.append((subset, item, system))

    log(f"Running {len(tasks)} evaluations ({len(all_items)} items × {len(systems)} systems)")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            result, error = future.result()
            completed += 1
            if error:
                errors += 1
                log(f"  Error: {error[:80]}")
            elif result:
                all_traces.append(result)

            if completed % 20 == 0:
                log(f"  Progress: {completed}/{len(tasks)} ({errors} errors)")
                # Update progress
                with open(progress_file, "w") as f:
                    json.dump({"completed": completed, "total": len(tasks), "errors": errors}, f)

    # Save traces
    with open(output_dir / "targeted_rerun_traces.jsonl", "w") as f:
        for t in all_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Compute scores
    scores = {}
    for t in all_traces:
        system = t.get("system_type", "unknown")
        if system not in scores:
            scores[system] = {"total": 0, "correct": 0, "judge_sum": 0, "ctx_sum": 0, "fallback_count": 0}
        scores[system]["total"] += 1
        if t.get("is_correct"):
            scores[system]["correct"] += 1
        scores[system]["judge_sum"] += t.get("judge_score", 0) or 0
        scores[system]["ctx_sum"] += t.get("selected_context_tokens", 0)
        if t.get("fallback_used"):
            scores[system]["fallback_count"] += 1

    # Save scores CSV
    with open(output_dir / "targeted_rerun_scores.csv", "w") as f:
        f.write("System,Total,Correct,Accuracy,AvgJudgeScore,AvgContextTokens,FallbackCount\n")
        for system, data in scores.items():
            acc = data["correct"] / max(data["total"], 1)
            avg_judge = data["judge_sum"] / max(data["total"], 1)
            avg_ctx = data["ctx_sum"] / max(data["total"], 1)
            f.write(f"{system},{data['total']},{data['correct']},{acc:.3f},{avg_judge:.3f},{avg_ctx:.0f},{data['fallback_count']}\n")

    # Print summary
    log(f"\n=== Phase 6.9 Targeted Rerun Complete ===")
    for system, data in scores.items():
        acc = data["correct"] / max(data["total"], 1)
        avg_judge = data["judge_sum"] / max(data["total"], 1)
        log(f"  {system}: acc={acc:.1%}, judge={avg_judge:.3f}, total={data['total']}, fallback={data['fallback_count']}")

    log(f"Total traces: {len(all_traces)}, Errors: {errors}")
    log(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
