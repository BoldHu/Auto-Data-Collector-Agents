"""Run Phase 6 baseline evaluation on CFBench.

Usage:
    python scripts/run_phase_6_baseline_evaluation.py \
        --run_id phase_6_baseline \
        --benchmark_root data/benchmark/subsets \
        --max_workers 32 \
        --judge_workers 16 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


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


def main():
    parser = argparse.ArgumentParser(description="Phase 6 baseline evaluation")
    parser.add_argument("--run_id", type=str, default="phase_6_baseline")
    parser.add_argument("--benchmark_root", type=str, default="data/benchmark/subsets")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--judge_workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test only")
    parser.add_argument("--dev_only", action="store_true", help="Run dev evaluation only")
    args = parser.parse_args()

    from src.autodata.evaluation.model_registry import load_model_registry, save_model_registry
    from src.autodata.evaluation.evaluation_runner import (
        run_evaluation, save_results, compute_summary, load_jsonl
    )
    from src.autodata.utils.model_pool import get_model_pool

    benchmark_root = Path(args.benchmark_root)
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_baseline_evaluation"
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6"

    # Load models
    models = load_model_registry()
    save_model_registry(models)
    print(f"Loaded {len(models)} models")

    # Judge pool (Xiaomi API_KEY1)
    judge_pool = get_model_pool(use_key2=False)

    # Define subsets to evaluate
    subsets = {
        "cfbench_text": benchmark_root / "cfbench_text_test.jsonl",
        "cfbench_exam": benchmark_root / "cfbench_exam_test.jsonl",
        "cfbench_core": benchmark_root / "cfbench_core_test.jsonl",
        "cfbench_hard": benchmark_root / "cfbench_hard_test.jsonl",
        "cfbench_agenttask": benchmark_root / "cfbench_agenttask_test.jsonl",
    }

    # Smoke test: 10 items per subset, 2 models
    if args.smoke_test:
        print("\n=== Smoke Test ===")
        smoke_models = models[:2]  # First 2 models
        smoke_results = []

        for subset_name, subset_path in subsets.items():
            items = load_jsonl(subset_path)[:10]
            if not items:
                continue
            print(f"  {subset_name}: {len(items)} items x {len(smoke_models)} models")

            checkpoint_path = report_dir / f"smoke_checkpoint_{subset_name}.json"
            results = run_evaluation(
                smoke_models, items, subset_name, args.run_id,
                eval_dir / "raw_outputs", max_workers=4,
                judge_pool=judge_pool, checkpoint_path=checkpoint_path,
            )
            smoke_results.extend(results)

        save_results(smoke_results, report_dir / "smoke_test_results.jsonl")

        # Summary
        summary = compute_summary(smoke_results)
        with open(report_dir / "smoke_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Smoke test: {summary['accuracy']:.2%} accuracy ({summary['correct']}/{summary['evaluated']})")
        print(f"Results: {report_dir / 'smoke_test_results.json'}")
        return

    # Full evaluation
    print("\n=== Full Evaluation ===")
    all_results = []
    start_time = time.time()

    for subset_name, subset_path in subsets.items():
        items = load_jsonl(subset_path)
        if not items:
            print(f"  {subset_name}: SKIP (no items)")
            continue

        print(f"  {subset_name}: {len(items)} items x {len(models)} models")

        checkpoint_path = report_dir / f"checkpoint_{subset_name}.json"
        results = run_evaluation(
            models, items, subset_name, args.run_id,
            eval_dir / "raw_outputs", max_workers=args.max_workers,
            judge_pool=judge_pool, checkpoint_path=checkpoint_path,
        )
        all_results.extend(results)

        # Save per-subset results
        save_results(results, eval_dir / "parsed_predictions" / f"{subset_name}_predictions.jsonl")

        # Per-subset summary
        summary = compute_summary(results)
        with open(report_dir / f"{subset_name}_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"    Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['evaluated']})")

    # Overall summary
    elapsed = time.time() - start_time
    overall = compute_summary(all_results)

    with open(report_dir / "test_results_summary.json", "w") as f:
        json.dump({
            "run_id": args.run_id,
            "elapsed_seconds": elapsed,
            "total_results": len(all_results),
            "overall": overall,
        }, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Total results: {len(all_results)}")
    print(f"Overall accuracy: {overall['accuracy']:.2%}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Results: {report_dir}")


if __name__ == "__main__":
    main()
