"""Phase 7: Evaluate fine-tuned model on CFBench.

Usage:
    python scripts/run_phase_7_evaluate_finetuned.py \
        --base_model <model_path> \
        --finetuned_model <model_path> \
        --benchmark_subset text \
        --max_items 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--base_model", type=str, default="", help="Base model path")
    parser.add_argument("--finetuned_model", type=str, default="", help="Fine-tuned model path")
    parser.add_argument("--benchmark_subset", type=str, default="text", help="Benchmark subset")
    parser.add_argument("--max_items", type=int, default=50, help="Max items to evaluate")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro", help="Judge model")
    args = parser.parse_args()

    from src.autodata.finetuning.evaluate_finetuned_model import (
        evaluate_model_on_benchmark, load_benchmark_subset,
    )
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark subset
    items = load_benchmark_subset(PROJECT_ROOT, args.benchmark_subset)
    if not items:
        print(f"No benchmark items found for subset: {args.benchmark_subset}")
        return

    print(f"Loaded {len(items)} items from {args.benchmark_subset}")
    if args.max_items > 0:
        items = items[:args.max_items]

    # Create clients
    judge_client = UnifiedModelClient(model_name=args.judge_model)

    results = {}

    # Evaluate base model
    if args.base_model:
        print(f"\nEvaluating base model: {args.base_model}")
        base_client = UnifiedModelClient(model_name=args.base_model)
        base_results = evaluate_model_on_benchmark(base_client, items, judge_client)
        results["base"] = base_results
        print(f"  Accuracy: {base_results['accuracy']:.1%}")

    # Evaluate fine-tuned model
    if args.finetuned_model:
        print(f"\nEvaluating fine-tuned model: {args.finetuned_model}")
        ft_client = UnifiedModelClient(model_name=args.finetuned_model)
        ft_results = evaluate_model_on_benchmark(ft_client, items, judge_client)
        results["finetuned"] = ft_results
        print(f"  Accuracy: {ft_results['accuracy']:.1%}")

    # Save results
    with open(report_dir / "finetuned_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nEvaluation results saved.")


if __name__ == "__main__":
    main()
