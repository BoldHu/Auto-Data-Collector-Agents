"""Run Phase 6.7 system ablation with corrected token accounting.

Usage:
    python scripts/run_phase_6_7_system_ablation.py \
        --model deepseek-v4-flash \
        --judge_model doubao-seed-2.0-pro \
        --max_tasks 100 \
        --max_workers 16 \
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


def main():
    parser = argparse.ArgumentParser(description="Phase 6.7 system ablation")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro")
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--judge_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    from src.autodata.evaluation.long_horizon_subset_builder import build_stress_subsets, save_subsets
    from src.autodata.evaluation.system_ablation_runner import run_ablation
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7"
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_7_ablation_robustness"

    # Build subsets
    print("Building stress-test subsets...")
    subsets = build_stress_subsets()
    subset_paths = save_subsets(subsets)
    print(f"  Standard: {subsets['stats']['standard']['count']}")
    print(f"  Long-context: {subsets['stats']['long_context']['count']}")
    print(f"  Stress: {subsets['stats']['stress']['count']}")

    # Systems
    systems = ["direct_llm", "single_react", "plan_execute", "broadcast", "static_router", "dtcg"]

    # Run on each subset
    all_results = {}
    for subset_name, items in subsets.items():
        if subset_name == "stats":
            continue

        if args.smoke_test:
            items = items[:5]

        print(f"\n=== Subset: {subset_name} ({len(items)} items) ===")

        subset_output = output_dir / subset_name
        subset_output.mkdir(parents=True, exist_ok=True)

        result = run_ablation(
            items=items,
            systems=systems,
            model_name=args.model,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
            judge_workers=args.judge_workers,
            output_dir=subset_output,
            report_dir=report_dir,
            resume=args.resume,
        )

        all_results[subset_name] = result

        # Print summary
        print(f"\n{subset_name} results:")
        for system, data in result.get("systems", {}).items():
            print(f"  {system}: acc={data.get('accuracy', 0):.2%}, judge={data.get('avg_judge_score', 0):.3f}, ctx={data.get('avg_context_tokens', 0)}")

    # Save combined results
    with open(report_dir / "phase_6_7_combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Phase 6.7 Complete ===")
    print(f"Results saved to: {report_dir}")


if __name__ == "__main__":
    main()
