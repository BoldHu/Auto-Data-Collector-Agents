"""Run Phase 6.6 system ablation.

Usage:
    python scripts/run_phase_6_6_system_ablation.py \
        --model deepseek-v4-flash \
        --judge_model doubao-seed-2.0-pro \
        --max_tasks 100 \
        --max_workers 16 \
        --judge_workers 8 \
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
    parser = argparse.ArgumentParser(description="Phase 6.6 system ablation")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro")
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--judge_workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    from src.autodata.evaluation.ablation_subset_builder import build_ablation_subset, save_ablation_subset
    from src.autodata.evaluation.system_ablation_runner import run_ablation

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_6"
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_6_system_ablation"

    # Build ablation subset
    print("Building ablation subset...")
    items, stats = build_ablation_subset()
    save_ablation_subset(items, stats)
    print(f"  Subset: {stats['total_items']} items")

    # Limit items based on stage
    if args.smoke_test:
        items = items[:5]
        print(f"  Smoke test: {len(items)} items")
    elif args.pilot:
        items = items[:30]
        print(f"  Pilot: {len(items)} items")
    else:
        items = items[:args.max_tasks]
        print(f"  Full run: {len(items)} items")

    # Systems to evaluate
    systems = ["direct_llm", "single_react", "plan_execute", "broadcast", "static_router", "dtcg"]

    # Run ablation
    print(f"\nStarting ablation with {len(systems)} systems on {len(items)} items...")
    print(f"Main model: {args.model}")
    print(f"Judge model: {args.judge_model}")
    print(f"Workers: {args.max_workers}")

    summary = run_ablation(
        items=items,
        systems=systems,
        model_name=args.model,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        judge_workers=args.judge_workers,
        output_dir=output_dir,
        report_dir=report_dir,
        resume=args.resume,
    )

    # Print results
    print(f"\n=== Phase 6.6 System Ablation Complete ===")
    print(f"Total elapsed: {summary.get('total_elapsed_formatted', 'N/A')}")
    print(f"\nSystem Results:")
    for system, data in summary.get("systems", {}).items():
        print(f"  {system}: accuracy={data.get('accuracy', 0):.2%}, judge={data.get('avg_judge_score', 0):.3f}, context_tokens={data.get('avg_context_tokens', 0)}, saving={data.get('context_saving_ratio', 0):.1%}")

    # Save to Phase 6.6 report dir
    with open(report_dir / "full_ablation_result.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
