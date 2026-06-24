"""Run Phase 6.8 system ablation on expanded long-context and stress subsets.

Usage:
    python scripts/run_phase_6_8_system_ablation.py \
        --model deepseek-v4-flash \
        --judge_model doubao-seed-2.0-pro \
        --max_workers 12 \
        --judge_workers 6
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
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 6.8 system ablation")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro")
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--judge_workers", type=int, default=6)
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from src.autodata.evaluation.system_ablation_runner import run_ablation

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_8"
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Systems to evaluate
    systems = ["direct_llm", "single_react", "plan_execute", "broadcast", "static_router", "dtcg"]

    # Load subsets
    long_context_path = output_dir / "long_context_expanded.jsonl"
    stress_path = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "ablation_subset_stress.jsonl"

    subsets = {}
    if long_context_path.exists():
        items = load_jsonl(long_context_path)
        subsets["long_context_expanded"] = items[:args.max_tasks]
    if stress_path.exists():
        items = load_jsonl(stress_path)
        subsets["stress"] = items[:args.max_tasks]

    if not subsets:
        print("ERROR: No subsets found. Run build_phase_6_8_long_context_subset.py first.")
        return

    # Run ablation on each subset
    all_results = {}
    for subset_name, items in subsets.items():
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
    with open(report_dir / "phase_6_8_combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Phase 6.8 Ablation Complete ===")
    print(f"Results saved to: {report_dir}")


if __name__ == "__main__":
    main()
