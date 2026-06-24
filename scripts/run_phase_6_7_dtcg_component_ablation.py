"""Run Phase 6.7 DTCG component ablation.

Usage:
    python scripts/run_phase_6_7_dtcg_component_ablation.py \
        --model deepseek-v4-flash \
        --judge_model doubao-seed-2.0-pro \
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
    parser = argparse.ArgumentParser(description="Phase 6.7 DTCG component ablation")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge_model", type=str, default="doubao-seed-2.0-pro")
    parser.add_argument("--max_items", type=int, default=50)
    args = parser.parse_args()

    from src.autodata.evaluation.long_horizon_subset_builder import build_stress_subsets
    from src.autodata.evaluation.dtcg_component_ablation import run_component_ablation, compute_component_scores
    from src.autodata.evaluation.unified_model_client import UnifiedModelClient

    output_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build subsets
    print("Building subsets...")
    subsets = build_stress_subsets()

    # Take items from each subset
    items = []
    items.extend(subsets["standard"][:20])
    items.extend(subsets["long_context"][:15])
    items.extend(subsets["stress"][:15])
    items = items[:args.max_items]

    print(f"Running DTCG component ablation on {len(items)} items...")

    # Create clients
    main_client = UnifiedModelClient(model_name=args.model)
    judge_client = UnifiedModelClient(model_name=args.judge_model)

    # Run ablation
    traces = run_component_ablation(items, main_client, judge_client)

    # Save traces
    traces_path = output_dir / "dtcg_component_ablation_traces.jsonl"
    with open(traces_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    # Compute scores
    scores = compute_component_scores(traces)

    # Save scores as CSV
    csv_path = output_dir / "dtcg_component_ablation_scores.csv"
    with open(csv_path, "w") as f:
        f.write("Variant,Total,Correct,Accuracy,AvgJudgeScore,AvgContextTokens\n")
        for variant, data in scores.items():
            judge = data.get("avg_judge_score", data.get("avg_judge", 0))
            ctx = data.get("avg_context_tokens", data.get("avg_context", 0))
            f.write(f"{variant},{data['total']},{data.get('correct',0)},{data['accuracy']:.3f},{judge:.3f},{ctx}\n")

    # Save report
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_7_ablation_robustness"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "dtcg_component_ablation_report.md", "w") as f:
        f.write("# DTCG Component Ablation Report\n\n")
        f.write("| Variant | Total | Correct | Accuracy | Avg Judge | Avg Context |\n")
        f.write("|---------|-------|---------|----------|-----------|-------------|\n")
        for variant, data in scores.items():
            f.write(f"| {variant} | {data['total']} | {data.get('correct',0)} | {data['accuracy']:.1%} | {data['avg_judge_score']:.3f} | {data['avg_context_tokens']} |\n")

    print(f"\nDTCG Component Ablation Complete:")
    for variant, data in scores.items():
        print(f"  {variant}: acc={data['accuracy']:.1%}, judge={data['avg_judge_score']:.3f}, ctx={data['avg_context_tokens']}")


if __name__ == "__main__":
    main()
