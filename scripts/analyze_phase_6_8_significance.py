"""Analyze statistical significance for Phase 6.8.

Usage:
    python scripts/analyze_phase_6_8_significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.evaluation.statistical_analysis import run_significance_analysis, bootstrap_ci


def load_traces(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_6_8_evidence_consolidation"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load traces from all subsets
    subsets = {
        "long_context_expanded": PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "long_context_expanded",
        "stress": PROJECT_ROOT / "data" / "evaluation" / "phase_6_8" / "stress",
    }

    # Also load Phase 6.7 traces for reference
    phase67_subsets = {
        "standard": PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "standard",
        "long_context": PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "long_context",
        "stress": PROJECT_ROOT / "data" / "evaluation" / "phase_6_7" / "stress",
    }

    all_results = {}

    # Analyze Phase 6.7 traces (already available)
    for subset_name, subset_dir in phase67_subsets.items():
        traces_path = subset_dir / "system_ablation_traces.jsonl"
        if not traces_path.exists():
            continue

        traces = load_traces(traces_path)
        if not traces:
            continue

        # Group by system
        traces_by_system = {}
        for t in traces:
            sys_name = t.get("system_type", "unknown")
            if sys_name not in traces_by_system:
                traces_by_system[sys_name] = []
            traces_by_system[sys_name].append(t)

        # Run significance analysis
        result = run_significance_analysis(traces_by_system)
        all_results[f"phase67_{subset_name}"] = result

        # Print summary
        print(f"\n=== Phase 6.7 {subset_name} ===")
        for system, ci in result.get("bootstrap_cis", {}).items():
            acc_ci = ci.get("accuracy", {})
            print(f"  {system}: accuracy={acc_ci.get('mean', 0):.3f} [{acc_ci.get('ci_lower', 0):.3f}, {acc_ci.get('ci_upper', 0):.3f}]")

        for pair, comparison in result.get("pairwise_comparisons", {}).items():
            mc = comparison.get("mcnemar", {})
            print(f"  {pair}: p={mc.get('p_value', 'N/A')}, significant={mc.get('significant', False)}")

    # Analyze Phase 6.8 traces if available
    for subset_name, subset_dir in subsets.items():
        traces_path = subset_dir / "system_ablation_traces.jsonl"
        if not traces_path.exists():
            continue

        traces = load_traces(traces_path)
        if not traces:
            continue

        traces_by_system = {}
        for t in traces:
            sys_name = t.get("system_type", "unknown")
            if sys_name not in traces_by_system:
                traces_by_system[sys_name] = []
            traces_by_system[sys_name].append(t)

        result = run_significance_analysis(traces_by_system)
        all_results[f"phase68_{subset_name}"] = result

        print(f"\n=== Phase 6.8 {subset_name} ===")
        for system, ci in result.get("bootstrap_cis", {}).items():
            acc_ci = ci.get("accuracy", {})
            print(f"  {system}: accuracy={acc_ci.get('mean', 0):.3f} [{acc_ci.get('ci_lower', 0):.3f}, {acc_ci.get('ci_upper', 0):.3f}]")

        for pair, comparison in result.get("pairwise_comparisons", {}).items():
            mc = comparison.get("mcnemar", {})
            print(f"  {pair}: p={mc.get('p_value', 'N/A')}, significant={mc.get('significant', False)}")

    # Save results
    with open(report_dir / "statistical_significance.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save markdown report
    with open(report_dir / "statistical_significance.md", "w") as f:
        f.write("# Statistical Significance Analysis\n\n")
        for subset_name, result in all_results.items():
            f.write(f"## {subset_name}\n\n")
            f.write("| System | Accuracy | 95% CI |\n|--------|----------|--------|\n")
            for system, ci in result.get("bootstrap_cis", {}).items():
                acc = ci.get("accuracy", {})
                f.write(f"| {system} | {acc.get('mean', 0):.3f} | [{acc.get('ci_lower', 0):.3f}, {acc.get('ci_upper', 0):.3f}] |\n")
            f.write("\n### Pairwise Comparisons\n\n")
            f.write("| Comparison | McNemar p-value | Significant |\n|------------|----------------|-------------|\n")
            for pair, comp in result.get("pairwise_comparisons", {}).items():
                mc = comp.get("mcnemar", {})
                f.write(f"| {pair} | {mc.get('p_value', 'N/A')} | {mc.get('significant', False)} |\n")
            f.write("\n")

    print(f"\nResults saved to: {report_dir}")


if __name__ == "__main__":
    main()
