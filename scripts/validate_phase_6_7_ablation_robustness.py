"""Validate Phase 6.7 outputs.

Usage:
    python scripts/validate_phase_6_7_ablation_robustness.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "phase_6_7_ablation_robustness"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase_6_7"


def main():
    checks = {}

    print("1. Checking file existence...")
    files = {
        "stress_stats": REPORT_DIR / "stress_subset_statistics.json",
        "token_audit": REPORT_DIR / "token_accounting_audit.json",
        "combined_results": REPORT_DIR / "phase_6_7_combined_results.json",
        "dtcg_component_report": REPORT_DIR / "dtcg_component_ablation_report.md",
        "standard_subset": EVAL_DIR / "ablation_subset_standard.jsonl",
        "long_context_subset": EVAL_DIR / "ablation_subset_long_context.jsonl",
        "stress_subset": EVAL_DIR / "ablation_subset_stress.jsonl",
        "dtcg_traces": EVAL_DIR / "dtcg_component_ablation_traces.jsonl",
        "dtcg_scores": EVAL_DIR / "dtcg_component_ablation_scores.csv",
    }
    for name, path in files.items():
        exists = path.exists()
        checks[f"exists_{name}"] = exists
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    # Check subset counts
    print("\n2. Checking subset counts...")
    for subset in ["standard", "long_context", "stress"]:
        path = EVAL_DIR / f"ablation_subset_{subset}.jsonl"
        if path.exists():
            count = sum(1 for line in open(path) if line.strip())
            checks[f"count_{subset}"] = count
            print(f"  {subset}: {count} items")

    # Check DTCG traces
    print("\n3. Checking DTCG component traces...")
    traces_path = EVAL_DIR / "dtcg_component_ablation_traces.jsonl"
    if traces_path.exists():
        count = sum(1 for line in open(traces_path) if line.strip())
        checks["dtcg_traces_count"] = count
        print(f"  Traces: {count}")

    # Check for API keys
    print("\n4. Checking no API keys...")
    for path in EVAL_DIR.rglob("*.jsonl"):
        with open(path) as f:
            content = f.read(5000)
            if "tp-" in content or "sk-" in content:
                checks["api_key_leak"] = True
                print(f"  WARNING: Possible API key in {path.name}")
                break
    else:
        checks["api_key_leak"] = False
        print("  OK: No API keys found")

    # Check report exists
    print("\n5. Checking report...")
    report_path = PROJECT_ROOT / "reports" / "phase_6_7_ablation_robustness" / "PHASE_6_7_REPORT.md"
    checks["report_exists"] = report_path.exists()
    print(f"  Report: {'OK' if report_path.exists() else 'MISSING'}")

    # Overall
    critical = [
        checks.get("exists_combined_results", False),
        checks.get("exists_dtcg_scores", False),
        checks.get("exists_standard_subset", False),
        checks.get("api_key_leak", True) is False,
    ]
    all_passed = all(critical)
    checks["validation_result"] = "PASS" if all_passed else "FAIL"
    checks["timestamp"] = time.time()

    print(f"\n=== Validation Result: {'PASS' if all_passed else 'FAIL'} ===")

    with open(REPORT_DIR / "validation_phase_6_7.json", "w") as f:
        json.dump(checks, f, indent=2)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
