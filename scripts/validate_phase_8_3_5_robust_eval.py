"""Phase 8.3.5: Validate outputs.

Usage:
    python scripts/validate_phase_8_3_5_robust_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_3_5_robust_eval"
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_3_5_robust_eval"
    checks = []
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            checks.append(f"[PASS] {name}")
        else:
            failed += 1
            checks.append(f"[FAIL] {name}")

    check("Canonical 150 rescore exists", (report_dir / "canonical_150_rescore_report.json").exists())
    check("Canonical significance exists", (report_dir / "significance_canonical_150.json").exists())
    check("Large eval manifest exists", (eval_dir / "large_eval_manifest.jsonl").exists())
    check("Large eval report exists", (report_dir / "large_eval_report.json").exists())
    check("Large eval significance exists", (report_dir / "significance_large_eval.json").exists())
    check("Data efficiency analysis exists", (report_dir / "data_efficiency_analysis.json").exists())
    check("Diagnostic analysis exists", (report_dir / "diagnostic_analysis.json").exists())
    check("Case summary exists", (eval_dir / "cases" / "case_summary.json").exists())
    check("Paper tables exist", (eval_dir / "paper_tables" / "table_main_finetuning_results.csv").exists())
    check("LaTeX tables exist", (PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_3_5_finetuning_results.tex").exists())
    check("Cloud decision exists", (PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_4_cloud_7b_decision.md").exists())

    # Check large eval outputs
    check("Base large eval outputs exist", (eval_dir / "large_eval" / "base_outputs.jsonl").exists())
    check("Gold100 large eval outputs exist", (eval_dir / "large_eval" / "gold100_outputs.jsonl").exists())
    check("V4full large eval outputs exist", (eval_dir / "large_eval" / "v4full_outputs.jsonl").exists())

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_3_5.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
