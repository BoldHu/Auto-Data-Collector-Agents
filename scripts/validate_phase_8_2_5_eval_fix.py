"""Phase 8.2.5: Validate outputs.

Usage:
    python scripts/validate_phase_8_2_5_eval_fix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_2_5_eval_fix"
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

    check("Denominator bug audit exists", (report_dir / "denominator_bug_audit.json").exists())
    check("Canonical manifest exists", (report_dir / "canonical_eval_manifest_stats.json").exists())
    check("Adapter activation check exists", (report_dir / "adapter_activation_check.json").exists())
    check("Re-evaluation report exists", (report_dir / "reevaluation_150_report.json").exists())
    check("Label masking verification exists", (report_dir / "label_masking_verification.json").exists())
    check("Root cause update exists", (report_dir / "root_cause_update.json").exists())
    check("Phase 8.2.5 report exists", (report_dir / "PHASE_8_2_5_REPORT.md").exists())

    # Check re-evaluation outputs
    eval_dir = PROJECT_ROOT / "data" / "evaluation" / "phase_8_2_5_eval_fix"
    check("Base outputs exist", (eval_dir / "base_outputs_150.jsonl").exists())
    check("Gold100 outputs exist", (eval_dir / "gold100_outputs_150.jsonl").exists())
    check("Goldfull outputs exist", (eval_dir / "goldfull_outputs_150.jsonl").exists())
    check("Formataligned outputs exist", (eval_dir / "formataligned_outputs_150.jsonl").exists())

    # Check comparison CSV
    check("Comparison CSV exists", (eval_dir / "base_vs_all_adapters_150.csv").exists())

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_2_5.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
