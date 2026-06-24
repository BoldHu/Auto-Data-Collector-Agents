"""Phase 8.2: Validate outputs.

Usage:
    python scripts/validate_phase_8_2_lora_degradation_debug.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_2_lora_degradation_debug"
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

    check("Result consistency audit exists", (report_dir / "result_consistency_audit.json").exists())
    check("Train/eval mismatch audit exists", (report_dir / "train_eval_mismatch_audit.json").exists())
    check("Label masking audit exists", (report_dir / "label_masking_audit.json").exists())
    check("Format-aligned SFT report exists", (report_dir / "format_aligned_sft_report.json").exists())
    check("Format-aligned training summary exists", (report_dir / "lora_format_aligned_training_summary.json").exists())
    check("Format-aligned eval report exists", (report_dir / "format_aligned_eval_report.json").exists())
    check("Root cause analysis exists", (report_dir / "root_cause_analysis.json").exists())
    check("Phase 8.2 report exists", (report_dir / "PHASE_8_2_REPORT.md").exists())

    # Check adapter exists
    adapter_path = PROJECT_ROOT / "outputs" / "phase_8_2_lora_degradation_debug" / "lora_format_aligned_200"
    check("Format-aligned adapter exists", adapter_path.exists())

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_2.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
