"""Phase 8.1B: Validate outputs.

Usage:
    python scripts/validate_phase_8_1b_gold_full_lora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1b_gold_full_lora"
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

    check("Phase 8.1A audit exists", (report_dir / "phase8_1a_consistency_audit.json").exists())
    check("Fixed eval set exists", (report_dir / "fixed_eval_set_stats.json").exists())
    check("Gold full data check exists", (report_dir / "gold_full_data_check.json").exists())
    check("Base v3 eval exists", (report_dir / "base_v3_eval_report.json").exists())
    check("Gold_100 v3 eval exists", (report_dir / "lora_gold100_v3_eval_report.json").exists())
    check("Gold full training exists", (report_dir / "lora_gold_full_training_summary.json").exists())
    check("Gold full eval exists", (report_dir / "lora_gold_full_eval_report.json").exists())
    check("Analysis exists", (report_dir / "phase_8_1b_analysis.json").exists())
    check("Phase 8.1B report exists", (report_dir / "PHASE_8_1B_REPORT.md").exists())

    # Check training success
    training_path = report_dir / "lora_gold_full_training_summary.json"
    if training_path.exists():
        with open(training_path) as f:
            training = json.load(f)
        check("Training completed", training.get("status") == "success")

    # Check adapter exists
    adapter_path = PROJECT_ROOT / "outputs" / "phase_8_1b_gold_full_lora" / "lora_gold_full"
    check("Adapter directory exists", adapter_path.exists())

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_1b.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
