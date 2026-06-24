"""Phase 8.3: Validate outputs.

Usage:
    python scripts/validate_phase_8_3_v4_full_lora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_3_v4_full_lora"
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

    check("Preflight exists", (report_dir / "preflight_phase_8_3.json").exists())
    check("Frozen baseline scores exist", (report_dir / "frozen_baseline_scores.json").exists())
    check("V4 full training summary exists", (report_dir / "lora_v4_full_training_summary.json").exists())
    check("V4 full eval report exists", (report_dir / "v4_full_eval_report.json").exists())
    check("Unified comparison exists", (report_dir / "unified_comparison_report.json").exists())
    check("Phase 8.3 report exists", (report_dir / "PHASE_8_3_REPORT.md").exists())

    # Check adapter exists
    adapter_path = PROJECT_ROOT / "outputs" / "phase_8_3_v4_full_lora" / "lora_v4_full"
    check("V4 full adapter exists", adapter_path.exists())

    # Check training success
    training_path = report_dir / "lora_v4_full_training_summary.json"
    if training_path.exists():
        with open(training_path) as f:
            training = json.load(f)
        check("Training completed", training.get("status") == "success")

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_3.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
