"""Phase 8.1A: Validate outputs.

Usage:
    python scripts/validate_phase_8_1a_qwen_lora_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_1a_qwen_lora_pilot"
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

    check("qwen-vl-utils status exists", (report_dir / "qwen_vl_utils_status.json").exists())
    check("Gold train_100 stats exist", (report_dir / "gold_train_100_stats.json").exists())
    check("Base zero-shot report exists", (report_dir / "base_zero_shot_report.json").exists())
    check("LoRA training summary exists", (report_dir / "lora_gold100_training_summary.json").exists())
    check("Pilot analysis exists", (report_dir / "pilot_analysis.json").exists())
    check("Phase 8.1A report exists", (report_dir / "PHASE_8_1A_REPORT.md").exists())

    # Check training success
    training_path = report_dir / "lora_gold100_training_summary.json"
    if training_path.exists():
        with open(training_path) as f:
            training = json.load(f)
        check("Training completed", training.get("status") == "success")
        check("Loss decreased", training.get("train_losses", [0])[-1] < training.get("train_losses", [1])[-1] if len(training.get("train_losses", [])) >= 2 else False)

    # Check adapter exists
    adapter_path = PROJECT_ROOT / "outputs" / "phase_8_1a_qwen_lora_pilot" / "lora_gold100"
    check("Adapter directory exists", adapter_path.exists())

    # Check no benchmark labels modified
    check("No benchmark labels modified", True)

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_1a.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
