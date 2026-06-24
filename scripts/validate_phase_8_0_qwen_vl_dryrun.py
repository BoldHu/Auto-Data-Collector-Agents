"""Phase 8.0: Validate outputs.

Usage:
    python scripts/validate_phase_8_0_qwen_vl_dryrun.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_0_qwen_vl_dryrun"
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

    check("Environment check exists", (report_dir / "environment_check.json").exists())
    check("Load test results exist", (report_dir / "qwen_vl_load_test_results.json").exists())
    check("Test samples exist", (PROJECT_ROOT / "data" / "evaluation" / "phase_8_0_qwen_vl" / "qwen_vl_test_samples.jsonl").exists())
    check("Smoke test report exists", (report_dir / "qwen_vl_smoke_test_report.json").exists())
    check("SFT compatibility exists", (report_dir / "sft_v4_qwen_compatibility.json").exists())
    check("LoRA dry-run exists", (report_dir / "qwen_vl_lora_dryrun_report.json").exists())
    check("Cloud migration exists", (report_dir / "cloud_migration_manifest.json").exists())
    check("Training plan exists", (PROJECT_ROOT / "reports" / "paper_ready" / "phase_8_1_qwen_training_plan.md").exists())
    check("Phase 8.0 report exists", (report_dir / "PHASE_8_0_REPORT.md").exists())

    # Check if model was downloaded
    model_path = PROJECT_ROOT / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct"
    model_downloaded = model_path.exists() and len(list(model_path.glob("*"))) > 5
    check("Model downloaded", model_downloaded)

    for c in checks:
        print(c)

    print(f"\nPassed: {passed}/{passed + failed}")
    print(f"Failed: {failed}/{passed + failed}")

    with open(report_dir / "validation_phase_8_0.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()
